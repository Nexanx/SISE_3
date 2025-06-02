from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ucimlrepo import fetch_ucirepo
import pickle
import zipfile
import io
import os


class TabNetModel:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cat_idxs: list,
        cat_dims: list,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        lambda_sparse: float = 1e-3,
        mask_type: str = 'entmax',
        cat_emb_dim: int = 1,
        n_independent: int = 2,
        n_shared: int = 2,
        momentum: float = 0.02,
        optimizer_fn: object = torch.optim.Adam,
        optimizer_params: dict = None,
        scheduler_fn: object = torch.optim.lr_scheduler.StepLR,
        scheduler_params: dict = None,
        verbose: int = 1,
        device_name: str = 'auto',
        n_shared_decoder: int = 1,
        n_indep_decoder: int = 1
    ):
        """
        Tworzy TabNetClassifier z podanymi hiperparametrami.

        Parametry modelu:
          - input_dim (int): liczba cech (kolumn) w X.
          - output_dim (int): liczba klas (np. 2 dla jadalne/trujące).
          - cat_idxs (list[int]): indeksy kolumn kategorycznych (0-based).
          - cat_dims (list[int]): liczba unikalnych wartości dla każdej kolumny kategorycznej.
          - n_d (int): szerokość warstwy decyzyjnej (domyślnie 8).
          - n_a (int): szerokość warstwy uwag (attention embedding), zwykle n_d = n_a.
          - n_steps (int): liczba kroków w architekturze (zwykle 3–10).
          - gamma (float): współczynnik ponownego wykorzystania cech (1.0–2.0, domyślnie 1.3).
          - lambda_sparse (float): współczynnik strat sparsity, domyślnie 1e-3.
          - mask_type (str): rodzaj maski – "sparsemax" lub "entmax" (domyślnie 'entmax').
          - cat_emb_dim (int): wymiar embeddingu dla każdej cechy kategorycznej (domyślnie 1).
          - n_independent (int): liczba niezależnych bloków GLU na każdym kroku (1–5, domyślnie 2).
          - n_shared (int): liczba współdzielonych bloków GLU na każdym kroku (1–5, domyślnie 2).
          - epsilon (float): małe epsilon do stabilności (zwykle 1e-15, nie ruszać).
          - seed (int): ziarno losowe do reprodukowalności (domyślnie 0).
          - momentum (float): momentum dla BatchNorm (0.01–0.4, domyślnie 0.02).
          - clip_value (float | None): jeśli ustawione, przycina gradient do ±clip_value.
          - optimizer_fn (torch.optim class): funkcja optymalizatora (domyślnie torch.optim.Adam).
          - optimizer_params (dict): słownik parametrów do przekazania do optimizer_fn (np. {"lr":2e-2, "weight_decay":1e-4}).
          - scheduler_fn (torch.optim.lr_scheduler class | None): funkcja harmonogramu LR (domyślnie StepLR).
          - scheduler_params (dict): parametry dla scheduler_fn (np. {"step_size":10, "gamma":0.9}).
          - model_name (str): nazwa modelu do zapisu na dysk (domyślnie 'TabNet').
          - verbose (int): poziom szczegółowości logów (0 lub 1).
          - device_name (str): 'cpu', 'gpu' lub 'auto' (domyślnie 'auto').
          - n_shared_decoder (int): dla TabNetPretrainer – liczba współdzielonych bloków w dekoderze.
          - n_indep_decoder (int): dla TabNetPretrainer – liczba niezależnych bloków w dekoderze.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 2e-2}
        if scheduler_params is None:
            scheduler_params = {"step_size": 10, "gamma": 0.9}

        self.model = TabNetClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            mask_type=mask_type,
            n_independent=n_independent,
            n_shared=n_shared,
            momentum=momentum,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            scheduler_fn=scheduler_fn,
            scheduler_params=scheduler_params,
            verbose=verbose,
            device_name=device_name,
            n_shared_decoder=n_shared_decoder,
            n_indep_decoder=n_indep_decoder
        )

    def load_secondary_mushroom():
        """
        Pobiera „Secondary Mushroom Dataset” z UCI (ID=848) przez ucimlrepo
        i zwraca od razu:
        - X: NumPy array (float32 dla cech numerycznych i int32 dla kodów kategorii)
        - y: NumPy array (int64; 0='e', 1='p')
        - cat_idxs: lista indeksów kolumn kategorycznych (0-based)
        - cat_dims: lista wymiarów (liczba unikalnych wartości) dla kolumn kategorycznych
        """
        # 1) Pobieramy cały zbiór (ID=848) przez ucimlrepo
        ds = fetch_ucirepo(id=848)

        # 2) Określamy DataFrame cech i potencjalny DataFrame/Series etykiet
        X_df = ds.data.features.copy()
        y_obj = ds.data.targets.copy()

        # 3) Jeśli y_obj to DataFrame z jedną kolumną, zamieniamy na Series
        if isinstance(y_obj, pd.DataFrame):
            y_ser = y_obj.squeeze()
        else:
            y_ser = y_obj

        # 4) Kodujemy y: 'e'→0, 'p'→1
        y = y_ser.map({'e': 0, 'p': 1}).values.astype(np.int64)
       
        # 5) Znajdujemy kolumny kategoryczne i tworzymy cat_idxs oraz cat_dims
        cat_idxs = []
        cat_dims = []
        for idx, col in enumerate(X_df.columns):
            if X_df[col].dtype == object:
                # traktujemy dtype=object jako category
                X_df[col] = X_df[col].astype('category')
                cat_idxs.append(idx)
                cat_dims.append(len(X_df[col].cat.categories))
                # kodujemy wartości na kody int32 i usuwamy ewentualne -1
                codes = X_df[col].cat.codes
                codes = codes.where(codes >= 0, 0).astype(np.int32)
                X_df[col] = codes
            else:
                # kolumna numeryczna → rzutujemy na float32
                X_df[col] = X_df[col].astype(np.float32)

        # 6) Konwertujemy na NumPy array
        X = X_df.values  # dtype: float32 i int32
        return X, y, cat_idxs, cat_dims

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
        max_epochs: int = 200,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        loss_fn=None,
        eval_metric: list = None,
        patience: int = 10,
        weights = 0,
        drop_last: bool = False,
        callbacks: list = None
    ):
        """
        Trenuje TabNetClassifier, wywołując wbudowaną metodę .fit(...).

        Parametry .fit:
          - X_train, y_train: numpy.array lub scipy.sparse.csr_matrix (float32, int64).
          - X_valid, y_valid: opcjonalne dane walidacyjne.
          - max_epochs (int): maksymalna liczba epok (domyślnie 200).
          - batch_size (int): liczba próbek w batchu (domyślnie 1024).
          - virtual_batch_size (int): wielkość ghost‐batch dla BatchNorm (domyślnie 128).
          - loss_fn: funkcja strat (domyślnie CrossEntropy dla klasyfikacji).
          - eval_metric (list[str]): lista nazw metryk, np. ["accuracy"]. Ostatnia
            metryka jest używana do early‐stopping.
          - patience (int): liczba epok bez poprawy metryki walidacyjnej, zanim przerwie trenowanie.
          - weights (int lub dict): do resamplingu klas (0→brak, 1→inverse class freq, dict→custom).
          - drop_last (bool): czy odrzucać ostatni nieduży batch (domyślnie False).
          - num_workers (int): liczba procesów dla DataLoadera (domyślnie 0).
          - mask_type (str): "sparsemax" lub "entmax" (można zmienić w trakcie trenowania).
          - callbacks (list[func]): lista dodatkowych callbacków (opcjonalna).
        """
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        if eval_metric is None:
            eval_metric = ["accuracy"]
        eval_set = []
        eval_name = []
        eval_metric_list = []
        if X_valid is not None and y_valid is not None:
            eval_set.append((X_valid, y_valid))
            eval_name.append("valid")
            eval_metric_list = eval_metric

        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=eval_metric_list,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            weights=weights,
            loss_fn=loss_fn,
            drop_last=drop_last,
            callbacks=callbacks
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Zwraca numpy.array z przewidywanymi etykietami (0,1,...).
        """
        return self.model.predict(X_test)
    
    def save(self,path:str) -> None:
        self.model.save_model("weights")
        pickle_bytes = pickle.dumps(self)

        with zipfile.ZipFile(path, "w") as z:
            z.write("weights.zip")
            z.writestr("config.pkl",pickle_bytes)
        
        os.remove("weights.zip")
    
    @staticmethod
    def load(path:str):
        with zipfile.ZipFile(path, "r") as z:
            
            pickle_bytes = z.read("config.pkl")
            obj = pickle.loads(pickle_bytes)
      
            weights_bytes = z.read("weights.zip")
            weights_buffer = io.BytesIO(weights_bytes)

        obj.model.load_model(weights_buffer)
               
        return obj