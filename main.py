from tabnet import *
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    X, y, cat_idxs, cat_dims = TabNetModel.load_secondary_mushroom()

    # 2) split 60/40:
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    # 3) split holdout 50/50 → walidacja/test (czyli 20%/20% całości):
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_holdout, y_holdout,
        test_size=0.5,
        random_state=42,
        stratify=y_holdout
    )

    # 4) Sprawdź rozmiary:
    print("X_train_full:", X_train.shape)  # ~60% danych
    print("X_valid:", X_valid.shape)            # ~20% danych
    print("X_test:", X_test.shape)              # ~20% danych

    model = TabNetModel(
        input_dim = X_train.shape[1],   
        output_dim = 2,                   
        cat_idxs = cat_idxs,
        cat_dims = cat_dims,
        n_d = 8,
        n_a = 8,
        n_steps = 3,
        gamma = 1.3,
        lambda_sparse = 1e-3,
        mask_type = "entmax",
        cat_emb_dim = 1,
        n_independent = 2,
        n_shared = 2,
        epsilon = 1e-15,
        seed = 1,
        momentum = 0.02,
        optimizer_fn = torch.optim.Adam,
        optimizer_params = {"lr": 2e-2, "weight_decay": 1e-4},
        scheduler_fn = torch.optim.lr_scheduler.StepLR,
        scheduler_params = {"step_size": 5, "gamma": 0.9},
        verbose = 1,
        device_name = "auto",
        n_shared_decoder = 1,
        n_indep_decoder = 1
    )

    # Trening z walidacją:
    model.train_model(
        X_train = X_train,
        y_train = y_train,
        X_valid = X_valid,
        y_valid = y_valid,
        max_epochs = 100,
        batch_size = 512,
        virtual_batch_size = 64,
        loss_fn = nn.CrossEntropyLoss(),
        eval_metric = ["accuracy"],
        patience = 0,
        weights = 1
    )

    # Po treningu:
    preds = model.predict(X_test)
    test_acc = (preds == y_test).mean()
    print("Test accuracy:", test_acc)
