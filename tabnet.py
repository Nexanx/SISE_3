from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

class TabNetModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TabNetModel, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = TabNetClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            lambda_sparse=0.001,
            mask_type='entmax',
        )

    def forward(self, x):
        logits, maskLoss = self.model(x)
        return logits, maskLoss
    

    def train_model(self, X_train, y_train, X_val=None, y_val=None,
                    epochs=1000, batch_size=256, lambda_sparse=0.001,):

        # 1. Tworzymy DataLoader (zmieniamy numpy → tensor dopiero przy batchu)
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits, mask_loss = self.forward(batch_X)
                clf_loss = self.criterion(logits, batch_y)
                loss = clf_loss + lambda_sparse * mask_loss

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
                _, preds = torch.max(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_X.size(0)

            self.scheduler.step()

            train_loss = epoch_loss / total
            train_acc = correct / total

            print(f'Epoch {epoch}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

    def predict(self, x, device='cuda'):
        self.model.eval()
        testTensor = torch.tensor(x, dtype=torch.float32).to(device)
        logits, _ = self.model(testTensor)
        _  , predictions = torch.max(logits, 1)
        return predictions.gpu().np()
    
    def load_mushroom_data(file_path, label_column='class', delimiter=';'):
 
        df = pd.read_csv(file_path, sep=delimiter)
        
        if label_column not in df.columns:
            raise ValueError(f"Brak kolumny etykiet '{label_column}' w pliku.")
        
        y_series = df[label_column]
        X_df = df.drop(columns=[label_column])
        
        cat_idxs = []
        cat_dims = []
        for idx, col in enumerate(X_df.columns):
            if X_df[col].dtype == object:
                X_df[col] = X_df[col].astype('category')
                cat_idxs.append(idx)
                cat_dims.append(len(X_df[col].cat.categories))
                X_df[col] = X_df[col].cat.codes
            else:
                try:
                    X_df[col] = X_df[col].astype(np.float32)
                except:
                    X_df[col] = X_df[col].astype('category')
                    cat_idxs.append(idx)
                    cat_dims.append(len(X_df[col].cat.categories))
                    X_df[col] = X_df[col].cat.codes
        
        y_series = y_series.astype('category')
        y = y_series.cat.codes.values.astype(np.int64)
        
        X = X_df.values.astype(np.float32)
        
        return X, y, cat_idxs, cat_dims
    

    