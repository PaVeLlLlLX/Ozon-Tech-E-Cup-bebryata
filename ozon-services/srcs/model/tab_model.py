import optuna
import joblib
import pandas as pd
import torch
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn


class CatBoostTabularModel:
    def __init__(self, artifacts_dir="artifacts", seed=42, n_trials=30):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        self.seed = seed
        self.n_trials = n_trials
        self.model = None

    def _objective(self, trial, X_train, y_train, X_val, y_val, cat_features):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
            "random_seed": self.seed,
            "loss_function": "Logloss",
            "eval_metric": "F1",
            "verbose": False,
            "task_type": "GPU",  # можно поменять на "CPU"
        }
        model = CatBoostClassifier(**params)
        # В этой задаче категориальных нет, используем только числовые признаки
        cat_features = []
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=False
        )
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    def fit(self, df: pd.DataFrame, target_col="resolution", id_col="id"):
        y = df[target_col]
        X = df.drop(columns=[target_col, id_col, 'description', 'name_rus'])

        # В этой задаче категориальных нет, используем только числовые признаки
        cat_features = []

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.seed
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self._objective(t, X_train, y_train, X_val, y_val, cat_features),
                       n_trials=self.n_trials)

        best_params = study.best_params
        best_params.update({
            "random_seed": self.seed,
            "loss_function": "Logloss",
            "eval_metric": "F1",
            "task_type": "GPU",
            "verbose": False,
        })

        self.model = CatBoostClassifier(**best_params)
        self.model.fit(X, y, cat_features=cat_features, verbose=False)

        joblib.dump(self.model, self.artifacts_dir / "catboost_model.pkl")
        return self

    def load(self):
        if self.model is None:
            self.model = joblib.load(self.artifacts_dir / "catboost_model.pkl")
        return self.model

    def predict_proba(self, df: pd.DataFrame, id_col="id"):
        self.load()
        X = df.drop(columns=[id_col], errors="ignore")
        return self.model.predict_proba(X)[:, 1]


class TabularNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = self.fc(x)
        #print("Tabular features", x)
        return x
    

class TabNetModel(nn.Module):
    def __init__(self, input_dim, output_dim=64, n_d=8, n_a=8, n_steps=3, 
                 gamma=1.3, n_independent=2, n_shared=2, momentum=0.02):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.tabnet_params = {
            'n_d': n_d,                    # Размерность prediction layer
            'n_a': n_a,                    # Размерность attention layer
            'n_steps': n_steps,            # Количество шагов внимания
            'gamma': gamma,                # Коэффициент масштабирования
            'n_independent': n_independent, # Независимые GLU слои
            'n_shared': n_shared,          # Общие GLU слои
            'momentum': momentum,          # Momentum для batch norm
            'mask_type': 'sparsemax',      # Тип маски
        }
        
        self.tabnet = TabNetClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            **self.tabnet_params
        )
        
        self.embedding_projection = nn.Sequential(
            nn.Linear(n_d, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim)
        )
        self.is_fitted = False
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            max_epochs=100, patience=10, batch_size=1024):
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        self.tabnet.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=128,
            eval_metric=['auc']
        )
        
        self.is_fitted = True
        return self
    
    def get_embeddings(self, x):
        if not self.is_fitted:
            raise RuntimeError("TabNet must be fitted before getting embeddings")
        
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        self.tabnet.network.eval()
        
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            
            output, M_loss = self.tabnet.network(x_tensor)
            
            embeddings = output[0]
            
            if isinstance(embeddings, torch.Tensor):
                embeddings = self.embedding_projection(embeddings)
            else:
                embeddings = self.embedding_projection(torch.tensor(embeddings))
        
        return embeddings
    
    def forward(self, x):
        return self.get_embeddings(x)
    
    def get_attention_masks(self, x):
        if not self.is_fitted:
            raise RuntimeError("TabNet must be fitted before getting attention masks")
        
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        explanations = self.tabnet.explain(x_np)
        return explanations