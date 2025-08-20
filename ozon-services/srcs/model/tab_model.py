import optuna
import joblib
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

'''
import os
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from transformers import BertModel
from torchvision.models import resnet18

class TabularNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)
'''