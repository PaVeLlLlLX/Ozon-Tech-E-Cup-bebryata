import os
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from transformers import BertModel
from torchvision.models import resnet18
from pytorch_tabnet.tab_network import TabNet
from sklearn.model_selection import train_test_split

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

class TabNetModel(nn.Module):
    def __init__(self, input_dim, output_dim=64, n_d=32, n_a=32, n_steps=5, 
                 gamma=1.3, n_independent=2, n_shared=2, momentum=0.02):
        super().__init__()

        group_matrix = torch.eye(input_dim).to('cuda')
        self.register_buffer('group_attention_matrix', group_matrix)
        
        self.tabnet = TabNet(
            input_dim=input_dim,
            output_dim=n_d + n_a,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            momentum=momentum,
            mask_type='sparsemax',
            group_attention_matrix=self.group_attention_matrix
        )
        
        self.embedding_projection = nn.Sequential(
            nn.Linear(n_d + n_a, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor):
        x_features, M_loss = self.tabnet(x)
        
        embeddings = self.embedding_projection(x_features)
        
        return embeddings, M_loss