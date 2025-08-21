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