import os
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from srcs.model.tab_model import TabularNet
from srcs.model.text_model import TextNet
from srcs.model.image_model import ImageNet


class MultimodalModel(nn.Module):
    def __init__(self, tabular_input_dim, text_model_name="cointegrated/rubert-tiny2"):
        super().__init__()
        self.tabular_net = TabularNet(tabular_input_dim)
        self.text_net = TextNet(text_model_name)
        self.image_net = ImageNet()
        
        self.classifier = nn.Sequential(
            nn.Linear(64 + 312 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
    
    def forward(self, image, input_ids, attention_mask, tabular):
        tab_out = self.tabular_net(tabular)
        text_out = self.text_net(input_ids, attention_mask)
        img_out = self.image_net(image)
        
        combined = torch.cat([tab_out, text_out, img_out], dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(-1)


def get_model(tabular_input_dim, text_model_name):
    model = MultimodalModel(tabular_input_dim, text_model_name)
    
    return model