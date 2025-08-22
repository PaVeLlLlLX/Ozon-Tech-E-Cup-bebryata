import os
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
#from srcs.model.tab_model import TabularNet
from srcs.model.tab_model import CatBoostTabularModel, TabNetModel
from srcs.model.text_model import TextNet
from srcs.model.image_model import ImageNet



class MultimodalModel(nn.Module):
    def __init__(self, tabular_input_dim, text_model_name="cointegrated/rubert-tiny2"):
        super().__init__()
        self.kitty = CatBoostTabularModel()
        #self.tabular_net = TabularNet(tabular_input_dim)
        tabnet_params=None
        self.tabnet = TabNetModel(
            input_dim=tabular_input_dim,
            output_dim=64,
            **(tabnet_params or {})
        )
        self.text_net = TextNet(text_model_name)
        self.image_net = ImageNet()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(2368, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.linear = nn.Linear()
        
    def forward(self, image, input_ids, attention_mask, tabular):
        tab_out = self.tabnet(tabular)
        #tab_out = self.tabular_net(tabular)
        text_out = self.text_net(input_ids, attention_mask)
        img_out = self.image_net(image)

        text_img_combined = torch.stack([text_out, img_out], dim=1)  # (batch_size, 2, 128)
        attended, _ = self.cross_attention(text_img_combined, text_img_combined, text_img_combined)
        attended_combined = attended.mean(dim=1)  # (batch_size, 128)
        
        combined = torch.cat([tab_out, text_out, img_out, attended_combined], dim=1)
        
        output = self.classifier(combined)
        return output.squeeze(-1)
        combined = torch.cat([tab_out, text_out, img_out], dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(-1)


def get_model(tabular_input_dim, text_model_name):
    model = MultimodalModel(tabular_input_dim, text_model_name)
    
    return model