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
from srcs.model.tab_model import TabNetModel

class MultimodalModel(nn.Module):
    def __init__(self, tabular_input_dim, text_model_name="sberbank-ai/ruBERT-base"):
        super().__init__()
        tabnet_params=None
        self.tabnet = TabNetModel(
            input_dim=tabular_input_dim,
        )
        self.text_net = TextNet(text_model_name)
        self.image_net = ImageNet()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True
        )
        self.proj= nn.Linear(768, 512)
        # ИСПРАВЛЕНО: Скорректирована размерность входа в классификатор
        # 64 (tab) + 512 (text) + 512 (image) + 512 (attention) = 1600
        self.classifier = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        
    def forward(self, image, input_ids, attention_mask, tabular):
        tab_embeddings, tab_m_loss = self.tabnet(tabular)
        text_out = self.text_net(input_ids, attention_mask)
        img_out = self.image_net(image)
        #text_out = self.proj(text_out)
        text_img_combined = torch.stack([text_out, img_out], dim=1)
        attended, _ = self.cross_attention(text_img_combined, text_img_combined, text_img_combined)
        attended_combined = attended.mean(dim=1)
        
        combined = torch.cat([tab_embeddings, text_out, img_out, attended_combined], dim=1)
        
        output = self.classifier(combined)
        return output.squeeze(-1), tab_m_loss


def get_model(tabular_input_dim, text_model_name="sberbank-ai/ruBERT-base"):
    model = MultimodalModel(tabular_input_dim, text_model_name)
    
    return model