import os
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from srcs.model.tab_model import TabularNet
from srcs.model.text_model import TextNet
from srcs.model.image_model import ImageNet
from srcs.model.tab_model import TabNetModel


class PositionalEncoding(nn.Module):
    def __init__(self, num_modalities, d_model):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(num_modalities, d_model))
        
    def forward(self, x):
        return x + self.pos_encoding.unsqueeze(0) # [batch_size, num_modalities, d_model]
    

class ModalityProjection(nn.Module):
    """Проекция каждой модальности в общее пространство"""
    def __init__(self, input_dims, d_model):
        super().__init__()
        self.projection_text = nn.Linear(512, d_model)
        self.projection_image = nn.Linear(512, d_model)
        self.projection_tabular = nn.Linear(128, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, text, image, tabular):
        text_proj = self.dropout(F.relu(self.projection_text(text)))
        image_proj = self.dropout(F.relu(self.projection_image(image)))
        tabular_proj = self.dropout(F.relu(self.projection_tabular(tabular)))
        
        # Объединяем в последовательность (batch_size, 3, d_model)
        combined = torch.stack([text_proj, image_proj, tabular_proj], dim=1)
        return self.layer_norm(combined)


class MultimodalTransformerClassifier(nn.Module):
    def __init__(self, input_dims, d_model=256, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # Проекция модальностей
        self.modality_projection = ModalityProjection(input_dims, d_model)
        
        # Позиционное кодирование для модальностей
        self.pos_encoder = PositionalEncoding(3, d_model)  # 3 модальности
        
        # Трансформер энкодер
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # [CLS] токен
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.d_model = d_model
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, text, image, tabular):
        # Проектируем модальности в общее пространство
        x = self.modality_projection(text, image, tabular)  # (batch_size, 3, d_model)
        
        # Добавляем позиционное кодирование
        x = self.pos_encoder(x)
        
        # Добавляем CLS токен
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, 4, d_model)
        
        # Пропускаем через трансформер
        transformer_output = self.transformer_encoder(x)  # (batch_size, 4, d_model)
        
        cls_output = transformer_output[:, 0, :]  # (batch_size, d_model)
        
        output = self.classifier(cls_output)
        return output.squeeze()
    

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
        # self.classifier = nn.Sequential(
        #     nn.Linear(1600, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.LayerNorm(512),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 1),
        # )
        d_model=256
        nhead=8
        num_layers=3
        dim_feedforward=512 
        dropout=0.1
        self.modality_projection = ModalityProjection(tabular_input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(3, d_model) 
        
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # [CLS] токен
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.d_model = d_model
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, image, input_ids, attention_mask, tabular):
        tab_embeddings, tab_m_loss = self.tabnet(tabular)
        text_out = self.text_net(input_ids, attention_mask)
        img_out = self.image_net(image)
        x = self.modality_projection(text_out, img_out, tab_embeddings)  # (batch_size, 3, d_model)
        
        x = self.pos_encoder(x)
        
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, 4, d_model)
        
        transformer_output = self.transformer_encoder(x)  # (batch_size, 4, d_model)
        
        cls_output = transformer_output[:, 0, :]  # (batch_size, d_model)
        
        output = self.classifier(cls_output)
        return output.squeeze(-1), tab_m_loss
        # #text_out = self.proj(text_out)
        # text_img_combined = torch.stack([text_out, img_out], dim=1)
        # attended, _ = self.cross_attention(text_img_combined, text_img_combined, text_img_combined)
        # attended_combined = attended.mean(dim=1)
        
        # combined = torch.cat([tab_embeddings, text_out, img_out, attended_combined], dim=1)
        
        # output = self.classifier(combined)
        # return output.squeeze(-1), tab_m_loss


def get_model(tabular_input_dim, text_model_name="sberbank-ai/ruBERT-base"):
    model = MultimodalModel(tabular_input_dim, text_model_name)
    
    return model