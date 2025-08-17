import os
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from transformers import BertModel
from torchvision.models import resnet18
from tab_model import TabularNet
from text_model import TextNet
from image_model import ImageNet


class MultimodalModel(nn.Module):
    def __init__(self, tabular_input_dim, text_model_name="bert-base-uncased"):
        super().__init__()
        self.tabular_net = TabularNet(tabular_input_dim)
        self.text_net = TextNet(text_model_name)
        self.image_net = ImageNet()
        
        self.classifier = nn.Sequential(
            nn.Linear(64 + 768 + 2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, tabular, input_ids, attention_mask, image):
        tab_out = self.tabular_net(tabular)  #[batch_size, 64]
        text_out = self.text_net(input_ids, attention_mask)  #[batch_size, 768]
        img_out = self.image_net(image)  #[batch_size, 2048]
        
        combined = torch.cat([tab_out, text_out, img_out], dim=1)  #[batch_size, 64+768+2048]
        return self.classifier(combined)


def get_model(tabular_input_dim):
    model = MultimodalModel(tabular_input_dim)
    
    return model