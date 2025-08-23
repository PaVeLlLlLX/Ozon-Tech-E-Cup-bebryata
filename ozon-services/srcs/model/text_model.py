import os
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18
from transformers import AutoTokenizer, AutoModel

class TextNet(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.projection = nn.Linear(self.bert.config.hidden_size, 512)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_embs = sum_embeddings / sum_mask
        
        # ИЗМЕНЕНИЕ: Применяем проекционный слой
        projected_embs = self.projection(mean_pooled_embs)
        
        return projected_embs