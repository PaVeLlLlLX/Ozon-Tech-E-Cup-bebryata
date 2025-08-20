import os
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from transformers import AutoTokenizer, AutoModel

class TextNet(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #print("Text features", outputs.last_hidden_state[:, 0, :])
        return outputs.last_hidden_state[:, 0, :]  #[batch_size, 768]
