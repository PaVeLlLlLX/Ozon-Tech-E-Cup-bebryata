import os
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from transformers import BertModel
from torchvision.models import resnet18


class ImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.cnn = resnet18(pretrained=True)
        # self.cnn.fc = nn.Identity()

        self.model = torchvision.models.efficientnet_v2_m(weights=torchvision.models.EfficientNet_V2_M_Weights.DEFAULT)
    
        for param in self.model.parameters():
            param.requires_grad = False


        # num_features = self.model.classifier[1].in_features
        self.proj = nn.Linear(1280, 512)
        self.model.classifier = nn.Identity()

        for param in self.model.classifier.parameters():
            param.requires_grad =True
    def forward(self, x):
        x = self.model(x)  #[batch_size, 1280]
        #print("Image features", x)
        x = self.proj(x)
        return x
    