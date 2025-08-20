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
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
    
    def forward(self, x):
        x = self.cnn(x)  #[batch_size, 512]
        #print("Image features", x)
        return x