import logging
import os
from typing import List

import cv2
import numpy as np
import torch
from srcs.model.model import get_model

class FakeClassifier:
    def __init__(self):
        self.model_ = None
        model_fname = os.path.join(os.path.dirname(__file__), 'model.pth')
        # Check if the model file exists
        if not os.path.isfile(model_fname):
            raise IOError(f'The file "{model_fname}" does not exist!')
        # Load the model
        checkpoint = torch.load(model_fname)
        self.model_ = get_model(tabular_input_dim=100)
        self.model_.load_state_dict(checkpoint)

        # Set up device and model
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_.eval().to(self.device_)

    def _preprocess(self, img):
        #img = transform(image=img)["image"]
        img = cv2.resize(img, (200, 200))
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img).float().to(self.device_)

    def predict(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = self._preprocess(image)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model_(image_tensor)
        return outputs

    def predict_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        image_tensors = torch.stack([self._preprocess(image) for image in images])
        with torch.no_grad():
            outputs = self.model_(image_tensors)
        return outputs
