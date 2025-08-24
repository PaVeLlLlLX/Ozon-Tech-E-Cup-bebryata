import cv2
import numpy as np
import torch
import re
import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import List
from PIL import Image


transform_train = A.Compose([
    A.Resize(384, 384), 
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=12, max_height=40, max_width=40, min_holes=8, fill_value=0, p=0.7),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MotionBlur(blur_limit=(3, 7), p=0.5),
    ], p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

transform_val = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class OzonDataset(Dataset):
    def __init__(self, df, images_dir, tokenizer, tabular_cols, target_col, id_col, text_cols, transform=None):
        self.df = df
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.tabular_cols = tabular_cols
        self.target_col = target_col
        self.id_col = id_col
        self.text_cols = text_cols

        self.image_paths = self.df[self.id_col].apply(lambda x: self.images_dir / f"{x}.png").tolist()
        
        self.texts = self.df[self.text_cols].apply(
            lambda row: ' [SEP] '.join(row.values.astype(str)), axis=1
        ).tolist()
         
        self.tabular_features = torch.FloatTensor(self.df[self.tabular_cols].values)
        self.labels = torch.FloatTensor(self.df[self.target_col].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = np.zeros((384, 384, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)['image']

        text = self.texts[idx]
        tokenized_text = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )

        tabular = self.tabular_features[idx]
        label = self.labels[idx].unsqueeze(0)
        
        return {
            "image": image,
            "input_ids": tokenized_text["input_ids"].squeeze(0),
            "attention_mask": tokenized_text["attention_mask"].squeeze(0),
            "tabular": tabular,
        }, label


def get_dataloaders(train_df, val_df, images_dir, tokenizer, tabular_cols, target_col, id_col, text_cols, batch_size, num_workers=None):
    train_dataset = OzonDataset(
        df=train_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        tabular_cols=tabular_cols,
        target_col=target_col,
        id_col=id_col,
        text_cols=text_cols,
        transform=transform_train
    )
    val_dataset = OzonDataset(
        df=val_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        tabular_cols=tabular_cols,
        target_col=target_col,
        id_col=id_col,
        text_cols=text_cols,
        transform=transform_val
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return train_loader, val_loader


def get_test_dataloader(test_df, images_dir, tokenizer, tabular_cols, target_col, id_col, text_cols, batch_size, num_workers=None):
    test_dataset = OzonDataset(
        df=test_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        tabular_cols=tabular_cols,
        target_col=target_col,
        id_col=id_col,
        text_cols=text_cols,
        transform=transform_val
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    return test_loader


def plot_augmented_samples(dataset, n=5):
    plt.figure(figsize=(20,10))
    for i in range(n):
        img, _ = dataset[np.random.randint(len(dataset))]
        plt.subplot(1,n,i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')