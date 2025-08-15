import cv2
import numpy as np
import torch
import re
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import List
from PIL import Image


transform_train = A.Compose([
    A.Rotate(limit=5, p=0.3),
    A.RandomFog(fog_coef_range=(0.1, 0.2), alpha_coef=0.1, p=0.4),
    A.Perspective(scale=(0.05, 0.15), p=0.25, keep_size=True),
    A.OneOf([
    A.CLAHE(p=0.25),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.RandomGamma(gamma_limit=(90, 110), p=0.3), 
    ], p=0.8),
    A.GaussNoise(std_range=(0.05, 0.1), p=0.4),
    A.RandomSunFlare(
        flare_roi=(0, 0, 1, 1),
        angle_range=(0,1),
        src_radius=100,
        src_color=(255, 255, 255),
        num_flare_circles_range = (1, 4),
        p=0.2
    ),

    A.Spatter(
        intensity=(0.05,0.1),
        p=0.2
    ),
    A.ColorJitter(
        brightness=0.05,
        contrast=0.05,
        saturation=0.05,
        hue=0.05,
        p=0.5
    ),
])

class OzonDataset(Dataset):
    def __init__(self, images_dir, txt_paths_file, tabular_data, texts, images, labels, tokenizer, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.texts = texts
        self.tabular_data = tabular_data
        self.images = images
        self.labels = labels
        self.tokenizer = tokenizer
        self.image_labels = self._load_image_labels(txt_paths_file)
        self.alphabet, self.char_to_idx = self._build_alphabet()

    def _load_image_labels(self, txt_paths_file):
        try:
            with open(txt_paths_file, "r", encoding="utf-8") as f:
                return f.read().split("РАЗДЕЛИТЕЛЬ")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки файла {txt_paths_file}: {str(e)}")

    
    def _build_alphabet(self):
        # Формирование словаря для эмбеддингов итд (NLP)
        unique_chars = set()
        
        for i, label in enumerate(self.image_labels):
            chars = set(label)
            unique_chars.update(chars)
        uc = unique_chars.copy()
        for char in unique_chars:
            if char in "":
                uc.remove(char)

        unique_chars = uc

        sorted_chars = sorted(list(unique_chars))

        special_tokens = ['@', '<SOS>', '<EOS>'] # '@' - PAD
        for sp in special_tokens:
            if sp in sorted_chars:
                sorted_chars.remove(sp)

        final_alphabet = special_tokens + sorted_chars
        char_mapping = {char: idx for idx, char in enumerate(final_alphabet)}

        self.pad_token_id = char_mapping['@']
        self.sos_token_id = char_mapping['<SOS>']
        self.eos_token_id = char_mapping['<EOS>']
        print(f"PAD ID: {self.pad_token_id}, SOS ID: {self.sos_token_id}, EOS ID: {self.eos_token_id}")
        print(f"Размер словаря: {len(final_alphabet)}")

        return final_alphabet, char_mapping

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        tabular = torch.FloatTensor(self.tabular_data[idx])
        text = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        image = self.transform(self.images[idx])
        label = torch.FloatTensor([self.labels[idx]])
        
        return {
            "tabular": tabular,
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "image": image,
            "label": label
        }

    def _load_image(self, path):
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки изображения {path}: {str(e)}")

    def _encode_text(self, text):
        return [self.char_to_idx.get(char, self.pad_token_id) for char in text]


def get_ozon_dataloader(
        path_train, path_val, txt_paths_file, tabular_data, texts, images, labels, tokenizer, batch_size, shuffle=True, num_workers=8,
    ):
    train_dataset = OzonDataset(path_train, txt_paths_file, tabular_data, texts, images, labels, tokenizer, transform=transform_train)
    val_dataset = OzonDataset(path_val, txt_paths_file, tabular_data, texts, images, labels, tokenizer)
    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)


def plot_augmented_samples(dataset, n=5):
    plt.figure(figsize=(20,10))
    for i in range(n):
        img, _ = dataset[np.random.randint(len(dataset))]
        plt.subplot(1,n,i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')


def get_ozon_test_dataloader(
        path_test, txt_paths_file, tabular_data, texts, images, labels, tokenizer, batch_size, num_workers=8):
    test_dataset = OzonDataset(path_test, txt_paths_file, tabular_data, texts, images, labels, tokenizer)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers
    }
    return DataLoader(test_dataset, **loader_args)
