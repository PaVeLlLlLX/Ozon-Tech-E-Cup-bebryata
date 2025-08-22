import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import hydra
from omegaconf import DictConfig
from srcs.model.text_model import TextNet
from srcs.model.image_model import ImageNet
from srcs.data_loader.data_loaders import OzonDataset, transform_train, transform_val
from srcs.utils import instantiate, get_logger, is_master
SEED = 123

class NeuralFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_net = TextNet("cointegrated/rubert-tiny2")
        self.image_net = ImageNet()
        
        for param in self.text_net.parameters():
            param.requires_grad = False
        for param in self.image_net.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(312 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, image, input_ids, attention_mask):
        with torch.no_grad():
            text_out = self.text_net(input_ids, attention_mask)
            img_out = self.image_net(image)
        
        combined = torch.cat([text_out, img_out], dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(-1)

@hydra.main(version_base=None, config_path='conf/', config_name='train')
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    
    df = pd.read_csv("../data/train_with_folds.csv", index_col='id')
    
    VAL_FOLD = 0
    train_df = df[df['fold'] != VAL_FOLD]
    val_df = df[df['fold'] == VAL_FOLD]
    # train_df, val_df = train_test_split(
    #     df,
    #     test_size=config.data.val_size,
    #     random_state=SEED,
    #     stratify=df[config.data.target_col]
    # )
    train_loader, val_loader = instantiate(config.data_loader, train_df=train_df, val_df=val_df)

    model = NeuralFeatureExtractor().to(device)
    
    optimizer = AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=1e-3)
    
    pos_weight = torch.tensor([4.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0
    
    for epoch in range(5):
        model.train()
        for batch_data, target in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            image = batch_data['image'].to(device)
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            target = target.to(device).float()

            optimizer.zero_grad()
            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_data, target in tqdm(val_loader, desc=f"Epoch {epoch+1} Valid"):
                image, input_ids, attention_mask = batch_data['image'].to(device), batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device)
                
                logits = model(image, input_ids, attention_mask)
                preds = torch.round(torch.sigmoid(logits))
                
                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

        f1 = f1_score(torch.cat(all_targets), torch.cat(all_preds))
        print(f"Epoch {epoch+1}, Validation F1-score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            print(f"Saving to best_neural_head.pth")
            torch.save(model.state_dict(), "best_neural_head.pth")

if __name__ == '__main__':
    main()
