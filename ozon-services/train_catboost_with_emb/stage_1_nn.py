import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import hydra, logging
from omegaconf import DictConfig
from srcs.model.text_model import TextNet
from srcs.model.image_model import ImageNet
from srcs.data_loader.data_loaders import OzonDataset, transform_train, transform_val
from srcs.utils import instantiate, get_logger, is_master
SEED = 123


logging.basicConfig(filename="stage_1_nn.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

class NeuralFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_net = TextNet("sberbank-ai/ruBERT-base")
        self.image_net = ImageNet()
        combined_dim = 768 + 1280
        for param in self.text_net.parameters():
            param.requires_grad = False
        for param in self.image_net.parameters():
            param.requires_grad = False

        self.attention = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, combined_dim),
            nn.Sigmoid() # Чтобы получить веса от 0 до 1
        )
            
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
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
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        logits = self.classifier(attended_features)
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
    model.load_state_dict(torch.load("best_neural_head_v3_with_transfer.pth"))

    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print(name)

    optimizer = AdamW([
        {"params": model.classifier.parameters(), "lr": 1e-5},
        {"params": model.attention.parameters(), "lr": 1e-5},
        {"params": model.image_net.model.features[8].parameters(), "lr": 2e-5},
        {"params": model.text_net.bert.pooler.parameters(), "lr": 4e-5},
        {"params": model.text_net.bert.encoder.layer[11].parameters(), "lr": 2e-5}
    ], lr=1e-4, weight_decay=1e-4)

    optimizer_2 = AdamW(params=[
        {"params": model.classifier.parameters(), "lr": 5e-4},
        {"params": model.attention.parameters(), "lr": 5e-4},
        {"params": model.image_net.model.features[8].parameters(), "lr": 1e-5},
        {"params": model.image_net.model.features[7].parameters(), "lr": 1e-5},
        {"params": model.text_net.bert.pooler.parameters(), "lr": 1e-5},
        {"params": model.text_net.bert.encoder.layer[10].parameters(), "lr": 1e-5},
        {"params": model.text_net.bert.encoder.layer[11].parameters(), "lr": 3e-6}
    ], weight_decay=1e-4)

    pos_weight = torch.tensor([4.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    print(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')\
        
    # for epoch in range(2):
    #     model.train()
    #     for name, param in model.named_parameters():
    #         if 'text_net.bert.encoder.layer.11.' in name or \
    #         'text_net.bert.pooler.' in name or \
    #         'image_net.model.features.8.' in name:
    #             param.requires_grad = True
    #     for batch_data, target in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
    #         image = batch_data['image'].to(device)
    #         input_ids = batch_data['input_ids'].to(device)
    #         attention_mask = batch_data['attention_mask'].to(device)
    #         target = target.to(device).float()
    #         optimizer.zero_grad()
    #         logits = model(image, input_ids, attention_mask)
    #         loss = criterion(logits, target)
    #         loss.backward()
    #         optimizer.step()

    #     model.eval()
    #     all_preds = []
    #     all_targets = []
    #     with torch.no_grad():
    #         for batch_data, target in tqdm(val_loader, desc=f"Epoch {epoch+1} Valid"):
    #             image, input_ids, attention_mask = batch_data['image'].to(device), batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device)
                
    #             logits = model(image, input_ids, attention_mask)
    #             preds = torch.round(torch.sigmoid(logits))
                
    #             all_preds.append(preds.cpu())
    #             all_targets.append(target.cpu())

    #     f1 = f1_score(torch.cat(all_targets), torch.cat(all_preds))
    #     precision = precision_score(torch.cat(all_targets), torch.cat(all_preds))
    #     recall = recall_score(torch.cat(all_targets), torch.cat(all_preds))
    #     accuracy = accuracy_score(torch.cat(all_targets), torch.cat(all_preds))
    #     print(f"Epoch {epoch+1}, Validation F1-score: {f1:.4f}, Validation accuracy-score: {accuracy:.4f}, Validation precision-score: {precision:.4f}, Validation recall-score: {recall:.4f}")
    #     logging.info(f"Epoch {epoch+1}, Validation F1-score: {f1:.4f}, Validation accuracy-score: {accuracy:.4f}, Validation precision-score: {precision:.4f}, Validation recall-score: {recall:.4f}")
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         print(f"Saving to best_neural_head.pth")
    #         torch.save(model.state_dict(), "best_neural_head_v3_with_transfer_v2.pth")
    
    for epoch in range(10):
        model.train()
        for name, param in model.named_parameters():
                if 'text_net.bert.encoder.layer.10.' in name or \
                'image_net.model.features.7.' in name:
                    param.requires_grad = True
        for batch_data, target in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            image = batch_data['image'].to(device)
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            target = target.to(device).float()
            optimizer_2.zero_grad()
            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits, target)
            loss.backward()
            optimizer_2.step()

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
        precision = precision_score(torch.cat(all_targets), torch.cat(all_preds))
        recall = recall_score(torch.cat(all_targets), torch.cat(all_preds))
        accuracy = accuracy_score(torch.cat(all_targets), torch.cat(all_preds))
        print(f"Epoch {epoch+1}, Validation F1-score: {f1:.4f}, Validation accuracy-score: {accuracy:.4f}, Validation precision-score: {precision:.4f}, Validation recall-score: {recall:.4f}")
        logging.info(f"Epoch {epoch+1}, Validation F1-score: {f1:.4f}, Validation accuracy-score: {accuracy:.4f}, Validation precision-score: {precision:.4f}, Validation recall-score: {recall:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            print(f"Saving to best_neural_head.pth")
            torch.save(model.state_dict(), "best_neural_head_v3_with_transfer_v2.pth")

if __name__ == '__main__':
    main()
