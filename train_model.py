# ==========================================
# Dataset: GAN-Balanced | Regularization: Label Smoothing + Cosine Annealing
# Combinations: B0+BERT, DenseNet+BERT, B7+BERT, ViT+BERT, B0+RoBERTa, DenseNet+RoBERTa

# ==========================================

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from tqdm import tqdm  # LIVE PROGRESS BAR IMPORT

# --- MOUNT GOOGLE DRIVE ---
from google.colab import drive
#drive.mount('/content/drive', force_remount=True)
print(" Google Drive Mounted Successfully!")

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'transformers'])
    from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" V14 SOTA BENCHMARK LOOP INITIALIZED ON {device.type.upper()}")

# ------------------------------------------
# 1. LOAD GAN-BALANCED DATASET
# ------------------------------------------
#source_real = '/content/drive/MyDrive/GAN_Balanced_Dataset/Real'
#source_fake = '/content/drive/MyDrive/GAN_Balanced_Dataset/Fake'

all_real = glob.glob(f"{source_real}/*.npy")
all_fake = glob.glob(f"{source_fake}/*.npy")

#df = pd.read_csv('/content/drive/MyDrive/Final_Thesis_Dataset_Real_Fake.csv')
df['content'] = df['content'].astype(str)
df = df.dropna(subset=['content', 'label'])

df_real = df[df.label == 0].sample(n=len(all_real), replace=True, random_state=42)
df_fake = df[df.label == 1].sample(n=len(all_fake), replace=True, random_state=42)
df_balanced_text = pd.concat([df_real, df_fake])

master_data = []
for path in all_real: master_data.append({'data': path, 'label': 0, 'modality': 0})
for path in all_fake: master_data.append({'data': path, 'label': 1, 'modality': 0})
for _, row in df_balanced_text.iterrows(): master_data.append({'data': row['content'], 'label': row['label'], 'modality': 1})

labels = [x['label'] for x in master_data]
train_data, temp_data, train_labels, temp_labels = train_test_split(master_data, labels, test_size=0.30, random_state=42, stratify=labels)
val_data, test_data, _, _ = train_test_split(temp_data, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels)

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
EPOCHS = 15
print(f"⚙️ Hyperparameters: BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")

class UnifiedMultimodalDataset(Dataset):
    def __init__(self, data_list, tokenizer, transform=None, max_len=128):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        modality, label = item['modality'], item['label']
        image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        input_ids = torch.zeros(self.max_len, dtype=torch.long)
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)

        if modality == 0:
            img_array = np.load(item['data']).astype(np.float32)
            if len(img_array.shape) > 2: img_array = img_array[img_array.shape[0]//2]
            img_min, img_max = img_array.min(), img_array.max()
            img_array = 255.0 * (img_array - img_min) / (img_max - img_min) if img_max > img_min else np.zeros_like(img_array)
            img_array = np.stack((img_array.astype(np.uint8),)*3, axis=-1)
            if self.transform: image_tensor = self.transform(Image.fromarray(img_array))
        elif modality == 1:
            encoding = self.tokenizer(item['data'], add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
            input_ids, attention_mask = encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()

        return {'image': image_tensor, 'input_ids': input_ids, 'attention_mask': attention_mask, 'label': torch.tensor(label, dtype=torch.long), 'modality': torch.tensor(modality, dtype=torch.long)}

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(15), transforms.Resize((224, 224)), transforms.ToTensor()])
val_test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# ------------------------------------------
# 2. DYNAMIC ARCHITECTURE
# ------------------------------------------
class DynamicFusionNetwork(nn.Module):
    def __init__(self, img_model_name, txt_model_name):
        super(DynamicFusionNetwork, self).__init__()

        self.img_model_name = img_model_name
        self.txt_model_name = txt_model_name

        if img_model_name == 'DenseNet':
            net = models.densenet121(weights='DEFAULT')
            self.img_net = net.features
            self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.img_dim = 1024
        elif img_model_name == 'EfficientNet': # B0
            net = models.efficientnet_b0(weights='DEFAULT')
            self.img_net = net.features
            self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.img_dim = 1280
        elif img_model_name == 'EfficientNet_B7':
            net = models.efficientnet_b7(weights='DEFAULT')
            self.img_net = net.features
            self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.img_dim = 2560
        elif img_model_name == 'ViT':
            net = models.vit_b_16(weights='DEFAULT')
            self.img_net = net
            self.img_dim = 768

        model_str = 'bert-base-uncased' if txt_model_name == 'BERT' else 'roberta-base'
        self.text_net = AutoModel.from_pretrained(model_str)
        self.txt_dim = 768

        self.classifier = nn.Sequential(
            nn.Linear(self.img_dim + self.txt_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )

    def forward(self, images, input_ids, attention_mask, modalities):
        batch_size = images.size(0)
        img_vecs = torch.zeros(batch_size, self.img_dim, device=images.device)
        txt_vecs = torch.zeros(batch_size, self.txt_dim, device=images.device)

        img_idx = (modalities == 0).nonzero(as_tuple=True)[0]
        txt_idx = (modalities == 1).nonzero(as_tuple=True)[0]

        if len(img_idx) > 0:
            if self.img_model_name == 'ViT':
                x = self.img_net._process_input(images[img_idx])
                n = x.shape[0]
                batch_class_token = self.img_net.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = self.img_net.encoder(x)
                feat = x[:, 0]
            else:
                feat = self.img_pool(self.img_net(images[img_idx])).view(len(img_idx), -1)
            img_vecs[img_idx] = feat.type_as(img_vecs)

        if len(txt_idx) > 0:
            outputs = self.text_net(input_ids[txt_idx], attention_mask=attention_mask[txt_idx])
            txt_feat = outputs.pooler_output if self.txt_model_name == 'BERT' else outputs.last_hidden_state[:, 0, :]
            txt_vecs[txt_idx] = txt_feat.type_as(txt_vecs)

        return self.classifier(torch.cat((img_vecs, txt_vecs), dim=1))

# ------------------------------------------
# 3. AUTOMATED COMBINATION LOOP
# ------------------------------------------
combinations = [
    ("EfficientNet", "BERT"),
    ("DenseNet", "BERT"),
    ("EfficientNet_B7", "BERT"),
    ("ViT", "BERT"),
    ("EfficientNet", "RoBERTa"),
    ("DenseNet", "RoBERTa")
]

for img_model, txt_model in combinations:
    print(f"\n" + "="*60)
    print(f" SOTA INITIALIZING: [ {img_model} + {txt_model} ]")
    print(f"="*60)

    tokenizer_name = 'bert-base-uncased' if txt_model == 'BERT' else 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # CRITICAL FIX: num_workers=0 to prevent Drive Crash
    train_loader = DataLoader(UnifiedMultimodalDataset(train_data, tokenizer, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(UnifiedMultimodalDataset(val_data, tokenizer, val_test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DynamicFusionNetwork(img_model, txt_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    save_path = f'/content/drive/MyDrive/Model_Final_GAN_{img_model}_{txt_model}.pth'
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        # LIVE PROGRESS BAR FOR TRAINING
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}", leave=False):
            optimizer.zero_grad()
            img, ids, mask = batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels, mods = batch['label'].to(device), batch['modality'].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(img, ids, mask, mods)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss_epoch = running_loss / len(train_loader)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            # LIVE PROGRESS BAR FOR VALIDATION
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{EPOCHS}", leave=False):
                img, ids, mask = batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device)
                labels, mods = batch['label'].to(device), batch['modality'].to(device)
                outputs = model(img, ids, mask, mods)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(avg_val_loss)

        scheduler.step()
        print(f" Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"    Saved Best {img_model}+{txt_model} to Drive!")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(range(1, EPOCHS+1), history['train_acc'], 'bo-', label='Train Accuracy')
    ax1.plot(range(1, EPOCHS+1), history['val_acc'], 'ro-', label='Val Accuracy')
    ax1.set_title(f'Acc: GAN Data | {img_model} + {txt_model} (SOTA)', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()

    ax2.plot(range(1, EPOCHS+1), history['train_loss'], 'bs-', label='Train Loss')
    ax2.plot(range(1, EPOCHS+1), history['val_loss'], 'rs-', label='Val Loss')
    ax2.set_title(f'Loss: GAN Data | {img_model} + {txt_model} (SOTA)', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.savefig(f'Learning_Curve_FINAL_{img_model}_{txt_model}.png', dpi=300)
    plt.show()
    print(f" Completed {img_model} + {txt_model}.\n")

    del model, optimizer, scaler, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

print(" ALL 6 FINAL SOTA BENCHMARKS COMPLETED SUCCESSFULLY!")
