# ==========================================
#TESTING & EVALUATION
# ==========================================

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_fscore_support
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from tqdm import tqdm

# --- MOUNT GOOGLE DRIVE ---
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print(" Google Drive Mounted Successfully!")

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'transformers'])
    from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" INFERENCE ENGINE INITIALIZED ON {device.type.upper()}")

# ------------------------------------------
# 1. LOAD DATASET (TO RECREATE THE EXACT TEST SET)
# ------------------------------------------
source_real = '/content/drive/MyDrive/GAN_Balanced_Dataset/Real'
source_fake = '/content/drive/MyDrive/GAN_Balanced_Dataset/Fake'

all_real = glob.glob(f"{source_real}/*.npy")
all_fake = glob.glob(f"{source_fake}/*.npy")

df = pd.read_csv('/content/drive/MyDrive/Final_Thesis_Dataset_Real_Fake.csv')
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
# Using the exact same random_state=42 ensures the Test set is 100% identical and unseen
_, temp_data, _, temp_labels = train_test_split(master_data, labels, test_size=0.30, random_state=42, stratify=labels)
_, test_data, _, _ = train_test_split(temp_data, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels)

val_test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

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
        elif img_model_name == 'EfficientNet':
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
# 3. EVALUATION LOOP (TESTING ONLY)
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
    save_path = f'/content/drive/MyDrive/Model_Final_GAN_{img_model}_{txt_model}.pth'

    # Check if the model actually exists before testing
    if not os.path.exists(save_path):
        print(f"\n SKIPPING: {img_model} + {txt_model} (Model file not found in Drive)")
        continue

    print(f"\n" + "="*60)
    print(f" TESTING SAVED MODEL: [ {img_model} + {txt_model} ]")
    print(f"="*60)

    tokenizer_name = 'bert-base-uncased' if txt_model == 'BERT' else 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Batch size can be larger for testing to save time
    test_loader = DataLoader(UnifiedMultimodalDataset(test_data, tokenizer, val_test_transform), batch_size=64, shuffle=False, num_workers=0)

    # Initialize model and load SAVED weights
    model = DynamicFusionNetwork(img_model, txt_model).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {img_model}+{txt_model}"):
            img, ids, mask = batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels, mods = batch['label'].to(device), batch['modality'].to(device)

            outputs = model(img, ids, mask, mods)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Get probabilities for class 1
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- METRICS CALCULATION ---
    test_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    print(f"\n TEST RESULTS FOR [{img_model} + {txt_model}]:")
    print(f" Accuracy:  {test_acc*100:.2f}%")
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" F1-Score:  {f1:.4f}\n")

    # --- PLOTTING CM & ROC ---
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    sns.set_theme(style="white")
    fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                xticklabels=['Authentic (0)', 'Synthetic (1)'],
                yticklabels=['Authentic (0)', 'Synthetic (1)'],
                annot_kws={"size": 14})
    ax_cm.set_title(f'Confusion Matrix\n{img_model} + {txt_model}', fontweight='bold', fontsize=14)
    ax_cm.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax_cm.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

    # Plot ROC Curve
    sns.set_theme(style="whitegrid")
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax_roc.set_title(f'ROC Curve\n{img_model} + {txt_model}', fontweight='bold', fontsize=14)
    ax_roc.legend(loc="lower right", fontsize=12)

    # Save and Show Figure
    plot_path = f'/content/Testing_Results_{img_model}_{txt_model}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved Confusion Matrix & ROC Curve to: {plot_path}")

    # Free up memory
    del model, test_loader
    torch.cuda.empty_cache()
    gc.collect()

print("\n ALL TESTING COMPLETED SUCCESSFULLY!")
