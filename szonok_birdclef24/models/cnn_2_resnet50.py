import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import torchvision.models as models
from torch.amp import autocast, GradScaler

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- AUGMENTATION FUNCTIONS ---
def time_mask(spec, max_width=30):
    t = spec.shape[1]
    if t <= max_width:
        return spec
    width = random.randint(1, max_width)
    t0 = random.randint(0, t - width)
    spec[:, t0:t0+width] = 0
    return spec

def freq_mask(spec, max_width=10):
    f = spec.shape[0]
    if f <= max_width:
        return spec
    width = random.randint(1, max_width)
    f0 = random.randint(0, f - width)
    spec[f0:f0+width, :] = 0
    return spec

# --- DATA PREP ---
train_df = pd.read_csv('train_pairs.csv')
val_df = pd.read_csv('val_pairs.csv')
spectr_dir = 'melspectr_train'

class BirdSpectrogramDataset(Dataset):
    def __init__(self, df, spectr_dir, target_length=5200, crop_width=224, random_crop=True):
        self.df = df.reset_index(drop=True)
        self.spectr_dir = spectr_dir
        self.target_length = target_length
        self.crop_width = crop_width
        self.random_crop = random_crop

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spectr = np.load(os.path.join(self.spectr_dir, row['spectrogram']))
        spectr = pad_or_truncate(spectr, self.target_length)
        # For training: random crop; for validation: left crop
        if self.crop_width < self.target_length:
            if self.random_crop:
                start = np.random.randint(0, self.target_length - self.crop_width + 1)
            else:
                start = 0
            spectr = spectr[:, start:start+self.crop_width]
        else:
            pad_width = self.crop_width - spectr.shape[1]
            spectr = np.pad(spectr, ((0, 0), (0, pad_width)), mode='constant')
        # --- AUGMENTATION ---
        if self.random_crop:
            if random.random() < 0.5:
                spectr = time_mask(spectr, max_width=30)
            if random.random() < 0.5:
                spectr = freq_mask(spectr, max_width=10)
        
        # Convert to tensor and replicate to 3 channels
        spectr = torch.tensor(spectr).float()
        
        # Per-sample normalization (z-score)
        mean = spectr.mean()
        std = spectr.std()
        if std > 0:
            spectr = (spectr - mean) / std
        
        # Replicate single channel to 3 channels for pre-trained models
        spectr = spectr.unsqueeze(0).repeat(3, 1, 1)  # [3, 192, 224]
        
        label = int(row['label_idx'])
        return spectr, label

def pad_or_truncate(spec, target_length=8000):
    if spec.shape[1] < target_length:
        pad_width = target_length - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spec = spec[:, :target_length]
    return spec

# --- SLIDING WINDOW FUNCTION ---
def sliding_window_predict(model, spectr, window_size=224, step_size=112, device='cuda'):
    model.eval()
    T = spectr.shape[-1]
    preds = []
    with torch.no_grad():
        for start in range(0, T - window_size + 1, step_size):
            window = spectr[:, :, start:start+window_size].unsqueeze(0).to(device)
            out = model(window)
            preds.append(torch.softmax(out, dim=1).cpu())
        if (T - window_size) % step_size != 0:
            window = spectr[:, :, -window_size:].unsqueeze(0).to(device)
            out = model(window)
            preds.append(torch.softmax(out, dim=1).cpu())
    preds = torch.cat(preds, dim=0)
    return preds.mean(dim=0)

# --- DATASET AND DATALOADER ---
target_length = 5200
train_dataset = BirdSpectrogramDataset(train_df, spectr_dir, target_length=5200, crop_width=224, random_crop=True)
val_dataset = BirdSpectrogramDataset(val_df, spectr_dir, target_length=5200, crop_width=224, random_crop=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
torch.backends.cudnn.benchmark = True

# --- MODEL ---
class SpectrogramResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Keep the original conv1 for 3-channel input
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        # x: [batch, 3, 192, 224] -> [batch, 3, 224, 224]
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(train_df['label_idx'].unique())
model = SpectrogramResNet50(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001)  # Start with original learning rate
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4)
best_val_acc = 0
epochs_no_improve = 0
early_stop_patience = 8
num_epochs = 90
scaler = GradScaler('cuda')

# --- TRAINING AND VALIDATION LOOP ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for spectr, label in train_bar:
        spectr, label = spectr.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(spectr)
            loss = criterion(outputs, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * spectr.size(0)
        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        train_bar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    # --- VALIDATION ---
    model.eval()
    val_correct = 0
    val_total = 0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for spectr, label in val_bar:
            spectr = spectr.to(device)
            label = label.to(device)
            all_windows = []
            window_counts = []
            for i in range(spectr.size(0)):
                windows = []
                T = spectr[i].shape[-1]
                for start in range(0, T - 224 + 1, 112):
                    windows.append(spectr[i:i+1, :, :, start:start+224])
                if (T - 224) % 112 != 0:
                    windows.append(spectr[i:i+1, :, :, -224:])
                all_windows.extend(windows)
                window_counts.append(len(windows))
            all_windows_tensor = torch.cat(all_windows, dim=0)
            outputs = model(all_windows_tensor)
            probs = torch.softmax(outputs, dim=1)
            idx = 0
            batch_preds = []
            for count in window_counts:
                pred = probs[idx:idx+count].mean(dim=0)
                batch_preds.append(pred)
                idx += count
            batch_preds = torch.stack(batch_preds, dim=0)
            predicted_labels = batch_preds.argmax(dim=1)
            val_total += label.size(0)
            val_correct += (predicted_labels == label).sum().item()
    val_acc = val_correct / val_total
    print(f"Validation Acc: {val_acc:.4f}")

    scheduler.step(val_acc)
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'resnet50_bird_best.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
    print(f"Best Validation Acc so far: {best_val_acc:.4f}")

torch.save(model.state_dict(), 'resnet50_bird_last.pth')
print("Training complete!")
