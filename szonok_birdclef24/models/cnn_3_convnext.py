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
import timm
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
        spectr = torch.tensor(spectr).float().unsqueeze(0)  # [1, 192, 224]
        label = int(row['label_idx'])
        return spectr, label

def pad_or_truncate(spec, target_length=8000):
    if spec.shape[1] < target_length:
        pad_width = target_length - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spec = spec[:, :target_length]
    return spec

# --- DATASET AND DATALOADER ---
target_length = 5200
train_dataset = BirdSpectrogramDataset(train_df, spectr_dir, target_length=5200, crop_width=224, random_crop=True)
val_dataset = BirdSpectrogramDataset(val_df, spectr_dir, target_length=5200, crop_width=224, random_crop=False)
# Reduced batch size for ConvNeXt to improve speed and memory usage
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
torch.backends.cudnn.benchmark = True

# --- MODEL ---
class SpectrogramConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import torchvision.models as models
        
        # Try a different approach: replicate single channel to 3 channels
        # This preserves all pretrained weights without modification
        self.backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        
        # Replace classifier for our number of classes
        self.backbone.classifier[2] = nn.Linear(self.backbone.classifier[2].in_features, num_classes)
        
        # Initialize the new classifier properly
        nn.init.xavier_uniform_(self.backbone.classifier[2].weight)
        nn.init.zeros_(self.backbone.classifier[2].bias)
    
    def forward(self, x):
        # x: [batch, 1, 192, 224] 
        # First interpolate to 224x224
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize each sample individually to preserve differences
        # This is crucial for ConvNeXt to work properly
        batch_size = x.shape[0]
        normalized_batch = []
        for i in range(batch_size):
            sample = x[i:i+1]  # Keep batch dimension
            # Normalize this single sample
            sample_norm = (sample - sample.mean()) / (sample.std() + 1e-8)
            # Replicate to 3 channels
            sample_3ch = sample_norm.repeat(1, 3, 1, 1)
            normalized_batch.append(sample_3ch)
        
        x = torch.cat(normalized_batch, dim=0)
        
        return self.backbone(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(train_df['label_idx'].unique())
model = SpectrogramConvNeXt(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4)
best_val_acc = 0
epochs_no_improve = 0
early_stop_patience = 8
num_epochs = 80
scaler = GradScaler(device='cuda')

print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Device: {device}")

# --- TRAINING AND VALIDATION LOOP ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Store initial weights to check if model is learning
    if epoch == 0:
        initial_classifier_weight = model.backbone.classifier[2].weight.data.clone()
    
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
    
    # Check if weights are updating
    if epoch == 0:
        weight_diff = (model.backbone.classifier[2].weight.data - initial_classifier_weight).abs().mean()
        print(f"Classifier weight change after epoch 1: {weight_diff:.6f}")
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    # --- VALIDATION ---
    model.eval()
    val_correct = 0
    val_total = 0
    all_predictions = []
    all_labels = []
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for spectr, label in val_bar:
            spectr = spectr.to(device)  # [batch, 1, 192, 224]
            label = label.to(device)
            
            # Debug: print shapes and first few predictions
            if epoch == 0 and val_total == 0:
                print(f"Val input shape: {spectr.shape}")
                print(f"Val label shape: {label.shape}")
                print(f"First few labels: {label[:5]}")
            
            outputs = model(spectr)
            _, predicted = torch.max(outputs, 1)
            
            if epoch == 0 and val_total == 0:
                print(f"Output shape: {outputs.shape}")
                print(f"First few predictions: {predicted[:5]}")
                print(f"First few raw outputs: {outputs[:2, :5]}")
                print(f"Output max values: {outputs.max(dim=1)[0][:5]}")
                print(f"Output min values: {outputs.min(dim=1)[0][:5]}")
                # Check if inputs are actually different
                print(f"Input spectrogram statistics:")
                print(f"  Sample 1 mean: {spectr[0].mean().item():.4f}, std: {spectr[0].std().item():.4f}")
                print(f"  Sample 2 mean: {spectr[1].mean().item():.4f}, std: {spectr[1].std().item():.4f}")
                print(f"  Are inputs identical? {torch.equal(spectr[0], spectr[1])}")
            
            val_total += label.size(0)
            val_correct += (predicted == label).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            # Debug: print running accuracy
            if epoch == 0 and val_total % 1000 == 0:
                current_acc = val_correct / val_total
                print(f"Val samples processed: {val_total}, Running acc: {current_acc:.4f}")
    
    val_acc = val_correct / val_total
    
    # Show prediction distribution for first epoch
    if epoch == 0:
        unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
        print(f"Validation prediction distribution (first 10):")
        for i in range(min(10, len(unique_preds))):
            print(f"  Class {unique_preds[i]}: {pred_counts[i]} times ({pred_counts[i]/len(all_predictions)*100:.1f}%)")
        
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        print(f"Validation label distribution (first 10):")
        for i in range(min(10, len(unique_labels))):
            print(f"  Class {unique_labels[i]}: {label_counts[i]} times ({label_counts[i]/len(all_labels)*100:.1f}%)")
    
    print(f"Validation Acc: {val_acc:.4f}")

    scheduler.step(val_acc)
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'convnext_tiny_bird_best.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
    print(f"Best Validation Acc so far: {best_val_acc:.4f}")

torch.save(model.state_dict(), 'convnext_tiny_bird_last.pth')
print("Training complete!")
