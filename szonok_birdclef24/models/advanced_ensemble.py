import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import random
import torchvision.models as models
import timm
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def load_melspectrogram(mel_path, target_height=192, target_width=5200):
    """Load mel-spectrogram exactly as done during training"""
    mel = np.load(mel_path)
    
    # Pad or truncate to target size (same as original training)
    if mel.shape[1] < target_width:
        # Pad with zeros if too short
        pad_width = target_width - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if too long
        mel = mel[:, :target_width]
    
    # Ensure height is correct (should be 192 for mel spectrograms)
    if mel.shape[0] != target_height:
        # Resize height if needed
        mel = np.resize(mel, (target_height, target_width))
    
    # NO Z-SCORE NORMALIZATION - use raw dB values as in training
    
    # Convert to tensor and add channel dimension
    mel_tensor = torch.tensor(mel, dtype=torch.float32)
    
    # Add channel dimension
    if mel_tensor.dim() == 2:
        mel_tensor = mel_tensor.unsqueeze(0)  # Add channel dimension: [1, 192, 5200]
    
    return mel_tensor

class BirdDataset(Dataset):
    def __init__(self, csv_file, mel_dir, num_classes=182):
        self.data = pd.read_csv(csv_file)
        self.mel_dir = mel_dir
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        mel_path = os.path.join(self.mel_dir, row['spectrogram'])
        mel = load_melspectrogram(mel_path)
        label = row['label_idx']
        return mel, torch.tensor(label, dtype=torch.long)

def sliding_window_predict_batch(model, mel_batch, window_size=(192, 224), stride=96, device='cuda'):
    """
    Perform sliding window prediction on a batch of mel-spectrograms
    mel_batch: shape [batch_size, C, H, W]
    Returns: predictions for each sample in the batch
    """
    batch_size, channels, height, width = mel_batch.shape
    all_batch_preds = []
    
    # Process each sample in the batch
    for b in range(batch_size):
        mel_tensor = mel_batch[b]  # [C, H, W]
        
        # Calculate sliding windows for this sample
        windows = []
        for h in range(0, height - window_size[0] + 1, stride):
            for w in range(0, width - window_size[1] + 1, stride):
                window = mel_tensor[:, h:h+window_size[0], w:w+window_size[1]]
                windows.append(window)
        
        if not windows:
            # If mel is smaller than window, pad it
            pad_h = max(0, window_size[0] - height)
            pad_w = max(0, window_size[1] - width)
            padded_mel = torch.nn.functional.pad(mel_tensor, (0, pad_w, 0, pad_h))
            windows = [padded_mel]
        
        # Stack all windows for this sample and run inference
        if windows:
            windows_tensor = torch.stack(windows).to(device)  # [num_windows, C, H, W]
            with torch.no_grad():
                outputs = model(windows_tensor)
                probs = torch.softmax(outputs, dim=1)
                # Average predictions across all windows
                sample_pred = torch.mean(probs, dim=0)
                all_batch_preds.append(sample_pred)
    
    return torch.stack(all_batch_preds)  # [batch_size, num_classes]

def predict_batch_efficient(model, mel_batch, device='cuda', max_chunks_per_batch=128, model_name=''):
    """
    More efficient prediction that matches training preprocessing exactly
    mel_batch: shape [batch_size, C, H, W] where W=5200
    """
    batch_size, channels, height, width = mel_batch.shape
    
    if model_name == 'efficientnet':
        # EfficientNet: EXACTLY as trained - NO normalization, keep 1 channel
        # From Dataprep_EfNB0.py: raw dB values, crop to 224, keep as [1, 192, 224]
        all_chunks = []
        chunk_indices = []
        
        for b in range(batch_size):
            # Multiple crops for better coverage (like sliding window but matching training)
            crop_width = 224
            stride = 180  # Some overlap for better coverage
            
            for start in range(0, width - crop_width + 1, stride):
                # Crop to 224 width - keep raw dB values (NO normalization)
                crop = mel_batch[b, :, :, start:start+crop_width]  # Keep as [1, 192, 224]
                all_chunks.append(crop)
                chunk_indices.append(b)
            
            # Add final crop if needed
            if width % stride != 0:
                crop = mel_batch[b, :, :, -crop_width:]  # [1, 192, 224]
                all_chunks.append(crop)
                chunk_indices.append(b)
    
    else:
        # ConvNeXt/ResNet/DenseNet: EXACTLY as trained - normalization + 3 channels
        # From cnn_3_convnext.py: crop to 224, normalize, convert to 3ch, interpolate to 224x224
        # IMPORTANT: Use single LEFT CROP (start=0) to match training validation exactly!
        all_chunks = []
        chunk_indices = []
        
        for b in range(batch_size):
            # SINGLE LEFT CROP - exactly matching training validation (random_crop=False, start=0)
            crop_width = 224
            start = 0  # LEFT CROP - matching training validation exactly
            
            # Crop to 224 width (matching training validation exactly)
            crop = mel_batch[b, 0, :, start:start+crop_width]  # Remove channel dim: [192, 224]
            
            # Per-sample z-score normalization (EXACT match to training)
            mean = crop.mean()
            std = crop.std()
            if std > 0:
                crop_norm = (crop - mean) / std
            else:
                crop_norm = crop - mean
            
            # Convert to 3 channels (EXACT match to training)
            crop_3ch = crop_norm.unsqueeze(0).repeat(3, 1, 1)  # [3, 192, 224]
            
            all_chunks.append(crop_3ch)
            chunk_indices.append(b)
    
    # Process all chunks in manageable batches
    if all_chunks:
        all_probs = []
        
        # Dynamic batch sizing based on GPU memory (very conservative to avoid OOM)
        if torch.cuda.is_available():
            # Start with very small batch size to be safe
            try:
                test_batch_size = min(32, len(all_chunks))  # Much smaller
                test_chunks = torch.stack(all_chunks[:test_batch_size]).to(device)
                with torch.no_grad():
                    _ = model(test_chunks)
                max_chunks_per_batch = 32  # Very conservative
                del test_chunks
                torch.cuda.empty_cache()
            except RuntimeError:
                max_chunks_per_batch = 16  # Extremely conservative fallback
                torch.cuda.empty_cache()
        
        with torch.no_grad():
            for i in range(0, len(all_chunks), max_chunks_per_batch):
                batch_chunks = all_chunks[i:i+max_chunks_per_batch]
                chunks_tensor = torch.stack(batch_chunks).to(device)  # [chunk_batch_size, C, H, W]
                
                outputs = model(chunks_tensor)
                probs = torch.softmax(outputs, dim=1).cpu()
                all_probs.append(probs)
                
                # Clear GPU cache more frequently with larger input batches
                if i % (max_chunks_per_batch * 2) == 0:  # More frequent clearing
                    torch.cuda.empty_cache()
        
        # Concatenate all probabilities
        all_probs = torch.cat(all_probs, dim=0)
        
        # Average predictions per sample
        sample_preds = []
        for b in range(batch_size):
            # Find all chunks belonging to this sample
            sample_chunk_preds = []
            for i, chunk_idx in enumerate(chunk_indices):
                if chunk_idx == b:
                    sample_chunk_preds.append(all_probs[i])
            
            if sample_chunk_preds:
                # Average across chunks for this sample
                sample_pred = torch.mean(torch.stack(sample_chunk_preds), dim=0)
                sample_preds.append(sample_pred)
        
        return torch.stack(sample_preds)  # [batch_size, num_classes]
    
    return torch.zeros(batch_size, 182)  # fallback

# Model architectures
class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes=182):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, in_chans=1)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes=182):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # x: [batch, 3, 192, 224] -> interpolate to [batch, 3, 224, 224]
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)

class ResNet50Model(nn.Module):
    def __init__(self, num_classes=182):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        # x: [batch, 3, 192, 224] -> interpolate to [batch, 3, 224, 224]
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)

class DenseNet121Model(nn.Module):
    def __init__(self, num_classes=182):
        super().__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, x):
        # x: [batch, 3, 192, 224] -> interpolate to [batch, 3, 224, 224]
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)

def load_models(device):
    """Load all trained models"""
    models_info = {
        'efficientnet': {
            'model': EfficientNetB0Model(),
            'checkpoint': '/home/des/nnet/EfNB0_bird_cnn_best.pth'
        },
        'convnext': {
            'model': ConvNeXtModel(),
            'checkpoint': '/home/des/nnet/convnext_tiny_bird_best.pth'
        },
        'resnet50': {
            'model': ResNet50Model(),
            'checkpoint': '/home/des/nnet/resnet50_bird_best.pth'
        },
        'densenet': {
            'model': DenseNet121Model(),
            'checkpoint': '/home/des/nnet/densenet121_bird_best.pth'
        }
    }
    
    loaded_models = {}
    for name, info in models_info.items():
        try:
            model = info['model']
            checkpoint = torch.load(info['checkpoint'], map_location=device, weights_only=True)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            loaded_models[name] = model
            print(f"✓ Loaded {name} from {info['checkpoint']}")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    return loaded_models

def evaluate_single_model(model, val_loader, device, model_name):
    """Evaluate a single model with efficient batch prediction"""
    model.eval()
    correct = 0
    total = 0
    
    print(f"\nEvaluating {model_name}...")
    with torch.no_grad():
        for mel, labels in tqdm(val_loader, desc=f"{model_name} evaluation"):
            # Use efficient batch prediction with model-specific preprocessing
            batch_preds = predict_batch_efficient(model, mel.to(device), device=device, model_name=model_name)
            predicted = torch.argmax(batch_preds, dim=1).cpu()  # Move predictions to CPU
            correct += (predicted == labels).sum().item()  # Both on CPU now
            total += mel.size(0)
    
    accuracy = correct / total
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    return accuracy

def get_all_predictions(models, val_loader, device):
    """Get predictions from all models for ensemble evaluation - OPTIMIZED VERSION"""
    all_model_preds = {name: [] for name in models.keys()}
    all_labels = []
    
    print("\nGetting predictions from all models (optimized)...")
    
    # Process each model separately to avoid memory issues
    for model_name, model in models.items():
        print(f"Getting predictions from {model_name}...")
        model_preds = []
        
        with torch.no_grad():
            for mel, labels in tqdm(val_loader, desc=f"{model_name} predictions"):
                mel_batch = mel.to(device)
                batch_preds = predict_batch_efficient(model, mel_batch, device=device, model_name=model_name)
                model_preds.append(batch_preds.cpu())
                
                # Only collect labels once
                if model_name == list(models.keys())[0]:  # First model
                    all_labels.append(labels.cpu())
        
        all_model_preds[model_name] = torch.cat(model_preds, dim=0)
        
        # Clear GPU cache after each model
        torch.cuda.empty_cache()
    
    all_labels = torch.cat(all_labels, dim=0)
    return all_model_preds, all_labels

def ensemble_equal_weights(predictions_dict):
    """Simple average of all model predictions"""
    pred_tensors = list(predictions_dict.values())
    return torch.mean(torch.stack(pred_tensors), dim=0)

def ensemble_performance_weights(predictions_dict, individual_accuracies):
    """Weight predictions by individual model performance"""
    weighted_preds = []
    total_weight = 0
    
    for model_name, preds in predictions_dict.items():
        weight = individual_accuracies[model_name]
        weighted_preds.append(preds * weight)
        total_weight += weight
    
    return torch.sum(torch.stack(weighted_preds), dim=0) / total_weight

def ensemble_top_k(predictions_dict, individual_accuracies, k=2):
    """Use only top-k performing models"""
    sorted_models = sorted(individual_accuracies.items(), key=lambda x: x[1], reverse=True)
    top_k_models = [name for name, _ in sorted_models[:k]]
    
    top_k_preds = [predictions_dict[name] for name in top_k_models]
    return torch.mean(torch.stack(top_k_preds), dim=0)

def find_optimal_weights(predictions_dict, labels, num_trials=50):
    """Find optimal weights using random search"""
    model_names = list(predictions_dict.keys())
    best_accuracy = 0
    best_weights = None
    
    for _ in range(num_trials):
        # Generate random weights and normalize
        weights = np.random.random(len(model_names))
        weights = weights / np.sum(weights)
        
        # Compute weighted ensemble prediction
        weighted_preds = []
        for i, model_name in enumerate(model_names):
            weighted_preds.append(predictions_dict[model_name] * weights[i])
        
        ensemble_pred = torch.sum(torch.stack(weighted_preds), dim=0)
        predicted_labels = torch.argmax(ensemble_pred, dim=1)
        accuracy = (predicted_labels == labels).float().mean().item()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights.copy()
    
    return best_weights, best_accuracy

def ensemble_stacking(predictions_dict, labels, test_size=0.3):
    """Use logistic regression as meta-learner"""
    # Prepare features (concatenate all model predictions)
    features = torch.cat(list(predictions_dict.values()), dim=1).numpy()
    labels_np = labels.numpy()
    
    # Split into train/test for meta-learner
    n_samples = len(features)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    train_idx, test_idx = indices[n_test:], indices[:n_test]
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels_np[train_idx], labels_np[test_idx]
    
    # Train meta-learner
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    meta_learner.fit(X_train, y_train)
    
    # Evaluate
    predictions = meta_learner.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return meta_learner, accuracy

class EnsembleModel(nn.Module):
    """Wrapper class for the best ensemble model"""
    def __init__(self, models, method, **kwargs):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.method = method
        self.kwargs = kwargs
        
    def forward(self, x):
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = torch.softmax(model(x), dim=1)
        
        # Apply ensemble method
        if self.method == 'equal_weights':
            return ensemble_equal_weights(predictions)
        elif self.method == 'performance_weights':
            return ensemble_performance_weights(predictions, self.kwargs['weights'])
        elif self.method == 'top_k':
            return ensemble_top_k(predictions, self.kwargs['accuracies'], self.kwargs['k'])
        elif self.method == 'optimal_weights':
            model_names = list(predictions.keys())
            weighted_preds = []
            for i, model_name in enumerate(model_names):
                weighted_preds.append(predictions[model_name] * self.kwargs['weights'][i])
            return torch.sum(torch.stack(weighted_preds), dim=0)
        else:
            return ensemble_equal_weights(predictions)

def save_ensemble_model(ensemble_model, config, save_path='best_ensemble_model.pth'):
    """Save the ensemble model and configuration"""
    torch.save({
        'ensemble_model': ensemble_model.state_dict(),
        'config': config
    }, save_path)
    print(f"Ensemble model saved to {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print GPU memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load validation dataset
    val_dataset = BirdDataset('/home/des/nnet/val_pairs.csv', '/home/des/nnet/melspectr_train/')
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=False)
    
    # Load all models
    models = load_models(device)
    
    if not models:
        print("No models loaded successfully. Exiting.")
        return
    
    print(f"\nSuccessfully loaded {len(models)} models: {list(models.keys())}")
    
    # Evaluate individual models
    print("\n" + "="*50)
    print("INDIVIDUAL MODEL EVALUATION")
    print("="*50)
    
    individual_accuracies = {}
    for model_name, model in models.items():
        accuracy = evaluate_single_model(model, val_loader, device, model_name)
        individual_accuracies[model_name] = accuracy
    
    # Get all predictions for ensemble evaluation
    all_predictions, all_labels = get_all_predictions(models, val_loader, device)
    
    # Ensemble evaluation
    print("\n" + "="*50)
    print("ENSEMBLE EVALUATION")
    print("="*50)
    
    ensemble_results = {}
    
    # 1. Equal weights ensemble
    equal_pred = ensemble_equal_weights(all_predictions)
    equal_accuracy = (torch.argmax(equal_pred, dim=1) == all_labels).float().mean().item()
    ensemble_results['equal_weights'] = equal_accuracy
    print(f"Equal weights ensemble accuracy: {equal_accuracy:.4f}")
    
    # 2. Performance-based weights
    perf_pred = ensemble_performance_weights(all_predictions, individual_accuracies)
    perf_accuracy = (torch.argmax(perf_pred, dim=1) == all_labels).float().mean().item()
    ensemble_results['performance_weights'] = perf_accuracy
    print(f"Performance-weighted ensemble accuracy: {perf_accuracy:.4f}")
    
    # 3. Top-2 models only
    top2_pred = ensemble_top_k(all_predictions, individual_accuracies, k=2)
    top2_accuracy = (torch.argmax(top2_pred, dim=1) == all_labels).float().mean().item()
    ensemble_results['top_2'] = top2_accuracy
    print(f"Top-2 ensemble accuracy: {top2_accuracy:.4f}")
    
    # 4. Optimal weights
    print("Finding optimal weights...")
    optimal_weights, optimal_accuracy = find_optimal_weights(all_predictions, all_labels)
    ensemble_results['optimal_weights'] = optimal_accuracy
    print(f"Optimal weights ensemble accuracy: {optimal_accuracy:.4f}")
    print(f"Optimal weights: {dict(zip(models.keys(), optimal_weights))}")
    
    # 5. Stacking
    print("Training stacking ensemble...")
    try:
        meta_learner, stacking_accuracy = ensemble_stacking(all_predictions, all_labels)
        ensemble_results['stacking'] = stacking_accuracy
        print(f"Stacking ensemble accuracy: {stacking_accuracy:.4f}")
    except Exception as e:
        print(f"Stacking failed: {e}")
        stacking_accuracy = 0
    
    # Find best ensemble method
    best_method = max(ensemble_results, key=ensemble_results.get)
    best_accuracy = ensemble_results[best_method]
    
    print(f"\n" + "="*50)
    print("BEST ENSEMBLE RESULTS")
    print("="*50)
    print(f"Best ensemble method: {best_method}")
    print(f"Best ensemble accuracy: {best_accuracy:.4f}")
    print(f"Improvement over best single model: {best_accuracy - max(individual_accuracies.values()):.4f}")
    
    # Create and save the best ensemble model
    if best_method == 'optimal_weights':
        config = {
            'method': 'optimal_weights',
            'weights': optimal_weights.tolist(),
            'model_names': list(models.keys())
        }
        ensemble_model = EnsembleModel(models, 'optimal_weights', weights=optimal_weights)
    elif best_method == 'performance_weights':
        config = {
            'method': 'performance_weights',
            'weights': individual_accuracies,
            'model_names': list(models.keys())
        }
        ensemble_model = EnsembleModel(models, 'performance_weights', weights=individual_accuracies)
    elif best_method == 'top_2':
        config = {
            'method': 'top_k',
            'k': 2,
            'accuracies': individual_accuracies,
            'model_names': list(models.keys())
        }
        ensemble_model = EnsembleModel(models, 'top_k', accuracies=individual_accuracies, k=2)
    else:  # equal_weights
        config = {
            'method': 'equal_weights',
            'model_names': list(models.keys())
        }
        ensemble_model = EnsembleModel(models, 'equal_weights')
    
    # Save the ensemble model and configuration
    save_ensemble_model(ensemble_model, config)
    
    # Save configuration as JSON for easy loading
    with open('ensemble_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nEnsemble configuration saved to ensemble_config.json")
    
    # Summary
    print(f"\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print("Individual model accuracies:")
    for model_name, acc in individual_accuracies.items():
        print(f"  {model_name}: {acc:.4f}")
    
    print(f"\nEnsemble method accuracies:")
    for method, acc in ensemble_results.items():
        marker = " ← BEST" if method == best_method else ""
        print(f"  {method}: {acc:.4f}{marker}")

if __name__ == "__main__":
    main()
