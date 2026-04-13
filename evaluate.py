import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import seaborn as sns
from dataset import DRDataset, val_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load validation dataset
val_dataset = DRDataset('data/valid.csv', 'data/val_images', val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
print(f"Validation samples: {len(val_dataset)}")

def load_resnet50(weights_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    return model

def load_efficientnet_b3(weights_path):
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    return model

def load_resnet50_ordinal(weights_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)  # 4 outputs for ordinal
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    return model

def ordinal_to_class(logits):
    """Convert ordinal logits to class predictions."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).sum(dim=1)
    return preds

def evaluate_model(model, loader, is_ordinal=False):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            if is_ordinal:
                predicted = ordinal_to_class(outputs)
            else:
                _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(labels, preds, title, filename):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Prolif.'],
                yticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Prolif.'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

# Evaluate all models
models_to_eval = [
    ("ResNet-50 Baseline", "resnet50_dr.pth", load_resnet50, False),
    ("EfficientNet-B3", "efficientnet_b3_dr.pth", load_efficientnet_b3, False),
    ("ResNet-50 Weighted", "resnet50_weighted_dr.pth", load_resnet50, False),
    ("ResNet-50 Ordinal", "resnet50_ordinal_dr.pth", load_resnet50_ordinal, True),
]

results = []

for name, weights, loader_fn, is_ordinal in models_to_eval:
    print(f"\nEvaluating {name}...")
    model = loader_fn(weights)
    labels, preds = evaluate_model(model, val_loader, is_ordinal)
    
    # Calculate metrics
    accuracy = (labels == preds).mean() * 100
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    
    # Per-class accuracy
    per_class_acc = []
    for c in range(5):
        mask = labels == c
        if mask.sum() > 0:
            class_acc = (preds[mask] == c).mean() * 100
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0)
    
    results.append({
        'Model': name,
        'Val Accuracy': f"{accuracy:.2f}%",
        'Quadratic Kappa': f"{kappa:.4f}",
        'Class 0 (No DR)': f"{per_class_acc[0]:.1f}%",
        'Class 1 (Mild)': f"{per_class_acc[1]:.1f}%",
        'Class 2 (Moderate)': f"{per_class_acc[2]:.1f}%",
        'Class 3 (Severe)': f"{per_class_acc[3]:.1f}%",
        'Class 4 (Prolif.)': f"{per_class_acc[4]:.1f}%",
    })
    
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Quadratic Kappa: {kappa:.4f}")
    print(f"  Per-class: {[f'{a:.1f}%' for a in per_class_acc]}")
    
    # Plot confusion matrix
    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    plot_confusion_matrix(labels, preds, f"{name} Confusion Matrix", f"confusion_{safe_name}.png")

# Save results summary
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(results_df.to_string(index=False))
results_df.to_csv('model_comparison.csv', index=False)
print("\nSaved results to model_comparison.csv")