import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
import time
from dataset import DRDataset, train_transform, val_transform

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load datasets
train_dataset = DRDataset('data/train_1.csv', 'data/train_images', train_transform)
val_dataset = DRDataset('data/valid.csv', 'data/val_images', val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Calculate class weights (inverse frequency)
train_df = pd.read_csv('data/train_1.csv')
class_counts = train_df['diagnosis'].value_counts().sort_index().values
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize
class_weights = class_weights.to(device)

print(f"\nClass counts: {class_counts}")
print(f"Class weights: {class_weights}")

# Load pretrained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(device)

# Weighted loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    train_acc = 100 * correct / total
    epoch_time = time.time() - start_time
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Time: {epoch_time:.1f}s")
    print(f"  Train Acc: {train_acc:.2f}%")
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    print(f"  Val Acc: {val_acc:.2f}%\n")

# Save the model
torch.save(model.state_dict(), 'resnet50_weighted_dr.pth')
print("Model saved to resnet50_weighted_dr.pth")