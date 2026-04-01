import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
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

# Load pretrained EfficientNet-B3
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

# Replace final layer for 5 classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

# Move to GPU
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
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
    
    try:
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"  Val Acc: {val_acc:.2f}%\n")
    except Exception as e:
        print(f"  Validation error: {e}\n")

# Save the model
torch.save(model.state_dict(), 'efficientnet_b3_dr.pth')
print("Model saved to efficientnet_b3_dr.pth")