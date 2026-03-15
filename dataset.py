import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path

class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Check for nested folder structure
        img_path = self.img_dir / f"{row['id_code']}.png"
        if not img_path.exists():
            # Try nested folder (folder name matches parent)
            for subfolder in self.img_dir.iterdir():
                if subfolder.is_dir():
                    nested = subfolder / f"{row['id_code']}.png"
                    if nested.exists():
                        img_path = nested
                        break
        
        image = Image.open(img_path).convert('RGB')
        label = row['diagnosis']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define preprocessing transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Test the dataset
if __name__ == "__main__":
    # Create dataset
    train_dataset = DRDataset(
        csv_file='data/train_1.csv',
        img_dir='data/train_images',
        transform=train_transform
    )
    
    print(f"Dataset size: {len(train_dataset)} images")
    
    # Load one sample
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Load one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Batch labels: {labels}")
    print("\nDataset and DataLoader working correctly!")