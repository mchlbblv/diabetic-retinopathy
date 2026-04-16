import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd

# Load one sample image
train_df = pd.read_csv('data/train_1.csv')
sample_id = train_df.iloc[15]['id_code']

# Find the image
img_dir = Path('data/train_images/train_images')
img_path = img_dir / f"{sample_id}.png"
original = Image.open(img_path).convert('RGB')

# Define individual transforms
augmentations = [
    ("Original", transforms.Compose([transforms.Resize((224, 224))])),
    ("Horizontal Flip", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0)])),
    ("Vertical Flip", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomVerticalFlip(p=1.0)])),
    ("Rotation (20°)", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(20)])),
    ("Color Jitter", transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.4, contrast=0.4)])),
    ("All Combined", transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4)
    ])),
]

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, transform) in enumerate(augmentations):
    img = transform(original)
    axes[idx].imshow(img)
    axes[idx].set_title(name, fontsize=12)
    axes[idx].axis('off')

plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to augmentation_examples.png")