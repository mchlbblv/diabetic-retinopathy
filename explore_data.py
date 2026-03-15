import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the training labels
train_df = pd.read_csv('data/train_1.csv')

# Display basic info
print("Dataset shape:", train_df.shape)
print("\nColumn names:", train_df.columns.tolist())
print("\nFirst few rows:")
print(train_df.head())

# Count images per severity class
print("\n--- Class Distribution ---")
class_counts = train_df['diagnosis'].value_counts().sort_index()
print(class_counts)

# Calculate percentages
print("\n--- Class Percentages ---")
class_percentages = (class_counts / len(train_df) * 100).round(1)
for level, pct in class_percentages.items():
    print(f"Class {level}: {pct}%")

# Visualize the distribution
plt.figure(figsize=(8, 5))
class_counts.plot(kind='bar', color=['green', 'yellow', 'orange', 'red', 'darkred'])
plt.xlabel('DR Severity Level')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Training Set')
plt.xticks(range(5), ['0: No DR', '1: Mild', '2: Moderate', '3: Severe', '4: Proliferative'], rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
print("\nSaved chart to class_distribution.png")