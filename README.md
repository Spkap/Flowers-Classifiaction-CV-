

---

# Flower Species Classification using ResNet50

## Project Overview

This project focuses on the classification of 102 flower species from the Oxford 102 Flowers dataset using a deep learning model based on the ResNet50 architecture. The goal is to accurately identify the species of flowers from images by leveraging the power of convolutional neural networks.

## Libraries Used

- **torch** and **torchvision**: Core libraries for deep learning in Python. `torch` provides the building blocks for model creation, and `torchvision` offers datasets, models, and image transformation utilities.
- **numpy** and **scipy**: Fundamental libraries for numerical operations. `numpy` is used for array manipulations, and `scipy` provides advanced mathematical functions and signal processing.
- **matplotlib** and **seaborn**: Libraries for data visualization. `matplotlib` is a plotting library, and `seaborn` builds on it for more complex visualizations.
- **PIL**: The Python Imaging Library, used for opening, manipulating, and saving image files.

## Methods and Workflow

### Data Acquisition
The dataset was downloaded from the University of Oxford's official website and extracted for use.

```bash
# Downloading all the data using wget command if not already downloaded
[ ! -f setid.mat ] && wget 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'
[ ! -f imagelabels.mat ] && wget 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
[ ! -f 102flowers.tgz ] && wget 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'

# Extracting the data from archived files if not already extracted
[ -f 102flowers.tgz ] && tar xvf 102flowers.tgz

# Removing the useless archived file
[ -f 102flowers.tgz ] && rm -rf 102flowers.tgz
```

### Data Loading
The labels are stored in MATLAB files, which were loaded using `scipy.io`.

```python
from scipy.io import loadmat
import numpy as np

# Load the labels from MATLAB file
labels = loadmat('imagelabels.mat')['labels'].squeeze()
print("Labels loaded:", labels.shape)
```

### Data Visualization
Visualized the distribution of flower categories to identify any imbalances.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Counting the labels and making a histogram to see their frequency
sns.histplot(labels)
plt.xlabel('Category Number')
plt.ylabel('Count')
plt.title('Distribution of Flower Categories')
plt.show()
```

### Data Preprocessing
Images were resized to 224x224 pixels and normalized. Data augmentation techniques were applied to enhance the training data variability.

```python
from torchvision import transforms

# Transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Dataset Preparation
The dataset was split into training, validation, and testing subsets.

```python
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

# Load dataset
images_dir = 'flowers/102'
full_dataset = ImageFolder(images_dir, transform=transform)

# Split dataset into training, validation, and testing sets
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### Model Architecture
The ResNet50 model was used, pre-trained on ImageNet, and fine-tuned for flower species classification.

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet50 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)

# Modify the final layer to match the number of flower categories
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 102)
model = model.to(device)
```

### Training Setup
Configured the training process with the Adam optimizer and cross-entropy loss.

```python
import torch.optim as optim

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
train_metrics = {'loss': [], 'accuracy': []}
val_metrics = {'loss': [], 'accuracy': []}
```

### Training the Model
Trained the model, monitoring performance on training and validation sets.

```python
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    train_metrics['loss'].append(epoch_loss)
    train_metrics['accuracy'].append(epoch_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_acc = val_correct / val_total
    val_metrics['loss'].append(val_epoch_loss)
    val_metrics['accuracy'].append(val_epoch_acc)
    print(f'Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}')
```

### Evaluation and Results
Evaluated the model on the test set to determine its accuracy and loss.

```python
# Evaluate the model on test data
model.eval()
test_running_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
test_epoch_loss = test_running_loss / len(test_loader.dataset)
test_epoch_acc = test_correct / test_total
print(f'Test Loss: {test_epoch_loss:.4f}, Accuracy: {test_epoch_acc:.4f}')
```

### Results Visualization
Visualized the training and validation loss and accuracy over epochs.

```python
# Plot training and validation loss
plt.subplot(121)
plt.plot(train_metrics['loss'], label='Training Loss')
plt.plot(val_metrics['loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Plot training and testing accuracy
plt.subplot(122)
plt.plot(train_metrics['accuracy'], label='Training Accuracy')
plt.plot(val_metrics['accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()
```

## Conclusion
This project demonstrates the application of deep learning techniques to classify flower species using the ResNet50 model. The high accuracy achieved reflects the model's effectiveness and the thoroughness of the preprocessing and training process. 

## Repository Structure
- `Flowers CV.ipynb`: The Jupyter notebook containing all the code and explanations for the project.
- `data/`: Directory containing the dataset.
- `README.md`: This file, providing an overview and detailed documentation of the project.

## How to Use
1. Clone this repository to your local machine.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter notebook `Flowers CV.ipynb` to see the entire workflow and results.

## Future Work
Future improvements could include experimenting with other deep learning architectures, applying transfer learning to different datasets, and further tuning hyperparameters for even better performance.

---
