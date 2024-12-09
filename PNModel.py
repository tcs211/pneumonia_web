import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import json
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(123)
np.random.seed(123)

# Custom Dataset class
class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
        # Create class to index mapping
        self.classes = sorted(dataframe['labels'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepaths']
        label = self.dataframe.iloc[idx]['labels']
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Convert label to tensor
        label_idx = self.class_to_idx[label]
        label_tensor = torch.tensor(label_idx)
        
        return image, label_tensor

# Model definition with configurable EfficientNet variant
class ChestXRayModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=2):
        super(ChestXRayModel, self).__init__()
        
        # Dictionary of available EfficientNet models and their input sizes
        self.efficientnet_configs = {
            'efficientnet_b0': (224, models.efficientnet_b0),
            'efficientnet_b1': (240, models.efficientnet_b1),
            'efficientnet_b2': (260, models.efficientnet_b2),
            'efficientnet_b3': (300, models.efficientnet_b3),
            'efficientnet_b4': (380, models.efficientnet_b4),
            'efficientnet_b5': (456, models.efficientnet_b5),
            'efficientnet_b6': (528, models.efficientnet_b6),
            'efficientnet_b7': (600, models.efficientnet_b7),
        }
        
        if model_name not in self.efficientnet_configs:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(self.efficientnet_configs.keys())}")
        
        # Load pre-trained EfficientNet
        self.input_size, model_fn = self.efficientnet_configs[model_name]
        self.efficientnet = model_fn(pretrained=True)
        
        # Get the number of features from the last layer
        if hasattr(self.efficientnet, 'classifier'):
            num_features = self.efficientnet.classifier[1].in_features
        else:
            num_features = self.efficientnet.fc.in_features
            
        # Modify classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, num_classes)
        )
        
        # Replace the classifier
        if hasattr(self.efficientnet, 'classifier'):
            self.efficientnet.classifier = self.classifier
        else:
            self.efficientnet.fc = self.classifier
            
    def forward(self, x):
        return self.efficientnet(x)
    
    def get_input_size(self):
        return self.input_size



class TrainingConfig:
    def __init__(self, model_name='efficientnet_b0', batch_size=16, num_epochs=20, learning_rate=0.001):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Add checkpoint directory to config
        self.checkpoint_dir = Path(f'../checkpoints/{model_name}')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Add training history file
        self.history_file = self.checkpoint_dir / 'training_history.json'

def train_model(config, train_df, valid_df, test_df):
    # Initialize model
    model = ChestXRayModel(model_name=config.model_name)
    input_size = model.get_input_size()
    
    # Create data loaders
    train_loader, valid_loader, test_loader, classes = create_data_loaders(
        train_df, valid_df, test_df, input_size, config.batch_size
    )
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=config.learning_rate)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Add progress bar for training
        pbar = tqdm(loader, desc='Training', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Add progress bar for validation
        pbar = tqdm(loader, desc='Validating', leave=False)
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    # Training loop
    best_val_acc = 0.0
    
    print(f"Training {config.model_name} for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint for each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        # Save epoch checkpoint
        torch.save(checkpoint, config.checkpoint_dir / f'epoch_{epoch+1}.pth')
        
        # Save best model separately
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, config.checkpoint_dir / 'best_model.pth')
            print(f'New best model saved! Validation Accuracy: {val_acc:.2f}%')
    
    # Save training history
    with open(config.history_file, 'w') as f:
        json.dump(history, f)
    
    # Load best model and evaluate
    best_checkpoint = torch.load(config.checkpoint_dir / 'best_model.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print('\nEvaluating best model on test set...')
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Get predictions for confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    
    print('\nGenerating classification report...')
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Processing test data'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    return model, test_acc, history

def plot_training_history(history):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    return plt


def create_data_loaders(train_df, valid_df, test_df, input_size, batch_size):
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ChestXRayDataset(train_df, transform=train_transform)
    valid_dataset = ChestXRayDataset(valid_df, transform=val_transform)
    test_dataset = ChestXRayDataset(test_df, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, train_dataset.classes


# Data loading functions
def get_data(data_dir):
    filepaths = []
    labels = []
    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)
    return filepaths, labels
