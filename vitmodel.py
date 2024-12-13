import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import json
from pathlib import Path
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# Set random seed for reproducibility
torch.manual_seed(123)
np.random.seed(123)

# Data loading function
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


class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
        # Create class to index mapping
        self.classes = sorted(dataframe['labels'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Compute weights for balanced sampling
        class_counts = dataframe['labels'].value_counts()
        self.weights = [1.0/class_counts[self.dataframe.iloc[idx]['labels']] for idx in range(len(self.dataframe))]
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepaths']
        label = self.dataframe.iloc[idx]['labels']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
            
        # Apply transforms if available
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label_idx = self.class_to_idx[label]
        label_tensor = torch.tensor(label_idx)
        
        return image, label_tensor

# 2. Enhanced data augmentation pipeline
def create_data_loaders(train_df, valid_df, test_df, batch_size=16):
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Larger initial size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=15
        ),
        transforms.CenterCrop(224),  # Center crop after augmentation
        # ColorJitter before ToTensor
        transforms.ColorJitter(
            brightness=0.25, 
            contrast=0.25,
            saturation=0.15,
            hue=0.1
        ),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        # ToTensor after all PIL-based transforms
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),  # RandomErasing works on tensors
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation/Test transforms without augmentation
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    train_dataset = ChestXRayDataset(train_df, transform=train_transform)
    valid_dataset = ChestXRayDataset(valid_df, transform=eval_transform)
    test_dataset = ChestXRayDataset(test_df, transform=eval_transform)
    
    # Create weighted sampler for training data
    sampler = WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader, train_dataset.classes
    

class TrainingConfig:
    def __init__(self, batch_size=16, num_epochs=10, learning_rate=2e-5):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_name = "vit-chest-xray"
        
        # Add regularization parameters
        self.weight_decay = 0.01
        self.dropout_rate = 0.2
        
        # Add learning rate scheduler parameters
        self.min_lr = 1e-6
        self.warmup_epochs = 2
        
        # Add checkpoint directory
        self.checkpoint_dir = Path(f'../checkpoints/{self.model_name}')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Add training history file
        self.history_file = self.checkpoint_dir / 'training_history.json'


class ChestXRayViT(nn.Module):
    def __init__(self, num_classes=2, pretrained_model="google/vit-base-patch16-224"):
        super(ChestXRayViT, self).__init__()
        
        self.vit = ViTForImageClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Add an extra classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        vit_output = self.vit(x, output_hidden_states=True)
        pooled_output = vit_output.hidden_states[-1][:, 0]  # Get CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train_model(config, train_df, valid_df, test_df):
    # Create data loaders
    train_loader, valid_loader, test_loader, classes = create_data_loaders(
        train_df, valid_df, test_df, config.batch_size
    )
    
    # Initialize model
    model = ChestXRayViT(num_classes=len(classes))
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
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
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
                
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    # Training loop
    best_val_acc = 0.0
    
    print(f"Training ViT for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        torch.save(checkpoint, config.checkpoint_dir / f'epoch_{epoch+1}.pth')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, config.checkpoint_dir / 'best_model.pth')
            print(f'New best model saved! Validation Accuracy: {val_acc:.2f}%')
    
    # Save training history
    with open(config.history_file, 'w') as f:
        json.dump(history, f)
    
    # Evaluate best model on test set
    best_checkpoint = torch.load(config.checkpoint_dir / 'best_model.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    print('\nEvaluating best model on test set...')
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Generate classification report
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
    
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    return model, test_acc, history
