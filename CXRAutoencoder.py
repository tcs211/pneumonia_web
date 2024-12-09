import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class AutoencoderDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepaths']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, image  # Return same image as input and target

class ChestXRayAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ChestXRayAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 14 * 14),
            nn.Unflatten(1, (256, 14, 14)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def train_autoencoder(train_df, valid_df, num_epochs=20, batch_size=32, latent_dim=128):
    # Initialize model and training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChestXRayAutoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create datasets and dataloaders
    train_dataset = AutoencoderDataset(train_df, transform=transform)
    valid_dataset = AutoencoderDataset(valid_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for images, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, _ in tqdm(valid_loader, desc='Validating'):
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                valid_loss += loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Valid Loss = {avg_valid_loss:.4f}')
        
        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), 'best_autoencoder.pth')
    
    return model, train_losses, valid_losses

def evaluate_image(model, image_path, threshold=0.1):
    """Evaluate a single image and return reconstruction error."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get reconstruction
    model.eval()
    with torch.no_grad():
        reconstruction = model(image_tensor)
    
    # Calculate reconstruction error
    error = nn.MSELoss()(reconstruction, image_tensor).item()
    
    # Determine if image is likely a chest X-ray
    is_xray = error < threshold
    
    return error, is_xray, reconstruction.cpu()

def test_on_external_dataset(model, xray_test_df, non_xray_dir):
    """Test the model on both X-ray and non-X-ray images."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Process X-ray images
    xray_errors = []
    for _, row in tqdm(xray_test_df.iterrows(), desc='Processing X-ray images'):
        image_path = row['filepaths']
        error, _, _ = evaluate_image(model, image_path)
        xray_errors.append(error)
    
    # Process non-X-ray images
    non_xray_errors = []
    for img_file in tqdm(os.listdir(non_xray_dir), desc='Processing non-X-ray images'):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(non_xray_dir, img_file)
            error, _, _ = evaluate_image(model, image_path)
            non_xray_errors.append(error)
    
    # Calculate ROC curve
    y_true = [1] * len(xray_errors) + [0] * len(non_xray_errors)
    y_scores = [-x for x in xray_errors + non_xray_errors]  # Negative because lower error = higher confidence
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for X-ray Detection')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    return roc_auc, xray_errors, non_xray_errors
