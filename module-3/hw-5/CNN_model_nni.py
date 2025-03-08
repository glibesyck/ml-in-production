import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import nni
import logging
from nni.utils import merge_parameter

logger = logging.getLogger('nni_cnn')

class SimpleCNN(pl.LightningModule):
    def __init__(self, lr, dropout_rate, conv1_channels, conv2_channels, fc_units):
        super().__init__()
        self.lr = lr
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # For 224x224 input: after two pooling layers -> 56x56
        self.fc1 = nn.Linear(conv2_channels * 56 * 56, fc_units)
        self.fc2 = nn.Linear(fc_units, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
    
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float().unsqueeze(1)
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        
        acc = ((outputs.sigmoid() > 0.5) == labels).float().mean()
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        
        self.train_losses.append(loss)
        self.train_accs.append(acc)
        
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        avg_acc = torch.stack(self.train_accs).mean()
        
        self.log("train/epoch_loss", avg_loss, prog_bar=True)
        self.log("train/epoch_acc", avg_acc, prog_bar=True)
        
        self.train_losses = []
        self.train_accs = []

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float().unsqueeze(1)
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        acc = ((outputs.sigmoid() > 0.5) == labels).float().mean()
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        
        self.val_losses.append(loss)
        self.val_accs.append(acc)
        
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        avg_acc = torch.stack(self.val_accs).mean()
        
        self.log("val/epoch_loss", avg_loss, prog_bar=True)
        self.log("val/epoch_acc", avg_acc, prog_bar=True)
        
        logger.info(f"Epoch {self.current_epoch}: Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.4f}")
        
        # Report metrics to NNI
        nni.report_intermediate_result(avg_acc.item())
        
        self.val_losses = []
        self.val_accs = []

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = ImageFolder(root=root_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label

def get_default_parameters():
    params = {
        'lr': 0.001,
        'batch_size': 128,
        'dropout_rate': 0.3,
        'conv1_channels': 16,
        'conv2_channels': 32,
        'fc_units': 128,
        'epochs': 5
    }
    return params

def main(params):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_data = ImageDataset("data/train", transform=transform)
    val_data = ImageDataset("data/val", transform=transform)

    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False, num_workers=0)

    model = SimpleCNN(
        lr=params['lr'],
        dropout_rate=params['dropout_rate'],
        conv1_channels=params['conv1_channels'],
        conv2_channels=params['conv2_channels'],
        fc_units=params['fc_units']
    )

    trainer = pl.Trainer(
        max_epochs=params['epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=False  # Disable wandb logging
    )

    trainer.fit(model, train_loader, val_loader)

    # Get final validation accuracy
    val_acc = torch.stack(model.val_accs).mean().item() if model.val_accs else 0
    
    # Report final result to NNI
    nni.report_final_result(val_acc)
    
    return val_acc

if __name__ == "__main__":
    try:
        # Get parameters from NNI tuner
        tuner_params = nni.get_next_parameter()
        logger.info(f"Received parameters from NNI: {tuner_params}")
        
        # Update default parameters with tuner suggestions
        params = get_default_parameters()
        params.update(tuner_params)
        
        main(params)
    except Exception as e:
        logger.error(f"Error in NNI experiment: {e}")
        raise