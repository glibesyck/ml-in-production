import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import numpy as np
import os

class SimpleCNN(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # For 224x224 input: after two pooling layers -> 56x56
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)
    
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
        self.log("step", self.current_epoch)
        
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
        self.log("step", self.current_epoch)
        
        self.print(f"Epoch {self.current_epoch}: Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.4f}")
        
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

def train():

    wandb.init()
    
    lr = wandb.config.lr  

    wandb_logger = WandbLogger(project="dead-leaves-binary-membership-classifier", log_model="all", name=f"SimpleCNN_lr_{lr}")

    model = SimpleCNN(lr=lr)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint(f'model_lr_{lr}.ckpt')

    wandb.finish()

if __name__ == "__main__":
    sweep_config = {
    "method": "grid",
    "metric": {"name": "val/epoch_loss", "goal": "minimize"},
    "parameters": {
        "lr": {"values": [1e-4, 5e-4, 1e-3]} 
    }
}
    sweep_id = wandb.sweep(sweep_config, project="dead-leaves-binary-membership-classifier")

    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])

    train_data = ImageDataset("data/train", transform=transform)
    val_data = ImageDataset("data/val", transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=0)

    wandb.agent(sweep_id, train, count=3) 