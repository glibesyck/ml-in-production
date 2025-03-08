import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import numpy as np
import ray
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    prepare_trainer,
)
import ray.train as ray_train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from filelock import FileLock
import argparse

class SimpleCNN(pl.LightningModule):
    def __init__(self, lr=0.001, dropout_rate=0.3, conv1_channels=16, conv2_channels=32, fc_units=128):
        super().__init__()
        self.save_hyperparameters()
        
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
        
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/acc", acc, prog_bar=True, sync_dist=True)
        
        self.train_losses.append(loss)
        self.train_accs.append(acc)
        
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        avg_acc = torch.stack(self.train_accs).mean()
        
        self.log("train/epoch_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("train/epoch_acc", avg_acc, prog_bar=True, sync_dist=True)
        
        self.train_losses = []
        self.train_accs = []

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float().unsqueeze(1)
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        acc = ((outputs.sigmoid() > 0.5) == labels).float().mean()
        
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)
        
        self.val_losses.append(loss)
        self.val_accs.append(acc)
        
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        avg_acc = torch.stack(self.val_accs).mean()
        
        self.log("val/epoch_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val/epoch_acc", avg_acc, prog_bar=True, sync_dist=True)
        
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

def train_func(config):
    batch_size = config.get("batch_size", 128)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Use FileLock to avoid conflicts when multiple processes access the same files
    with FileLock(os.path.expanduser("~/.data.lock")):
        train_dataset = ImageDataset("data/train", transform=transform)
        val_dataset = ImageDataset("data/val", transform=transform)
    
    model = SimpleCNN(
        lr=config.get("lr", 0.001),
        dropout_rate=config.get("dropout_rate", 0.3),
        conv1_channels=config.get("conv1_channels", 16),
        conv2_channels=config.get("conv2_channels", 32),
        fc_units=config.get("fc_units", 128)
    )
    
    strategy = RayDDPStrategy(find_unused_parameters=False)
    
    trainer = pl.Trainer(
        default_root_dir=config.get("save_dir", "./checkpoints"),
        max_epochs=config.get("epochs", 10),
        accelerator="mps",
        devices="auto",
        strategy=strategy,
        enable_progress_bar=True,
        logger=True
    )
    
    trainer = prepare_trainer(trainer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 4),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    model_path = os.path.join(config.get("save_dir", "./checkpoints"), "final_model.ckpt")
    trainer.save_checkpoint(model_path)
    
    print(f"Model saved to {model_path}")

def run_distributed_training(
    num_workers=2,
    use_gpu=True,
    lr=0.001,
    batch_size=128,
    epochs=10,
    save_dir="./distributed_checkpoints",
    dropout_rate=0.3,
    conv1_channels=16,
    conv2_channels=32,
    fc_units=128,
    num_data_workers=1
):
    if not ray.is_initialized():
        ray.init()
    
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": 2, "GPU": 1 if use_gpu else 0},
    )
    
    config = {
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "save_dir": save_dir,
        "dropout_rate": dropout_rate,
        "conv1_channels": conv1_channels,
        "conv2_channels": conv2_channels,
        "fc_units": fc_units,
        "num_workers": num_data_workers,
    }
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
    )
    
    result = trainer.fit()
    
    print(f"Training completed! Results: {result}")
    print(f"Model saved to {os.path.join(save_dir, 'final_model.ckpt')}")
    
    return result

if __name__ == "__main__":
    
    run_distributed_training()