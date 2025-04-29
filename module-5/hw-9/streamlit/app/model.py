import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import pytorch_lightning as pl

class SimpleCNN(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr

        self.conv1 = nn.Conv2d(1, 16, kernel_size=11, padding=5)
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


def load_model(checkpoint_path="checkpoints/cnn-2layers-11-3-kernel.ckpt"):
    model = SimpleCNN.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def predict(model, image: Image.Image) -> bool:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img_tensor = transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        output = model(img_tensor)
        return bool(torch.sigmoid(output).item() > 0.5)
