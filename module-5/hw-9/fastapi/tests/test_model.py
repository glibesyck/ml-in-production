import sys
import os
import pytest
from PIL import Image
import torch
from pathlib import Path

# Add the parent directory to the path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import SimpleCNN, predict

@pytest.fixture
def model():
    model = SimpleCNN()
    return model

def test_model_forward_pass(model):
    # Create a random input tensor
    x = torch.randn(1, 1, 224, 224)
    output = model(x)
    
    # Check shape and dtype
    assert output.shape == (1, 1)
    assert output.dtype == torch.float32

def test_predict_function():
    # Create a dummy model
    model = SimpleCNN()
    
    # Create a blank image
    image = Image.new('L', (224, 224), color=128)
    
    # Test prediction functionality
    result = predict(model, image)
    
    # Check that result is a boolean
    assert isinstance(result, bool)