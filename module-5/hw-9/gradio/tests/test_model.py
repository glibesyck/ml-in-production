import pytest
import torch
from PIL import Image
import numpy as np
from app.model import SimpleCNN, predict

def test_simplecnn_forward():
    model = SimpleCNN()
    x = torch.randn(1, 1, 224, 224)
    output = model(x)
    assert output.shape == (1, 1)

def test_predict_function():
    model = SimpleCNN()
    # Create a dummy image
    image = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
    result = predict(model, image)
    assert isinstance(result, bool)