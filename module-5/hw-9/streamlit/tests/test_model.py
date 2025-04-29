from app.model import load_model, predict
from PIL import Image
import numpy as np

def test_model_prediction():
    model = load_model()
    dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224) * 255), 'L')
    result = predict(model, dummy_img)
    assert isinstance(result, bool)