import torch
import numpy as np

# Test ML Engineer Tools extension
def test_model():
    model = torch.nn.Linear(10, 1)
    x = torch.randn(5, 10)
    y = model(x)
    return y

if __name__ == "__main__":
    result = test_model()
    print(f"Model output shape: {result.shape}")
