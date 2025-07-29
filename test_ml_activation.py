# Test ML Engineer Tools Extension Activation
import torch
import numpy as np

# This should trigger the ML extension activation
def test_ml_features():
    # Create a simple model
    model = torch.nn.Linear(10, 1)

    # Create some data
    x = torch.randn(5, 10)
    y = model(x)

    # This should show tensor shape info on hover
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    return y

if __name__ == "__main__":
    result = test_ml_features()
    print("ML extension should be active now!")
