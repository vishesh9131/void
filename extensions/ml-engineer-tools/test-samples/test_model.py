import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Test Shape Inspector - hover over these tensors
X_train = torch.randn(100, 784)  # Should show (100, 784)
y_train = torch.randint(0, 10, (100,))  # Should show (100,)
data = torch.zeros([32, 3, 224, 224])  # Should show (32, 3, 224, 224)
features = torch.ones(64, 512, 7, 7)  # Should show (64, 512, 7, 7)

# Test GPU Toggle - click the ⚡ icon or use Ctrl+Shift+G
model = nn.Linear(784, 10).cpu()
X_train = X_train.cpu()
criterion = nn.CrossEntropyLoss()
device = torch.device('cpu')

# Test NaN Detection - these should be highlighted with warnings
problematic_layer = nn.Sigmoid()  # High risk - vanishing gradients
risky_operation = torch.log(X_train)  # High risk - log of potentially negative values
division_risk = X_train / torch.zeros_like(X_train)  # High risk - division by zero
large_lr = 1.0  # High risk - large learning rate
lstm_layer = nn.LSTM(100, 50)  # High risk - exploding gradients
sqrt_negative = torch.sqrt(torch.randn(10))  # Medium risk - sqrt of negative

# Test Memory Monitor - should show memory estimates beside these lines
big_tensor = torch.randn(1000, 1000, 100)  # Should show ~400MB
conv_layer = nn.Conv2d(256, 512, 3)  # Should show parameter memory
transformer_layer = nn.MultiheadAttention(512, 8)  # Should show attention memory
batch_size = 128  # Should warn if too large
huge_linear = nn.Linear(10000, 10000)  # Should show large memory usage

# Test Hyperparameter Tweaker - Alt+click these values to open sliders
learning_rate = 0.001
batch_size = 32
epochs = 100
hidden_size = 256
dropout = 0.1
weight_decay = 1e-4
momentum = 0.9
num_layers = 3
temperature = 2.0

# Test Loss Plotter - click the 📈 icon to visualize
def train_model():
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)  # Should detect loss pattern
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loss = loss.item()  # Should detect loss pattern
            
        val_loss = evaluate_model()  # Should detect loss pattern
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

# Test Seed Synchronizer - use Ctrl+Shift+S to sync all seeds
torch.manual_seed(123)
np.random.seed(456)
random.seed(789)
# Should sync all to default seed (42)

# Test Import Cleaner - save file to trigger auto-cleaning
import unused_library  # Should be removed on save
import another_unused_module  # Should be removed on save
import matplotlib.pyplot as plt  # Should be kept (used above)

# Test Tensor Selection and Multi-cursor - double-click or Alt+click
class TestModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Test layer selection
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        # Test tensor operation selection
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.batch_norm(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Test Code Colorizer - should highlight different ML constructs
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Model layers should be highlighted in teal
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 56 * 56, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Data operations should be highlighted in orange
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Tensor reshape operation
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return x

# Test different frameworks for compatibility
def tensorflow_example():
    # This should trigger TensorFlow import suggestions when pasted
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Should detect loss pattern
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Test data loading patterns
def create_data_loader():
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # Multi-cursor target
        shuffle=True,
        num_workers=4
    )
    return train_loader

# Test optimizer patterns
def setup_optimizer():
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,  # Multi-cursor target
        weight_decay=weight_decay  # Multi-cursor target
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    
    return optimizer, scheduler

# Test evaluation patterns
def evaluate_model():
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)  # Loss pattern
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

# Test gradient operations
def training_step():
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()  # Should show gradient computation memory
    
    # Test gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss.item()

# Test mixed precision training
def mixed_precision_training():
    from torch.cuda.amp import autocast, GradScaler
    
    scaler = GradScaler()
    
    for data, targets in train_loader:
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# Test attention mechanisms
class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.norm(attn_output + x)

# Test various tensor operations for shape analysis
def tensor_operations():
    # Various tensor creation patterns
    zeros = torch.zeros(32, 128)
    ones = torch.ones([64, 256])
    randn = torch.randn(16, 3, 224, 224)
    
    # Tensor manipulations
    reshaped = zeros.view(-1, 32)
    permuted = randn.permute(0, 2, 3, 1)
    concatenated = torch.cat([zeros, ones[:32]], dim=1)
    
    return reshaped, permuted, concatenated

# Test error-prone patterns for NaN detection
def risky_operations():
    # These should all be flagged by NaN detector
    x = torch.randn(10, 10)
    
    # Division by zero risk
    result1 = x / torch.zeros_like(x)
    
    # Log of negative values
    result2 = torch.log(x)
    
    # Square root of negative values
    result3 = torch.sqrt(x)
    
    # Sigmoid saturation
    result4 = torch.sigmoid(x * 100)
    
    return result1, result2, result3, result4

# Main execution
if __name__ == "__main__":
    # Create model instance
    model = TestModel(784, hidden_size, 10)
    
    # Setup training components
    train_loader = create_data_loader()
    optimizer, scheduler = setup_optimizer()
    
    # Training loop
    for epoch in range(epochs):
        loss = training_step()
        
        if epoch % 10 == 0:
            val_loss = evaluate_model()
            print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
            
        scheduler.step()
    
    print("Training completed!")