# Google Colab Test Code - Copy and paste this into test_model.py to test Smart Paste

# These should be removed/commented by Smart Paste
!pip install torch torchvision torchaudio
!pip install transformers datasets
!apt-get update
!apt-get install -y graphviz

# Colab-specific imports that should be removed
from google.colab import drive, files, auth
from google.colab.patches import cv2_imshow
import google.colab.data_table

# Magic commands that should be removed
%matplotlib inline
%load_ext tensorboard
%reload_ext autoreload
%autoreload 2

# Drive mounting that should be removed
drive.mount('/content/drive')
files.upload()

# This should trigger auto-import detection when pasted
model.fit(X_train, y_train, epochs=10, validation_split=0.2)  # Should suggest: import tensorflow as tf
np.array([1, 2, 3])  # Should suggest: import numpy as np
pd.read_csv("data.csv")  # Should suggest: import pandas as pd
plt.plot([1, 2, 3])  # Should suggest: import matplotlib.pyplot as plt

# Test path fixing - these paths should be updated to project structure
data = pd.read_csv("/content/drive/MyDrive/dataset.csv")  # Should fix to ./data/dataset.csv
model_path = "/content/drive/MyDrive/models/best_model.pth"  # Should fix to ./models/best_model.pth
config_file = "/content/config.json"  # Should fix to ./config.json

# Test framework detection and import suggestions
# TensorFlow code without imports
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation='softmax')
])

# PyTorch code without imports
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Hugging Face code without imports
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Test shape mismatch detection
layer1 = nn.Linear(784, 256)  # Output: 256
layer2 = nn.Linear(128, 10)   # Input: 128 - MISMATCH! Should warn

# Test version conflicts
import tensorflow as tf  # TF 2.x
import keras  # Separate keras - should warn about conflict

# Regular Python code that should remain unchanged
def process_data(data):
    """This function should not be modified by Smart Paste."""
    processed = data * 2
    return processed

# This is normal ML code that should work fine
X = torch.randn(100, 784)
y = torch.randint(0, 10, (100,))