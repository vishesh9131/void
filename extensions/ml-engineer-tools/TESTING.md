# 🧪 ML Engineer Tools - Testing Guide

This guide provides step-by-step instructions for testing all features of the ML Engineer Tools extension.

## 🚀 Quick Setup

1. **Install the Extension**
   ```bash
   cd extensions/ml-engineer-tools
   npm install
   npm run compile
   ```

2. **Launch Extension Development Host**
   - Press `F5` in VS Code to launch a new Extension Development Host window
   - Or use `Ctrl+Shift+P` → "Debug: Start Debugging"

3. **Create Test Files**
   Create these sample files in your test workspace:

## 📁 Test Files Setup

### `test_model.py` - Main Test File
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Test Shape Inspector - hover over these tensors
X_train = torch.randn(100, 784)  # Should show (100, 784)
y_train = torch.randint(0, 10, (100,))  # Should show (100,)
data = torch.zeros([32, 3, 224, 224])  # Should show (32, 3, 224, 224)

# Test GPU Toggle - click the ⚡ icon or use Ctrl+Shift+G
model = nn.Linear(784, 10).cpu()
X_train = X_train.cpu()
criterion = nn.CrossEntropyLoss()

# Test NaN Detection - these should be highlighted
problematic_layer = nn.Sigmoid()  # High risk - vanishing gradients
risky_operation = torch.log(X_train)  # High risk - log of potentially negative values
large_lr = 1.0  # High risk - large learning rate
lstm_layer = nn.LSTM(100, 50)  # High risk - exploding gradients

# Test Memory Monitor - should show memory estimates
big_tensor = torch.randn(1000, 1000, 100)  # Should show ~400MB
conv_layer = nn.Conv2d(256, 512, 3)  # Should show parameter memory
batch_size = 128  # Should warn if too large

# Test Hyperparameter Tweaker - Alt+click these values
learning_rate = 0.001
batch_size = 32
epochs = 100
hidden_size = 256
dropout = 0.1
weight_decay = 1e-4

# Test Loss Plotter - click the 📈 icon
def train_model():
    for epoch in range(epochs):
        loss = criterion(model(X_train), y_train)  # Should detect loss pattern
        train_loss = loss.item()  # Should detect loss pattern
        print(f"Epoch {epoch}, Loss: {train_loss}")

# Test Seed Synchronizer - use Ctrl+Shift+S
torch.manual_seed(123)
np.random.seed(456)
# Should sync all to default seed (42)

# Test Import Cleaner - save file to trigger
import unused_library  # Should be removed on save
import matplotlib.pyplot as plt  # Should be kept (used above)

class TestModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### `colab_code.py` - Test Smart Paste
```python
# Copy this code and paste it into test_model.py to test Smart Paste
!pip install torch torchvision  # Should be removed
from google.colab import drive  # Should be removed
drive.mount('/content/drive')   # Should be removed
%matplotlib inline              # Should be removed

# This should trigger auto-import detection
model.fit(X_train, y_train)  # Should suggest: import tensorflow as tf
np.array([1, 2, 3])         # Should suggest: import numpy as np
pd.read_csv("data.csv")     # Should suggest: import pandas as pd

# Test path fixing
data = pd.read_csv("/content/drive/MyDrive/data.csv")  # Should fix to ./data/data.csv
```

### `shape_test.py` - Test Tensor Selection
```python
import torch
import torch.nn as nn

# Test Tensor Block Selection - double-click Conv2d to select entire layer
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 56 * 56, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Test Multi-cursor - Alt+click on 'batch_size' to add cursors to all occurrences
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_batch_size = batch_size // 2
```

## 🧪 Feature Testing Instructions

### 1. **Shape Inspector Testing**
1. Open `test_model.py`
2. Hover over tensor variables (`X_train`, `y_train`, `data`)
3. **Expected**: Tooltip showing shape, dtype, device, memory usage, framework
4. **Verify**: Shape format like `(100, 784)`, memory estimates, quick action links

### 2. **GPU Toggle Testing**
1. Look for ⚡ icon in editor title bar
2. Click the icon or press `Ctrl+Shift+G`
3. **Expected**: All `.cpu()` calls switch to `.cuda()` and vice versa
4. **Verify**: Code changes from `model.cpu()` to `model.cuda()`
5. **Test selection**: Select specific lines and toggle only those

### 3. **Import Cleaner Testing**
1. Add unused imports to `test_model.py`:
   ```python
   import unused_module
   import another_unused
   ```
2. Save the file (`Ctrl+S`)
3. **Expected**: Unused imports are automatically removed
4. **Verify**: Only used imports remain, formatting is preserved

### 4. **Smart Paste Testing**
1. Copy code from `colab_code.py`
2. Paste into `test_model.py`
3. **Expected**: 
   - Colab commands removed/commented
   - Missing imports added automatically
   - File paths updated to project structure
   - Warning notifications about changes
4. **Verify**: Check imports section and notification messages

### 5. **Seed Synchronizer Testing**
1. Press `Ctrl+Shift+S` or use Command Palette → "ML Tools: Sync All Seeds"
2. **Expected**: All seed values change to 42 (or configured default)
3. **Test presets**: `Ctrl+Shift+P` → "ML Tools: Show Seed Presets"
4. **Verify**: `torch.manual_seed(42)`, `np.random.seed(42)`, etc.

### 6. **Loss Plotter Testing**
1. Click 📈 icon in editor title bar or use Command Palette
2. **Expected**: Webview panel opens with interactive loss plot
3. **Test features**:
   - Toggle training/validation loss
   - Reset zoom
   - Export plot
4. **Verify**: Plotly chart with mock loss data, responsive controls

### 7. **NaN Detector Testing**
1. Open `test_model.py`
2. Look for wavy underlines on risky operations
3. Hover over highlighted code
4. **Expected**: 
   - Red underlines for high risk (Sigmoid, large learning rates)
   - Yellow for medium risk
   - Hover tooltips with suggestions
5. **Test auto-fix**: Click "Fix Automatically" links in tooltips

### 8. **Memory Monitor Testing**
1. Look for 🧠 annotations next to memory-intensive operations
2. Hover over memory indicators
3. **Expected**: Memory usage estimates, optimization suggestions
4. **Test report**: Command Palette → "ML Tools: Check Memory Usage"
5. **Verify**: Memory calculations for different layer types

### 9. **Tensor Selector Testing**
1. Double-click on `Conv2d` in `shape_test.py`
2. **Expected**: Entire layer definition selected
3. **Test multi-cursor**: Alt+click on `batch_size`
4. **Expected**: Cursors added to all `batch_size` occurrences
5. **Test block selection**: Place cursor in class definition, use Command Palette → "ML Tools: Select Tensor Block"

### 10. **Hyperparameter Tweaker Testing**
1. Alt+click on numeric hyperparameters (`learning_rate = 0.001`)
2. **Expected**: Webview panel with interactive sliders
3. **Test features**:
   - Move sliders to change values
   - Click preset buttons (Conservative, Aggressive, etc.)
   - Use quick value buttons
   - Randomize all parameters
4. **Verify**: Code updates in real-time as sliders move

### 11. **Code Colorizer Testing**
1. Open `test_model.py`
2. **Expected**: 
   - Tensor variables highlighted in orange
   - Data operations in bright orange italic
   - Model layers in teal with underlines
   - Loss functions in red with background
3. **Test mini-map**: Command Palette → "ML Tools: Show Architecture"
4. **Verify**: Syntax highlighting follows ML semantics

## 🎛️ Configuration Testing

Test different configuration options in VS Code settings:

```json
{
  "mlTools.enableShapeInspection": false,  // Disable shape hover
  "mlTools.enableGPUToggle": false,       // Hide GPU toggle button
  "mlTools.enableAutoImportClean": false, // Disable auto-cleaning
  "mlTools.enableNaNDetection": false,    // Hide NaN warnings
  "mlTools.enableMemoryMonitoring": false, // Hide memory annotations
  "mlTools.defaultSeed": 123,             // Change default seed
  "mlTools.enableSmartPaste": false,      // Disable smart paste
  "mlTools.enableShapeMatching": false    // Disable shape warnings
}
```

## 🐛 Common Issues & Troubleshooting

### Issue: Features not working
**Solution**: 
1. Ensure file is saved as `.py` (Python language mode)
2. Check that extension is activated (look for ML Tools in Extensions panel)
3. Reload VS Code window (`Ctrl+Shift+P` → "Developer: Reload Window")

### Issue: Hover tooltips not showing
**Solution**:
1. Verify `mlTools.enableShapeInspection` is `true`
2. Try hovering over different tensor patterns
3. Check VS Code's hover delay settings

### Issue: GPU toggle not visible
**Solution**:
1. Ensure file contains CUDA/CPU patterns
2. Check `mlTools.enableGPUToggle` setting
3. Look in editor title bar (may be collapsed in `...` menu)

### Issue: Import cleaning too aggressive
**Solution**:
1. Check import usage patterns (imports used in comments/strings may be removed)
2. Temporarily disable with `mlTools.enableAutoImportClean: false`
3. Manually trigger with Command Palette instead of auto-save

### Issue: Memory estimates seem wrong
**Solution**:
1. Memory estimates are approximations based on tensor shapes
2. Actual memory usage depends on many factors (gradients, intermediate tensors)
3. Use for relative comparisons, not absolute measurements

## 📊 Performance Testing

### Large File Testing
1. Create a file with 1000+ lines of ML code
2. Test that features remain responsive
3. Verify decorations update efficiently

### Memory Usage Testing
1. Monitor VS Code's memory usage with large models
2. Test with files containing many tensor operations
3. Ensure webview panels don't leak memory

## 🔍 Manual Verification Checklist

- [ ] Shape inspection shows correct tensor dimensions
- [ ] GPU toggle works for all CUDA/CPU patterns
- [ ] Import cleaning preserves necessary imports
- [ ] Smart paste handles Colab code correctly
- [ ] Seed synchronization updates all frameworks
- [ ] Loss plotter displays interactive charts
- [ ] NaN detection highlights risky operations
- [ ] Memory monitor shows realistic estimates
- [ ] Tensor selection works for ML constructs
- [ ] Hyperparameter sliders update code in real-time
- [ ] Code colorization follows ML semantics
- [ ] All keyboard shortcuts work as expected
- [ ] Configuration options take effect
- [ ] Extension works in different VS Code themes
- [ ] Features work with both PyTorch and TensorFlow code

## 📝 Test Results Template

```markdown
## Test Results - [Date]

### Environment
- VS Code Version: 
- Extension Version: 
- OS: 

### Feature Test Results
- [ ] Shape Inspector: ✅/❌ - Notes:
- [ ] GPU Toggle: ✅/❌ - Notes:
- [ ] Import Cleaner: ✅/❌ - Notes:
- [ ] Smart Paste: ✅/❌ - Notes:
- [ ] Seed Synchronizer: ✅/❌ - Notes:
- [ ] Loss Plotter: ✅/❌ - Notes:
- [ ] NaN Detector: ✅/❌ - Notes:
- [ ] Memory Monitor: ✅/❌ - Notes:
- [ ] Tensor Selector: ✅/❌ - Notes:
- [ ] Hyperparameter Tweaker: ✅/❌ - Notes:
- [ ] Code Colorizer: ✅/❌ - Notes:

### Performance
- Large file handling: ✅/❌
- Memory usage: ✅/❌
- Response time: ✅/❌

### Issues Found
1. 
2. 
3. 

### Suggestions
1. 
2. 
3. 
```

---

Happy Testing! 🚀 If you find any issues or have suggestions for improvements, please create an issue in the repository.