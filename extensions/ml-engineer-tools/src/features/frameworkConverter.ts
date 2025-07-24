import * as vscode from 'vscode';

export class FrameworkConverter {
    private tensorflowToPytorch = new Map([
        // Basic operations
        ['tf.Variable', 'torch.tensor'],
        ['tf.constant', 'torch.tensor'],
        ['tf.zeros', 'torch.zeros'],
        ['tf.ones', 'torch.ones'],
        ['tf.random.normal', 'torch.randn'],
        ['tf.random.uniform', 'torch.rand'],
        
        // Layers
        ['tf.keras.layers.Dense', 'nn.Linear'],
        ['tf.keras.layers.Conv2D', 'nn.Conv2d'],
        ['tf.keras.layers.MaxPooling2D', 'nn.MaxPool2d'],
        ['tf.keras.layers.BatchNormalization', 'nn.BatchNorm2d'],
        ['tf.keras.layers.Dropout', 'nn.Dropout'],
        ['tf.keras.layers.ReLU', 'nn.ReLU'],
        ['tf.keras.layers.Sigmoid', 'nn.Sigmoid'],
        ['tf.keras.layers.Tanh', 'nn.Tanh'],
        
        // Activations
        ['tf.nn.relu', 'torch.relu'],
        ['tf.nn.sigmoid', 'torch.sigmoid'],
        ['tf.nn.tanh', 'torch.tanh'],
        ['tf.nn.softmax', 'torch.softmax'],
        
        // Loss functions
        ['tf.keras.losses.SparseCategoricalCrossentropy', 'nn.CrossEntropyLoss'],
        ['tf.keras.losses.MeanSquaredError', 'nn.MSELoss'],
        ['tf.keras.losses.BinaryCrossentropy', 'nn.BCELoss'],
        
        // Optimizers
        ['tf.keras.optimizers.Adam', 'torch.optim.Adam'],
        ['tf.keras.optimizers.SGD', 'torch.optim.SGD'],
        ['tf.keras.optimizers.RMSprop', 'torch.optim.RMSprop'],
        
        // Math operations
        ['tf.reduce_mean', 'torch.mean'],
        ['tf.reduce_sum', 'torch.sum'],
        ['tf.reduce_max', 'torch.max'],
        ['tf.matmul', 'torch.matmul'],
        ['tf.transpose', 'torch.transpose'],
        ['tf.reshape', 'torch.reshape'],
    ]);

    private pytorchToTensorflow = new Map([
        // Basic operations
        ['torch.tensor', 'tf.constant'],
        ['torch.zeros', 'tf.zeros'],
        ['torch.ones', 'tf.ones'],
        ['torch.randn', 'tf.random.normal'],
        ['torch.rand', 'tf.random.uniform'],
        
        // Layers
        ['nn.Linear', 'tf.keras.layers.Dense'],
        ['nn.Conv2d', 'tf.keras.layers.Conv2D'],
        ['nn.MaxPool2d', 'tf.keras.layers.MaxPooling2D'],
        ['nn.BatchNorm2d', 'tf.keras.layers.BatchNormalization'],
        ['nn.Dropout', 'tf.keras.layers.Dropout'],
        ['nn.ReLU', 'tf.keras.layers.ReLU'],
        ['nn.Sigmoid', 'tf.keras.layers.Sigmoid'],
        ['nn.Tanh', 'tf.keras.layers.Tanh'],
        
        // Activations
        ['torch.relu', 'tf.nn.relu'],
        ['torch.sigmoid', 'tf.nn.sigmoid'],
        ['torch.tanh', 'tf.nn.tanh'],
        ['torch.softmax', 'tf.nn.softmax'],
        
        // Loss functions
        ['nn.CrossEntropyLoss', 'tf.keras.losses.SparseCategoricalCrossentropy'],
        ['nn.MSELoss', 'tf.keras.losses.MeanSquaredError'],
        ['nn.BCELoss', 'tf.keras.losses.BinaryCrossentropy'],
        
        // Optimizers
        ['torch.optim.Adam', 'tf.keras.optimizers.Adam'],
        ['torch.optim.SGD', 'tf.keras.optimizers.SGD'],
        ['torch.optim.RMSprop', 'tf.keras.optimizers.RMSprop'],
        
        // Math operations
        ['torch.mean', 'tf.reduce_mean'],
        ['torch.sum', 'tf.reduce_sum'],
        ['torch.max', 'tf.reduce_max'],
        ['torch.matmul', 'tf.matmul'],
        ['torch.transpose', 'tf.transpose'],
        ['torch.reshape', 'tf.reshape'],
    ]);

    async convert() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const document = editor.document;
        const selection = editor.selection;
        
        // Determine the source framework
        const text = selection.isEmpty ? document.getText() : document.getText(selection);
        const sourceFramework = this.detectFramework(text);
        
        if (sourceFramework === 'unknown') {
            vscode.window.showErrorMessage('Could not detect framework. Please select code containing TensorFlow or PyTorch.');
            return;
        }

        // Show conversion options
        const targetFramework = await this.showConversionOptions(sourceFramework);
        if (!targetFramework) {
            return;
        }

        // Perform conversion
        const convertedCode = this.performConversion(text, sourceFramework, targetFramework);
        
        if (convertedCode !== text) {
            await this.applyConversion(editor, selection, convertedCode, sourceFramework, targetFramework);
        } else {
            vscode.window.showInformationMessage('No convertible code patterns found.');
        }
    }

    private detectFramework(text: string): 'tensorflow' | 'pytorch' | 'unknown' {
        const tfPatterns = /tf\.|tensorflow|keras/g;
        const torchPatterns = /torch\.|nn\.|optim\./g;
        
        const tfCount = (text.match(tfPatterns) || []).length;
        const torchCount = (text.match(torchPatterns) || []).length;
        
        if (tfCount > torchCount && tfCount > 0) {
            return 'tensorflow';
        } else if (torchCount > 0) {
            return 'pytorch';
        }
        
        return 'unknown';
    }

    private async showConversionOptions(sourceFramework: string): Promise<string | undefined> {
        const options = sourceFramework === 'tensorflow' 
            ? [{ label: 'Convert to PyTorch', value: 'pytorch' }]
            : [{ label: 'Convert to TensorFlow', value: 'tensorflow' }];

        const selected = await vscode.window.showQuickPick(options, {
            placeHolder: `Convert from ${sourceFramework.charAt(0).toUpperCase() + sourceFramework.slice(1)}`
        });

        return selected?.value;
    }

    private performConversion(text: string, sourceFramework: string, targetFramework: string): string {
        let convertedText = text;
        
        if (sourceFramework === 'tensorflow' && targetFramework === 'pytorch') {
            convertedText = this.convertTensorFlowToPyTorch(convertedText);
        } else if (sourceFramework === 'pytorch' && targetFramework === 'tensorflow') {
            convertedText = this.convertPyTorchToTensorFlow(convertedText);
        }
        
        return convertedText;
    }

    private convertTensorFlowToPyTorch(text: string): string {
        let result = text;
        
        // Convert imports
        result = result.replace(/import tensorflow as tf/g, 'import torch\nimport torch.nn as nn\nimport torch.optim as optim');
        result = result.replace(/from tensorflow import keras/g, '# Converted to PyTorch - no direct equivalent');
        
        // Convert basic patterns
        for (const [tfPattern, pytorchReplacement] of this.tensorflowToPytorch) {
            const regex = new RegExp(tfPattern.replace(/\./g, '\\.'), 'g');
            result = result.replace(regex, pytorchReplacement);
        }
        
        // Convert model definition patterns
        result = this.convertTFModelToPyTorch(result);
        
        // Convert training loop patterns
        result = this.convertTFTrainingToPyTorch(result);
        
        // Add necessary imports if they don't exist
        if (result.includes('nn.') && !result.includes('import torch.nn')) {
            result = 'import torch.nn as nn\n' + result;
        }
        
        return result;
    }

    private convertPyTorchToTensorFlow(text: string): string {
        let result = text;
        
        // Convert imports
        result = result.replace(/import torch/g, 'import tensorflow as tf');
        result = result.replace(/import torch\.nn as nn/g, '# Converted to TensorFlow - layers are in tf.keras.layers');
        result = result.replace(/import torch\.optim as optim/g, '# Converted to TensorFlow - optimizers are in tf.keras.optimizers');
        
        // Convert basic patterns
        for (const [pytorchPattern, tfReplacement] of this.pytorchToTensorflow) {
            const regex = new RegExp(pytorchPattern.replace(/\./g, '\\.'), 'g');
            result = result.replace(regex, tfReplacement);
        }
        
        // Convert model definition patterns
        result = this.convertPyTorchModelToTF(result);
        
        // Convert training loop patterns
        result = this.convertPyTorchTrainingToTF(result);
        
        return result;
    }

    private convertTFModelToPyTorch(text: string): string {
        let result = text;
        
        // Convert Sequential model
        result = result.replace(
            /tf\.keras\.Sequential\(\[([\s\S]*?)\]\)/g,
            (match, layers) => {
                const convertedLayers = layers.replace(/tf\.keras\.layers\./g, 'nn.');
                return `nn.Sequential(\n${convertedLayers}\n)`;
            }
        );
        
        // Convert functional API model
        result = result.replace(
            /model = tf\.keras\.Model\(inputs=(\w+), outputs=(\w+)\)/g,
            '# PyTorch uses class-based model definition - convert to nn.Module'
        );
        
        // Convert model compilation
        result = result.replace(
            /model\.compile\(([\s\S]*?)\)/g,
            '# PyTorch: Define optimizer and loss function separately\n# optimizer = torch.optim.Adam(model.parameters())\n# criterion = nn.CrossEntropyLoss()'
        );
        
        return result;
    }

    private convertPyTorchModelToTF(text: string): string {
        let result = text;
        
        // Convert nn.Sequential
        result = result.replace(
            /nn\.Sequential\(([\s\S]*?)\)/g,
            (match, layers) => {
                const convertedLayers = layers.replace(/nn\./g, 'tf.keras.layers.');
                return `tf.keras.Sequential([\n${convertedLayers}\n])`;
            }
        );
        
        // Convert class-based model to functional API
        result = result.replace(
            /class (\w+)\(nn\.Module\):/g,
            '# TensorFlow: Convert to functional API or subclassing\n# class $1(tf.keras.Model):'
        );
        
        return result;
    }

    private convertTFTrainingToPyTorch(text: string): string {
        let result = text;
        
        // Convert model.fit()
        result = result.replace(
            /model\.fit\(([\s\S]*?)\)/g,
            '# PyTorch training loop:\n# for epoch in range(epochs):\n#     for batch in dataloader:\n#         optimizer.zero_grad()\n#         output = model(batch)\n#         loss = criterion(output, target)\n#         loss.backward()\n#         optimizer.step()'
        );
        
        // Convert model.evaluate()
        result = result.replace(
            /model\.evaluate\(([\s\S]*?)\)/g,
            '# PyTorch evaluation:\n# model.eval()\n# with torch.no_grad():\n#     for batch in test_loader:\n#         output = model(batch)\n#         # compute metrics'
        );
        
        return result;
    }

    private convertPyTorchTrainingToTF(text: string): string {
        let result = text;
        
        // Convert manual training loop to model.fit()
        const trainingLoopPattern = /for epoch in range\(.*?\):([\s\S]*?)optimizer\.step\(\)/g;
        result = result.replace(
            trainingLoopPattern,
            '# TensorFlow: Use model.fit() for training\n# model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))'
        );
        
        // Convert optimizer.zero_grad()
        result = result.replace(/optimizer\.zero_grad\(\)/g, '# TensorFlow handles gradients automatically');
        
        // Convert loss.backward()
        result = result.replace(/loss\.backward\(\)/g, '# TensorFlow handles backpropagation automatically');
        
        return result;
    }

    private async applyConversion(
        editor: vscode.TextEditor, 
        selection: vscode.Selection, 
        convertedCode: string,
        sourceFramework: string,
        targetFramework: string
    ) {
        // Show preview first
        const showPreview = await vscode.window.showQuickPick(
            ['Apply conversion', 'Show preview first', 'Cancel'],
            { placeHolder: `Convert ${sourceFramework} → ${targetFramework}` }
        );

        if (showPreview === 'Cancel') {
            return;
        }

        if (showPreview === 'Show preview first') {
            await this.showConversionPreview(convertedCode, sourceFramework, targetFramework);
            return;
        }

        // Apply the conversion
        await editor.edit(editBuilder => {
            if (selection.isEmpty) {
                const fullRange = new vscode.Range(
                    editor.document.positionAt(0),
                    editor.document.positionAt(editor.document.getText().length)
                );
                editBuilder.replace(fullRange, convertedCode);
            } else {
                editBuilder.replace(selection, convertedCode);
            }
        });

        // Add necessary imports
        await this.addRequiredImports(editor, targetFramework);

        vscode.window.showInformationMessage(
            `🔄 Converted from ${sourceFramework} to ${targetFramework}!`
        );
    }

    private async showConversionPreview(convertedCode: string, sourceFramework: string, targetFramework: string) {
        const panel = vscode.window.createWebviewPanel(
            'frameworkConversion',
            `Framework Conversion: ${sourceFramework} → ${targetFramework}`,
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = this.getPreviewHTML(convertedCode, sourceFramework, targetFramework);
    }

    private getPreviewHTML(convertedCode: string, sourceFramework: string, targetFramework: string): string {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Framework Conversion Preview</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    padding: 20px; 
                    background: #1e1e1e; 
                    color: #d4d4d4; 
                }
                .header {
                    background: #252526;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    border-left: 4px solid #007acc;
                }
                .conversion-info {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .framework-badge {
                    padding: 5px 15px;
                    border-radius: 15px;
                    font-weight: bold;
                }
                .tensorflow { background: #ff6f00; color: white; }
                .pytorch { background: #ee4c2c; color: white; }
                .code-container {
                    background: #252526;
                    border-radius: 8px;
                    padding: 20px;
                    overflow-x: auto;
                }
                pre {
                    margin: 0;
                    white-space: pre-wrap;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                }
                .warning {
                    background: #3c1e1e;
                    border: 1px solid #d73a49;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                }
                .tips {
                    background: #1e3c1e;
                    border: 1px solid #28a745;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🔄 Framework Conversion Preview</h2>
                <div class="conversion-info">
                    <span class="framework-badge ${sourceFramework}">${sourceFramework.toUpperCase()}</span>
                    <span style="margin: 0 20px;">→</span>
                    <span class="framework-badge ${targetFramework}">${targetFramework.toUpperCase()}</span>
                </div>
            </div>

            <div class="code-container">
                <h3>Converted Code:</h3>
                <pre><code>${this.escapeHtml(convertedCode)}</code></pre>
            </div>

            <div class="warning">
                <h3>⚠️ Important Notes</h3>
                <ul>
                    <li>This is an automated conversion and may require manual adjustments</li>
                    <li>Some framework-specific features may not have direct equivalents</li>
                    <li>Please review and test the converted code thoroughly</li>
                    <li>Data loading and preprocessing patterns may need adaptation</li>
                </ul>
            </div>

            <div class="tips">
                <h3>💡 Conversion Tips</h3>
                <ul>
                    ${this.getConversionTips(sourceFramework, targetFramework)}
                </ul>
            </div>
        </body>
        </html>
        `;
    }

    private escapeHtml(text: string): string {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    private getConversionTips(sourceFramework: string, targetFramework: string): string {
        if (sourceFramework === 'tensorflow' && targetFramework === 'pytorch') {
            return `
                <li>PyTorch uses explicit training loops instead of model.fit()</li>
                <li>Remember to call optimizer.zero_grad() before each backward pass</li>
                <li>Use model.train() and model.eval() to switch between training and evaluation modes</li>
                <li>PyTorch tensors require .cuda() to move to GPU</li>
                <li>Data loading uses torch.utils.data.DataLoader</li>
            `;
        } else {
            return `
                <li>TensorFlow uses model.fit() for training instead of explicit loops</li>
                <li>Model compilation requires optimizer, loss, and metrics</li>
                <li>TensorFlow handles gradient computation automatically</li>
                <li>Use tf.data for efficient data pipeline</li>
                <li>Consider using tf.keras.Model for complex architectures</li>
            `;
        }
    }

    private async addRequiredImports(editor: vscode.TextEditor, targetFramework: string) {
        const document = editor.document;
        const text = document.getText();
        
        const importsToAdd: string[] = [];
        
        if (targetFramework === 'pytorch') {
            if (text.includes('torch.') && !text.includes('import torch')) {
                importsToAdd.push('import torch');
            }
            if (text.includes('nn.') && !text.includes('import torch.nn')) {
                importsToAdd.push('import torch.nn as nn');
            }
            if (text.includes('optim.') && !text.includes('import torch.optim')) {
                importsToAdd.push('import torch.optim as optim');
            }
        } else if (targetFramework === 'tensorflow') {
            if (text.includes('tf.') && !text.includes('import tensorflow')) {
                importsToAdd.push('import tensorflow as tf');
            }
        }
        
        if (importsToAdd.length > 0) {
            await editor.edit(editBuilder => {
                editBuilder.insert(new vscode.Position(0, 0), importsToAdd.join('\n') + '\n\n');
            });
        }
    }
}