import * as vscode from 'vscode';

export class HyperparameterTweaker {
    private activeSliders = new Map<string, SliderInfo>();

    async openSlider(range: vscode.Range) {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const document = editor.document;
        const text = document.getText(range);
        
        const paramInfo = this.parseHyperparameter(text, range);
        if (paramInfo) {
            await this.showSliderInterface(paramInfo, editor);
        } else {
            vscode.window.showErrorMessage('No valid hyperparameter found at this location');
        }
    }

    async tweakValue(direction: 'increase' | 'decrease') {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const position = editor.selection.active;
        const range = this.findHyperparameterRange(editor.document, position);
        
        if (range) {
            const paramInfo = this.parseHyperparameter(editor.document.getText(range), range);
            if (paramInfo) {
                const newValue = this.adjustValue(paramInfo.value, paramInfo.type, direction);
                await this.updateParameterValue(editor, range, newValue);
            }
        }
    }

    async showPresetButtons() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const presets = [
            { label: '🚀 FP16 Training', description: 'Enable mixed precision', action: 'fp16' },
            { label: '⚡ XLA Compilation', description: 'Enable XLA acceleration', action: 'xla' },
            { label: '🔧 AMP Training', description: 'Automatic Mixed Precision', action: 'amp' },
            { label: '📊 Learning Rate Schedule', description: 'Add LR scheduling', action: 'lr_schedule' },
            { label: '🎯 Early Stopping', description: 'Add early stopping', action: 'early_stop' },
            { label: '💾 Checkpointing', description: 'Add model checkpoints', action: 'checkpoint' }
        ];

        const selected = await vscode.window.showQuickPick(presets, {
            placeHolder: 'Select optimization preset to add'
        });

        if (selected) {
            await this.applyPreset(selected.action, editor);
        }
    }

    async compareRuns() {
        const panel = vscode.window.createWebviewPanel(
            'hyperparameterComparison',
            '📊 Hyperparameter Comparison',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = this.getComparisonHTML();
    }

    async randomizeParameters() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const document = editor.document;
        const text = document.getText();
        const parameters = this.findAllHyperparameters(text);

        if (parameters.length === 0) {
            vscode.window.showInformationMessage('No hyperparameters found to randomize');
            return;
        }

        const selected = await vscode.window.showQuickPick(
            parameters.map(p => ({
                label: `${p.name}: ${p.value}`,
                description: `Line ${p.range.start.line + 1}`,
                parameter: p
            })),
            {
                placeHolder: 'Select parameters to randomize',
                canPickMany: true
            }
        );

        if (selected && selected.length > 0) {
            await this.randomizeSelectedParameters(selected.map(s => s.parameter), editor);
        }
    }

    private parseHyperparameter(text: string, range: vscode.Range): HyperparameterInfo | null {
        // Common hyperparameter patterns
        const patterns = [
            { regex: /(\w+)\s*=\s*([\d.]+)/, type: 'number' },
            { regex: /(\w+)\s*=\s*"([^"]+)"/, type: 'string' },
            { regex: /(\w+)\s*=\s*(True|False)/, type: 'boolean' }
        ];

        for (const pattern of patterns) {
            const match = pattern.regex.exec(text);
            if (match) {
                return {
                    name: match[1],
                    value: match[2],
                    type: pattern.type,
                    range,
                    suggestions: this.getSuggestions(match[1], match[2], pattern.type)
                };
            }
        }

        return null;
    }

    private getSuggestions(name: string, currentValue: string, type: string): any[] {
        const suggestions = [];
        
        if (type === 'number') {
            const numValue = parseFloat(currentValue);
            
            if (name.includes('learning_rate') || name === 'lr') {
                suggestions.push(0.001, 0.01, 0.1, 0.0001, 0.3);
            } else if (name.includes('batch_size')) {
                suggestions.push(16, 32, 64, 128, 256);
            } else if (name.includes('dropout')) {
                suggestions.push(0.1, 0.2, 0.3, 0.5);
            } else if (name.includes('epochs')) {
                suggestions.push(10, 50, 100, 200);
            } else {
                // Generate reasonable variations
                suggestions.push(
                    numValue * 0.1,
                    numValue * 0.5,
                    numValue * 2,
                    numValue * 10
                );
            }
        } else if (type === 'boolean') {
            suggestions.push(true, false);
        }

        return suggestions.filter(s => s.toString() !== currentValue);
    }

    private findHyperparameterRange(document: vscode.TextDocument, position: vscode.Position): vscode.Range | null {
        const line = document.lineAt(position.line);
        const lineText = line.text;
        
        // Look for parameter assignment on current line
        const paramPattern = /(\w+)\s*=\s*([\d.]+|"[^"]*"|True|False)/g;
        let match;
        
        while ((match = paramPattern.exec(lineText)) !== null) {
            const startCol = match.index;
            const endCol = match.index + match[0].length;
            
            if (position.character >= startCol && position.character <= endCol) {
                return new vscode.Range(
                    new vscode.Position(position.line, startCol),
                    new vscode.Position(position.line, endCol)
                );
            }
        }

        return null;
    }

    private adjustValue(value: string, type: string, direction: 'increase' | 'decrease'): string {
        if (type === 'number') {
            const numValue = parseFloat(value);
            const factor = direction === 'increase' ? 1.1 : 0.9;
            
            // Smart adjustment based on magnitude
            let newValue: number;
            if (numValue < 0.001) {
                newValue = direction === 'increase' ? numValue * 2 : numValue * 0.5;
            } else if (numValue < 1) {
                newValue = numValue * factor;
            } else {
                newValue = Math.round(numValue * factor);
            }
            
            return newValue.toString();
        } else if (type === 'boolean') {
            return value === 'True' ? 'False' : 'True';
        }
        
        return value;
    }

    private async updateParameterValue(editor: vscode.TextEditor, range: vscode.Range, newValue: string) {
        await editor.edit(editBuilder => {
            const originalText = editor.document.getText(range);
            const newText = originalText.replace(/(=\s*)([\d.]+|"[^"]*"|True|False)/, `$1${newValue}`);
            editBuilder.replace(range, newText);
        });

        vscode.window.showInformationMessage(`🎛️ Updated parameter to: ${newValue}`);
    }

    private async showSliderInterface(paramInfo: HyperparameterInfo, editor: vscode.TextEditor) {
        const panel = vscode.window.createWebviewPanel(
            'hyperparameterSlider',
            `🎛️ ${paramInfo.name} Slider`,
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = this.getSliderHTML(paramInfo);
        
        // Handle messages from webview
        panel.webview.onDidReceiveMessage(async (message) => {
            if (message.command === 'updateValue') {
                await this.updateParameterValue(editor, paramInfo.range, message.value);
            }
        });
    }

    private getSliderHTML(paramInfo: HyperparameterInfo): string {
        const isNumber = paramInfo.type === 'number';
        const currentValue = parseFloat(paramInfo.value);
        
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hyperparameter Tweaker</title>
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
                .slider-container {
                    background: #252526;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }
                .slider {
                    width: 100%;
                    height: 10px;
                    border-radius: 5px;
                    background: #3e3e42;
                    outline: none;
                    margin: 20px 0;
                }
                .slider::-webkit-slider-thumb {
                    appearance: none;
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    background: #007acc;
                    cursor: pointer;
                }
                .value-display {
                    font-size: 2em;
                    font-weight: bold;
                    color: #4ec9b0;
                    text-align: center;
                    margin: 20px 0;
                    padding: 15px;
                    background: #2d2d30;
                    border-radius: 8px;
                }
                .suggestions {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
                    gap: 10px;
                    margin-top: 20px;
                }
                .suggestion-btn {
                    padding: 10px;
                    background: #3e3e42;
                    border: none;
                    border-radius: 5px;
                    color: #d4d4d4;
                    cursor: pointer;
                    transition: background 0.3s;
                }
                .suggestion-btn:hover {
                    background: #007acc;
                }
                .controls {
                    display: flex;
                    gap: 10px;
                    margin-top: 20px;
                }
                .control-btn {
                    flex: 1;
                    padding: 10px;
                    background: #007acc;
                    border: none;
                    border-radius: 5px;
                    color: white;
                    cursor: pointer;
                    font-weight: bold;
                }
                .control-btn:hover {
                    background: #005a8b;
                }
                .info {
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
                <h2>🎛️ ${paramInfo.name} Tweaker</h2>
                <p><strong>Type:</strong> ${paramInfo.type}</p>
                <p><strong>Current:</strong> ${paramInfo.value}</p>
            </div>

            <div class="slider-container">
                <div class="value-display" id="valueDisplay">${paramInfo.value}</div>
                
                ${isNumber ? `
                <input type="range" 
                       class="slider" 
                       id="valueSlider"
                       min="${Math.max(0, currentValue * 0.1)}" 
                       max="${currentValue * 10}"
                       step="${currentValue < 1 ? currentValue * 0.01 : 1}"
                       value="${currentValue}">
                ` : ''}

                <div class="suggestions">
                    <h4 style="grid-column: 1/-1;">Quick Values:</h4>
                    ${paramInfo.suggestions.map(suggestion => `
                        <button class="suggestion-btn" onclick="setValue('${suggestion}')">${suggestion}</button>
                    `).join('')}
                </div>

                <div class="controls">
                    <button class="control-btn" onclick="applyValue()">Apply to Code</button>
                    <button class="control-btn" onclick="resetValue()">Reset</button>
                </div>
            </div>

            <div class="info">
                <h3>💡 Tips for ${paramInfo.name}</h3>
                ${this.getParameterTips(paramInfo.name)}
            </div>

            <script>
                const vscode = acquireVsCodeApi();
                const slider = document.getElementById('valueSlider');
                const display = document.getElementById('valueDisplay');
                const originalValue = '${paramInfo.value}';
                let currentValue = originalValue;

                if (slider) {
                    slider.addEventListener('input', function() {
                        currentValue = this.value;
                        display.textContent = currentValue;
                    });
                }

                function setValue(value) {
                    currentValue = value;
                    display.textContent = value;
                    if (slider) {
                        slider.value = value;
                    }
                }

                function applyValue() {
                    vscode.postMessage({
                        command: 'updateValue',
                        value: currentValue
                    });
                }

                function resetValue() {
                    setValue(originalValue);
                }
            </script>
        </body>
        </html>
        `;
    }

    private getParameterTips(paramName: string): string {
        const tips = {
            'learning_rate': 'Start with 0.001-0.01. Lower for fine-tuning, higher for initial training.',
            'batch_size': 'Powers of 2 work best (16, 32, 64, 128). Larger batches need higher learning rates.',
            'epochs': 'Use early stopping to avoid overfitting. Start with 10-100.',
            'dropout': 'Typically 0.1-0.5. Higher values prevent overfitting but may hurt performance.',
            'momentum': 'Usually 0.9 for SGD. Higher values smooth training but may overshoot.',
            'weight_decay': 'L2 regularization. Start with 1e-4 to 1e-2.'
        };

        const lowerName = paramName.toLowerCase();
        for (const [key, tip] of Object.entries(tips)) {
            if (lowerName.includes(key)) {
                return `<p>${tip}</p>`;
            }
        }

        return '<p>Experiment with different values and monitor validation metrics.</p>';
    }

    private findAllHyperparameters(text: string): HyperparameterInfo[] {
        const parameters: HyperparameterInfo[] = [];
        const lines = text.split('\n');
        
        for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
            const line = lines[lineIndex];
            const paramPattern = /(\w+)\s*=\s*([\d.]+|"[^"]*"|True|False)/g;
            let match;
            
            while ((match = paramPattern.exec(line)) !== null) {
                const paramName = match[1];
                const paramValue = match[2];
                
                // Check if it looks like a hyperparameter
                if (this.isLikelyHyperparameter(paramName)) {
                    const range = new vscode.Range(
                        new vscode.Position(lineIndex, match.index),
                        new vscode.Position(lineIndex, match.index + match[0].length)
                    );
                    
                    parameters.push({
                        name: paramName,
                        value: paramValue,
                        type: this.inferType(paramValue),
                        range,
                        suggestions: this.getSuggestions(paramName, paramValue, this.inferType(paramValue))
                    });
                }
            }
        }

        return parameters;
    }

    private isLikelyHyperparameter(name: string): boolean {
        const hyperparamNames = [
            'learning_rate', 'lr', 'batch_size', 'epochs', 'dropout', 'momentum',
            'weight_decay', 'beta1', 'beta2', 'epsilon', 'num_layers', 'hidden_size',
            'num_heads', 'dropout_rate', 'temperature', 'top_k', 'top_p'
        ];
        
        return hyperparamNames.some(param => 
            name.toLowerCase().includes(param) || param.includes(name.toLowerCase())
        );
    }

    private inferType(value: string): string {
        if (value === 'True' || value === 'False') return 'boolean';
        if (value.startsWith('"') && value.endsWith('"')) return 'string';
        if (!isNaN(parseFloat(value))) return 'number';
        return 'unknown';
    }

    private async randomizeSelectedParameters(parameters: HyperparameterInfo[], editor: vscode.TextEditor) {
        for (const param of parameters) {
            if (param.type === 'number') {
                const randomValue = this.generateRandomValue(param.name, param.value);
                await this.updateParameterValue(editor, param.range, randomValue.toString());
            }
        }
        
        vscode.window.showInformationMessage(`🎲 Randomized ${parameters.length} parameters`);
    }

    private generateRandomValue(name: string, currentValue: string): number {
        const numValue = parseFloat(currentValue);
        
        if (name.includes('learning_rate') || name === 'lr') {
            const logMin = Math.log10(1e-5);
            const logMax = Math.log10(1e-1);
            return Math.pow(10, Math.random() * (logMax - logMin) + logMin);
        } else if (name.includes('batch_size')) {
            const sizes = [8, 16, 32, 64, 128, 256];
            return sizes[Math.floor(Math.random() * sizes.length)];
        } else if (name.includes('dropout')) {
            return Math.random() * 0.5; // 0 to 0.5
        } else {
            // General random variation
            const factor = 0.1 + Math.random() * 1.9; // 0.1x to 2x
            return numValue * factor;
        }
    }

    private async applyPreset(action: string, editor: vscode.TextEditor) {
        const document = editor.document;
        const text = document.getText();
        
        let insertionCode = '';
        
        switch (action) {
            case 'fp16':
                insertionCode = '# Enable FP16 training\nfrom torch.cuda.amp import autocast, GradScaler\nscaler = GradScaler()\n';
                break;
            case 'xla':
                insertionCode = '# Enable XLA compilation\nimport torch_xla.core.xla_model as xm\ndevice = xm.xla_device()\n';
                break;
            case 'amp':
                insertionCode = '# Enable Automatic Mixed Precision\ntorch.backends.cudnn.allow_tf32 = True\n';
                break;
            case 'lr_schedule':
                insertionCode = '# Add learning rate scheduler\nfrom torch.optim.lr_scheduler import StepLR\nscheduler = StepLR(optimizer, step_size=10, gamma=0.1)\n';
                break;
            case 'early_stop':
                insertionCode = '# Early stopping setup\nbest_loss = float(\'inf\')\npatience = 10\npatience_counter = 0\n';
                break;
            case 'checkpoint':
                insertionCode = '# Model checkpointing\ncheckpoint_dir = "checkpoints/"\nos.makedirs(checkpoint_dir, exist_ok=True)\n';
                break;
        }
        
        if (insertionCode) {
            const position = new vscode.Position(0, 0);
            await editor.edit(editBuilder => {
                editBuilder.insert(position, insertionCode + '\n');
            });
            
            vscode.window.showInformationMessage(`✅ Added ${action} preset`);
        }
    }

    private getComparisonHTML(): string {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hyperparameter Comparison</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    padding: 20px; 
                    background: #1e1e1e; 
                    color: #d4d4d4; 
                }
                .comparison-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                .comparison-table th,
                .comparison-table td {
                    border: 1px solid #3e3e42;
                    padding: 12px;
                    text-align: left;
                }
                .comparison-table th {
                    background: #252526;
                    font-weight: bold;
                }
                .best-value {
                    background: #1e3c1e;
                    color: #4ec9b0;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <h2>📊 Hyperparameter Comparison</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Run 1</th>
                        <th>Run 2</th>
                        <th>Run 3</th>
                        <th>Best</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Learning Rate</td>
                        <td>0.001</td>
                        <td class="best-value">0.01</td>
                        <td>0.1</td>
                        <td>0.01</td>
                    </tr>
                    <tr>
                        <td>Batch Size</td>
                        <td>32</td>
                        <td>64</td>
                        <td class="best-value">128</td>
                        <td>128</td>
                    </tr>
                    <tr>
                        <td>Validation Accuracy</td>
                        <td>85.2%</td>
                        <td class="best-value">92.1%</td>
                        <td>78.9%</td>
                        <td>92.1%</td>
                    </tr>
                </tbody>
            </table>
            <p><em>This is a demo comparison. In practice, this would show actual run results.</em></p>
        </body>
        </html>
        `;
    }
}

interface HyperparameterInfo {
    name: string;
    value: string;
    type: string;
    range: vscode.Range;
    suggestions: any[];
}

interface SliderInfo {
    parameter: HyperparameterInfo;
    panel: vscode.WebviewPanel;
}