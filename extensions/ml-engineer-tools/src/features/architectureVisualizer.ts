import * as vscode from 'vscode';

export class ArchitectureVisualizer {
    private architecturePanel: vscode.WebviewPanel | undefined;

    async show() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const document = editor.document;
        const text = document.getText();

        // Analyze the code to extract model architecture
        const architecture = this.analyzeArchitecture(text);
        
        if (architecture.layers.length === 0) {
            vscode.window.showInformationMessage('No model architecture detected in current file');
            return;
        }

        await this.showArchitectureVisualization(architecture);
    }

    async showMiniMap() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        // Create a mini-map in the top-right corner
        const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        statusBarItem.text = "🧠 Model Flow";
        statusBarItem.tooltip = "Click to show full architecture";
        statusBarItem.command = 'mlTools.showArchitecture';
        statusBarItem.show();

        // Auto-hide after 10 seconds
        setTimeout(() => {
            statusBarItem.dispose();
        }, 10000);
    }

    private analyzeArchitecture(text: string): ModelArchitecture {
        const architecture: ModelArchitecture = {
            framework: this.detectFramework(text),
            modelName: this.extractModelName(text),
            layers: [],
            connections: [],
            parameters: 0,
            memoryUsage: '0 MB'
        };

        // Extract layers
        architecture.layers = this.extractLayers(text);
        
        // Calculate connections
        architecture.connections = this.calculateConnections(architecture.layers);
        
        // Calculate parameters and memory
        architecture.parameters = this.calculateTotalParameters(architecture.layers);
        architecture.memoryUsage = this.estimateMemoryUsage(architecture.parameters);

        return architecture;
    }

    private detectFramework(text: string): 'pytorch' | 'tensorflow' | 'unknown' {
        if (text.includes('torch.') || text.includes('nn.')) {
            return 'pytorch';
        } else if (text.includes('tf.') || text.includes('keras')) {
            return 'tensorflow';
        }
        return 'unknown';
    }

    private extractModelName(text: string): string {
        // Look for class definitions
        const classMatch = text.match(/class\s+(\w+)\s*\(/);
        if (classMatch) {
            return classMatch[1];
        }

        // Look for model variable assignments
        const modelMatch = text.match(/(\w+)\s*=\s*(?:nn\.Sequential|tf\.keras\.Sequential)/);
        if (modelMatch) {
            return modelMatch[1];
        }

        return 'Model';
    }

    private extractLayers(text: string): Layer[] {
        const layers: Layer[] = [];
        let layerIndex = 0;

        // PyTorch patterns
        const pytorchPatterns = [
            { pattern: /nn\.Linear\s*\(\s*(\d+),\s*(\d+)/g, type: 'Linear' },
            { pattern: /nn\.Conv2d\s*\(\s*(\d+),\s*(\d+),\s*(\d+)/g, type: 'Conv2d' },
            { pattern: /nn\.Conv1d\s*\(\s*(\d+),\s*(\d+),\s*(\d+)/g, type: 'Conv1d' },
            { pattern: /nn\.MaxPool2d\s*\(\s*(\d+)/g, type: 'MaxPool2d' },
            { pattern: /nn\.AvgPool2d\s*\(\s*(\d+)/g, type: 'AvgPool2d' },
            { pattern: /nn\.BatchNorm2d\s*\(\s*(\d+)/g, type: 'BatchNorm2d' },
            { pattern: /nn\.BatchNorm1d\s*\(\s*(\d+)/g, type: 'BatchNorm1d' },
            { pattern: /nn\.Dropout\s*\(\s*([\d.]+)/g, type: 'Dropout' },
            { pattern: /nn\.ReLU\s*\(/g, type: 'ReLU' },
            { pattern: /nn\.Sigmoid\s*\(/g, type: 'Sigmoid' },
            { pattern: /nn\.Tanh\s*\(/g, type: 'Tanh' },
            { pattern: /nn\.Softmax\s*\(/g, type: 'Softmax' },
            { pattern: /nn\.LSTM\s*\(\s*(\d+),\s*(\d+)/g, type: 'LSTM' },
            { pattern: /nn\.GRU\s*\(\s*(\d+),\s*(\d+)/g, type: 'GRU' },
        ];

        // TensorFlow patterns
        const tensorflowPatterns = [
            { pattern: /tf\.keras\.layers\.Dense\s*\(\s*(\d+)/g, type: 'Dense' },
            { pattern: /tf\.keras\.layers\.Conv2D\s*\(\s*(\d+),\s*\((\d+),\s*(\d+)\)/g, type: 'Conv2D' },
            { pattern: /tf\.keras\.layers\.MaxPooling2D\s*\(\s*\((\d+),\s*(\d+)\)/g, type: 'MaxPooling2D' },
            { pattern: /tf\.keras\.layers\.BatchNormalization\s*\(/g, type: 'BatchNormalization' },
            { pattern: /tf\.keras\.layers\.Dropout\s*\(\s*([\d.]+)/g, type: 'Dropout' },
            { pattern: /tf\.keras\.layers\.ReLU\s*\(/g, type: 'ReLU' },
            { pattern: /tf\.keras\.layers\.Sigmoid\s*\(/g, type: 'Sigmoid' },
            { pattern: /tf\.keras\.layers\.LSTM\s*\(\s*(\d+)/g, type: 'LSTM' },
        ];

        const allPatterns = [...pytorchPatterns, ...tensorflowPatterns];

        for (const { pattern, type } of allPatterns) {
            let match;
            while ((match = pattern.exec(text)) !== null) {
                const layer: Layer = {
                    id: `layer_${layerIndex++}`,
                    name: type,
                    type: this.categorizeLayerType(type),
                    inputShape: this.extractInputShape(match, type),
                    outputShape: this.extractOutputShape(match, type),
                    parameters: this.calculateLayerParameters(match, type),
                    position: { x: 0, y: layerIndex * 80 } // Will be calculated later
                };
                layers.push(layer);
            }
        }

        // Calculate positions for better visualization
        this.calculateLayerPositions(layers);

        return layers;
    }

    private categorizeLayerType(type: string): 'input' | 'dense' | 'conv' | 'pool' | 'norm' | 'activation' | 'output' | 'recurrent' {
        if (type.includes('Conv')) return 'conv';
        if (type.includes('Pool')) return 'pool';
        if (type.includes('BatchNorm')) return 'norm';
        if (type.includes('ReLU') || type.includes('Sigmoid') || type.includes('Tanh') || type.includes('Softmax')) return 'activation';
        if (type.includes('Linear') || type.includes('Dense')) return 'dense';
        if (type.includes('LSTM') || type.includes('GRU')) return 'recurrent';
        return 'dense';
    }

    private extractInputShape(match: RegExpExecArray, type: string): number[] {
        // Simplified shape extraction - would need more sophisticated parsing in real implementation
        if (type.includes('Conv2d') && match[1]) {
            return [parseInt(match[1])]; // Input channels
        }
        if (type.includes('Linear') && match[1]) {
            return [parseInt(match[1])]; // Input features
        }
        return [];
    }

    private extractOutputShape(match: RegExpExecArray, type: string): number[] {
        if (type.includes('Conv2d') && match[2]) {
            return [parseInt(match[2])]; // Output channels
        }
        if (type.includes('Linear') && match[2]) {
            return [parseInt(match[2])]; // Output features
        }
        return [];
    }

    private calculateLayerParameters(match: RegExpExecArray, type: string): number {
        if (type.includes('Linear') && match[1] && match[2]) {
            const input = parseInt(match[1]);
            const output = parseInt(match[2]);
            return input * output + output; // weights + bias
        }
        if (type.includes('Conv2d') && match[1] && match[2] && match[3]) {
            const inChannels = parseInt(match[1]);
            const outChannels = parseInt(match[2]);
            const kernelSize = parseInt(match[3]);
            return inChannels * outChannels * kernelSize * kernelSize + outChannels;
        }
        return 0;
    }

    private calculateLayerPositions(layers: Layer[]) {
        const spacing = 120;
        let currentY = 50;
        
        for (let i = 0; i < layers.length; i++) {
            layers[i].position = {
                x: 100 + (i % 3) * 200, // Arrange in columns
                y: currentY + Math.floor(i / 3) * spacing
            };
        }
    }

    private calculateConnections(layers: Layer[]): Connection[] {
        const connections: Connection[] = [];
        
        for (let i = 0; i < layers.length - 1; i++) {
            connections.push({
                from: layers[i].id,
                to: layers[i + 1].id,
                dataFlow: this.analyzeDataFlow(layers[i], layers[i + 1])
            });
        }
        
        return connections;
    }

    private analyzeDataFlow(fromLayer: Layer, toLayer: Layer): DataFlow {
        return {
            shapeTransformation: `${fromLayer.outputShape.join('×')} → ${toLayer.inputShape.join('×')}`,
            compatible: this.areShapesCompatible(fromLayer.outputShape, toLayer.inputShape)
        };
    }

    private areShapesCompatible(outputShape: number[], inputShape: number[]): boolean {
        // Simplified compatibility check
        if (outputShape.length === 0 || inputShape.length === 0) return true;
        return outputShape[outputShape.length - 1] === inputShape[0];
    }

    private calculateTotalParameters(layers: Layer[]): number {
        return layers.reduce((total, layer) => total + layer.parameters, 0);
    }

    private estimateMemoryUsage(parameters: number): string {
        const bytes = parameters * 4; // Assuming float32
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    }

    private async showArchitectureVisualization(architecture: ModelArchitecture) {
        if (this.architecturePanel) {
            this.architecturePanel.dispose();
        }

        this.architecturePanel = vscode.window.createWebviewPanel(
            'architectureVisualizer',
            `🧠 ${architecture.modelName} Architecture`,
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        this.architecturePanel.webview.html = this.getArchitectureHTML(architecture);
        
        vscode.window.showInformationMessage(
            `📊 Visualizing ${architecture.modelName} with ${architecture.layers.length} layers`
        );
    }

    private getArchitectureHTML(architecture: ModelArchitecture): string {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Architecture</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: #1e1e1e; 
                    color: #d4d4d4; 
                    overflow: hidden;
                }
                .header {
                    background: #252526;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    border-left: 4px solid #007acc;
                }
                .stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 10px;
                    margin-bottom: 20px;
                }
                .stat-item {
                    background: #2d2d30;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                }
                .architecture-container {
                    background: #252526;
                    border-radius: 8px;
                    padding: 20px;
                    height: 70vh;
                    overflow: auto;
                    position: relative;
                }
                .layer {
                    position: absolute;
                    background: #3c3c3c;
                    border: 2px solid #555;
                    border-radius: 8px;
                    padding: 10px;
                    min-width: 120px;
                    text-align: center;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                .layer:hover {
                    border-color: #007acc;
                    transform: scale(1.05);
                    z-index: 10;
                }
                .layer-dense { border-color: #4CAF50; }
                .layer-conv { border-color: #FF9800; }
                .layer-pool { border-color: #2196F3; }
                .layer-norm { border-color: #9C27B0; }
                .layer-activation { border-color: #F44336; }
                .layer-recurrent { border-color: #607D8B; }
                
                .layer-name {
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                .layer-shape {
                    font-size: 12px;
                    color: #888;
                }
                .layer-params {
                    font-size: 11px;
                    color: #aaa;
                    margin-top: 5px;
                }
                .connection {
                    position: absolute;
                    height: 2px;
                    background: #555;
                    z-index: 1;
                }
                .connection-compatible { background: #4CAF50; }
                .connection-incompatible { 
                    background: #F44336; 
                    animation: pulse 1s infinite;
                }
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                .legend {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #2d2d30;
                    padding: 15px;
                    border-radius: 8px;
                    font-size: 12px;
                }
                .legend-item {
                    display: flex;
                    align-items: center;
                    margin: 5px 0;
                }
                .legend-color {
                    width: 16px;
                    height: 16px;
                    border-radius: 3px;
                    margin-right: 8px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🧠 ${architecture.modelName} Architecture</h2>
                <p><strong>Framework:</strong> ${architecture.framework.toUpperCase()}</p>
            </div>

            <div class="stats">
                <div class="stat-item">
                    <strong>Layers</strong><br>
                    <span style="color: #4ec9b0;">${architecture.layers.length}</span>
                </div>
                <div class="stat-item">
                    <strong>Parameters</strong><br>
                    <span style="color: #dcdcaa;">${architecture.parameters.toLocaleString()}</span>
                </div>
                <div class="stat-item">
                    <strong>Memory</strong><br>
                    <span style="color: #f9c513;">${architecture.memoryUsage}</span>
                </div>
                <div class="stat-item">
                    <strong>Connections</strong><br>
                    <span style="color: #c586c0;">${architecture.connections.length}</span>
                </div>
            </div>

            <div class="architecture-container" id="architectureContainer">
                ${this.generateLayersHTML(architecture.layers)}
                ${this.generateConnectionsHTML(architecture.layers, architecture.connections)}
            </div>

            <div class="legend">
                <h4>Layer Types</h4>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4CAF50;"></div>
                    Dense/Linear
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF9800;"></div>
                    Convolution
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196F3;"></div>
                    Pooling
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #9C27B0;"></div>
                    Normalization
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #F44336;"></div>
                    Activation
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #607D8B;"></div>
                    Recurrent
                </div>
            </div>

            <script>
                // Add interactivity
                const layers = document.querySelectorAll('.layer');
                layers.forEach(layer => {
                    layer.addEventListener('click', () => {
                        const layerName = layer.querySelector('.layer-name').textContent;
                        const layerParams = layer.querySelector('.layer-params').textContent;
                        alert(\`Layer: \${layerName}\\nParameters: \${layerParams}\`);
                    });
                });

                // Auto-arrange layers if they overlap
                function checkOverlaps() {
                    // Simplified overlap detection
                    // In a real implementation, you'd use a more sophisticated layout algorithm
                }

                checkOverlaps();
            </script>
        </body>
        </html>
        `;
    }

    private generateLayersHTML(layers: Layer[]): string {
        return layers.map(layer => `
            <div class="layer layer-${layer.type}" 
                 style="left: ${layer.position.x}px; top: ${layer.position.y}px;"
                 data-layer-id="${layer.id}">
                <div class="layer-name">${layer.name}</div>
                <div class="layer-shape">
                    ${layer.inputShape.length > 0 ? layer.inputShape.join('×') : '?'} → 
                    ${layer.outputShape.length > 0 ? layer.outputShape.join('×') : '?'}
                </div>
                <div class="layer-params">${layer.parameters.toLocaleString()} params</div>
            </div>
        `).join('');
    }

    private generateConnectionsHTML(layers: Layer[], connections: Connection[]): string {
        return connections.map(connection => {
            const fromLayer = layers.find(l => l.id === connection.from);
            const toLayer = layers.find(l => l.id === connection.to);
            
            if (!fromLayer || !toLayer) return '';
            
            const x1 = fromLayer.position.x + 60; // Center of layer
            const y1 = fromLayer.position.y + 30;
            const x2 = toLayer.position.x + 60;
            const y2 = toLayer.position.y + 30;
            
            const length = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
            const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
            
            const compatibilityClass = connection.dataFlow.compatible ? 'connection-compatible' : 'connection-incompatible';
            
            return `
                <div class="connection ${compatibilityClass}"
                     style="left: ${x1}px; top: ${y1}px; width: ${length}px; transform: rotate(${angle}deg); transform-origin: 0 50%;"
                     title="${connection.dataFlow.shapeTransformation}">
                </div>
            `;
        }).join('');
    }
}

interface ModelArchitecture {
    framework: 'pytorch' | 'tensorflow' | 'unknown';
    modelName: string;
    layers: Layer[];
    connections: Connection[];
    parameters: number;
    memoryUsage: string;
}

interface Layer {
    id: string;
    name: string;
    type: 'input' | 'dense' | 'conv' | 'pool' | 'norm' | 'activation' | 'output' | 'recurrent';
    inputShape: number[];
    outputShape: number[];
    parameters: number;
    position: { x: number; y: number };
}

interface Connection {
    from: string;
    to: string;
    dataFlow: DataFlow;
}

interface DataFlow {
    shapeTransformation: string;
    compatible: boolean;
}