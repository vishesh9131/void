import * as vscode from 'vscode';

export class GradientVisualizer {
    async visualize() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const selection = editor.selection;
        const document = editor.document;
        const selectedText = document.getText(selection);

        if (!selectedText) {
            vscode.window.showErrorMessage('Please select a layer or model to visualize gradients');
            return;
        }

        // Analyze the selected code to determine the layer type
        const layerInfo = this.analyzeLayer(selectedText);
        
        if (layerInfo) {
            await this.showGradientVisualization(layerInfo);
        } else {
            vscode.window.showErrorMessage('Selected code does not appear to be a neural network layer');
        }
    }

    private analyzeLayer(code: string): LayerInfo | null {
        const layerPatterns = [
            { pattern: /nn\.Linear\s*\(\s*(\d+),\s*(\d+)/, type: 'Linear', framework: 'PyTorch' },
            { pattern: /nn\.Conv2d\s*\(\s*(\d+),\s*(\d+)/, type: 'Conv2d', framework: 'PyTorch' },
            { pattern: /tf\.keras\.layers\.Dense\s*\(\s*(\d+)/, type: 'Dense', framework: 'TensorFlow' },
            { pattern: /tf\.keras\.layers\.Conv2D\s*\(\s*(\d+)/, type: 'Conv2D', framework: 'TensorFlow' },
        ];

        for (const { pattern, type, framework } of layerPatterns) {
            const match = pattern.exec(code);
            if (match) {
                return {
                    type,
                    framework,
                    inputSize: parseInt(match[1]),
                    outputSize: match[2] ? parseInt(match[2]) : parseInt(match[1]),
                    code
                };
            }
        }

        return null;
    }

    private async showGradientVisualization(layerInfo: LayerInfo) {
        const panel = vscode.window.createWebviewPanel(
            'gradientVisualizer',
            `Gradient Analysis - ${layerInfo.type}`,
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = this.getGradientVisualizationHTML(layerInfo);
        
        // Show information message
        vscode.window.showInformationMessage(
            `🔍 Visualizing gradients for ${layerInfo.type} layer (${layerInfo.inputSize} → ${layerInfo.outputSize})`
        );
    }

    private getGradientVisualizationHTML(layerInfo: LayerInfo): string {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gradient Visualization</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    padding: 20px; 
                    background: #1e1e1e; 
                    color: #d4d4d4; 
                }
                .gradient-container { 
                    display: flex; 
                    flex-direction: column; 
                    gap: 20px; 
                }
                .layer-info { 
                    background: #252526; 
                    padding: 15px; 
                    border-radius: 8px; 
                    border-left: 4px solid #007acc;
                }
                .gradient-plot { 
                    background: #252526; 
                    padding: 20px; 
                    border-radius: 8px; 
                    text-align: center;
                    min-height: 300px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                }
                .gradient-bar { 
                    height: 20px; 
                    margin: 5px 0; 
                    border-radius: 3px; 
                    position: relative;
                }
                .gradient-label { 
                    font-size: 12px; 
                    margin-bottom: 5px; 
                }
                .stats { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                    gap: 10px; 
                    margin-top: 20px;
                }
                .stat-item { 
                    background: #2d2d30; 
                    padding: 10px; 
                    border-radius: 5px; 
                    text-align: center;
                }
                .potential-issues { 
                    background: #3c1e1e; 
                    border: 1px solid #d73a49; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin-top: 20px;
                }
                .issue-item { 
                    margin: 5px 0; 
                    padding: 5px 0;
                }
                .recommendations { 
                    background: #1e3c1e; 
                    border: 1px solid #28a745; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="gradient-container">
                <div class="layer-info">
                    <h2>🧠 ${layerInfo.type} Layer Analysis</h2>
                    <p><strong>Framework:</strong> ${layerInfo.framework}</p>
                    <p><strong>Input Size:</strong> ${layerInfo.inputSize}</p>
                    <p><strong>Output Size:</strong> ${layerInfo.outputSize}</p>
                    <p><strong>Parameters:</strong> ${this.calculateParameters(layerInfo)}</p>
                </div>

                <div class="gradient-plot">
                    <h3>📊 Gradient Flow Visualization</h3>
                    <div id="gradientBars">
                        ${this.generateGradientBars(layerInfo)}
                    </div>
                </div>

                <div class="stats">
                    <div class="stat-item">
                        <strong>Gradient Norm</strong><br>
                        <span style="color: #4ec9b0;">${this.estimateGradientNorm(layerInfo)}</span>
                    </div>
                    <div class="stat-item">
                        <strong>Memory Usage</strong><br>
                        <span style="color: #dcdcaa;">${this.estimateMemoryUsage(layerInfo)}</span>
                    </div>
                    <div class="stat-item">
                        <strong>Vanishing Risk</strong><br>
                        <span style="color: ${this.getVanishingRiskColor(layerInfo)}">${this.assessVanishingRisk(layerInfo)}</span>
                    </div>
                    <div class="stat-item">
                        <strong>Exploding Risk</strong><br>
                        <span style="color: ${this.getExplodingRiskColor(layerInfo)}">${this.assessExplodingRisk(layerInfo)}</span>
                    </div>
                </div>

                ${this.generatePotentialIssues(layerInfo)}
                ${this.generateRecommendations(layerInfo)}
            </div>

            <script>
                // Add some interactive features
                const gradientBars = document.querySelectorAll('.gradient-bar');
                gradientBars.forEach(bar => {
                    bar.addEventListener('mouseover', () => {
                        bar.style.opacity = '0.8';
                    });
                    bar.addEventListener('mouseout', () => {
                        bar.style.opacity = '1';
                    });
                });
            </script>
        </body>
        </html>
        `;
    }

    private calculateParameters(layerInfo: LayerInfo): string {
        let params = 0;
        if (layerInfo.type === 'Linear' || layerInfo.type === 'Dense') {
            params = layerInfo.inputSize * layerInfo.outputSize + layerInfo.outputSize; // weights + bias
        } else if (layerInfo.type === 'Conv2d' || layerInfo.type === 'Conv2D') {
            // Assume 3x3 kernel for estimation
            params = layerInfo.inputSize * layerInfo.outputSize * 9 + layerInfo.outputSize;
        }
        return params.toLocaleString();
    }

    private generateGradientBars(layerInfo: LayerInfo): string {
        const layers = Math.min(10, Math.max(3, Math.floor(layerInfo.outputSize / 10)));
        let bars = '';
        
        for (let i = 0; i < layers; i++) {
            const intensity = Math.random() * 0.8 + 0.2; // Random for demo
            const color = this.getGradientColor(intensity);
            bars += `
                <div class="gradient-label">Layer ${i + 1}</div>
                <div class="gradient-bar" style="background: linear-gradient(to right, ${color}20, ${color}); width: ${intensity * 100}%;"></div>
            `;
        }
        
        return bars;
    }

    private getGradientColor(intensity: number): string {
        if (intensity < 0.3) return '#4ec9b0'; // Good gradients
        if (intensity < 0.7) return '#dcdcaa'; // Moderate gradients
        return '#d73a49'; // High gradients (potential exploding)
    }

    private estimateGradientNorm(layerInfo: LayerInfo): string {
        // Simplified estimation based on layer size
        const norm = Math.sqrt(layerInfo.inputSize * layerInfo.outputSize) / 100;
        return norm.toFixed(4);
    }

    private estimateMemoryUsage(layerInfo: LayerInfo): string {
        const params = layerInfo.inputSize * layerInfo.outputSize;
        const bytes = params * 4; // Assuming float32
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    private assessVanishingRisk(layerInfo: LayerInfo): string {
        if (layerInfo.inputSize > 1000 || layerInfo.outputSize > 1000) return 'High';
        if (layerInfo.inputSize > 100 || layerInfo.outputSize > 100) return 'Medium';
        return 'Low';
    }

    private assessExplodingRisk(layerInfo: LayerInfo): string {
        const ratio = layerInfo.outputSize / layerInfo.inputSize;
        if (ratio > 4 || ratio < 0.25) return 'High';
        if (ratio > 2 || ratio < 0.5) return 'Medium';
        return 'Low';
    }

    private getVanishingRiskColor(layerInfo: LayerInfo): string {
        const risk = this.assessVanishingRisk(layerInfo);
        if (risk === 'High') return '#d73a49';
        if (risk === 'Medium') return '#f9c513';
        return '#4ec9b0';
    }

    private getExplodingRiskColor(layerInfo: LayerInfo): string {
        const risk = this.assessExplodingRisk(layerInfo);
        if (risk === 'High') return '#d73a49';
        if (risk === 'Medium') return '#f9c513';
        return '#4ec9b0';
    }

    private generatePotentialIssues(layerInfo: LayerInfo): string {
        const issues = [];
        
        if (layerInfo.inputSize > 1000) {
            issues.push('⚠️ Large input dimension may cause vanishing gradients');
        }
        
        if (layerInfo.outputSize > layerInfo.inputSize * 4) {
            issues.push('⚠️ Large expansion ratio may cause exploding gradients');
        }
        
        if (layerInfo.type === 'Linear' && layerInfo.inputSize > 10000) {
            issues.push('⚠️ Very large linear layer - consider using smaller dimensions');
        }

        if (issues.length === 0) {
            return '<div class="recommendations"><h3>✅ No Issues Detected</h3><p>This layer configuration looks healthy!</p></div>';
        }

        return `
            <div class="potential-issues">
                <h3>⚠️ Potential Issues</h3>
                ${issues.map(issue => `<div class="issue-item">${issue}</div>`).join('')}
            </div>
        `;
    }

    private generateRecommendations(layerInfo: LayerInfo): string {
        const recommendations = [];
        
        if (layerInfo.inputSize > 1000) {
            recommendations.push('🔧 Consider using batch normalization or layer normalization');
            recommendations.push('🔧 Use gradient clipping to prevent exploding gradients');
        }
        
        if (layerInfo.type === 'Linear') {
            recommendations.push('🔧 Initialize weights using Xavier/He initialization');
            recommendations.push('🔧 Monitor gradient norms during training');
        }

        if (layerInfo.type.includes('Conv')) {
            recommendations.push('🔧 Consider using residual connections for deep networks');
        }

        return `
            <div class="recommendations">
                <h3>💡 Recommendations</h3>
                ${recommendations.map(rec => `<div class="issue-item">${rec}</div>`).join('')}
            </div>
        `;
    }
}

interface LayerInfo {
    type: string;
    framework: string;
    inputSize: number;
    outputSize: number;
    code: string;
}