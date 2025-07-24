import * as vscode from 'vscode';

export class TrainingTimeEstimator {
    async estimate() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const document = editor.document;
        const position = editor.selection.active;
        const line = document.lineAt(position);

        // Check if we're hovering over a training call
        const estimateInfo = this.analyzeTrainingCode(line.text, document.getText());
        
        if (estimateInfo) {
            await this.showTimeEstimation(estimateInfo);
        } else {
            vscode.window.showInformationMessage('🕐 No training code detected. Place cursor on model.fit() or training loop.');
        }
    }

    private analyzeTrainingCode(lineText: string, fullText: string): TrainingEstimate | null {
        // Detect different training patterns
        if (lineText.includes('model.fit(')) {
            return this.estimateKerasTraining(lineText, fullText);
        } else if (lineText.includes('for epoch in range')) {
            return this.estimatePyTorchTraining(lineText, fullText);
        } else if (lineText.includes('train_loop') || lineText.includes('training_step')) {
            return this.estimateCustomTraining(lineText, fullText);
        }

        return null;
    }

    private estimateKerasTraining(lineText: string, fullText: string): TrainingEstimate {
        const epochs = this.extractEpochs(lineText, fullText);
        const batchSize = this.extractBatchSize(lineText, fullText);
        const datasetSize = this.estimateDatasetSize(fullText);
        const modelComplexity = this.estimateModelComplexity(fullText);

        const stepsPerEpoch = Math.ceil(datasetSize / batchSize);
        const timePerStep = this.estimateTimePerStep(modelComplexity, batchSize);
        const totalTime = epochs * stepsPerEpoch * timePerStep;

        return {
            framework: 'TensorFlow/Keras',
            epochs,
            batchSize,
            datasetSize,
            stepsPerEpoch,
            timePerStep,
            totalTime,
            estimatedDuration: this.formatDuration(totalTime),
            breakdown: this.generateBreakdown(epochs, stepsPerEpoch, timePerStep),
            recommendations: this.generateRecommendations(epochs, batchSize, modelComplexity)
        };
    }

    private estimatePyTorchTraining(lineText: string, fullText: string): TrainingEstimate {
        const epochs = this.extractEpochs(lineText, fullText);
        const batchSize = this.extractBatchSize(lineText, fullText);
        const datasetSize = this.estimateDatasetSize(fullText);
        const modelComplexity = this.estimateModelComplexity(fullText);

        const stepsPerEpoch = Math.ceil(datasetSize / batchSize);
        const timePerStep = this.estimateTimePerStep(modelComplexity, batchSize);
        const totalTime = epochs * stepsPerEpoch * timePerStep;

        return {
            framework: 'PyTorch',
            epochs,
            batchSize,
            datasetSize,
            stepsPerEpoch,
            timePerStep,
            totalTime,
            estimatedDuration: this.formatDuration(totalTime),
            breakdown: this.generateBreakdown(epochs, stepsPerEpoch, timePerStep),
            recommendations: this.generateRecommendations(epochs, batchSize, modelComplexity)
        };
    }

    private estimateCustomTraining(lineText: string, fullText: string): TrainingEstimate {
        // Simplified estimation for custom training loops
        const epochs = this.extractEpochs(lineText, fullText) || 10;
        const batchSize = this.extractBatchSize(lineText, fullText) || 32;
        const datasetSize = this.estimateDatasetSize(fullText);
        const modelComplexity = this.estimateModelComplexity(fullText);

        const stepsPerEpoch = Math.ceil(datasetSize / batchSize);
        const timePerStep = this.estimateTimePerStep(modelComplexity, batchSize);
        const totalTime = epochs * stepsPerEpoch * timePerStep;

        return {
            framework: 'Custom',
            epochs,
            batchSize,
            datasetSize,
            stepsPerEpoch,
            timePerStep,
            totalTime,
            estimatedDuration: this.formatDuration(totalTime),
            breakdown: this.generateBreakdown(epochs, stepsPerEpoch, timePerStep),
            recommendations: this.generateRecommendations(epochs, batchSize, modelComplexity)
        };
    }

    private extractEpochs(lineText: string, fullText: string): number {
        // Look for epochs parameter in the current line
        let epochsMatch = lineText.match(/epochs\s*=\s*(\d+)/);
        if (epochsMatch) {
            return parseInt(epochsMatch[1]);
        }

        // Look for range(epochs) pattern
        epochsMatch = lineText.match(/range\s*\(\s*(\d+)\s*\)/);
        if (epochsMatch) {
            return parseInt(epochsMatch[1]);
        }

        // Look for epochs variable definition in the full text
        epochsMatch = fullText.match(/epochs\s*=\s*(\d+)/);
        if (epochsMatch) {
            return parseInt(epochsMatch[1]);
        }

        // Default estimate
        return 10;
    }

    private extractBatchSize(lineText: string, fullText: string): number {
        // Look for batch_size parameter
        let batchMatch = lineText.match(/batch_size\s*=\s*(\d+)/);
        if (batchMatch) {
            return parseInt(batchMatch[1]);
        }

        // Look in full text
        batchMatch = fullText.match(/batch_size\s*=\s*(\d+)/);
        if (batchMatch) {
            return parseInt(batchMatch[1]);
        }

        // Look for DataLoader batch_size
        batchMatch = fullText.match(/DataLoader\([^)]*batch_size\s*=\s*(\d+)/);
        if (batchMatch) {
            return parseInt(batchMatch[1]);
        }

        // Default estimate
        return 32;
    }

    private estimateDatasetSize(fullText: string): number {
        // Look for dataset size indicators
        const sizePatterns = [
            /len\s*\(\s*(\w+)\s*\)/,
            /\.shape\[0\]/,
            /size\s*=\s*(\d+)/,
            /(\d+)\s*samples/i
        ];

        for (const pattern of sizePatterns) {
            const match = fullText.match(pattern);
            if (match && match[1] && !isNaN(parseInt(match[1]))) {
                return parseInt(match[1]);
            }
        }

        // Check for common dataset names and estimate sizes
        if (fullText.includes('MNIST')) return 60000;
        if (fullText.includes('CIFAR')) return 50000;
        if (fullText.includes('ImageNet')) return 1000000;

        // Default estimate
        return 10000;
    }

    private estimateModelComplexity(fullText: string): 'simple' | 'medium' | 'complex' {
        let complexity = 0;

        // Count layers
        const layerPatterns = [
            /nn\.Linear/g,
            /nn\.Conv/g,
            /tf\.keras\.layers\./g,
            /Dense\(/g,
            /Conv2D\(/g
        ];

        for (const pattern of layerPatterns) {
            const matches = fullText.match(pattern);
            if (matches) {
                complexity += matches.length;
            }
        }

        // Check for complex architectures
        if (fullText.includes('ResNet') || fullText.includes('Transformer') || fullText.includes('BERT')) {
            complexity += 10;
        }

        if (complexity < 5) return 'simple';
        if (complexity < 15) return 'medium';
        return 'complex';
    }

    private estimateTimePerStep(complexity: string, batchSize: number): number {
        // Base time per step in seconds (rough estimates)
        let baseTime = 0.01; // 10ms for simple operations

        switch (complexity) {
            case 'simple':
                baseTime = 0.005; // 5ms
                break;
            case 'medium':
                baseTime = 0.02; // 20ms
                break;
            case 'complex':
                baseTime = 0.1; // 100ms
                break;
        }

        // Scale with batch size (not linear due to GPU parallelization)
        const batchScaling = Math.sqrt(batchSize / 32);
        
        return baseTime * batchScaling;
    }

    private formatDuration(seconds: number): string {
        if (seconds < 60) {
            return `${Math.round(seconds)} seconds`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = Math.round(seconds % 60);
            return `${minutes}m ${remainingSeconds}s`;
        } else if (seconds < 86400) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        } else {
            const days = Math.floor(seconds / 86400);
            const hours = Math.floor((seconds % 86400) / 3600);
            return `${days}d ${hours}h`;
        }
    }

    private generateBreakdown(epochs: number, stepsPerEpoch: number, timePerStep: number): string[] {
        return [
            `${epochs} epochs × ${stepsPerEpoch} steps = ${epochs * stepsPerEpoch} total steps`,
            `${timePerStep.toFixed(3)}s per step`,
            `${this.formatDuration(stepsPerEpoch * timePerStep)} per epoch`,
            `Estimated completion: ${new Date(Date.now() + epochs * stepsPerEpoch * timePerStep * 1000).toLocaleTimeString()}`
        ];
    }

    private generateRecommendations(epochs: number, batchSize: number, complexity: string): string[] {
        const recommendations = [];

        if (batchSize < 16) {
            recommendations.push('🚀 Consider increasing batch size for better GPU utilization');
        }

        if (batchSize > 128) {
            recommendations.push('⚠️ Large batch size may hurt convergence - consider reducing');
        }

        if (epochs > 100) {
            recommendations.push('⏰ Many epochs detected - consider early stopping');
        }

        if (complexity === 'complex') {
            recommendations.push('🔧 Complex model - consider mixed precision training for speedup');
            recommendations.push('💾 Monitor GPU memory usage during training');
        }

        recommendations.push('📊 Use learning rate scheduling for better convergence');
        recommendations.push('💿 Consider saving checkpoints periodically');

        return recommendations;
    }

    private async showTimeEstimation(estimate: TrainingEstimate) {
        const panel = vscode.window.createWebviewPanel(
            'trainingTimeEstimator',
            `⏱️ Training Time Estimation`,
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = this.getEstimationHTML(estimate);
        
        vscode.window.showInformationMessage(
            `⏱️ Estimated training time: ${estimate.estimatedDuration}`
        );
    }

    private getEstimationHTML(estimate: TrainingEstimate): string {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Time Estimation</title>
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
                .estimate-container {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .estimate-card {
                    background: #252526;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #3e3e42;
                }
                .time-display {
                    font-size: 2em;
                    font-weight: bold;
                    color: #4ec9b0;
                    text-align: center;
                    margin: 20px 0;
                    padding: 20px;
                    background: #2d2d30;
                    border-radius: 8px;
                }
                .breakdown {
                    background: #252526;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }
                .breakdown ul {
                    list-style-type: none;
                    padding: 0;
                }
                .breakdown li {
                    padding: 8px 0;
                    border-bottom: 1px solid #3e3e42;
                }
                .breakdown li:last-child {
                    border-bottom: none;
                }
                .recommendations {
                    background: #1e3c1e;
                    border: 1px solid #28a745;
                    padding: 15px;
                    border-radius: 8px;
                }
                .recommendation-item {
                    margin: 8px 0;
                    padding: 5px 0;
                }
                .warning {
                    background: #3c1e1e;
                    border: 1px solid #d73a49;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                }
                .progress-simulation {
                    background: #252526;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                }
                .progress-bar {
                    background: #3e3e42;
                    height: 20px;
                    border-radius: 10px;
                    overflow: hidden;
                    margin: 10px 0;
                }
                .progress-fill {
                    background: linear-gradient(90deg, #4ec9b0, #007acc);
                    height: 100%;
                    width: 0%;
                    transition: width 0.5s ease;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>⏱️ Training Time Estimation</h2>
                <p><strong>Framework:</strong> ${estimate.framework}</p>
            </div>

            <div class="time-display">
                ${estimate.estimatedDuration}
            </div>

            <div class="estimate-container">
                <div class="estimate-card">
                    <h3>📊 Training Parameters</h3>
                    <p><strong>Epochs:</strong> ${estimate.epochs}</p>
                    <p><strong>Batch Size:</strong> ${estimate.batchSize}</p>
                    <p><strong>Dataset Size:</strong> ${estimate.datasetSize.toLocaleString()}</p>
                    <p><strong>Steps per Epoch:</strong> ${estimate.stepsPerEpoch}</p>
                </div>
                
                <div class="estimate-card">
                    <h3>⚡ Performance Metrics</h3>
                    <p><strong>Time per Step:</strong> ${estimate.timePerStep.toFixed(3)}s</p>
                    <p><strong>Total Steps:</strong> ${(estimate.epochs * estimate.stepsPerEpoch).toLocaleString()}</p>
                    <p><strong>Steps per Hour:</strong> ${Math.round(3600 / estimate.timePerStep).toLocaleString()}</p>
                </div>
            </div>

            <div class="breakdown">
                <h3>📋 Time Breakdown</h3>
                <ul>
                    ${estimate.breakdown.map(item => `<li>${item}</li>`).join('')}
                </ul>
            </div>

            <div class="progress-simulation">
                <h3>🎯 Training Progress Simulation</h3>
                <p>Simulated progress bar (for demonstration):</p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <p id="progressText">Training will start...</p>
            </div>

            <div class="recommendations">
                <h3>💡 Optimization Recommendations</h3>
                ${estimate.recommendations.map(rec => `<div class="recommendation-item">${rec}</div>`).join('')}
            </div>

            <div class="warning">
                <h3>⚠️ Important Notes</h3>
                <ul>
                    <li>These are rough estimates based on code analysis</li>
                    <li>Actual training time depends on hardware, data complexity, and convergence</li>
                    <li>GPU utilization and memory bandwidth significantly affect performance</li>
                    <li>Consider using profiling tools for more accurate measurements</li>
                </ul>
            </div>

            <script>
                // Simulate training progress
                let progress = 0;
                const progressFill = document.getElementById('progressFill');
                const progressText = document.getElementById('progressText');
                const totalTime = ${estimate.totalTime};
                const epochs = ${estimate.epochs};
                
                function updateProgress() {
                    if (progress < 100) {
                        progress += 0.5;
                        progressFill.style.width = progress + '%';
                        
                        const currentEpoch = Math.floor((progress / 100) * epochs) + 1;
                        const remainingTime = (totalTime * (100 - progress)) / 100;
                        progressText.textContent = \`Epoch \${currentEpoch}/\${epochs} - ETA: \${formatDuration(remainingTime)}\`;
                        
                        setTimeout(updateProgress, 100);
                    } else {
                        progressText.textContent = 'Training completed! 🎉';
                    }
                }
                
                function formatDuration(seconds) {
                    if (seconds < 60) return Math.round(seconds) + 's';
                    if (seconds < 3600) return Math.floor(seconds/60) + 'm ' + Math.round(seconds%60) + 's';
                    return Math.floor(seconds/3600) + 'h ' + Math.floor((seconds%3600)/60) + 'm';
                }
                
                // Start simulation after a short delay
                setTimeout(updateProgress, 1000);
            </script>
        </body>
        </html>
        `;
    }
}

interface TrainingEstimate {
    framework: string;
    epochs: number;
    batchSize: number;
    datasetSize: number;
    stepsPerEpoch: number;
    timePerStep: number;
    totalTime: number;
    estimatedDuration: string;
    breakdown: string[];
    recommendations: string[];
}