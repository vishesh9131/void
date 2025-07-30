/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { Action2, registerAction2 } from '../../../../platform/actions/common/actions.js';
import { ServicesAccessor } from '../../../../platform/instantiation/common/instantiation.js';
import { localize2 } from '../../../../nls.js';
import { INotificationService } from '../../../../platform/notification/common/notification.js';
import { IEditorService } from '../../../services/editor/common/editorService.js';
import { IFileService } from '../../../../platform/files/common/files.js';
import { URI } from '../../../../base/common/uri.js';
import { VSBuffer } from '../../../../base/common/buffer.js';
import { IUntitledTextResourceEditorInput } from '../../../common/editor.js';

// Convert Python to Notebook
class ConvertPythonToNotebookAction extends Action2 {
    constructor() {
        super({
            id: 'vsaware.ml.convertNotebook',
            title: localize2('convertNotebook', 'ML: Convert Python to Notebook'),
            f1: true,
            category: localize2('mlCategory', 'ML Tools')
        });
    }

    async run(accessor: ServicesAccessor): Promise<void> {
        const notificationService = accessor.get(INotificationService);
        const editorService = accessor.get(IEditorService);
        const fileService = accessor.get(IFileService);

        try {
            const activeEditor = editorService.activeEditor;
            if (!activeEditor || !activeEditor.resource) {
                notificationService.warn('Please open a Python file to convert to notebook format.');
                return;
            }

            const pythonContent = await fileService.readFile(activeEditor.resource);
            const pythonCode = pythonContent.value.toString();

            // Convert Python code to notebook format
            const notebookContent = this.convertPythonToNotebook(pythonCode);

            // Create new notebook file
            const notebookUri = URI.parse(activeEditor.resource.toString().replace('.py', '.ipynb'));

            await fileService.writeFile(notebookUri, VSBuffer.fromString(JSON.stringify(notebookContent, null, 2)));

            // Open the new notebook
            await editorService.openEditor({
                resource: notebookUri,
                options: { pinned: true }
            });

            notificationService.info('Successfully converted Python file to Jupyter notebook!');
        } catch (error) {
            notificationService.error(`Failed to convert Python to notebook: ${error}`);
        }
    }

    private convertPythonToNotebook(pythonCode: string): any {
        const lines = pythonCode.split('\n');
        const cells: any[] = [];
        let currentCell: string[] = [];
        let cellType = 'code';

        for (const line of lines) {
            if (line.trim().startsWith('# %%') || line.trim().startsWith('#%%')) {
                // New cell marker
                if (currentCell.length > 0) {
                    cells.push({
                        cell_type: cellType,
                        source: currentCell,
                        metadata: {},
                        outputs: [],
                        execution_count: null
                    });
                }
                currentCell = [];
                cellType = 'code';
            } else if (line.trim().startsWith('"""') && currentCell.length === 0) {
                // Markdown cell
                cellType = 'markdown';
                currentCell.push(line.replace('"""', '').trim());
            } else {
                currentCell.push(line);
            }
        }

        // Add the last cell
        if (currentCell.length > 0) {
            cells.push({
                cell_type: cellType,
                source: currentCell,
                metadata: {},
                outputs: [],
                execution_count: null
            });
        }

        return {
            cells: cells,
            metadata: {
                kernelspec: {
                    display_name: 'Python 3',
                    language: 'python',
                    name: 'python3'
                },
                language_info: {
                    name: 'python',
                    version: '3.8.0'
                }
            },
            nbformat: 4,
            nbformat_minor: 4
        };
    }
}

// Neural Network Playground
class NeuralNetworkPlaygroundAction extends Action2 {
    constructor() {
        super({
            id: 'vsaware.ml.neuralPlayground',
            title: localize2('neuralPlayground', 'ML: Neural Network Playground'),
            f1: true,
            category: localize2('mlCategory', 'ML Tools')
        });
    }

    async run(accessor: ServicesAccessor): Promise<void> {
        const notificationService = accessor.get(INotificationService);
        const editorService = accessor.get(IEditorService);

        try {
            const playgroundHtml = this.generateNeuralNetworkPlayground();

            // Create a new untitled HTML file
            await editorService.openEditor({
                resource: URI.parse('untitled:neural-playground.html'),
                options: {
                    pinned: true,
                    override: 'default'
                }
            } as IUntitledTextResourceEditorInput);

            // Insert the playground content
            const activeEditor = editorService.activeTextEditorControl;
            if (activeEditor) {
                (activeEditor as any).setValue(playgroundHtml);
            }

            notificationService.info('Neural Network Playground opened! Train models interactively.');
        } catch (error) {
            notificationService.error(`Failed to open Neural Network Playground: ${error}`);
        }
    }

    private generateNeuralNetworkPlayground(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <title>Neural Network Playground</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .controls { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .control-group { display: flex; flex-direction: column; gap: 5px; }
        .canvas-container { display: flex; gap: 20px; margin-bottom: 20px; }
        canvas { border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 20px; background: #007acc; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #005999; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
        .metric { padding: 10px; background: #f0f0f0; border-radius: 4px; text-align: center; }
        .layer { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Neural Network Playground</h1>

        <div class="controls">
            <div class="control-group">
                <label>Learning Rate:</label>
                <input type="range" id="learningRate" min="0.001" max="0.1" step="0.001" value="0.01">
                <span id="learningRateValue">0.01</span>
            </div>
            <div class="control-group">
                <label>Hidden Layers:</label>
                <input type="range" id="hiddenLayers" min="1" max="5" step="1" value="2">
                <span id="hiddenLayersValue">2</span>
            </div>
            <div class="control-group">
                <label>Neurons per Layer:</label>
                <input type="range" id="neuronsPerLayer" min="4" max="20" step="1" value="10">
                <span id="neuronsPerLayerValue">10</span>
            </div>
            <div class="control-group">
                <label>Activation Function:</label>
                <select id="activation">
                    <option value="relu">ReLU</option>
                    <option value="sigmoid">Sigmoid</option>
                    <option value="tanh">Tanh</option>
                </select>
            </div>
            <div class="control-group">
                <button onclick="trainModel()">Train Model</button>
                <button onclick="resetModel()">Reset</button>
                <button onclick="generateData()">New Data</button>
            </div>
        </div>

        <div class="canvas-container">
            <div>
                <h3>Data Visualization</h3>
                <canvas id="dataCanvas" width="300" height="300"></canvas>
            </div>
            <div>
                <h3>Network Architecture</h3>
                <div id="networkViz" style="width: 300px; height: 300px; border: 1px solid #ddd; padding: 10px;">
                    <div class="layer">Input Layer (2 neurons)</div>
                    <div class="layer">Hidden Layer 1 (10 neurons)</div>
                    <div class="layer">Hidden Layer 2 (10 neurons)</div>
                    <div class="layer">Output Layer (1 neuron)</div>
                </div>
            </div>
            <div>
                <h3>Training Progress</h3>
                <canvas id="lossCanvas" width="300" height="300"></canvas>
            </div>
        </div>

        <div class="metrics">
            <div class="metric">
                <h4>Epoch</h4>
                <div id="epoch">0</div>
            </div>
            <div class="metric">
                <h4>Loss</h4>
                <div id="loss">N/A</div>
            </div>
            <div class="metric">
                <h4>Accuracy</h4>
                <div id="accuracy">N/A</div>
            </div>
            <div class="metric">
                <h4>Learning Rate</h4>
                <div id="currentLR">0.01</div>
            </div>
        </div>
    </div>

    <script>
        let model;
        let trainingData = [];
        let labels = [];
        let isTraining = false;
        let lossHistory = [];

        // Initialize
        generateData();
        createModel();

        // Event listeners
        document.getElementById('learningRate').addEventListener('input', function(e) {
            document.getElementById('learningRateValue').textContent = e.target.value;
            document.getElementById('currentLR').textContent = e.target.value;
        });

        document.getElementById('hiddenLayers').addEventListener('input', function(e) {
            document.getElementById('hiddenLayersValue').textContent = e.target.value;
            updateNetworkViz();
        });

        document.getElementById('neuronsPerLayer').addEventListener('input', function(e) {
            document.getElementById('neuronsPerLayerValue').textContent = e.target.value;
            updateNetworkViz();
        });

        function generateData() {
            trainingData = [];
            labels = [];

            // Generate spiral data
            for (let i = 0; i < 200; i++) {
                const r = Math.random() * 2 + 0.5;
                const t = Math.random() * 4 * Math.PI;
                const x = r * Math.cos(t) + (Math.random() - 0.5) * 0.5;
                const y = r * Math.sin(t) + (Math.random() - 0.5) * 0.5;
                const label = (Math.sin(t) > 0) ? 1 : 0;

                trainingData.push([x, y]);
                labels.push(label);
            }

            visualizeData();
        }

        function createModel() {
            const hiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
            const neuronsPerLayer = parseInt(document.getElementById('neuronsPerLayer').value);
            const activation = document.getElementById('activation').value;

            model = tf.sequential();

            // Input layer
            model.add(tf.layers.dense({
                inputShape: [2],
                units: neuronsPerLayer,
                activation: activation
            }));

            // Hidden layers
            for (let i = 1; i < hiddenLayers; i++) {
                model.add(tf.layers.dense({
                    units: neuronsPerLayer,
                    activation: activation
                }));
            }

            // Output layer
            model.add(tf.layers.dense({
                units: 1,
                activation: 'sigmoid'
            }));

            const learningRate = parseFloat(document.getElementById('learningRate').value);
            model.compile({
                optimizer: tf.train.adam(learningRate),
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });
        }

        function updateNetworkViz() {
            const hiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
            const neuronsPerLayer = parseInt(document.getElementById('neuronsPerLayer').value);

            let html = '<div class="layer">Input Layer (2 neurons)</div>';
            for (let i = 0; i < hiddenLayers; i++) {
                html += \`<div class="layer">Hidden Layer \${i + 1} (\${neuronsPerLayer} neurons)</div>\`;
            }
            html += '<div class="layer">Output Layer (1 neuron)</div>';

            document.getElementById('networkViz').innerHTML = html;
        }

        function visualizeData() {
            const canvas = document.getElementById('dataCanvas');
            const ctx = canvas.getContext('2d');

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            trainingData.forEach((point, i) => {
                ctx.fillStyle = labels[i] === 1 ? '#ff6b6b' : '#4ecdc4';
                ctx.beginPath();
                ctx.arc(
                    (point[0] + 3) * canvas.width / 6,
                    (point[1] + 3) * canvas.height / 6,
                    4, 0, 2 * Math.PI
                );
                ctx.fill();
            });
        }

        async function trainModel() {
            if (isTraining) return;

            isTraining = true;
            const trainButton = document.querySelector('button');
            trainButton.textContent = 'Training...';

            createModel();

            const xs = tf.tensor2d(trainingData);
            const ys = tf.tensor2d(labels, [labels.length, 1]);

            lossHistory = [];

            await model.fit(xs, ys, {
                epochs: 100,
                batchSize: 32,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        document.getElementById('epoch').textContent = epoch + 1;
                        document.getElementById('loss').textContent = logs.loss.toFixed(4);
                        document.getElementById('accuracy').textContent = (logs.acc * 100).toFixed(2) + '%';

                        lossHistory.push(logs.loss);
                        visualizeLoss();
                    }
                }
            });

            xs.dispose();
            ys.dispose();

            isTraining = false;
            trainButton.textContent = 'Train Model';
        }

        function visualizeLoss() {
            const canvas = document.getElementById('lossCanvas');
            const ctx = canvas.getContext('2d');

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (lossHistory.length < 2) return;

            ctx.strokeStyle = '#007acc';
            ctx.lineWidth = 2;
            ctx.beginPath();

            lossHistory.forEach((loss, i) => {
                const x = (i / (lossHistory.length - 1)) * canvas.width;
                const y = canvas.height - (loss / Math.max(...lossHistory)) * canvas.height;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });

            ctx.stroke();
        }

        function resetModel() {
            if (model) {
                model.dispose();
            }
            document.getElementById('epoch').textContent = '0';
            document.getElementById('loss').textContent = 'N/A';
            document.getElementById('accuracy').textContent = 'N/A';
            lossHistory = [];

            const lossCanvas = document.getElementById('lossCanvas');
            const ctx = lossCanvas.getContext('2d');
            ctx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

            createModel();
        }
    </script>
</body>
</html>`;
    }
}

// Dataset Visualizer
class DatasetVisualizerAction extends Action2 {
    constructor() {
        super({
            id: 'vsaware.ml.datasetVisualizer',
            title: localize2('datasetVisualizer', 'ML: Dataset Visualizer'),
            f1: true,
            category: localize2('mlCategory', 'ML Tools')
        });
    }

    async run(accessor: ServicesAccessor): Promise<void> {
        const notificationService = accessor.get(INotificationService);
        const editorService = accessor.get(IEditorService);

        try {
            const visualizerHtml = this.generateDatasetVisualizer();

            await editorService.openEditor({
                resource: URI.parse('untitled:dataset-visualizer.html'),
                options: {
                    pinned: true,
                    override: 'default'
                }
            } as IUntitledTextResourceEditorInput);

            const activeEditor = editorService.activeTextEditorControl;
            if (activeEditor) {
                (activeEditor as any).setValue(visualizerHtml);
            }

            notificationService.info('Dataset Visualizer opened! Upload CSV files and explore your data.');
        } catch (error) {
            notificationService.error(`Failed to open Dataset Visualizer: ${error}`);
        }
    }

    private generateDatasetVisualizer(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <title>Dataset Visualizer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .upload-area {
            border: 2px dashed #007acc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover { background: #f0f8ff; }
        .upload-area.dragover { background: #e6f3ff; border-color: #0056b3; }
        .charts-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .chart-card { background: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }
        .stats-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .stats-table th, .stats-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .stats-table th { background: #f0f0f0; }
        .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        select, button { padding: 8px 16px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007acc; color: white; border: none; cursor: pointer; }
        button:hover { background: #005999; }
        .correlation-matrix { margin-top: 20px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Dataset Visualizer</h1>

        <div class="upload-area" id="uploadArea">
            <h3>📁 Drop CSV file here or click to upload</h3>
            <p>Supports CSV files with headers</p>
            <input type="file" id="fileInput" accept=".csv" style="display: none;">
        </div>

        <div id="datasetInfo" class="hidden">
            <div class="controls">
                <select id="xAxis">
                    <option value="">Select X Axis</option>
                </select>
                <select id="yAxis">
                    <option value="">Select Y Axis</option>
                </select>
                <select id="chartType">
                    <option value="scatter">Scatter Plot</option>
                    <option value="line">Line Chart</option>
                    <option value="bar">Bar Chart</option>
                    <option value="histogram">Histogram</option>
                </select>
                <button onclick="updateChart()">Update Chart</button>
                <button onclick="exportAnalysis()">Export Analysis</button>
            </div>

            <div class="charts-container">
                <div class="chart-card">
                    <h3>Main Visualization</h3>
                    <canvas id="mainChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Distribution Analysis</h3>
                    <canvas id="distributionChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Correlation Heatmap</h3>
                    <div id="correlationHeatmap"></div>
                </div>

                <div class="chart-card">
                    <h3>Dataset Statistics</h3>
                    <table class="stats-table" id="statsTable">
                        <thead>
                            <tr><th>Column</th><th>Type</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let dataset = [];
        let columns = [];
        let mainChart, distributionChart;

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                loadFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                loadFile(e.target.files[0]);
            }
        });

        function loadFile(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const csv = e.target.result;
                parseCSV(csv);
            };
            reader.readAsText(file);
        }

        function parseCSV(csv) {
            const lines = csv.split('\\n');
            const headers = lines[0].split(',').map(h => h.trim());

            dataset = [];
            for (let i = 1; i < lines.length; i++) {
                if (lines[i].trim()) {
                    const row = {};
                    const values = lines[i].split(',');
                    headers.forEach((header, j) => {
                        const value = values[j]?.trim();
                        row[header] = isNaN(value) ? value : parseFloat(value);
                    });
                    dataset.push(row);
                }
            }

            columns = headers;
            initializeInterface();
            generateStatistics();
            createCorrelationMatrix();
        }

        function initializeInterface() {
            document.getElementById('datasetInfo').classList.remove('hidden');

            const xAxis = document.getElementById('xAxis');
            const yAxis = document.getElementById('yAxis');

            xAxis.innerHTML = '<option value="">Select X Axis</option>';
            yAxis.innerHTML = '<option value="">Select Y Axis</option>';

            columns.forEach(col => {
                xAxis.innerHTML += \`<option value="\${col}">\${col}</option>\`;
                yAxis.innerHTML += \`<option value="\${col}">\${col}</option>\`;
            });
        }

        function generateStatistics() {
            const stats = {};

            columns.forEach(col => {
                const values = dataset.map(row => row[col]).filter(v => !isNaN(v));
                const numericValues = values.filter(v => typeof v === 'number');

                stats[col] = {
                    type: numericValues.length > values.length * 0.8 ? 'Numeric' : 'Categorical',
                    count: values.length,
                    mean: numericValues.length > 0 ? (numericValues.reduce((a, b) => a + b, 0) / numericValues.length).toFixed(2) : 'N/A',
                    std: numericValues.length > 0 ? Math.sqrt(numericValues.reduce((sq, n) => sq + Math.pow(n - stats[col]?.mean || 0, 2), 0) / numericValues.length).toFixed(2) : 'N/A',
                    min: numericValues.length > 0 ? Math.min(...numericValues).toFixed(2) : 'N/A',
                    max: numericValues.length > 0 ? Math.max(...numericValues).toFixed(2) : 'N/A'
                };
            });

            const tbody = document.querySelector('#statsTable tbody');
            tbody.innerHTML = '';

            Object.entries(stats).forEach(([col, stat]) => {
                tbody.innerHTML += \`
                    <tr>
                        <td>\${col}</td>
                        <td>\${stat.type}</td>
                        <td>\${stat.count}</td>
                        <td>\${stat.mean}</td>
                        <td>\${stat.std}</td>
                        <td>\${stat.min}</td>
                        <td>\${stat.max}</td>
                    </tr>
                \`;
            });
        }

        function createCorrelationMatrix() {
            const numericColumns = columns.filter(col => {
                const values = dataset.map(row => row[col]).filter(v => !isNaN(v) && typeof v === 'number');
                return values.length > dataset.length * 0.8;
            });

            if (numericColumns.length < 2) {
                document.getElementById('correlationHeatmap').innerHTML = '<p>Not enough numeric columns for correlation analysis</p>';
                return;
            }

            const correlationMatrix = [];
            numericColumns.forEach(col1 => {
                const row = [];
                numericColumns.forEach(col2 => {
                    const correlation = calculateCorrelation(col1, col2);
                    row.push(correlation);
                });
                correlationMatrix.push(row);
            });

            // Create heatmap with D3
            const margin = {top: 20, right: 20, bottom: 30, left: 40};
            const width = 350 - margin.left - margin.right;
            const height = 300 - margin.top - margin.bottom;

            d3.select('#correlationHeatmap').selectAll('*').remove();

            const svg = d3.select('#correlationHeatmap')
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom)
                .append('g')
                .attr('transform', \`translate(\${margin.left},\${margin.top})\`);

            const cellSize = Math.min(width, height) / numericColumns.length;

            const colorScale = d3.scaleSequential(d3.interpolateRdYlBu)
                .domain([-1, 1]);

            correlationMatrix.forEach((row, i) => {
                row.forEach((corr, j) => {
                    svg.append('rect')
                        .attr('x', j * cellSize)
                        .attr('y', i * cellSize)
                        .attr('width', cellSize)
                        .attr('height', cellSize)
                        .attr('fill', colorScale(corr))
                        .attr('stroke', 'white');

                    svg.append('text')
                        .attr('x', j * cellSize + cellSize/2)
                        .attr('y', i * cellSize + cellSize/2)
                        .attr('text-anchor', 'middle')
                        .attr('dy', '.35em')
                        .attr('fill', Math.abs(corr) > 0.5 ? 'white' : 'black')
                        .text(corr.toFixed(2));
                });
            });

            // Add labels
            svg.selectAll('.col-label')
                .data(numericColumns)
                .enter().append('text')
                .attr('class', 'col-label')
                .attr('x', (d, i) => i * cellSize + cellSize/2)
                .attr('y', -5)
                .attr('text-anchor', 'middle')
                .text(d => d.substring(0, 8));

            svg.selectAll('.row-label')
                .data(numericColumns)
                .enter().append('text')
                .attr('class', 'row-label')
                .attr('x', -5)
                .attr('y', (d, i) => i * cellSize + cellSize/2)
                .attr('dy', '.35em')
                .attr('text-anchor', 'end')
                .text(d => d.substring(0, 8));
        }

        function calculateCorrelation(col1, col2) {
            const pairs = dataset.map(row => [row[col1], row[col2]])
                .filter(pair => !isNaN(pair[0]) && !isNaN(pair[1]));

            if (pairs.length < 2) return 0;

            const mean1 = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
            const mean2 = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;

            let numerator = 0, denom1 = 0, denom2 = 0;

            pairs.forEach(pair => {
                const diff1 = pair[0] - mean1;
                const diff2 = pair[1] - mean2;
                numerator += diff1 * diff2;
                denom1 += diff1 * diff1;
                denom2 += diff2 * diff2;
            });

            const denominator = Math.sqrt(denom1 * denom2);
            return denominator === 0 ? 0 : numerator / denominator;
        }

        function updateChart() {
            const xCol = document.getElementById('xAxis').value;
            const yCol = document.getElementById('yAxis').value;
            const chartType = document.getElementById('chartType').value;

            if (!xCol) return;

            // Main chart
            const ctx = document.getElementById('mainChart').getContext('2d');
            if (mainChart) mainChart.destroy();

            const xData = dataset.map(row => row[xCol]);
            const yData = yCol ? dataset.map(row => row[yCol]) : null;

            let chartData;
            if (chartType === 'scatter' && yData) {
                chartData = {
                    datasets: [{
                        label: \`\${yCol} vs \${xCol}\`,
                        data: xData.map((x, i) => ({x: x, y: yData[i]})),
                        backgroundColor: 'rgba(0, 122, 204, 0.6)',
                        borderColor: 'rgba(0, 122, 204, 1)'
                    }]
                };
            } else if (chartType === 'histogram') {
                const bins = createHistogramBins(xData);
                chartData = {
                    labels: bins.labels,
                    datasets: [{
                        label: 'Frequency',
                        data: bins.counts,
                        backgroundColor: 'rgba(0, 122, 204, 0.6)',
                        borderColor: 'rgba(0, 122, 204, 1)',
                        borderWidth: 1
                    }]
                };
            } else {
                // Bar or line chart
                const aggregated = aggregateData(xData, yData);
                chartData = {
                    labels: aggregated.labels,
                    datasets: [{
                        label: yCol || 'Count',
                        data: aggregated.values,
                        backgroundColor: 'rgba(0, 122, 204, 0.6)',
                        borderColor: 'rgba(0, 122, 204, 1)',
                        borderWidth: 1,
                        fill: false
                    }]
                };
            }

            mainChart = new Chart(ctx, {
                type: chartType === 'scatter' ? 'scatter' : (chartType === 'histogram' ? 'bar' : chartType),
                data: chartData,
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: xCol } },
                        y: { title: { display: true, text: yCol || 'Count' } }
                    }
                }
            });

            // Distribution chart
            updateDistributionChart(xCol);
        }

        function createHistogramBins(data) {
            const numericData = data.filter(d => !isNaN(d)).sort((a, b) => a - b);
            if (numericData.length === 0) return {labels: [], counts: []};

            const binCount = Math.min(20, Math.ceil(Math.sqrt(numericData.length)));
            const min = Math.min(...numericData);
            const max = Math.max(...numericData);
            const binWidth = (max - min) / binCount;

            const bins = Array(binCount).fill(0);
            const labels = [];

            for (let i = 0; i < binCount; i++) {
                labels.push(\`\${(min + i * binWidth).toFixed(2)}\`);
            }

            numericData.forEach(value => {
                const binIndex = Math.min(Math.floor((value - min) / binWidth), binCount - 1);
                bins[binIndex]++;
            });

            return {labels, counts: bins};
        }

        function aggregateData(xData, yData) {
            const groups = {};

            xData.forEach((x, i) => {
                if (!groups[x]) {
                    groups[x] = [];
                }
                groups[x].push(yData ? yData[i] : 1);
            });

            const labels = Object.keys(groups);
            const values = labels.map(label => {
                const vals = groups[label];
                return yData ? vals.reduce((a, b) => a + b, 0) / vals.length : vals.length;
            });

            return {labels, values};
        }

        function updateDistributionChart(column) {
            const ctx = document.getElementById('distributionChart').getContext('2d');
            if (distributionChart) distributionChart.destroy();

            const data = dataset.map(row => row[column]).filter(v => !isNaN(v));
            const bins = createHistogramBins(data);

            distributionChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: bins.labels,
                    datasets: [{
                        label: \`Distribution of \${column}\`,
                        data: bins.counts,
                        backgroundColor: 'rgba(255, 107, 107, 0.6)',
                        borderColor: 'rgba(255, 107, 107, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: column } },
                        y: { title: { display: true, text: 'Frequency' } }
                    }
                }
            });
        }

        function exportAnalysis() {
            const analysis = {
                dataset_info: {
                    rows: dataset.length,
                    columns: columns.length,
                    column_names: columns
                },
                statistics: {},
                correlations: {}
            };

            // Add statistics
            columns.forEach(col => {
                const values = dataset.map(row => row[col]).filter(v => !isNaN(v));
                const numericValues = values.filter(v => typeof v === 'number');

                analysis.statistics[col] = {
                    type: numericValues.length > values.length * 0.8 ? 'numeric' : 'categorical',
                    count: values.length,
                    unique_values: new Set(values).size
                };

                if (numericValues.length > 0) {
                    analysis.statistics[col].mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
                    analysis.statistics[col].min = Math.min(...numericValues);
                    analysis.statistics[col].max = Math.max(...numericValues);
                }
            });

            const blob = new Blob([JSON.stringify(analysis, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'dataset_analysis.json';
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>`;
    }
}

// Quick Model Builder
class QuickModelBuilderAction extends Action2 {
    constructor() {
        super({
            id: 'vsaware.ml.quickModel',
            title: localize2('quickModel', 'ML: Quick Model Builder'),
            f1: true,
            category: localize2('mlCategory', 'ML Tools')
        });
    }

    async run(accessor: ServicesAccessor): Promise<void> {
        const notificationService = accessor.get(INotificationService);
        const editorService = accessor.get(IEditorService);

        try {
            const modelCode = this.generateModelBoilerplate();

            // Create a new Python file
            const modelUri = URI.parse('untitled:ml_model.py');

            await editorService.openEditor({
                resource: modelUri,
                options: {
                    pinned: true,
                    override: 'default'
                }
            } as IUntitledTextResourceEditorInput);

            const activeEditor = editorService.activeTextEditorControl;
            if (activeEditor) {
                (activeEditor as any).getModel()?.setValue(modelCode);
            }

            notificationService.info('Quick ML model template generated! Customize the model architecture and training parameters.');
        } catch (error) {
            notificationService.error(`Failed to generate model: ${error}`);
        }
    }

    private generateModelBoilerplate(): string {
        return `# 🤖 Quick ML Model Builder - Generated Template
# Customize this template for your specific machine learning task

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class QuickMLModel:
    def __init__(self, task_type='classification'):
        """
        Quick ML Model Builder

        Args:
            task_type (str): 'classification' or 'regression'
        """
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

        # Model options
        if task_type == 'classification':
            self.models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42),
                'svm': SVC(random_state=42)
            }
        else:
            self.models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression(),
                'svm': SVR()
            }

    def load_data(self, file_path=None, data=None, target_column=None):
        """
        Load and prepare data

        Args:
            file_path (str): Path to CSV file
            data (DataFrame): Pandas DataFrame
            target_column (str): Name of target column
        """
        if file_path:
            data = pd.read_csv(file_path)

        if data is None:
            raise ValueError("Please provide either file_path or data")

        print(f"📊 Dataset shape: {data.shape}")
        print(f"📋 Columns: {list(data.columns)}")
        print(f"🔍 Missing values: {data.isnull().sum().sum()}")

        # Basic data exploration
        self.explore_data(data)

        # Prepare features and target
        if target_column:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            # Assume last column is target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

        self.feature_names = X.columns.tolist()

        # Handle categorical variables
        X = self.preprocess_features(X)

        # Encode target if classification
        if self.task_type == 'classification' and y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if self.task_type == 'classification' else None
        )

        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"✅ Data prepared successfully!")
        print(f"   Training set: {self.X_train.shape}")
        print(f"   Test set: {self.X_test.shape}")

    def explore_data(self, data):
        """Basic data exploration"""
        print("\\n📈 Data Exploration:")
        print(data.describe())
        print(f"\\n📊 Data types:")
        print(data.dtypes.value_counts())

        # Check for missing values
        if data.isnull().sum().sum() > 0:
            print(f"\\n⚠️ Missing values by column:")
            print(data.isnull().sum()[data.isnull().sum() > 0])

    def preprocess_features(self, X):
        """Preprocess features"""
        X_processed = X.copy()

        # Handle categorical variables
        categorical_columns = X_processed.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            # Simple label encoding for categorical variables
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))

        # Handle missing values
        X_processed = X_processed.fillna(X_processed.mean())

        return X_processed

    def train_model(self, model_name='random_forest', hyperparameter_tuning=False):
        """
        Train the selected model

        Args:
            model_name (str): Model to use ('random_forest', 'logistic_regression', 'svm', etc.)
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        """
        if self.X_train is None:
            raise ValueError("Please load data first using load_data()")

        self.model = self.models[model_name]

        if hyperparameter_tuning:
            print(f"🔧 Performing hyperparameter tuning for {model_name}...")
            self.hyperparameter_tuning(model_name)
        else:
            print(f"🎯 Training {model_name}...")
            self.model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Evaluate model
        self.evaluate_model(y_pred)

        print(f"✅ Model training completed!")

    def hyperparameter_tuning(self, model_name):
        """Perform hyperparameter tuning"""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }

        if model_name in param_grids:
            grid_search = GridSearchCV(
                self.model,
                param_grids[model_name],
                cv=5,
                scoring='accuracy' if self.task_type == 'classification' else 'r2',
                n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_
            print(f"🎯 Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self, y_pred):
        """Evaluate model performance"""
        print("\\n📊 Model Evaluation:")

        if self.task_type == 'classification':
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"🎯 Accuracy: {accuracy:.4f}")
            print("\\n📋 Classification Report:")
            print(classification_report(self.y_test, y_pred))

            # Cross-validation
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
            print(f"🔄 Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        else:
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            print(f"📉 Mean Squared Error: {mse:.4f}")
            print(f"📈 R² Score: {r2:.4f}")

            # Cross-validation
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='r2')
            print(f"🔄 Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\\n🔍 Feature Importance:")
            print(importance_df.head(10))

            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            plt.show()

            return importance_df

    def predict(self, new_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Please train a model first")

        # Preprocess new data
        new_data_processed = self.preprocess_features(new_data)
        new_data_scaled = self.scaler.transform(new_data_processed)

        predictions = self.model.predict(new_data_scaled)

        # Decode predictions if classification
        if self.task_type == 'classification' and hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)

        return predictions

    def save_model(self, filename):
        """Save the trained model"""
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder if hasattr(self.label_encoder, 'classes_') else None,
            'feature_names': self.feature_names,
            'task_type': self.task_type
        }
        joblib.dump(model_data, filename)
        print(f"💾 Model saved to {filename}")

    def load_model(self, filename):
        """Load a saved model"""
        import joblib
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.task_type = model_data['task_type']
        print(f"📂 Model loaded from {filename}")

# 🚀 Example Usage
if __name__ == "__main__":
    # Example 1: Classification Task
    print("🔵 Classification Example:")
    print("=" * 50)

    # Create sample classification data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    data['target'] = y

    # Initialize classifier
    clf = QuickMLModel(task_type='classification')

    # Load data
    clf.load_data(data=data, target_column='target')

    # Train model with hyperparameter tuning
    clf.train_model(model_name='random_forest', hyperparameter_tuning=True)

    # Get feature importance
    clf.get_feature_importance()

    # Save model
    clf.save_model('classification_model.pkl')

    print("\\n" + "=" * 50)
    print("🔴 Regression Example:")
    print("=" * 50)

    # Example 2: Regression Task
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    data['target'] = y

    # Initialize regressor
    reg = QuickMLModel(task_type='regression')

    # Load data
    reg.load_data(data=data, target_column='target')

    # Train model
    reg.train_model(model_name='random_forest', hyperparameter_tuning=True)

    # Get feature importance
    reg.get_feature_importance()

    # Save model
    reg.save_model('regression_model.pkl')

    print("\\n✨ Model building completed! Customize the code above for your specific use case.")
    print("💡 Tips:")
    print("   - Replace the sample data with your own dataset")
    print("   - Experiment with different models and hyperparameters")
    print("   - Add custom preprocessing steps for your data")
    print("   - Use the predict() method for new predictions")
`;
    }
}

// Tensor Shape Analyzer
class TensorShapeAnalyzerAction extends Action2 {
    constructor() {
        super({
            id: 'vsaware.ml.tensorAnalyzer',
            title: localize2('tensorAnalyzer', 'ML: Tensor Shape Analyzer'),
            f1: true,
            category: localize2('mlCategory', 'ML Tools')
        });
    }

    async run(accessor: ServicesAccessor): Promise<void> {
        const notificationService = accessor.get(INotificationService);
        const editorService = accessor.get(IEditorService);

        const analyzerHtml = this.generateTensorShapeAnalyzer();

        try {
            await editorService.openEditor({
                resource: URI.parse('untitled:tensor-shape-analyzer.html'),
                options: {
                    pinned: true,
                    override: 'default'
                }
            } as IUntitledTextResourceEditorInput);

            const activeEditor = editorService.activeTextEditorControl;
            if (activeEditor) {
                (activeEditor as any).setValue(analyzerHtml);
            }

            notificationService.info('Tensor Shape Analyzer opened! Analyze tensor shapes and dimensions.');
        } catch (error) {
            notificationService.error(`Failed to open Tensor Shape Analyzer: ${error}`);
        }
    }

    private generateTensorShapeAnalyzer(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <title>Tensor Shape Analyzer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .upload-area {
            border: 2px dashed #007acc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover { background: #f0f8ff; }
        .upload-area.dragover { background: #e6f3ff; border-color: #0056b3; }
        .charts-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .chart-card { background: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }
        .stats-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .stats-table th, .stats-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .stats-table th { background: #f0f0f0; }
        .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        select, button { padding: 8px 16px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007acc; color: white; border: none; cursor: pointer; }
        button:hover { background: #005999; }
        .correlation-matrix { margin-top: 20px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Tensor Shape Analyzer</h1>

        <div class="upload-area" id="uploadArea">
            <h3>📁 Drop a TensorFlow model file (.h5, .pb, .saved_model) here or click to upload</h3>
            <p>Supports TensorFlow SavedModel, HDF5, and Protocol Buffer formats.</p>
            <input type="file" id="fileInput" accept=".h5,.pb,.saved_model" style="display: none;">
        </div>

        <div id="modelInfo" class="hidden">
            <div class="controls">
                <button onclick="analyzeModel()">Analyze Model</button>
                <button onclick="resetModel()">Reset</button>
            </div>

            <div class="charts-container">
                <div class="chart-card">
                    <h3>Model Summary</h3>
                    <div id="modelSummary" style="white-space: pre-wrap; font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9;"></div>
                </div>

                <div class="chart-card">
                    <h3>Input Shapes</h3>
                    <canvas id="inputShapesChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Output Shapes</h3>
                    <canvas id="outputShapesChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Model Layers</h3>
                    <div id="modelLayers"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let model;
        let modelSummary = '';
        let inputShapesChart, outputShapesChart;

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                loadModel(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                loadModel(e.target.files[0]);
            }
        });

        function loadModel(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const modelBuffer = e.target.result;
                try {
                    model = tf.loadLayersModel(modelBuffer);
                    modelSummary = 'Model loaded successfully!';
                    document.getElementById('modelSummary').textContent = modelSummary;
                    document.getElementById('modelInfo').classList.remove('hidden');
                    analyzeModel(); // Analyze the loaded model
                } catch (error) {
                    modelSummary = 'Error loading model: ' + error;
                    document.getElementById('modelSummary').textContent = modelSummary;
                    console.error(error);
                }
            };
            reader.readAsArrayBuffer(file);
        }

        function analyzeModel() {
            if (!model) {
                alert('Please load a model file first.');
                return;
            }

            modelSummary = '';
            modelSummary += 'Model Name: ' + model.name + '\\n';
            modelSummary += 'Model Type: ' + model.constructor.name + '\\n';
            modelSummary += 'Model Architecture:\\n';
            model.summary().then(summary => {
                modelSummary += summary;
                document.getElementById('modelSummary').textContent = modelSummary;
            });

            const inputShapes = model.inputs.map(input => {
                const shape = input.shape;
                return { name: input.name, shape: shape.map(s => s.toString()).join('x') };
            });
            const outputShapes = model.outputs.map(output => {
                const shape = output.shape;
                return { name: output.name, shape: shape.map(s => s.toString()).join('x') };
            });

            updateInputShapesChart(inputShapes);
            updateOutputShapesChart(outputShapes);
        }

        function updateInputShapesChart(data) {
            const ctx = document.getElementById('inputShapesChart').getContext('2d');
            if (inputShapesChart) inputShapesChart.destroy();

            const labels = data.map(item => item.name);
            const shapes = data.map(item => item.shape);

            inputShapesChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Input Shapes',
                        data: shapes,
                        backgroundColor: 'rgba(0, 122, 204, 0.6)',
                        borderColor: 'rgba(0, 122, 204, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'Input Layer' } },
                        y: { title: { display: true, text: 'Shape' } }
                    }
                }
            });
        }

        function updateOutputShapesChart(data) {
            const ctx = document.getElementById('outputShapesChart').getContext('2d');
            if (outputShapesChart) outputShapesChart.destroy();

            const labels = data.map(item => item.name);
            const shapes = data.map(item => item.shape);

            outputShapesChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Output Shapes',
                        data: shapes,
                        backgroundColor: 'rgba(255, 107, 107, 0.6)',
                        borderColor: 'rgba(255, 107, 107, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'Output Layer' } },
                        y: { title: { display: true, text: 'Shape' } }
                    }
                }
            });
        }

        function resetModel() {
            if (model) {
                model.dispose();
                model = null;
            }
            modelSummary = '';
            document.getElementById('modelSummary').textContent = modelSummary;
            document.getElementById('modelInfo').classList.add('hidden');

            const inputShapesChart = document.getElementById('inputShapesChart');
            const ctx = inputShapesChart.getContext('2d');
            ctx.clearRect(0, 0, inputShapesChart.width, inputShapesChart.height);

            const outputShapesChart = document.getElementById('outputShapesChart');
            const ctx2 = outputShapesChart.getContext('2d');
            ctx2.clearRect(0, 0, outputShapesChart.width, outputShapesChart.height);
        }
    </script>
</body>
</html>`;
    }
}

// Experiment Tracker
class ExperimentTrackerAction extends Action2 {
    constructor() {
        super({
            id: 'vsaware.ml.experimentTracker',
            title: localize2('experimentTracker', 'ML: Experiment Tracker'),
            f1: true,
            category: localize2('mlCategory', 'ML Tools')
        });
    }

    async run(accessor: ServicesAccessor): Promise<void> {
        const notificationService = accessor.get(INotificationService);
        const editorService = accessor.get(IEditorService);

        try {
            const trackerHtml = this.generateExperimentTracker();

            await editorService.openEditor({
                resource: URI.parse('untitled:experiment-tracker.html'),
                options: {
                    pinned: true,
                    override: 'default'
                }
            } as IUntitledTextResourceEditorInput);

            const activeEditor = editorService.activeTextEditorControl;
            if (activeEditor) {
                (activeEditor as any).getModel()?.setValue(trackerHtml);
            }

            notificationService.info('Experiment Tracker opened! Track and compare ML experiments.');
        } catch (error) {
            notificationService.error(`Failed to open Experiment Tracker: ${error}`);
        }
    }

    private generateExperimentTracker(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <title>Experiment Tracker</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .upload-area {
            border: 2px dashed #007acc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover { background: #f0f8ff; }
        .upload-area.dragover { background: #e6f3ff; border-color: #0056b3; }
        .charts-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .chart-card { background: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }
        .stats-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .stats-table th, .stats-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .stats-table th { background: #f0f0f0; }
        .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        select, button { padding: 8px 16px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007acc; color: white; border: none; cursor: pointer; }
        button:hover { background: #005999; }
        .correlation-matrix { margin-top: 20px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Experiment Tracker</h1>

        <div class="upload-area" id="uploadArea">
            <h3>📁 Drop experiment results (.json) here or click to upload</h3>
            <p>Supports JSON files containing experiment data.</p>
            <input type="file" id="fileInput" accept=".json" style="display: none;">
        </div>

        <div id="experimentData" class="hidden">
            <div class="controls">
                <select id="xAxis">
                    <option value="">Select X Axis</option>
                </select>
                <select id="yAxis">
                    <option value="">Select Y Axis</option>
                </select>
                <select id="chartType">
                    <option value="line">Line Chart</option>
                    <option value="bar">Bar Chart</option>
                    <option value="scatter">Scatter Plot</option>
                </select>
                <button onclick="updateChart()">Update Chart</button>
                <button onclick="exportData()">Export Data</button>
            </div>

            <div class="charts-container">
                <div class="chart-card">
                    <h3>Experiment Results</h3>
                    <canvas id="experimentChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Model Performance</h3>
                    <canvas id="modelPerformanceChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Hyperparameter Impact</h3>
                    <div id="hyperparameterImpact"></div>
                </div>

                <div class="chart-card">
                    <h3>Dataset Statistics</h3>
                    <table class="stats-table" id="statsTable">
                        <thead>
                            <tr><th>Column</th><th>Type</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let experimentData = [];
        let columns = [];
        let experimentChart, modelPerformanceChart;

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                loadExperimentData(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                loadExperimentData(e.target.files[0]);
            }
        });

        function loadExperimentData(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const jsonData = JSON.parse(e.target.result);
                experimentData = jsonData;
                columns = Object.keys(jsonData[0] || {}); // Assuming all data has the same keys
                initializeInterface();
                generateStatistics();
                updateChart();
                analyzeModelPerformance();
                analyzeHyperparameterImpact();
            };
            reader.readAsText(file);
        }

        function initializeInterface() {
            document.getElementById('experimentData').classList.remove('hidden');

            const xAxis = document.getElementById('xAxis');
            const yAxis = document.getElementById('yAxis');

            xAxis.innerHTML = '<option value="">Select X Axis</option>';
            yAxis.innerHTML = '<option value="">Select Y Axis</option>';

            columns.forEach(col => {
                xAxis.innerHTML += \`<option value="\${col}">\${col}</option>\`;
                yAxis.innerHTML += \`<option value="\${col}">\${col}</option>\`;
            });
        }

        function generateStatistics() {
            const stats = {};

            columns.forEach(col => {
                const values = experimentData.map(exp => exp[col]).filter(v => !isNaN(v));
                const numericValues = values.filter(v => typeof v === 'number');

                stats[col] = {
                    type: numericValues.length > values.length * 0.8 ? 'Numeric' : 'Categorical',
                    count: values.length,
                    mean: numericValues.length > 0 ? (numericValues.reduce((a, b) => a + b, 0) / numericValues.length).toFixed(2) : 'N/A',
                    std: numericValues.length > 0 ? Math.sqrt(numericValues.reduce((sq, n) => sq + Math.pow(n - stats[col]?.mean || 0, 2), 0) / numericValues.length).toFixed(2) : 'N/A',
                    min: numericValues.length > 0 ? Math.min(...numericValues).toFixed(2) : 'N/A',
                    max: numericValues.length > 0 ? Math.max(...numericValues).toFixed(2) : 'N/A'
                };
            });

            const tbody = document.querySelector('#statsTable tbody');
            tbody.innerHTML = '';

            Object.entries(stats).forEach(([col, stat]) => {
                tbody.innerHTML += \`
                    <tr>
                        <td>\${col}</td>
                        <td>\${stat.type}</td>
                        <td>\${stat.count}</td>
                        <td>\${stat.mean}</td>
                        <td>\${stat.std}</td>
                        <td>\${stat.min}</td>
                        <td>\${stat.max}</td>
                    </tr>
                \`;
            });
        }

        function updateChart() {
            const xCol = document.getElementById('xAxis').value;
            const yCol = document.getElementById('yAxis').value;
            const chartType = document.getElementById('chartType').value;

            if (!xCol) return;

            // Experiment chart
            const ctx = document.getElementById('experimentChart').getContext('2d');
            if (experimentChart) experimentChart.destroy();

            const xData = experimentData.map(exp => exp[xCol]);
            const yData = yCol ? experimentData.map(exp => exp[yCol]) : null;

            let chartData;
            if (chartType === 'scatter' && yData) {
                chartData = {
                    labels: xData,
                    datasets: [{
                        label: \`\${yCol} vs \${xCol}\`,
                        data: xData.map((x, i) => ({x: x, y: yData[i]})),
                        backgroundColor: 'rgba(0, 122, 204, 0.6)',
                        borderColor: 'rgba(0, 122, 204, 1)'
                    }]
                };
            } else if (chartType === 'bar') {
                const aggregated = aggregateData(xData, yData);
                chartData = {
                    labels: aggregated.labels,
                    datasets: [{
                        label: yCol || 'Count',
                        data: aggregated.values,
                        backgroundColor: 'rgba(0, 122, 204, 0.6)',
                        borderColor: 'rgba(0, 122, 204, 1)',
                        borderWidth: 1,
                        fill: false
                    }]
                };
            } else { // Line chart
                chartData = {
                    labels: xData,
                    datasets: [{
                        label: yCol || 'Value',
                        data: yData,
                        borderColor: 'rgba(0, 122, 204, 1)',
                        borderWidth: 2,
                        fill: false
                    }]
                };
            }

            experimentChart = new Chart(ctx, {
                type: chartType,
                data: chartData,
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: xCol } },
                        y: { title: { display: true, text: yCol || 'Value' } }
                    }
                }
            });
        }

        function aggregateData(xData, yData) {
            const groups = {};

            xData.forEach((x, i) => {
                if (!groups[x]) {
                    groups[x] = [];
                }
                groups[x].push(yData ? yData[i] : 1);
            });

            const labels = Object.keys(groups);
            const values = labels.map(label => {
                const vals = groups[label];
                return yData ? vals.reduce((a, b) => a + b, 0) / vals.length : vals.length;
            });

            return {labels, values};
        }

        function analyzeModelPerformance() {
            const ctx = document.getElementById('modelPerformanceChart').getContext('2d');
            if (modelPerformanceChart) modelPerformanceChart.destroy();

            const modelPerformance = experimentData.map(exp => ({
                epoch: exp.epoch,
                loss: exp.loss,
                accuracy: exp.accuracy,
                learning_rate: exp.learning_rate
            }));

            modelPerformanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: modelPerformance.map(item => item.epoch),
                    datasets: [
                        {
                            label: 'Loss',
                            data: modelPerformance.map(item => item.loss),
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            fill: true
                        },
                        {
                            label: 'Accuracy',
                            data: modelPerformance.map(item => item.accuracy),
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'Epoch' } },
                        y: { title: { display: true, text: 'Value' } }
                    }
                }
            });
        }

        function analyzeHyperparameterImpact() {
            const hyperparameterImpact = {};
            columns.forEach(col => {
                if (col.toLowerCase().includes('learning_rate') || col.toLowerCase().includes('lr')) {
                    const values = experimentData.map(exp => exp[col]).filter(v => !isNaN(v));
                    if (values.length > 0) {
                        hyperparameterImpact[col] = {
                            mean: values.reduce((a, b) => a + b, 0) / values.length,
                            std: values.length > 1 ? Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - hyperparameterImpact[col]?.mean || 0, 2), 0) / (values.length - 1)) : 0
                        };
                    }
                }
            });

            const impactHtml = '<h3>Hyperparameter Impact</h3>';
            Object.entries(hyperparameterImpact).forEach(([param, stats]) => {
                impactHtml += \`<p><strong>\${param}:</strong> Mean = \${stats.mean.toFixed(4)}, Std = \${stats.std.toFixed(4)}</p>\`;
            });
            document.getElementById('hyperparameterImpact').innerHTML = impactHtml;
        }

        function exportData() {
            const analysis = {
                experiment_data: experimentData,
                dataset_info: {
                    rows: experimentData.length,
                    columns: columns.length,
                    column_names: columns
                },
                statistics: {}
            };

            // Add statistics
            columns.forEach(col => {
                const values = experimentData.map(exp => exp[col]).filter(v => !isNaN(v));
                const numericValues = values.filter(v => typeof v === 'number');

                analysis.statistics[col] = {
                    type: numericValues.length > values.length * 0.8 ? 'numeric' : 'categorical',
                    count: values.length,
                    unique_values: new Set(values).size
                };

                if (numericValues.length > 0) {
                    analysis.statistics[col].mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
                    analysis.statistics[col].min = Math.min(...numericValues);
                    analysis.statistics[col].max = Math.max(...numericValues);
                }
            });

            const blob = new Blob([JSON.stringify(analysis, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'experiment_results.json';
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>`;
    }
}

// ML Code Quality Checker
class MLCodeQualityAction extends Action2 {
    constructor() {
        super({
            id: 'vsaware.ml.codeChecker',
            title: localize2('codeChecker', 'ML: Code Quality Checker'),
            f1: true,
            category: localize2('mlCategory', 'ML Tools')
        });
    }

    async run(accessor: ServicesAccessor): Promise<void> {
        const notificationService = accessor.get(INotificationService);
        const editorService = accessor.get(IEditorService);

        const checkerHtml = this.generateCodeChecker();

        try {
            await editorService.openEditor({
                resource: URI.parse('untitled:code-checker.html'),
                options: {
                    pinned: true,
                    override: 'default'
                }
            } as IUntitledTextResourceEditorInput);

            const activeEditor = editorService.activeTextEditorControl;
            if (activeEditor) {
                (activeEditor as any).setValue(checkerHtml);
            }

            notificationService.info('ML Code Quality Checker opened! Analyze your ML code for best practices.');
        } catch (error) {
            notificationService.error(`Failed to open ML Code Quality Checker: ${error}`);
        }
    }

    private generateCodeChecker(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <title>ML Code Quality Checker</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .upload-area {
            border: 2px dashed #007acc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover { background: #f0f8ff; }
        .upload-area.dragover { background: #e6f3ff; border-color: #0056b3; }
        .charts-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .chart-card { background: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }
        .stats-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .stats-table th, .stats-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .stats-table th { background: #f0f0f0; }
        .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        select, button { padding: 8px 16px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007acc; color: white; border: none; cursor: pointer; }
        button:hover { background: #005999; }
        .correlation-matrix { margin-top: 20px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 ML Code Quality Checker</h1>

        <div class="upload-area" id="uploadArea">
            <h3>📁 Drop a Python file (.py) here or click to upload</h3>
            <p>Supports Python files containing ML code.</p>
            <input type="file" id="fileInput" accept=".py" style="display: none;">
        </div>

        <div id="codeAnalysis" class="hidden">
            <div class="controls">
                <button onclick="analyzeCode()">Analyze Code</button>
                <button onclick="resetAnalysis()">Reset</button>
            </div>

            <div class="charts-container">
                <div class="chart-card">
                    <h3>Code Metrics</h3>
                    <div id="codeMetrics" style="white-space: pre-wrap; font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9;"></div>
                </div>

                <div class="chart-card">
                    <h3>Code Complexity</h3>
                    <canvas id="complexityChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Code Duplication</h3>
                    <div id="duplicationImpact"></div>
                </div>

                <div class="chart-card">
                    <h3>Dataset Statistics</h3>
                    <table class="stats-table" id="statsTable">
                        <thead>
                            <tr><th>Column</th><th>Type</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let codeContent = '';
        let codeMetrics = '';
        let complexityChart;
        let duplicationImpact = '';

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                loadCode(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                loadCode(e.target.files[0]);
            }
        });

        function loadCode(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                codeContent = e.target.result;
                document.getElementById('codeMetrics').textContent = codeContent;
                document.getElementById('codeAnalysis').classList.remove('hidden');
                analyzeCode();
            };
            reader.readAsText(file);
        }

        function analyzeCode() {
            if (!codeContent) {
                alert('Please load a Python file first.');
                return;
            }

            codeMetrics = '';
            codeMetrics += 'File: ' + fileInput.files[0].name + '\\n';
            codeMetrics += 'Lines of Code: ' + codeContent.split('\\n').length + '\\n';
            codeMetrics += 'Characters: ' + codeContent.length + '\\n';
            codeMetrics += '\\nCode Quality Metrics:\\n';

            // Placeholder for actual analysis logic
            // This would involve parsing the Python code,
            // using a linter (e.g., pylint, flake8, etc.),
            // and extracting metrics.
            // For now, we'll just show a placeholder.
            codeMetrics += 'Placeholder for ML Code Quality Analysis.\\n';
            codeMetrics += 'Please implement a proper ML code linter and metric extractor.\\n';
            codeMetrics += 'This feature is under development.';

            document.getElementById('codeMetrics').textContent = codeMetrics;

            // Example of how to extract metrics (requires a proper linter)
            // try {
            //     const metrics = await analyzePythonCode(codeContent);
            //     codeMetrics += \`\n\`;
            //     Object.entries(metrics).forEach(([key, value]) => {
            //         codeMetrics += \`\${key}: \${value}\n\`;
            //     });
            // } catch (error) {
            //     codeMetrics += \`\nError analyzing code: \${error}\`;
            // }
        }

        function resetAnalysis() {
            codeContent = '';
            document.getElementById('codeMetrics').textContent = '';
            document.getElementById('codeAnalysis').classList.add('hidden');
            if (complexityChart) complexityChart.destroy();
            if (duplicationImpact) duplicationImpact = '';
        }

        function updateComplexityChart() {
            const ctx = document.getElementById('complexityChart').getContext('2d');
            if (complexityChart) complexityChart.destroy();

            // Placeholder for complexity analysis
            complexityChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Lines of Code', 'Cyclomatic Complexity', 'Halstead Volume'],
                    datasets: [{
                        label: 'Code Quality Metrics',
                        data: [0, 0, 0], // Placeholder data
                        backgroundColor: 'rgba(0, 122, 204, 0.6)',
                        borderColor: 'rgba(0, 122, 204, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'Metric' } },
                        y: { title: { display: true, text: 'Score' } }
                    }
                }
            });
        }

        function analyzeHyperparameterImpact() {
            // Placeholder for hyperparameter impact analysis
            duplicationImpact = '<h3>Hyperparameter Impact (Placeholder)</h3>';
            duplicationImpact += '<p>This feature is under development. It would analyze how changes in hyperparameters affect model performance.</p>';
            document.getElementById('duplicationImpact').innerHTML = duplicationImpact;
        }
    </script>
</body>
</html>`;
    }
}

// Data Generator
class DataGeneratorAction extends Action2 {
    constructor() {
        super({
            id: 'vsaware.ml.dataGenerator',
            title: localize2('dataGenerator', 'ML: Data Generator'),
            f1: true,
            category: localize2('mlCategory', 'ML Tools')
        });
    }

    async run(accessor: ServicesAccessor): Promise<void> {
        const notificationService = accessor.get(INotificationService);
        const editorService = accessor.get(IEditorService);

        try {
            const generatorHtml = this.generateDataGenerator();

            await editorService.openEditor({
                resource: URI.parse('untitled:data-generator.html'),
                options: {
                    pinned: true,
                    override: 'default'
                }
            } as IUntitledTextResourceEditorInput);

            const activeEditor = editorService.activeTextEditorControl;
            if (activeEditor) {
                (activeEditor as any).setValue(generatorHtml);
            }

            notificationService.info('Data Generator opened! Generate synthetic datasets for ML training.');
        } catch (error) {
            notificationService.error(`Failed to open Data Generator: ${error}`);
        }
    }

    private generateDataGenerator(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <title>Data Generator</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .upload-area {
            border: 2px dashed #007acc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover { background: #f0f8ff; }
        .upload-area.dragover { background: #e6f3ff; border-color: #0056b3; }
        .charts-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .chart-card { background: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }
        .stats-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .stats-table th, .stats-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .stats-table th { background: #f0f0f0; }
        .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        select, button { padding: 8px 16px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007acc; color: white; border: none; cursor: pointer; }
        button:hover { background: #005999; }
        .correlation-matrix { margin-top: 20px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Data Generator</h1>

        <div class="upload-area" id="uploadArea">
            <h3>📁 Drop a dataset description (.json) here or click to upload</h3>
            <p>Supports JSON files containing dataset specifications.</p>
            <input type="file" id="fileInput" accept=".json" style="display: none;">
        </div>

        <div id="datasetSpec" class="hidden">
            <div class="controls">
                <button onclick="generateData()">Generate Data</button>
                <button onclick="resetGenerator()">Reset</button>
            </div>

            <div class="charts-container">
                <div class="chart-card">
                    <h3>Generated Dataset</h3>
                    <canvas id="generatedDatasetChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Dataset Statistics</h3>
                    <table class="stats-table" id="statsTable">
                        <thead>
                            <tr><th>Column</th><th>Type</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let datasetSpec = {};
        let generatedDataset = [];
        let generatedDatasetChart;

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                loadDatasetSpec(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                loadDatasetSpec(e.target.files[0]);
            }
        });

        function loadDatasetSpec(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                datasetSpec = JSON.parse(e.target.result);
                document.getElementById('datasetSpec').classList.remove('hidden');
                generateData(); // Generate data based on loaded spec
            };
            reader.readAsText(file);
        }

        function generateData() {
            if (!datasetSpec || !datasetSpec.columns || datasetSpec.columns.length === 0) {
                alert('Please load a dataset specification file first.');
                return;
            }

            generatedDataset = [];
            const numRows = datasetSpec.num_samples || 100; // Default to 100

            for (let i = 0; i < numRows; i++) {
                const row = {};
                datasetSpec.columns.forEach(col => {
                    if (col.type === 'numeric') {
                        row[col.name] = generateRandomNumber(col.min, col.max, col.distribution);
                    } else if (col.type === 'categorical') {
                        row[col.name] = generateRandomCategorical(col.values);
                    } else if (col.type === 'date') {
                        row[col.name] = generateRandomDate(col.min_date, col.max_date);
                    }
                });
                generatedDataset.push(row);
            }

            updateGeneratedDatasetChart();
            generateStatistics();
        }

        function generateRandomNumber(min, max, distribution) {
            if (distribution === 'uniform') {
                return min + (max - min) * Math.random();
            } else if (distribution === 'normal') {
                const mean = (min + max) / 2;
                const stdDev = (max - min) / 6; // Standard deviation for a reasonable range
                return mean + stdDev * Math.random();
            } else { // Default to uniform
                return min + (max - min) * Math.random();
            }
        }

        function generateRandomCategorical(values) {
            const index = Math.floor(Math.random() * values.length);
            return values[index];
        }

        function generateRandomDate(minDate, maxDate) {
            const min = new Date(minDate).getTime();
            const max = new Date(maxDate).getTime();
            const random = min + Math.random() * (max - min);
            return new Date(random).toISOString().slice(0, 10); // YYYY-MM-DD
        }

        function updateGeneratedDatasetChart() {
            const ctx = document.getElementById('generatedDatasetChart').getContext('2d');
            if (generatedDatasetChart) generatedDatasetChart.destroy();

            const labels = datasetSpec.columns.map(col => col.name);
            const data = datasetSpec.columns.map(col => {
                if (col.type === 'numeric') {
                    return generatedDataset.map(row => row[col.name]);
                } else if (col.type === 'categorical') {
                    return generatedDataset.map(row => row[col.name]);
                } else if (col.type === 'date') {
                    return generatedDataset.map(row => row[col.name]);
                }
                return [];
            });

            generatedDatasetChart = new Chart(ctx, {
                type: 'scatter', // Default to scatter for simplicity, can be changed
                data: {
                    labels: labels,
                    datasets: data.map((colData, index) => ({
                        label: datasetSpec.columns[index].name,
                        data: colData.map((value, i) => ({x: i, y: value})),
                        backgroundColor: 'rgba(0, 122, 204, 0.6)',
                        borderColor: 'rgba(0, 122, 204, 1)',
                        borderWidth: 1
                    }))
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'Row Index' } },
                        y: { title: { display: true, text: 'Value' } }
                    }
                }
            });
        }

        function generateStatistics() {
            const stats = {};

            datasetSpec.columns.forEach(col => {
                const values = generatedDataset.map(row => row[col.name]).filter(v => !isNaN(v));
                const numericValues = values.filter(v => typeof v === 'number');

                stats[col.name] = {
                    type: numericValues.length > values.length * 0.8 ? 'Numeric' : 'Categorical',
                    count: values.length,
                    mean: numericValues.length > 0 ? (numericValues.reduce((a, b) => a + b, 0) / numericValues.length).toFixed(2) : 'N/A',
                    std: numericValues.length > 0 ? Math.sqrt(numericValues.reduce((sq, n) => sq + Math.pow(n - stats[col.name]?.mean || 0, 2), 0) / numericValues.length).toFixed(2) : 'N/A',
                    min: numericValues.length > 0 ? Math.min(...numericValues).toFixed(2) : 'N/A',
                    max: numericValues.length > 0 ? Math.max(...numericValues).toFixed(2) : 'N/A'
                };
            });

            const tbody = document.querySelector('#statsTable tbody');
            tbody.innerHTML = '';

            Object.entries(stats).forEach(([col, stat]) => {
                tbody.innerHTML += \`
                    <tr>
                        <td>\${col}</td>
                        <td>\${stat.type}</td>
                        <td>\${stat.count}</td>
                        <td>\${stat.mean}</td>
                        <td>\${stat.std}</td>
                        <td>\${stat.min}</td>
                        <td>\${stat.max}</td>
                    </tr>
                \`;
            });
        }

        function resetGenerator() {
            datasetSpec = {};
            generatedDataset = [];
            document.getElementById('datasetSpec').classList.add('hidden');
            document.getElementById('generatedDatasetChart').getContext('2d').clearRect(0, 0, document.getElementById('generatedDatasetChart').width, document.getElementById('generatedDatasetChart').height);
            document.getElementById('statsTable').innerHTML = '';
        }
    </script>
</body>
</html>`;
    }
}

// Model Comparator
class ModelComparatorAction extends Action2 {
    constructor() {
        super({
            id: 'vsaware.ml.modelComparator',
            title: localize2('modelComparator', 'ML: Model Comparator'),
            f1: true,
            category: localize2('mlCategory', 'ML Tools')
        });
    }

    async run(accessor: ServicesAccessor): Promise<void> {
        const notificationService = accessor.get(INotificationService);
        const editorService = accessor.get(IEditorService);

        try {
            const comparatorHtml = this.generateModelComparator();

            await editorService.openEditor({
                resource: URI.parse('untitled:model-comparator.html'),
                options: {
                    pinned: true,
                    override: 'default'
                }
            } as IUntitledTextResourceEditorInput);

            const activeEditor = editorService.activeTextEditorControl;
            if (activeEditor) {
                (activeEditor as any).setValue(comparatorHtml);
            }

            notificationService.info('Model Comparator opened! Compare ML model performance.');
        } catch (error) {
            notificationService.error(`Failed to open Model Comparator: ${error}`);
        }
    }

    private generateModelComparator(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <title>Model Comparator</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .upload-area {
            border: 2px dashed #007acc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover { background: #f0f8ff; }
        .upload-area.dragover { background: #e6f3ff; border-color: #0056b3; }
        .charts-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .chart-card { background: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }
        .stats-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .stats-table th, .stats-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .stats-table th { background: #f0f0f0; }
        .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        select, button { padding: 8px 16px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007acc; color: white; border: none; cursor: pointer; }
        button:hover { background: #005999; }
        .correlation-matrix { margin-top: 20px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Model Comparator</h1>

        <div class="upload-area" id="uploadArea">
            <h3>📁 Drop two model files (.pkl, .h5, .pb, .saved_model) here or click to upload</h3>
            <p>Supports multiple model formats.</p>
            <input type="file" id="fileInput" accept=".pkl,.h5,.pb,.saved_model" multiple style="display: none;">
        </div>

        <div id="modelComparison" class="hidden">
            <div class="controls">
                <button onclick="compareModels()">Compare Models</button>
                <button onclick="resetComparison()">Reset</button>
            </div>

            <div class="charts-container">
                <div class="chart-card">
                    <h3>Model Performance Comparison</h3>
                    <canvas id="performanceComparisonChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Model Size Comparison</h3>
                    <canvas id="sizeComparisonChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Model Complexity Comparison</h3>
                    <div id="complexityComparison"></div>
                </div>

                <div class="chart-card">
                    <h3>Dataset Statistics</h3>
                    <table class="stats-table" id="statsTable">
                        <thead>
                            <tr><th>Column</th><th>Type</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let models = [];
        let modelNames = [];
        let performanceData = [];
        let sizeData = [];
        let complexityData = {};
        let datasetSpec = {}; // Reusing datasetSpec for model comparison

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                loadModels(files);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                loadModels(e.target.files);
            }
        });

        function loadModels(files) {
            models = [];
            modelNames = [];
            for (let i = 0; i < files.length; i++) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    try {
                        const modelData = JSON.parse(e.target.result);
                        models.push(modelData);
                        modelNames.push(files[i].name);
                        if (models.length === files.length) {
                            document.getElementById('modelComparison').classList.remove('hidden');
                            compareModels();
                        }
                    } catch (error) {
                        alert('Error loading model ' + files[i].name + ': ' + error);
                    }
                };
                reader.readAsText(files[i]);
            }
        }

        function compareModels() {
            if (models.length < 2) {
                alert('Please upload at least two model files to compare.');
                return;
            }

            performanceData = [];
            sizeData = [];
            complexityData = {};

            for (let i = 0; i < models.length; i++) {
                const model = models[i];
                const name = modelNames[i];
                const performance = {};
                const size = {};

                if (model.model) {
                    try {
                        // Placeholder for actual performance analysis
                        // This would involve loading the model, generating dummy data, and testing it.
                        // For now, we'll just show a placeholder.
                        performance[name] = 'Placeholder for model performance analysis.';
                        size[name] = 'Placeholder for model size.';
                    } catch (error) {
                                                 performance[name] = 'Error loading model for analysis: ' + error;
                         size[name] = 'Error loading model for analysis: ' + error;
                    }
                } else {
                    performance[name] = 'Model data not found.';
                    size[name] = 'Model data not found.';
                }
                performanceData.push(performance);
                sizeData.push(size);
            }

            updatePerformanceComparisonChart();
            updateSizeComparisonChart();
            analyzeComplexity();
            generateStatistics();
        }

        function updatePerformanceComparisonChart() {
            const ctx = document.getElementById('performanceComparisonChart').getContext('2d');
            if (performanceChart) performanceChart.destroy();

            const labels = modelNames;
            const data = performanceData.map(model => Object.values(model));

            performanceChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: data.map((modelData, index) => ({
                        label: modelNames[index],
                        data: modelData,
                        backgroundColor: 'rgba(0, 122, 204, 0.6)',
                        borderColor: 'rgba(0, 122, 204, 1)',
                        borderWidth: 1
                    }))
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'Model' } },
                        y: { title: { display: true, text: 'Performance Metric' } }
                    }
                }
            });
        }

        function updateSizeComparisonChart() {
            const ctx = document.getElementById('sizeComparisonChart').getContext('2d');
            if (sizeChart) sizeChart.destroy();

            const labels = modelNames;
            const data = sizeData.map(model => Object.values(model));

            sizeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: data.map((modelData, index) => ({
                        label: modelNames[index],
                        data: modelData,
                        backgroundColor: 'rgba(255, 107, 107, 0.6)',
                        borderColor: 'rgba(255, 107, 107, 1)',
                        borderWidth: 1
                    }))
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'Model' } },
                        y: { title: { display: true, text: 'Size (Bytes)' } }
                    }
                }
            });
        }

        function analyzeComplexity() {
            complexityData = {};
            for (let i = 0; i < models.length; i++) {
                const model = models[i];
                const name = modelNames[i];
                const complexity = {};

                if (model.model) {
                    try {
                        // Placeholder for actual complexity analysis
                        // This would involve loading the model and analyzing its architecture.
                        complexity[name] = 'Placeholder for model complexity analysis.';
                    } catch (error) {
                                                 complexity[name] = 'Error loading model for analysis: ' + error;
                    }
                } else {
                    complexity[name] = 'Model data not found.';
                }
                complexityData[name] = complexity;
            }

            let complexityHtml = '<h3>Model Complexity Comparison</h3>';
            Object.entries(complexityData).forEach(([modelName, metrics]) => {
                                 complexityHtml += '<p><strong>' + modelName + ':</strong></p>';
                Object.entries(metrics).forEach(([metricName, value]) => {
                                         complexityHtml += '<p>' + metricName + ': ' + value + '</p>';
    });
                complexityHtml += '<hr>';
            });
document.getElementById('complexityComparison').innerHTML = complexityHtml;
        }

function generateStatistics() {
    const stats = {};

    datasetSpec.columns.forEach(col => {
        const values = generatedDataset.map(row => row[col.name]).filter(v => !isNaN(v));
        const numericValues = values.filter(v => typeof v === 'number');

        stats[col.name] = {
            type: numericValues.length > values.length * 0.8 ? 'Numeric' : 'Categorical',
            count: values.length,
            mean: numericValues.length > 0 ? (numericValues.reduce((a, b) => a + b, 0) / numericValues.length).toFixed(2) : 'N/A',
            std: numericValues.length > 0 ? Math.sqrt(numericValues.reduce((sq, n) => sq + Math.pow(n - stats[col.name]?.mean || 0, 2), 0) / numericValues.length).toFixed(2) : 'N/A',
            min: numericValues.length > 0 ? Math.min(...numericValues).toFixed(2) : 'N/A',
            max: numericValues.length > 0 ? Math.max(...numericValues).toFixed(2) : 'N/A'
        };
    });

    const tbody = document.querySelector('#statsTable tbody');
    tbody.innerHTML = '';

    Object.entries(stats).forEach(([col, stat]) => {
        tbody.innerHTML += \`
                    <tr>
                        <td>\${col}</td>
                        <td>\${stat.type}</td>
                        <td>\${stat.count}</td>
                        <td>\${stat.mean}</td>
                        <td>\${stat.std}</td>
                        <td>\${stat.min}</td>
                        <td>\${stat.max}</td>
                    </tr>
                \`;
            });
        }

        function resetComparison() {
            models = [];
            modelNames = [];
            performanceData = [];
            sizeData = [];
            complexityData = {};
            document.getElementById('modelComparison').classList.add('hidden');
            document.getElementById('performanceComparisonChart').getContext('2d').clearRect(0, 0, document.getElementById('performanceComparisonChart').width, document.getElementById('performanceComparisonChart').height);
            document.getElementById('sizeComparisonChart').getContext('2d').clearRect(0, 0, document.getElementById('sizeComparisonChart').width, document.getElementById('sizeComparisonChart').height);
            document.getElementById('complexityComparison').innerHTML = '';
            document.getElementById('statsTable').innerHTML = '';
        }
    </script>
</body>
</html>`;
    }
}

// Hyperparameter Tuner
class HyperparameterTunerAction extends Action2 {
    constructor() {
        super({
            id: 'vsaware.ml.hyperTuner',
            title: localize2('hyperTuner', 'ML: Hyperparameter Tuner'),
            f1: true,
            category: localize2('mlCategory', 'ML Tools')
        });
    }

    async run(accessor: ServicesAccessor): Promise<void> {
        const notificationService = accessor.get(INotificationService);
        const editorService = accessor.get(IEditorService);

        try {
            const tunerHtml = this.generateHyperparameterTuner();

            await editorService.openEditor({
                resource: URI.parse('untitled:hyperparameter-tuner.html'),
                options: {
                    pinned: true,
                    override: 'default'
                }
            } as IUntitledTextResourceEditorInput);

            const activeEditor = editorService.activeTextEditorControl;
            if (activeEditor) {
                (activeEditor as any).setValue(tunerHtml);
            }

            notificationService.info('Hyperparameter Tuner opened! Optimize model hyperparameters.');
        } catch (error) {
            notificationService.error(`Failed to open Hyperparameter Tuner: ${error}`);
        }
    }

    private generateHyperparameterTuner(): string {
        return `<!DOCTYPE html>
<html>
<head>
    <title>Hyperparameter Tuner</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .upload-area {
            border: 2px dashed #007acc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover { background: #f0f8ff; }
        .upload-area.dragover { background: #e6f3ff; border-color: #0056b3; }
        .charts-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .chart-card { background: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }
        .stats-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .stats-table th, .stats-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .stats-table th { background: #f0f0f0; }
        .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        select, button { padding: 8px 16px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007acc; color: white; border: none; cursor: pointer; }
        button:hover { background: #005999; }
        .correlation-matrix { margin-top: 20px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Hyperparameter Tuner</h1>

        <div class="upload-area" id="uploadArea">
            <h3>📁 Drop a dataset specification (.json) and a model template (.py) here or click to upload</h3>
            <p>Supports JSON files containing dataset specifications and Python files for model templates.</p>
            <input type="file" id="fileInput" accept=".json,.py" multiple style="display: none;">
        </div>

        <div id="tunerSetup" class="hidden">
            <div class="controls">
                <button onclick="setupTuning()">Setup Tuning</button>
                <button onclick="resetTuner()">Reset</button>
            </div>

            <div class="charts-container">
                <div class="chart-card">
                    <h3>Model Architecture</h3>
                    <div id="modelArchitecture" style="white-space: pre-wrap; font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9;"></div>
                </div>

                <div class="chart-card">
                    <h3>Hyperparameter Grid</h3>
                    <div id="hyperparameterGrid"></div>
                </div>

                <div class="chart-card">
                    <h3>Dataset Statistics</h3>
                    <table class="stats-table" id="statsTable">
                        <thead>
                            <tr><th>Column</th><th>Type</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>

        <div id="tuningProcess" class="hidden">
            <div class="controls">
                <button onclick="startTuning()">Start Tuning</button>
                <button onclick="stopTuning()">Stop Tuning</button>
                <button onclick="resetTuner()">Reset</button>
            </div>

            <div class="charts-container">
                <div class="chart-card">
                    <h3>Training Progress</h3>
                    <canvas id="tuningProgressChart" width="400" height="300"></canvas>
                </div>

                <div class="chart-card">
                    <h3>Best Model</h3>
                    <div id="bestModelSummary" style="white-space: pre-wrap; font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9;"></div>
                </div>

                <div class="chart-card">
                    <h3>Hyperparameter Impact</h3>
                    <div id="hyperparameterImpact"></div>
                </div>

                <div class="chart-card">
                    <h3>Dataset Statistics</h3>
                    <table class="stats-table" id="statsTable">
                        <thead>
                            <tr><th>Column</th><th>Type</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let datasetSpec = {};
        let modelCode = '';
        let modelArchitecture = '';
        let hyperparameterGrid = {};
        let isTuning = false;
        let tuningProgress = [];
        let bestModel = null;

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                loadFiles(files);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                loadFiles(e.target.files);
            }
        });

        function loadFiles(files) {
            datasetSpec = {};
            modelCode = '';
            modelArchitecture = '';
            hyperparameterGrid = {};
            tuningProgress = [];
            bestModel = null;

            for (let i = 0; i < files.length; i++) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    if (files[i].name.endsWith('.json')) {
                        try {
                            datasetSpec = JSON.parse(e.target.result);
                            document.getElementById('tunerSetup').classList.remove('hidden');
                            setupTuning();
                        } catch (error) {
                                                         alert('Error loading dataset spec file ' + files[i].name + ': ' + error);
                        }
                    } else if (files[i].name.endsWith('.py')) {
                        try {
                            modelCode = e.target.result;
                            document.getElementById('modelArchitecture').textContent = modelCode;
                            document.getElementById('tunerSetup').classList.remove('hidden');
                            setupTuning();
                        } catch (error) {
                                                         alert('Error loading model template file ' + files[i].name + ': ' + error);
                        }
                    }
                };
                reader.readAsText(files[i]);
            }
        }

        function setupTuning() {
            if (!datasetSpec || !datasetSpec.columns || datasetSpec.columns.length === 0) {
                alert('Please upload a dataset specification file (.json) first.');
                return;
            }
            if (!modelCode) {
                alert('Please upload a model template file (.py) first.');
                return;
            }

            modelArchitecture = '';
            hyperparameterGrid = {};
            try {
                const model = tf.sequential(); // Placeholder for actual model loading
                modelArchitecture = model.summary();
                document.getElementById('modelArchitecture').textContent = modelArchitecture;

                // Placeholder for hyperparameter grid generation
                // This would involve parsing the model template and identifying hyperparameters.
                // For now, we'll just show a placeholder.
                hyperparameterGrid = {
                    'learning_rate': {
                        type: 'range',
                        min: 0.001,
                        max: 0.1,
                        step: 0.001,
                        values: [0.001, 0.01, 0.1]
                    },
                    'hidden_layers': {
                        type: 'range',
                        min: 1,
                        max: 5,
                        step: 1,
                        values: [1, 2, 3, 4, 5]
                    },
                    'neurons_per_layer': {
                        type: 'range',
                        min: 4,
                        max: 20,
                        step: 1,
                        values: [4, 10, 20]
                    },
                    'activation_function': {
                        type: 'select',
                        values: ['relu', 'sigmoid', 'tanh']
                    }
                };
                document.getElementById('hyperparameterGrid').innerHTML = this.generateHyperparameterGridHtml(hyperparameterGrid);

            } catch (error) {
                                 modelArchitecture = 'Error loading model architecture: ' + error;
                document.getElementById('modelArchitecture').textContent = modelArchitecture;
                console.error(error);
            }
        }

        function generateHyperparameterGridHtml(grid) {
            let html = '<h4>Hyperparameter Grid</h4>';
            html += '<p>Please specify the range of values for each hyperparameter to tune.</p>';
            html += '<table class="stats-table">';
            html += '<thead><tr><th>Hyperparameter</th><th>Type</th><th>Values</th></tr></thead>';
            html += '<tbody>';
            Object.entries(grid).forEach(([param, options]) => {
                                 html += '<tr><td>' + param + '</td><td>' + (options.type === 'range' ? 'Range' : 'Select') + '</td><td>';
                if (options.type === 'range') {
                                         html += 'Min: ' + options.min.toFixed(4) + ', Max: ' + options.max.toFixed(4) + ', Step: ' + options.step.toFixed(4);
                } else if (options.type === 'select') {
                                         html += options.values.map(val => '<span style="padding: 2px 5px; background-color: #e0e0e0; border-radius: 3px;">' + val + '</span>').join(', ');
    }
                html += '</td></tr>';
});
html += '</tbody></table>';
return html;
        }

function startTuning() {
    if (isTuning) return;
    if (!datasetSpec || !datasetSpec.columns || datasetSpec.columns.length === 0) {
        alert('Please setup the hyperparameter grid first.');
        return;
    }
    if (!modelCode) {
        alert('Please setup the model architecture first.');
        return;
    }

    isTuning = true;
    document.getElementById('tuningProcess').classList.remove('hidden');
    document.getElementById('tunerSetup').classList.add('hidden');
    document.getElementById('tuningProgressChart').getContext('2d').clearRect(0, 0, document.getElementById('tuningProgressChart').width, document.getElementById('tuningProgressChart').height);
    document.getElementById('bestModelSummary').textContent = '';
    document.getElementById('hyperparameterImpact').innerHTML = '';
    tuningProgress = [];
    bestModel = null;

    const modelName = 'tuned_model'; // Placeholder for model name
    const taskType = 'classification'; // Placeholder for task type
    const model = this.createModelFromCode(modelCode, taskType);

    const xs = tf.tensor2d(datasetSpec.data); // Assuming datasetSpec.data is the training data
    const ys = tf.tensor2d(datasetSpec.labels); // Assuming datasetSpec.labels is the training labels

    const objective = (params) => {
        const model = this.createModelFromCode(modelCode, taskType, params);
        return new Promise(resolve => {
            model.fit(xs, ys, {
                epochs: 10, // Short epochs for quick tuning
                batchSize: 32,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        tuningProgress.push({
                            epoch: epoch,
                            loss: logs.loss,
                            accuracy: logs.acc,
                            learning_rate: params.learning_rate
                        });
                        updateTuningProgressChart();
                    }
                }
            }).then(logs => {
                model.dispose();
                xs.dispose();
                ys.dispose();
                resolve(logs.acc); // Return accuracy as objective
            });
        });
    };

    const search = new d3.RandomForestSearch({
        objective: objective,
        parameters: hyperparameterGrid,
        numIterations: 100, // Number of tuning iterations
        numElites: 10, // Number of best results to keep
        numThreads: 4 // Number of parallel threads
    });

    search.start(results => {
        bestModel = results.best;
        document.getElementById('bestModelSummary').textContent = JSON.stringify(bestModel, null, 2);
        analyzeHyperparameterImpact();
                 alert('Tuning complete! Best model found with accuracy: ' + bestModel.value.toFixed(4));
        isTuning = false;
    });
}

function stopTuning() {
    if (isTuning) {
        search.stop(); // Assuming search is defined globally or passed as an argument
        alert('Tuning stopped.');
        isTuning = false;
    }
}

function createModelFromCode(code, taskType, params = {}) {
    const model = tf.sequential();
    const lines = code.split('\n');
    let currentCell = [];
    let cellType = 'code';

    for (const line of lines) {
        if (line.trim().startsWith('# %%') || line.trim().startsWith('#%%')) {
            // New cell marker
            if (currentCell.length > 0) {
                processCell(model, currentCell, taskType, params);
            }
            currentCell = [];
            cellType = 'code';
        } else if (line.trim().startsWith('"""') && currentCell.length === 0) {
            // Markdown cell
            cellType = 'markdown';
            currentCell.push(line.replace('"""', '').trim());
        } else {
            currentCell.push(line);
        }
    }
    // Add the last cell
    if (currentCell.length > 0) {
        processCell(model, currentCell, taskType, params);
    }

    return model;
}

function processCell(model, source, taskType, params) {
    const cellType = source[0].trim().startsWith('# %%') || source[0].trim().startsWith('#%%') ? 'code' : 'markdown';
    const cellContent = source.join('\n').trim();

    if (cellType === 'markdown') {
        // For markdown cells, we can't directly execute code.
        // We'll just add it as a comment or display it.
        model.add(tf.layers.dense({
            units: 1, // Dummy layer to keep the structure
            activation: 'linear',
            useBias: false,
            kernelInitializer: 'zeros'
        }));
        return;
    }

    try {
        model.add(tf.layers.dense({
            units: 1, // Dummy layer to keep the structure
            activation: 'linear',
            useBias: false,
            kernelInitializer: 'zeros'
        }));
        console.log('Executed cell(placeholder): ' + cellContent);
    } catch (error) {
        console.error('Error executing cell: ' + cellContent + '\n' + error);
        model.add(tf.layers.dense({
            units: 1, // Dummy layer to keep the structure
            activation: 'linear',
            useBias: false,
            kernelInitializer: 'zeros'
        }));
    }
}

function updateTuningProgressChart() {
    const ctx = document.getElementById('tuningProgressChart').getContext('2d');
    if (tuningProgressChart) tuningProgressChart.destroy();

    const labels = tuningProgress.map(item => item.epoch);
    const lossData = tuningProgress.map(item => item.loss);
    const accuracyData = tuningProgress.map(item => item.accuracy);
    const lrData = tuningProgress.map(item => item.learning_rate);

    tuningProgressChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Loss',
                    data: lossData,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    fill: true
                },
                {
                    label: 'Accuracy',
                    data: accuracyData,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    fill: true
                },
                {
                    label: 'Learning Rate',
                    data: lrData,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Iteration' } },
                y: { title: { display: true, text: 'Value' } }
            }
        }
    });
}

function analyzeHyperparameterImpact() {
    const impact = {};
    Object.entries(hyperparameterGrid).forEach(([param, options]) => {
        if (options.type === 'range') {
            const values = tuningProgress.map(item => item[param]);
            if (values.length > 0) {
                impact[param] = {
                    mean: values.reduce((a, b) => a + b, 0) / values.length,
                    std: values.length > 1 ? Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - impact[param]?.mean || 0, 2), 0) / (values.length - 1)) : 0
                };
            }
        } else if (options.type === 'select') {
            const values = tuningProgress.map(item => item[param]);
            if (values.length > 0) {
                impact[param] = {
                    mean: values.reduce((a, b) => a + b, 0) / values.length,
                    std: values.length > 1 ? Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - impact[param]?.mean || 0, 2), 0) / (values.length - 1)) : 0
                };
            }
        }
    });

    let impactHtml = '<h3>Hyperparameter Impact</h3>';
    Object.entries(impact).forEach(([param, stats]) => {
        impactHtml += '<p><strong>' + param + ':</strong> Mean = ' + stats.mean.toFixed(4) + ', Std = ' + stats.std.toFixed(4) + '</p>';
    });
    document.getElementById('hyperparameterImpact').innerHTML = impactHtml;
}

function resetTuner() {
    datasetSpec = {};
    modelCode = '';
    modelArchitecture = '';
    hyperparameterGrid = {};
    tuningProgress = [];
    bestModel = null;
    document.getElementById('tunerSetup').classList.add('hidden');
    document.getElementById('tuningProcess').classList.add('hidden');
    document.getElementById('tuningProgressChart').getContext('2d').clearRect(0, 0, document.getElementById('tuningProgressChart').width, document.getElementById('tuningProgressChart').height);
    document.getElementById('bestModelSummary').textContent = '';
    document.getElementById('hyperparameterImpact').innerHTML = '';
    document.getElementById('statsTable').innerHTML = '';
}
</script>
    </body>
    </html>`;
    }
}

// Register all ML actions
registerAction2(ConvertPythonToNotebookAction);
registerAction2(NeuralNetworkPlaygroundAction);
registerAction2(DatasetVisualizerAction);
registerAction2(QuickModelBuilderAction);
registerAction2(TensorShapeAnalyzerAction);
registerAction2(ExperimentTrackerAction);
registerAction2(MLCodeQualityAction);
registerAction2(DataGeneratorAction);
registerAction2(ModelComparatorAction);
registerAction2(HyperparameterTunerAction);

console.log('VS Aware: 10 ML Tools registered successfully!');
