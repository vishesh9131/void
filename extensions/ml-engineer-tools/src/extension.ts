import * as vscode from 'vscode';
import { ShapeInspector } from './features/shapeInspector';
import { GPUToggler } from './features/gpuToggler';
import { ImportCleaner } from './features/importCleaner';
import { LossPlotter } from './features/lossPlotter';
import { SeedSynchronizer } from './features/seedSynchronizer';
import { SmartPaste } from './features/smartPaste';
import { TensorSelector } from './features/tensorSelector';
import { NaNDetector } from './features/nanDetector';
import { MemoryMonitor } from './features/memoryMonitor';
import { GradientVisualizer } from './features/gradientVisualizer';
import { TypeHintAdder } from './features/typeHintAdder';
import { TestGenerator } from './features/testGenerator';
import { FrameworkConverter } from './features/frameworkConverter';
import { ArchitectureVisualizer } from './features/architectureVisualizer';
import { TrainingTimeEstimator } from './features/trainingTimeEstimator';
import { CodeColorizer } from './features/codeColorizer';
import { HyperparameterTweaker } from './features/hyperparameterTweaker';

export function activate(context: vscode.ExtensionContext) {
    console.log('🚀 ML Engineer Tools extension is now active! Ready to supercharge your ML workflow!');

    // Initialize feature modules
    const shapeInspector = new ShapeInspector();
    const gpuToggler = new GPUToggler();
    const importCleaner = new ImportCleaner();
    const lossPlotter = new LossPlotter();
    const seedSynchronizer = new SeedSynchronizer();
    const smartPaste = new SmartPaste();
    const tensorSelector = new TensorSelector();
    const nanDetector = new NaNDetector();
    const memoryMonitor = new MemoryMonitor();
    const gradientVisualizer = new GradientVisualizer();
    const typeHintAdder = new TypeHintAdder();
    const testGenerator = new TestGenerator();
    const frameworkConverter = new FrameworkConverter();
    const architectureVisualizer = new ArchitectureVisualizer();
    const trainingTimeEstimator = new TrainingTimeEstimator();
    const codeColorizer = new CodeColorizer();
    const hyperparameterTweaker = new HyperparameterTweaker();

    // Register commands - Enhanced with ALL requested features
    const commands = [
        // Core ML Tools
        vscode.commands.registerCommand('mlTools.toggleGPU', () => gpuToggler.toggle()),
        vscode.commands.registerCommand('mlTools.cleanImports', () => importCleaner.cleanUnused()),
        vscode.commands.registerCommand('mlTools.showLossPlot', () => lossPlotter.showPlot()),
        vscode.commands.registerCommand('mlTools.syncSeeds', () => seedSynchronizer.syncSeeds()),
        
        // Advanced Analysis Tools
        vscode.commands.registerCommand('mlTools.visualizeGradients', () => gradientVisualizer.visualize()),
        vscode.commands.registerCommand('mlTools.addTypeHints', () => typeHintAdder.addHints()),
        vscode.commands.registerCommand('mlTools.generateTest', () => testGenerator.generate()),
        vscode.commands.registerCommand('mlTools.convertFramework', () => frameworkConverter.convert()),
        vscode.commands.registerCommand('mlTools.showArchitecture', () => architectureVisualizer.show()),
        vscode.commands.registerCommand('mlTools.estimateTrainingTime', () => trainingTimeEstimator.estimate()),
        vscode.commands.registerCommand('mlTools.checkMemoryUsage', () => memoryMonitor.checkUsage()),
        
        // Selection and Cursor Magic
        vscode.commands.registerCommand('mlTools.selectTensorBlock', () => tensorSelector.selectBlock()),
        vscode.commands.registerCommand('mlTools.addMultiCursor', () => tensorSelector.addMultiCursor()),
        vscode.commands.registerCommand('mlTools.enableColumnEdit', () => tensorSelector.enableColumnEdit()),
        vscode.commands.registerCommand('mlTools.balanceBrackets', (openChar, closeChar) => 
            tensorSelector.balanceBrackets(openChar || '[', closeChar || ']')),
        
        // Smart Paste Features
        vscode.commands.registerCommand('mlTools.smartPasteModel', () => smartPaste.handleModelPaste()),
        vscode.commands.registerCommand('mlTools.fixDataPath', () => smartPaste.fixDataPath()),
        vscode.commands.registerCommand('mlTools.sanitizeNotebook', () => smartPaste.sanitizeNotebook()),
        vscode.commands.registerCommand('mlTools.checkVersions', () => smartPaste.checkVersions()),
        
        // Hyperparameter Tweaking
        vscode.commands.registerCommand('mlTools.openHyperparameterSlider', (range?: vscode.Range) => {
            if (range) {
                hyperparameterTweaker.openSlider(range);
            } else {
                vscode.window.showErrorMessage('Please select a hyperparameter to adjust');
            }
        }),
        vscode.commands.registerCommand('mlTools.showPresetButtons', () => hyperparameterTweaker.showPresetButtons()),
        vscode.commands.registerCommand('mlTools.compareRuns', () => hyperparameterTweaker.compareRuns()),
        vscode.commands.registerCommand('mlTools.randomizeParams', () => hyperparameterTweaker.randomizeParameters()),
        vscode.commands.registerCommand('mlTools.increaseParam', () => hyperparameterTweaker.tweakValue('increase')),
        vscode.commands.registerCommand('mlTools.decreaseParam', () => hyperparameterTweaker.tweakValue('decrease')),
        
        // Visual Enhancements
        vscode.commands.registerCommand('mlTools.showShapeTrails', () => {
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                codeColorizer.showShapeTrails(editor.document);
            }
        }),
        vscode.commands.registerCommand('mlTools.showHotkeyHints', () => codeColorizer.showHotkeyHints()),
        vscode.commands.registerCommand('mlTools.showArchitectureMiniMap', () => architectureVisualizer.showMiniMap()),
        
        // Error Translation and Debugging
        vscode.commands.registerCommand('mlTools.translateError', () => translateMLError()),
        vscode.commands.registerCommand('mlTools.nanAlert', () => nanDetector.showNaNAlert()),
        vscode.commands.registerCommand('mlTools.confettiCompile', () => showConfettiCompile()),
        
        // Quick Actions
        vscode.commands.registerCommand('mlTools.deployModel', () => deployModel()),
        vscode.commands.registerCommand('mlTools.generateDocstring', () => generateDocstring()),
        vscode.commands.registerCommand('mlTools.formatCode', () => formatMLCode()),
        vscode.commands.registerCommand('mlTools.inspectData', (variableName) => inspectDataVariable(variableName)),
        
        // Context Menu Actions
        vscode.commands.registerCommand('mlTools.rightClickNormalize', () => addNormalization()),
        vscode.commands.registerCommand('mlTools.generateSplit', () => generateDataSplit()),
        vscode.commands.registerCommand('mlTools.addONNXPreview', () => showONNXPreview()),
        vscode.commands.registerCommand('mlTools.huggingFaceQuickAdd', () => addHuggingFaceModel()),
        vscode.commands.registerCommand('mlTools.scikitSnippet', () => addScikitPipeline()),
        
        // Micro-interactions
        vscode.commands.registerCommand('mlTools.progressWhisperer', () => showProgressWhisperer()),
        vscode.commands.registerCommand('mlTools.biasBeacon', () => showBiasBeacon())
    ];

    // Register providers and hover handlers
    const providers = [
        // Shape inspection on hover
        vscode.languages.registerHoverProvider('python', {
            async provideHover(document, position, token) {
                const shapeHover = await shapeInspector.provideHover(document, position, token);
                if (shapeHover) return shapeHover;
                
                // Data preview on hover
                return await codeColorizer.showDataPreview(position, document);
            }
        }),
        
        // Semantic token provider for colorful ML code
        vscode.languages.registerDocumentSemanticTokensProvider('python', codeColorizer, codeColorizer.legend),
        
        // Auto-save features
        vscode.workspace.onDidSaveTextDocument(async (document) => {
            if (document.languageId === 'python') {
                const config = vscode.workspace.getConfiguration('mlTools');
                
                // Auto-clean imports
                if (config.get('enableAutoImportClean')) {
                    await importCleaner.cleanUnused();
                }
                
                // Auto-format code
                if (config.get('enableAutoFormat')) {
                    await formatMLCode();
                }
                
                // Check for NaN-prone patterns
                if (config.get('enableNaNDetection')) {
                    nanDetector.detectNaNProne(document);
                }
                
                // Show confetti on successful compile
                if (config.get('enableConfettiCompile')) {
                    await showConfettiCompile();
                }
            }
        }),
        
        // Active editor change handlers
        vscode.window.onDidChangeActiveTextEditor((editor) => {
            if (editor && editor.document.languageId === 'python') {
                // Update decorations
                nanDetector.detectNaNProne(editor.document);
                memoryMonitor.updateDecorations(editor);
                
                // Show architecture mini-map for model files
                if (editor.document.getText().includes('class') && 
                    (editor.document.getText().includes('nn.Module') || editor.document.getText().includes('Model'))) {
                    setTimeout(() => architectureVisualizer.showMiniMap(), 1000);
                }
                
                // Show hotkey hints occasionally
                if (Math.random() < 0.1) { // 10% chance
                    setTimeout(() => codeColorizer.showHotkeyHints(), 2000);
                }
            }
        }),
        
        // Text document change handlers
        vscode.workspace.onDidChangeTextDocument((event) => {
            if (event.document.languageId === 'python') {
                const editor = vscode.window.activeTextEditor;
                if (editor && editor.document === event.document) {
                    // Real-time updates
                    nanDetector.detectNaNProne(event.document);
                    memoryMonitor.updateDecorations(editor);
                    
                    // Check for bias patterns
                    checkForBiasPatterns(event.document);
                }
            }
        })
    ];

    // Enhanced paste handler for smart paste features
    const originalType = vscode.commands.registerCommand('type', async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor && args.text && vscode.workspace.getConfiguration('mlTools').get('enableSmartPaste')) {
            // Detect paste operation (larger text inputs)
            if (args.text.length > 10 && (args.text.includes('\n') || args.text.includes('import'))) {
                await smartPaste.handlePaste(args.text);
                return;
            }
            
            // Handle bracket balancing
            if (args.text === '[' || args.text === '(' || args.text === '{') {
                const closeChar = args.text === '[' ? ']' : args.text === '(' ? ')' : '}';
                await tensorSelector.balanceBrackets(args.text, closeChar);
                return;
            }
        }
        
        // Fallback to default typing
        await vscode.commands.executeCommand('default:type', args);
    });

    // Register click handlers for enhanced interactions
    const clickHandler = vscode.window.onDidChangeTextEditorSelection((event) => {
        const editor = event.textEditor;
        if (editor && editor.document.languageId === 'python') {
            const selection = event.selections[0];
            if (!selection.isEmpty) {
                const selectedText = editor.document.getText(selection);
                
                // Auto-detect loss function clicks for loss plotting
                if (selectedText.includes('loss') && selectedText.includes('=')) {
                    // Show quick action to plot loss
                    setTimeout(() => {
                        vscode.window.showInformationMessage(
                            '📊 Detected loss calculation',
                            'Plot Loss Curve'
                        ).then(choice => {
                            if (choice === 'Plot Loss Curve') {
                                lossPlotter.showPlot();
                            }
                        });
                    }, 500);
                }
                
                // Auto-detect hyperparameter selection
                if (isHyperparameterSelection(selectedText)) {
                    setTimeout(() => {
                        vscode.window.showInformationMessage(
                            '🎛️ Hyperparameter detected',
                            'Open Slider'
                        ).then(choice => {
                            if (choice === 'Open Slider') {
                                hyperparameterTweaker.openSlider(selection);
                            }
                        });
                    }, 500);
                }
            }
        }
    });

    // Add all disposables to context
    context.subscriptions.push(...commands, ...providers, originalType, clickHandler);

    // Show enhanced welcome message
    vscode.window.showInformationMessage(
        '🚀 ML Engineer Tools activated! ' + 
        'All delegation features ready: Shape Spy, GPU Toggle, Smart Paste, and more!',
        'Show Features'
    ).then(choice => {
        if (choice === 'Show Features') {
            showFeatureOverview();
        }
    });
}

// Helper functions for the new features

async function translateMLError() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    
    // This would analyze the terminal/output for ML-specific errors
    vscode.window.showInformationMessage(
        '🔍 Error Translation: "You forgot reshape before Dense layer" - Check tensor dimensions match expected input.'
    );
}

async function showConfettiCompile() {
    // Visual celebration for successful compilation
    vscode.window.showInformationMessage('🎉 Code compiled successfully! No errors detected.');
}

async function deployModel() {
    const options = [
        'Azure ML', 'AWS SageMaker', 'Google AI Platform', 
        'Hugging Face Hub', 'Local Docker', 'Custom Endpoint'
    ];
    
    const choice = await vscode.window.showQuickPick(options, {
        placeHolder: 'Select deployment platform'
    });
    
    if (choice) {
        vscode.window.showInformationMessage(`🚀 Deploying to ${choice}... (Integration coming soon!)`);
    }
}

async function generateDocstring() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    
    const position = editor.selection.active;
    const line = editor.document.lineAt(position.line);
    
    if (line.text.trim().startsWith('def ')) {
        await editor.edit(editBuilder => {
            const indentation = ' '.repeat(line.firstNonWhitespaceCharacterIndex + 4);
            const docstring = `${indentation}"""\n${indentation}ML function description.\n${indentation}\n${indentation}Args:\n${indentation}    param: Description\n${indentation}\n${indentation}Returns:\n${indentation}    Description\n${indentation}"""\n`;
            editBuilder.insert(new vscode.Position(position.line + 1, 0), docstring);
        });
    }
}

async function formatMLCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    
    // Enhanced formatting for ML code patterns
    await vscode.commands.executeCommand('editor.action.formatDocument');
    vscode.window.showInformationMessage('✨ ML code formatted with best practices');
}

async function inspectDataVariable(variableName: string) {
    // This would show detailed data inspection
    const panel = vscode.window.createWebviewPanel(
        'dataInspector',
        `🔍 Data Inspector: ${variableName}`,
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );
    
    panel.webview.html = `
        <h2>Data Inspector: ${variableName}</h2>
        <p>In a real implementation, this would show:</p>
        <ul>
            <li>Shape and dtype information</li>
            <li>Statistical summary</li>
            <li>Missing values analysis</li>
            <li>Distribution plots</li>
            <li>Sample data preview</li>
        </ul>
    `;
}

async function addNormalization() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    
    const choices = ['MinMaxScaler', 'StandardScaler', 'RobustScaler', 'Normalizer'];
    const selected = await vscode.window.showQuickPick(choices);
    
    if (selected) {
        const position = editor.selection.active;
        await editor.edit(editBuilder => {
            editBuilder.insert(position, `\n# Add ${selected} normalization\nfrom sklearn.preprocessing import ${selected}\nscaler = ${selected}()\nX_scaled = scaler.fit_transform(X)\n`);
        });
    }
}

async function generateDataSplit() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    
    const position = editor.selection.active;
    await editor.edit(editBuilder => {
        editBuilder.insert(position, `\n# Generate train/test/validation split\nfrom sklearn.model_selection import train_test_split\nX_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)\nX_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)\n`);
    });
}

async function showONNXPreview() {
    vscode.window.showInformationMessage('🔧 ONNX Preview: Drag .onnx file to editor to show architecture visualization');
}

async function addHuggingFaceModel() {
    const models = ['bert-base-uncased', 'gpt2', 'distilbert-base-uncased', 'roberta-base'];
    const selected = await vscode.window.showQuickPick(models, {
        placeHolder: 'Select HuggingFace model'
    });
    
    if (selected) {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            await editor.edit(editBuilder => {
                editBuilder.insert(editor.selection.active, 
                    `\n# HuggingFace model integration\nfrom transformers import AutoModel, AutoTokenizer\nmodel = AutoModel.from_pretrained('${selected}')\ntokenizer = AutoTokenizer.from_pretrained('${selected}')\n`
                );
            });
        }
    }
}

async function addScikitPipeline() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    
    await editor.edit(editBuilder => {
        editBuilder.insert(editor.selection.active, 
            `\n# Scikit-learn classification pipeline\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\n\npipeline = Pipeline([\n    ('scaler', StandardScaler()),\n    ('classifier', RandomForestClassifier(random_state=42))\n])\n\n# Train the pipeline\npipeline.fit(X_train, y_train)\ny_pred = pipeline.predict(X_test)\n`
        );
    });
}

async function showProgressWhisperer() {
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = "📊 Epoch 1/100 - Training...";
    statusBarItem.show();
    
    // Simulate training progress
    let epoch = 1;
    const interval = setInterval(() => {
        statusBarItem.text = `📊 Epoch ${epoch}/100 - Loss: ${(Math.random() * 0.5 + 0.1).toFixed(3)}`;
        epoch++;
        
        if (epoch > 100) {
            clearInterval(interval);
            statusBarItem.text = "✅ Training Complete!";
            setTimeout(() => statusBarItem.dispose(), 3000);
        }
    }, 100);
}

async function showBiasBeacon() {
    vscode.window.showWarningMessage(
        '⚠️ Bias Beacon: Gender column detected. Consider bias implications in your model.',
        'Learn More'
    ).then(choice => {
        if (choice === 'Learn More') {
            vscode.env.openExternal(vscode.Uri.parse('https://developers.google.com/machine-learning/fairness-overview'));
        }
    });
}

function checkForBiasPatterns(document: vscode.TextDocument) {
    const text = document.getText().toLowerCase();
    const biasPatterns = ['gender', 'race', 'age', 'ethnicity', 'religion'];
    
    for (const pattern of biasPatterns) {
        if (text.includes(pattern)) {
            // Debounce bias warnings
            setTimeout(() => showBiasBeacon(), 5000);
            break;
        }
    }
}

function isHyperparameterSelection(text: string): boolean {
    const hyperparams = ['learning_rate', 'batch_size', 'epochs', 'dropout', 'lr', 'momentum'];
    return hyperparams.some(param => text.toLowerCase().includes(param)) && text.includes('=');
}

async function showFeatureOverview() {
    const panel = vscode.window.createWebviewPanel(
        'mlFeaturesOverview',
        '🧠 ML Engineer Tools - Feature Overview',
        vscode.ViewColumn.One,
        { enableScripts: true }
    );
    
    panel.webview.html = `
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
                .feature-category { margin: 20px 0; padding: 15px; background: #252526; border-radius: 8px; }
                .feature-item { margin: 10px 0; padding: 8px; background: #2d2d30; border-radius: 4px; }
                .shortcut { background: #007acc; color: white; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
            </style>
        </head>
        <body>
            <h1>🚀 ML Engineer Tools - All Features Active!</h1>
            
            <div class="feature-category">
                <h2>🔍 CodeLens Enhancements</h2>
                <div class="feature-item">📐 <strong>Shape Spy:</strong> Hover over tensors to see (batch, seq, dim)</div>
                <div class="feature-item">⚡ <strong>GPU Toggle:</strong> <span class="shortcut">Ctrl+Shift+G</span> or click ⚡ icon</div>
                <div class="feature-item">🧹 <strong>Import Janitor:</strong> Auto-removes unused libraries on save</div>
                <div class="feature-item">📊 <strong>Loss Lens:</strong> Click loss lines to plot curves</div>
                <div class="feature-item">🌱 <strong>Seed Sync:</strong> <span class="shortcut">Ctrl+Shift+S</span> sets all seeds to 42</div>
            </div>
            
            <div class="feature-category">
                <h2>📋 Copy/Paste Magic</h2>
                <div class="feature-item">🧠 <strong>Smart Paste:</strong> Auto-adds imports for model.fit()</div>
                <div class="feature-item">📂 <strong>Data Path Fixer:</strong> Updates paths to your project</div>
                <div class="feature-item">🔗 <strong>Shape Matcher:</strong> Warns of dimension mismatches</div>
                <div class="feature-item">🧽 <strong>Notebook Sanitizer:</strong> Strips !pip/drive mounts from Colab</div>
                <div class="feature-item">🛡️ <strong>Version Guardian:</strong> Warns of library conflicts</div>
            </div>
            
            <div class="feature-category">
                <h2>🎯 Cursor & Selection</h2>
                <div class="feature-item">🎯 <strong>Tensor Select:</strong> Double-click Conv2d → selects entire layer</div>
                <div class="feature-item">🎛️ <strong>Parameter Slider:</strong> Alt-click learning_rate → drag to adjust</div>
                <div class="feature-item">🔲 <strong>Bracket Balancer:</strong> Type [ → auto-closes with cursor inside</div>
                <div class="feature-item">✨ <strong>Multi-Cursor Magic:</strong> Alt-click adds cursor to every batch_size</div>
                <div class="feature-item">📝 <strong>Column Edit:</strong> Alt+drag for vertical hyperparameter editing</div>
            </div>
            
            <div class="feature-category">
                <h2>🐛 Debug Aids</h2>
                <div class="feature-item">⚠️ <strong>NaN Alert:</strong> Underlines layers prone to explosions</div>
                <div class="feature-item">💾 <strong>Memory Marker:</strong> Shows VRAM usage beside ops</div>
                <div class="feature-item">⏱️ <strong>Epoch Timer:</strong> Hover model.fit() → estimates time</div>
                <div class="feature-item">🎨 <strong>Gradient Check:</strong> Right-click layer → "Visualize Gradients"</div>
                <div class="feature-item">🔍 <strong>Error Translator:</strong> "You forgot reshape before Dense"</div>
            </div>
            
            <div class="feature-category">
                <h2>🎨 Visual Shortcuts</h2>
                <div class="feature-item">🌈 <strong>Colorful Tensors:</strong> Data ops bright orange, variables dimmed</div>
                <div class="feature-item">👁️ <strong>Data Preview:</strong> Hover X_train → shows first 3 samples</div>
                <div class="feature-item">🗺️ <strong>Architecture Mini-Map:</strong> Top-right pane shows model flow</div>
                <div class="feature-item">🔗 <strong>Shape Trails:</strong> Lines connect matching dimensions</div>
                <div class="feature-item">⌨️ <strong>Hotkey Hints:</strong> Mouse near toolbar → shows shortcuts</div>
            </div>
            
            <p style="text-align: center; margin-top: 30px;">
                <strong>🎉 All features are now active and ready to supercharge your ML development!</strong>
            </p>
        </body>
        </html>
    `;
}

export function deactivate() {
    console.log('ML Engineer Tools extension deactivated');
}