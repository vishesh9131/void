import * as vscode from 'vscode';

export class CodeColorizer implements vscode.DocumentSemanticTokensProvider {
    readonly legend = new vscode.SemanticTokensLegend(
        ['mlTensor', 'mlDataOp', 'mlModel', 'mlOptimizer', 'mlLoss', 'mlMetric', 'mlHyperParam'],
        ['bold', 'italic']
    );

    private tensorPatterns = [
        /\b\w*tensor\w*\b/gi,
        /\b\w*Tensor\w*\b/g,
        /torch\.(?:zeros|ones|randn|rand|empty|tensor)\b/g,
        /tf\.(?:Variable|constant|zeros|ones)\b/g,
        /np\.(?:array|zeros|ones|empty)\b/g
    ];

    private dataOpPatterns = [
        /\.(?:reshape|view|permute|transpose|squeeze|unsqueeze|flatten)\b/g,
        /\.(?:sum|mean|max|min|std|var)\b/g,
        /torch\.(?:cat|stack|split)\b/g,
        /tf\.(?:concat|stack|split)\b/g,
        /np\.(?:concatenate|stack|split)\b/g
    ];

    private modelPatterns = [
        /\b\w*(?:model|Model|net|Net|network|Network)\w*\b/g,
        /nn\.(?:Module|Sequential|Linear|Conv|LSTM|GRU)\b/g,
        /tf\.keras\.(?:Model|Sequential|layers)\b/g
    ];

    private optimizerPatterns = [
        /\b\w*(?:optimizer|Optimizer|optim)\w*\b/g,
        /torch\.optim\.\w+\b/g,
        /tf\.keras\.optimizers\.\w+\b/g
    ];

    private lossPatterns = [
        /\b\w*(?:loss|Loss|criterion|error|Error)\w*\b/g,
        /nn\.(?:CrossEntropyLoss|MSELoss|BCELoss)\b/g,
        /tf\.keras\.losses\.\w+\b/g
    ];

    private hyperParamPatterns = [
        /\b(?:learning_rate|lr|batch_size|epochs|dropout|momentum)\b/g,
        /\b(?:weight_decay|beta1|beta2|epsilon)\b/g
    ];

    async provideDocumentSemanticTokens(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): Promise<vscode.SemanticTokens> {
        const tokensBuilder = new vscode.SemanticTokensBuilder(this.legend);
        
        if (!vscode.workspace.getConfiguration('mlTools').get('enableColorfulTensors')) {
            return tokensBuilder.build();
        }

        const text = document.getText();
        const lines = text.split('\n');

        for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
            const line = lines[lineIndex];
            
            // Highlight tensors
            this.highlightPatterns(tokensBuilder, line, lineIndex, this.tensorPatterns, 'mlTensor');
            
            // Highlight data operations
            this.highlightPatterns(tokensBuilder, line, lineIndex, this.dataOpPatterns, 'mlDataOp');
            
            // Highlight models
            this.highlightPatterns(tokensBuilder, line, lineIndex, this.modelPatterns, 'mlModel');
            
            // Highlight optimizers
            this.highlightPatterns(tokensBuilder, line, lineIndex, this.optimizerPatterns, 'mlOptimizer');
            
            // Highlight loss functions
            this.highlightPatterns(tokensBuilder, line, lineIndex, this.lossPatterns, 'mlLoss');
            
            // Highlight hyperparameters
            this.highlightPatterns(tokensBuilder, line, lineIndex, this.hyperParamPatterns, 'mlHyperParam');
        }

        return tokensBuilder.build();
    }

    private highlightPatterns(
        builder: vscode.SemanticTokensBuilder,
        line: string,
        lineIndex: number,
        patterns: RegExp[],
        tokenType: string
    ) {
        for (const pattern of patterns) {
            pattern.lastIndex = 0; // Reset regex
            let match;
            
            while ((match = pattern.exec(line)) !== null) {
                const startPos = match.index;
                const length = match[0].length;
                
                builder.push(
                    new vscode.Range(
                        new vscode.Position(lineIndex, startPos),
                        new vscode.Position(lineIndex, startPos + length)
                    ),
                    tokenType
                );
            }
        }
    }

    // Data preview on hover
    async showDataPreview(position: vscode.Position, document: vscode.TextDocument): Promise<vscode.Hover | undefined> {
        const wordRange = document.getWordRangeAtPosition(position);
        if (!wordRange) {
            return;
        }

        const word = document.getText(wordRange);
        const line = document.lineAt(position.line).text;

        // Check if it's a variable that might contain data
        if (this.isDataVariable(word, line)) {
            const previewInfo = this.generateDataPreview(word, line);
            
            if (previewInfo) {
                const markdown = new vscode.MarkdownString();
                markdown.isTrusted = true;
                markdown.appendMarkdown(`**📊 Data Preview: \`${word}\`**\n\n`);
                markdown.appendMarkdown(previewInfo);
                
                return new vscode.Hover(markdown, wordRange);
            }
        }

        return;
    }

    private isDataVariable(word: string, line: string): boolean {
        // Check if variable name suggests data
        const dataNames = ['X', 'y', 'data', 'dataset', 'X_train', 'X_test', 'y_train', 'y_test', 'features', 'labels'];
        
        if (dataNames.some(name => word.toLowerCase().includes(name.toLowerCase()))) {
            return true;
        }

        // Check if line contains data loading operations
        const dataOperations = ['pd.read_csv', 'np.load', 'torch.load', 'tf.data', '.reshape', '.values'];
        
        return dataOperations.some(op => line.includes(op));
    }

    private generateDataPreview(variableName: string, line: string): string | null {
        // Simulate data preview (in real implementation, this would execute code safely)
        const samples = this.generateSampleData(variableName, line);
        
        if (samples) {
            let preview = `\`\`\`\n${samples}\`\`\`\n\n`;
            preview += `**Shape:** (estimated)\n`;
            preview += `**Type:** ${this.inferDataType(variableName, line)}\n`;
            preview += `**Memory:** ${this.estimateMemoryUsage(variableName, line)}\n\n`;
            preview += `[Inspect Full Data](command:mlTools.inspectData?${variableName})`;
            
            return preview;
        }

        return null;
    }

    private generateSampleData(variableName: string, line: string): string | null {
        // Generate realistic sample data based on context
        if (variableName.toLowerCase().includes('x') || variableName.includes('feature')) {
            return `[[0.1, 0.2, 0.3],\n [0.4, 0.5, 0.6],\n [0.7, 0.8, 0.9]]\n...`;
        }
        
        if (variableName.toLowerCase().includes('y') || variableName.includes('label')) {
            return `[0, 1, 2, 1, 0, 2, ...]\n(3 samples shown)`;
        }
        
        if (line.includes('pd.read_csv') || variableName.includes('df')) {
            return `   feature1  feature2  feature3\n0      1.23      4.56      7.89\n1      2.34      5.67      8.90\n2      3.45      6.78      9.01\n...`;
        }

        return null;
    }

    private inferDataType(variableName: string, line: string): string {
        if (line.includes('torch.')) return 'torch.Tensor';
        if (line.includes('tf.') || line.includes('tensorflow')) return 'tf.Tensor';
        if (line.includes('np.') || line.includes('numpy')) return 'numpy.ndarray';
        if (line.includes('pd.') || line.includes('pandas')) return 'pandas.DataFrame';
        
        return 'unknown';
    }

    private estimateMemoryUsage(variableName: string, line: string): string {
        // Simple estimation based on variable name patterns
        if (variableName.includes('train')) return '~50-100 MB';
        if (variableName.includes('test')) return '~10-20 MB';
        if (variableName.includes('batch')) return '~1-5 MB';
        
        return '~unknown';
    }

    // Shape trails - connect variables with matching dimensions
    async showShapeTrails(document: vscode.TextDocument): Promise<void> {
        const decorationType = vscode.window.createTextEditorDecorationType({
            backgroundColor: new vscode.ThemeColor('mlTools.shapeTrail'),
            border: '1px solid',
            borderColor: new vscode.ThemeColor('mlTools.shapeTrailBorder')
        });

        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document !== document) {
            return;
        }

        const text = document.getText();
        const shapeConnections = this.findShapeConnections(text);
        
        const decorations: vscode.DecorationOptions[] = [];
        
        for (const connection of shapeConnections) {
            // Add decorations to show connected variables
            for (const position of connection.positions) {
                decorations.push({
                    range: new vscode.Range(position, position.translate(0, connection.variableName.length)),
                    hoverMessage: `Connected to: ${connection.connectedVariables.join(', ')}`
                });
            }
        }

        editor.setDecorations(decorationType, decorations);

        // Auto-remove decorations after 5 seconds
        setTimeout(() => {
            decorationType.dispose();
        }, 5000);
    }

    private findShapeConnections(text: string): ShapeConnection[] {
        // Simplified shape connection analysis
        const connections: ShapeConnection[] = [];
        const lines = text.split('\n');
        
        // This would be more sophisticated in a real implementation
        // For now, just find variables that might be connected by shape
        
        return connections;
    }

    // Hotkey hints near toolbar
    async showHotkeyHints(): Promise<void> {
        const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 200);
        statusBarItem.text = "🎯 ML Shortcuts";
        statusBarItem.tooltip = this.getHotkeyHintsTooltip();
        statusBarItem.show();

        // Auto-hide after 8 seconds
        setTimeout(() => {
            statusBarItem.dispose();
        }, 8000);
    }

    private getHotkeyHintsTooltip(): string {
        return `ML Engineer Shortcuts:
Ctrl+Shift+G - Toggle GPU/CPU
Ctrl+Shift+S - Sync Seeds
Ctrl+Shift+I - Clean Imports
Alt+Click - Multi-cursor on variables
Double-click Conv2d - Select tensor block
Right-click layer - Visualize gradients`;
    }
}

interface ShapeConnection {
    variableName: string;
    positions: vscode.Position[];
    connectedVariables: string[];
    shape: string;
}