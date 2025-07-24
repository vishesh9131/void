import * as vscode from 'vscode';

export class TensorSelector {
    private tensorBlockPatterns = [
        // PyTorch layer blocks
        /(?:nn\.)?\w*(?:Conv|Linear|BatchNorm|LayerNorm|Dropout|ReLU|Sigmoid|Tanh)\d*\([^)]*\)/g,
        // Model definition blocks
        /class\s+\w+\((?:nn\.)?Module\):/g,
        // Sequential blocks
        /nn\.Sequential\s*\(/g
    ];

    async selectBlock() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const position = editor.selection.active;
        const document = editor.document;
        const block = this.findTensorBlock(document, position);

        if (block) {
            editor.selection = new vscode.Selection(block.start, block.end);
            vscode.window.showInformationMessage('🎯 Tensor block selected');
        }
    }

    async addMultiCursor() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const document = editor.document;
        const word = document.getText(document.getWordRangeAtPosition(editor.selection.active));
        
        if (!word) {
            return;
        }

        // Find all occurrences of the word (like batch_size, learning_rate, etc.)
        const positions = this.findAllOccurrences(document, word);
        
        if (positions.length > 1) {
            const selections = positions.map(pos => new vscode.Selection(pos, pos));
            editor.selections = selections;
            vscode.window.showInformationMessage(`✨ Added ${positions.length} cursors for '${word}'`);
        }
    }

    async enableColumnEdit() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        // This would be used with Alt+drag gesture
        // For now, implement a command to enable column editing mode
        const startPosition = editor.selection.start;
        const endPosition = editor.selection.end;

        if (startPosition.line === endPosition.line) {
            vscode.window.showInformationMessage('📝 Select multiple lines for column editing');
            return;
        }

        // Create cursors at the same column for each line
        const selections: vscode.Selection[] = [];
        const column = startPosition.character;

        for (let line = startPosition.line; line <= endPosition.line; line++) {
            const pos = new vscode.Position(line, column);
            selections.push(new vscode.Selection(pos, pos));
        }

        editor.selections = selections;
        vscode.window.showInformationMessage('📐 Column edit mode enabled');
    }

    private findTensorBlock(document: vscode.TextDocument, position: vscode.Position): vscode.Range | null {
        const line = document.lineAt(position);
        const lineText = line.text;

        // Check if current line contains a tensor operation
        for (const pattern of this.tensorBlockPatterns) {
            pattern.lastIndex = 0;
            if (pattern.test(lineText)) {
                return this.expandToFullBlock(document, position);
            }
        }

        return null;
    }

    private expandToFullBlock(document: vscode.TextDocument, position: vscode.Position): vscode.Range {
        let startLine = position.line;
        let endLine = position.line;

        // Expand upward to find the start of the block
        while (startLine > 0) {
            const prevLine = document.lineAt(startLine - 1);
            if (this.isPartOfTensorBlock(prevLine.text) || this.isIndentedContinuation(prevLine.text, document.lineAt(position).text)) {
                startLine--;
            } else {
                break;
            }
        }

        // Expand downward to find the end of the block
        while (endLine < document.lineCount - 1) {
            const nextLine = document.lineAt(endLine + 1);
            if (this.isPartOfTensorBlock(nextLine.text) || this.isIndentedContinuation(nextLine.text, document.lineAt(position).text)) {
                endLine++;
            } else {
                break;
            }
        }

        return new vscode.Range(
            new vscode.Position(startLine, 0),
            new vscode.Position(endLine, document.lineAt(endLine).text.length)
        );
    }

    private isPartOfTensorBlock(line: string): boolean {
        const trimmed = line.trim();
        return trimmed.includes('nn.') || 
               trimmed.includes('torch.') || 
               trimmed.includes('tf.') ||
               trimmed.includes('layers.') ||
               trimmed.match(/^\s*[,)]/) !== null; // Continuation lines
    }

    private isIndentedContinuation(line: string, referenceLine: string): boolean {
        const lineIndent = line.length - line.trimStart().length;
        const refIndent = referenceLine.length - referenceLine.trimStart().length;
        
        return lineIndent > refIndent && line.trim().length > 0;
    }

    private findAllOccurrences(document: vscode.TextDocument, word: string): vscode.Position[] {
        const positions: vscode.Position[] = [];
        const text = document.getText();
        const wordRegex = new RegExp(`\\b${word}\\b`, 'g');
        
        let match;
        while ((match = wordRegex.exec(text)) !== null) {
            const position = document.positionAt(match.index);
            positions.push(position);
        }

        return positions;
    }

    // Bracket balancer functionality
    async balanceBrackets(openChar: string, closeChar: string) {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const position = editor.selection.active;
        const newPosition = new vscode.Position(position.line, position.character + 1);
        
        await editor.edit(editBuilder => {
            editBuilder.insert(position, openChar + closeChar);
        });

        // Move cursor between brackets
        editor.selection = new vscode.Selection(newPosition, newPosition);
    }
}