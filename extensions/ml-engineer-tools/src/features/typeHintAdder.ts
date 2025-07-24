import * as vscode from 'vscode';

export class TypeHintAdder {
    private mlTypeHints = new Map([
        // PyTorch types
        ['torch.Tensor', 'torch.Tensor'],
        ['tensor', 'torch.Tensor'],
        ['model', 'torch.nn.Module'],
        ['optimizer', 'torch.optim.Optimizer'],
        ['loss_fn', 'torch.nn.Module'],
        ['criterion', 'torch.nn.Module'],
        
        // NumPy types
        ['array', 'np.ndarray'],
        ['ndarray', 'np.ndarray'],
        ['X', 'np.ndarray'],
        ['y', 'np.ndarray'],
        ['X_train', 'np.ndarray'],
        ['X_test', 'np.ndarray'],
        ['y_train', 'np.ndarray'],
        ['y_test', 'np.ndarray'],
        
        // Common ML types
        ['batch_size', 'int'],
        ['learning_rate', 'float'],
        ['lr', 'float'],
        ['epochs', 'int'],
        ['num_classes', 'int'],
        ['hidden_size', 'int'],
        ['input_size', 'int'],
        ['output_size', 'int'],
        ['dropout', 'float'],
        
        // DataFrame types
        ['df', 'pd.DataFrame'],
        ['data', 'pd.DataFrame'],
        ['dataset', 'pd.DataFrame'],
        
        // Device types
        ['device', 'torch.device'],
    ]);

    async addHints() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const document = editor.document;
        const selection = editor.selection;

        if (selection.isEmpty) {
            // Add type hints to entire function or class
            await this.addHintsToScope(editor);
        } else {
            // Add type hints to selected code
            await this.addHintsToSelection(editor, selection);
        }
    }

    private async addHintsToScope(editor: vscode.TextEditor) {
        const document = editor.document;
        const position = editor.selection.active;
        
        // Find the function or class that contains the current position
        const scope = this.findContainingScope(document, position);
        
        if (scope) {
            await this.processScope(editor, scope);
        } else {
            vscode.window.showInformationMessage('🔍 No function or class found at current position');
        }
    }

    private async addHintsToSelection(editor: vscode.TextEditor, selection: vscode.Selection) {
        const document = editor.document;
        const selectedText = document.getText(selection);
        
        const typeHints = this.analyzeCodeForTypeHints(selectedText);
        
        if (typeHints.length > 0) {
            await this.applyTypeHints(editor, typeHints, selection);
        } else {
            vscode.window.showInformationMessage('🤔 No suitable variables found for type hints in selection');
        }
    }

    private findContainingScope(document: vscode.TextDocument, position: vscode.Position): ScopeInfo | null {
        // Search backwards to find function or class definition
        for (let line = position.line; line >= 0; line--) {
            const lineText = document.lineAt(line).text;
            
            // Check for function definition
            const funcMatch = /^(\s*)def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.+?))?:/g.exec(lineText);
            if (funcMatch) {
                const indentation = funcMatch[1];
                const funcName = funcMatch[2];
                const params = funcMatch[3];
                const returnType = funcMatch[4];
                
                const endLine = this.findScopeEnd(document, line, indentation.length);
                
                return {
                    type: 'function',
                    name: funcName,
                    startLine: line,
                    endLine,
                    parameters: params,
                    returnType,
                    indentation: indentation.length
                };
            }
            
            // Check for class definition
            const classMatch = /^(\s*)class\s+(\w+)(?:\(.*?\))?:/g.exec(lineText);
            if (classMatch) {
                const indentation = classMatch[1];
                const className = classMatch[2];
                
                const endLine = this.findScopeEnd(document, line, indentation.length);
                
                return {
                    type: 'class',
                    name: className,
                    startLine: line,
                    endLine,
                    indentation: indentation.length
                };
            }
        }
        
        return null;
    }

    private findScopeEnd(document: vscode.TextDocument, startLine: number, baseIndentation: number): number {
        for (let line = startLine + 1; line < document.lineCount; line++) {
            const lineText = document.lineAt(line).text;
            
            if (lineText.trim().length === 0) {
                continue; // Skip empty lines
            }
            
            const lineIndentation = lineText.length - lineText.trimStart().length;
            if (lineIndentation <= baseIndentation) {
                return line - 1;
            }
        }
        
        return document.lineCount - 1;
    }

    private async processScope(editor: vscode.TextEditor, scope: ScopeInfo) {
        const document = editor.document;
        const typeHints: TypeHint[] = [];
        
        if (scope.type === 'function') {
            // Process function parameters
            if (scope.parameters) {
                const paramHints = this.analyzeParameters(scope.parameters);
                typeHints.push(...paramHints);
            }
            
            // Process function body
            const bodyText = this.getScopeBody(document, scope);
            const bodyHints = this.analyzeCodeForTypeHints(bodyText);
            typeHints.push(...bodyHints);
            
            // Process return type
            if (!scope.returnType) {
                const returnTypeHint = this.inferReturnType(bodyText);
                if (returnTypeHint) {
                    typeHints.push(returnTypeHint);
                }
            }
        }
        
        if (typeHints.length > 0) {
            await this.applyTypeHints(editor, typeHints, new vscode.Selection(
                new vscode.Position(scope.startLine, 0),
                new vscode.Position(scope.endLine, document.lineAt(scope.endLine).text.length)
            ));
        }
    }

    private analyzeParameters(parameters: string): TypeHint[] {
        const hints: TypeHint[] = [];
        const params = parameters.split(',').map(p => p.trim()).filter(p => p.length > 0);
        
        for (const param of params) {
            const [name] = param.split('=')[0].split(':')[0].trim().split(' ');
            
            if (name === 'self' || name === 'cls') {
                continue;
            }
            
            const suggestedType = this.inferTypeFromName(name);
            if (suggestedType) {
                hints.push({
                    variableName: name,
                    suggestedType,
                    context: 'parameter',
                    line: -1 // Will be resolved when applying
                });
            }
        }
        
        return hints;
    }

    private getScopeBody(document: vscode.TextDocument, scope: ScopeInfo): string {
        let body = '';
        for (let line = scope.startLine + 1; line <= scope.endLine; line++) {
            body += document.lineAt(line).text + '\n';
        }
        return body;
    }

    private analyzeCodeForTypeHints(code: string): TypeHint[] {
        const hints: TypeHint[] = [];
        const lines = code.split('\n');
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // Find variable assignments
            const assignmentPattern = /^\s*(\w+)\s*=\s*(.+)/;
            const match = assignmentPattern.exec(line);
            
            if (match) {
                const varName = match[1];
                const assignment = match[2];
                
                // Skip if already has type annotation
                if (line.includes(':')) {
                    continue;
                }
                
                const inferredType = this.inferTypeFromAssignment(assignment) || this.inferTypeFromName(varName);
                
                if (inferredType) {
                    hints.push({
                        variableName: varName,
                        suggestedType: inferredType,
                        context: 'assignment',
                        line: i
                    });
                }
            }
        }
        
        return hints;
    }

    private inferTypeFromName(name: string): string | null {
        // Check exact matches
        if (this.mlTypeHints.has(name)) {
            return this.mlTypeHints.get(name)!;
        }
        
        // Check pattern matches
        if (name.includes('tensor') || name.endsWith('_tensor')) {
            return 'torch.Tensor';
        }
        
        if (name.includes('model') || name.endsWith('_model')) {
            return 'torch.nn.Module';
        }
        
        if (name.includes('optimizer') || name.endsWith('_optimizer')) {
            return 'torch.optim.Optimizer';
        }
        
        if (name.includes('loss') || name.endsWith('_loss')) {
            return 'torch.nn.Module';
        }
        
        if (name.includes('size') || name.includes('dim') || name.includes('num_')) {
            return 'int';
        }
        
        if (name.includes('rate') || name.includes('prob') || name.includes('ratio')) {
            return 'float';
        }
        
        if (name.startsWith('is_') || name.startsWith('has_') || name.startsWith('should_')) {
            return 'bool';
        }
        
        return null;
    }

    private inferTypeFromAssignment(assignment: string): string | null {
        assignment = assignment.trim();
        
        // PyTorch tensors
        if (assignment.includes('torch.') && (assignment.includes('tensor') || assignment.includes('zeros') || assignment.includes('ones'))) {
            return 'torch.Tensor';
        }
        
        // NumPy arrays
        if (assignment.includes('np.') && (assignment.includes('array') || assignment.includes('zeros') || assignment.includes('ones'))) {
            return 'np.ndarray';
        }
        
        // Pandas DataFrames
        if (assignment.includes('pd.') && assignment.includes('DataFrame')) {
            return 'pd.DataFrame';
        }
        
        // PyTorch models
        if (assignment.includes('nn.') && (assignment.includes('Module') || assignment.includes('Sequential'))) {
            return 'torch.nn.Module';
        }
        
        // Optimizers
        if (assignment.includes('optim.')) {
            return 'torch.optim.Optimizer';
        }
        
        // Lists and tuples
        if (assignment.startsWith('[') && assignment.endsWith(']')) {
            return 'List';
        }
        
        if (assignment.startsWith('(') && assignment.endsWith(')')) {
            return 'Tuple';
        }
        
        // Literals
        if (assignment.match(/^\d+$/)) {
            return 'int';
        }
        
        if (assignment.match(/^\d+\.\d+$/)) {
            return 'float';
        }
        
        if (assignment === 'True' || assignment === 'False') {
            return 'bool';
        }
        
        if (assignment.startsWith('"') || assignment.startsWith("'")) {
            return 'str';
        }
        
        return null;
    }

    private inferReturnType(bodyText: string): TypeHint | null {
        const returnStatements = bodyText.match(/return\s+(.+)/g);
        
        if (returnStatements) {
            for (const stmt of returnStatements) {
                const returnValue = stmt.replace('return', '').trim();
                const inferredType = this.inferTypeFromAssignment(returnValue);
                
                if (inferredType) {
                    return {
                        variableName: 'return',
                        suggestedType: inferredType,
                        context: 'return',
                        line: -1
                    };
                }
            }
        }
        
        return null;
    }

    private async applyTypeHints(editor: vscode.TextEditor, hints: TypeHint[], scope: vscode.Selection) {
        if (hints.length === 0) {
            return;
        }
        
        // Show quick pick to let user select which hints to apply
        const items = hints.map(hint => ({
            label: `${hint.variableName}: ${hint.suggestedType}`,
            description: hint.context,
            hint
        }));
        
        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select type hints to add',
            canPickMany: true
        });
        
        if (selected && selected.length > 0) {
            await editor.edit(editBuilder => {
                for (const item of selected) {
                    this.applyTypeHint(editBuilder, editor.document, item.hint, scope);
                }
            });
            
            vscode.window.showInformationMessage(`✨ Added ${selected.length} type hints`);
        }
    }

    private applyTypeHint(editBuilder: vscode.TextEditorEdit, document: vscode.TextDocument, hint: TypeHint, scope: vscode.Selection) {
        // This is a simplified implementation
        // In a real implementation, you'd need to handle various cases more carefully
        
        if (hint.context === 'parameter') {
            // Find the function definition and add parameter type
            // This would require more complex parsing
        } else if (hint.context === 'assignment') {
            // Add type annotation to variable assignment
            const lines = document.getText(scope).split('\n');
            if (hint.line >= 0 && hint.line < lines.length) {
                const line = lines[hint.line];
                const assignmentPattern = new RegExp(`^(\\s*)(${hint.variableName})(\\s*=\\s*.+)`);
                const match = assignmentPattern.exec(line);
                
                if (match) {
                    const newLine = `${match[1]}${match[2]}: ${hint.suggestedType}${match[3]}`;
                    const lineNumber = scope.start.line + hint.line;
                    const range = new vscode.Range(
                        new vscode.Position(lineNumber, 0),
                        new vscode.Position(lineNumber, line.length)
                    );
                    editBuilder.replace(range, newLine);
                }
            }
        }
    }
}

interface ScopeInfo {
    type: 'function' | 'class';
    name: string;
    startLine: number;
    endLine: number;
    parameters?: string;
    returnType?: string;
    indentation: number;
}

interface TypeHint {
    variableName: string;
    suggestedType: string;
    context: 'parameter' | 'assignment' | 'return';
    line: number;
}