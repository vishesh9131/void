import * as vscode from 'vscode';
import * as path from 'path';

export class TestGenerator {
    async generate() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const document = editor.document;
        const selection = editor.selection;

        // Find the function or class to test
        const codeToTest = this.findTestableCode(document, selection);
        
        if (codeToTest) {
            const testCode = this.generateTestCode(codeToTest);
            await this.createTestFile(document, testCode, codeToTest.name);
        } else {
            vscode.window.showErrorMessage('Please select a function or class to generate tests for');
        }
    }

    private findTestableCode(document: vscode.TextDocument, selection: vscode.Selection): TestableCode | null {
        const position = selection.isEmpty ? selection.active : selection.start;
        
        // Look for function definition
        for (let line = position.line; line >= 0; line--) {
            const lineText = document.lineAt(line).text;
            
            // Function pattern
            const funcMatch = /^(\s*)def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.+?))?:/g.exec(lineText);
            if (funcMatch) {
                const indentation = funcMatch[1];
                const funcName = funcMatch[2];
                const params = funcMatch[3];
                const returnType = funcMatch[4];
                
                const endLine = this.findScopeEnd(document, line, indentation.length);
                const body = this.getFunctionBody(document, line, endLine);
                
                return {
                    type: 'function',
                    name: funcName,
                    parameters: params,
                    returnType,
                    body,
                    startLine: line,
                    endLine,
                    isMLFunction: this.isMLFunction(body)
                };
            }
            
            // Class pattern
            const classMatch = /^(\s*)class\s+(\w+)(?:\(.*?\))?:/g.exec(lineText);
            if (classMatch) {
                const indentation = classMatch[1];
                const className = classMatch[2];
                
                const endLine = this.findScopeEnd(document, line, indentation.length);
                const body = this.getFunctionBody(document, line, endLine);
                
                return {
                    type: 'class',
                    name: className,
                    body,
                    startLine: line,
                    endLine,
                    isMLFunction: this.isMLFunction(body)
                };
            }
        }
        
        return null;
    }

    private findScopeEnd(document: vscode.TextDocument, startLine: number, baseIndentation: number): number {
        for (let line = startLine + 1; line < document.lineCount; line++) {
            const lineText = document.lineAt(line).text;
            
            if (lineText.trim().length === 0) {
                continue;
            }
            
            const lineIndentation = lineText.length - lineText.trimStart().length;
            if (lineIndentation <= baseIndentation) {
                return line - 1;
            }
        }
        
        return document.lineCount - 1;
    }

    private getFunctionBody(document: vscode.TextDocument, startLine: number, endLine: number): string {
        let body = '';
        for (let line = startLine; line <= endLine; line++) {
            body += document.lineAt(line).text + '\n';
        }
        return body;
    }

    private isMLFunction(body: string): boolean {
        const mlPatterns = [
            /torch\./,
            /tensorflow|tf\./,
            /numpy|np\./,
            /sklearn/,
            /pandas|pd\./,
            /model\./,
            /\.fit\(/,
            /\.predict\(/,
            /\.forward\(/,
            /\.backward\(/,
            /\.train\(/,
            /\.eval\(/,
            /nn\./,
            /optim\./,
            /cuda\(/,
            /to\(device\)/
        ];
        
        return mlPatterns.some(pattern => pattern.test(body));
    }

    private generateTestCode(code: TestableCode): string {
        if (code.type === 'function') {
            return this.generateFunctionTest(code);
        } else {
            return this.generateClassTest(code);
        }
    }

    private generateFunctionTest(code: TestableCode): string {
        const testMethods = [];
        const imports = this.generateImports(code);
        const fixtures = this.generateFixtures(code);
        
        // Basic test
        testMethods.push(this.generateBasicTest(code));
        
        // Edge case tests
        if (code.isMLFunction) {
            testMethods.push(this.generateMLEdgeCaseTests(code));
        }
        
        // Property-based tests
        testMethods.push(this.generatePropertyTests(code));
        
        return `${imports}

${fixtures}

class Test${this.capitalize(code.name)}:
${testMethods.join('\n\n')}
`;
    }

    private generateClassTest(code: TestableCode): string {
        const imports = this.generateImports(code);
        const fixtures = this.generateFixtures(code);
        
        return `${imports}

${fixtures}

class Test${code.name}:
    def test_initialization(self):
        """Test that ${code.name} can be initialized correctly."""
        # TODO: Add initialization parameters
        instance = ${code.name}()
        assert instance is not None
    
    def test_basic_functionality(self):
        """Test basic functionality of ${code.name}."""
        instance = ${code.name}()
        # TODO: Add basic functionality tests
        pass
    
    ${code.isMLFunction ? this.generateMLClassTests(code) : ''}
`;
    }

    private generateImports(code: TestableCode): string {
        const baseImports = [
            'import pytest',
            'import numpy as np'
        ];
        
        if (code.isMLFunction) {
            if (code.body.includes('torch')) {
                baseImports.push('import torch');
                baseImports.push('import torch.nn as nn');
            }
            
            if (code.body.includes('tensorflow') || code.body.includes('tf.')) {
                baseImports.push('import tensorflow as tf');
            }
            
            if (code.body.includes('sklearn')) {
                baseImports.push('from sklearn.metrics import accuracy_score, mean_squared_error');
            }
            
            if (code.body.includes('pandas') || code.body.includes('pd.')) {
                baseImports.push('import pandas as pd');
            }
        }
        
        // Add import for the function/class being tested
        baseImports.push(`from your_module import ${code.name}  # TODO: Update import path`);
        
        return baseImports.join('\n');
    }

    private generateFixtures(code: TestableCode): string {
        if (!code.isMLFunction) {
            return '';
        }
        
        const fixtures = [];
        
        // Common ML fixtures
        fixtures.push(`
@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y`);
        
        if (code.body.includes('torch')) {
            fixtures.push(`
@pytest.fixture
def sample_tensor():
    """Generate sample PyTorch tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(10, 5)`);
            
            fixtures.push(`
@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')`);
        }
        
        if (code.body.includes('tensorflow') || code.body.includes('tf.')) {
            fixtures.push(`
@pytest.fixture
def sample_tf_data():
    """Generate sample TensorFlow data for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((100, 10))`);
        }
        
        return fixtures.join('\n');
    }

    private generateBasicTest(code: TestableCode): string {
        const params = this.parseParameters(code.parameters || '');
        const testParams = this.generateTestParameters(params);
        
        return `    def test_${code.name}_basic(self):
        """Test basic functionality of ${code.name}."""
        # Arrange
        ${testParams.join('\n        ')}
        
        # Act
        result = ${code.name}(${params.map(p => p.name).join(', ')})
        
        # Assert
        assert result is not None
        ${this.generateBasicAssertions(code)}`;
    }

    private generateMLEdgeCaseTests(code: TestableCode): string {
        const tests = [];
        
        if (code.body.includes('torch') || code.body.includes('tensor')) {
            tests.push(`    def test_${code.name}_empty_tensor(self):
        """Test ${code.name} with empty tensor."""
        empty_tensor = torch.empty(0)
        # TODO: Add test logic for empty tensor
        pass`);
            
            tests.push(`    def test_${code.name}_gpu_compatibility(self, device):
        """Test ${code.name} GPU compatibility."""
        if device.type == 'cuda':
            # TODO: Add GPU-specific tests
            pass`);
        }
        
        if (code.body.includes('numpy') || code.body.includes('np.')) {
            tests.push(`    def test_${code.name}_nan_values(self):
        """Test ${code.name} with NaN values."""
        data_with_nan = np.array([1, 2, np.nan, 4])
        # TODO: Add NaN handling test
        pass`);
        }
        
        tests.push(`    def test_${code.name}_large_input(self, sample_data):
        """Test ${code.name} with large input."""
        X, y = sample_data
        # Scale up the data
        large_X = np.repeat(X, 100, axis=0)
        # TODO: Add large input test
        pass`);
        
        return tests.join('\n\n');
    }

    private generatePropertyTests(code: TestableCode): string {
        return `    def test_${code.name}_deterministic(self):
        """Test that ${code.name} produces deterministic results."""
        # TODO: Set seeds and test deterministic behavior
        pass
    
    def test_${code.name}_input_validation(self):
        """Test input validation for ${code.name}."""
        # TODO: Test with invalid inputs
        with pytest.raises((ValueError, TypeError)):
            ${code.name}(None)`;
    }

    private generateMLClassTests(code: TestableCode): string {
        return `
    def test_forward_pass(self):
        """Test forward pass of the model."""
        model = ${code.name}()
        # TODO: Add forward pass test
        pass
    
    def test_backward_pass(self):
        """Test backward pass of the model."""
        model = ${code.name}()
        # TODO: Add backward pass test
        pass
    
    def test_model_parameters(self):
        """Test model parameters."""
        model = ${code.name}()
        # TODO: Test parameter count, shapes, etc.
        pass`;
    }

    private parseParameters(parameters: string): Parameter[] {
        if (!parameters.trim()) {
            return [];
        }
        
        return parameters.split(',').map(param => {
            const [name, defaultValue] = param.trim().split('=');
            const [varName, typeHint] = name.split(':');
            
            return {
                name: varName.trim(),
                type: typeHint?.trim(),
                defaultValue: defaultValue?.trim()
            };
        });
    }

    private generateTestParameters(params: Parameter[]): string[] {
        return params.map(param => {
            if (param.defaultValue) {
                return `${param.name} = ${param.defaultValue}  # Using default value`;
            } else {
                return `${param.name} = ${this.generateTestValue(param)}  # TODO: Provide appropriate test value`;
            }
        });
    }

    private generateTestValue(param: Parameter): string {
        if (param.type) {
            switch (param.type) {
                case 'int': return '10';
                case 'float': return '0.5';
                case 'str': return '"test_string"';
                case 'bool': return 'True';
                case 'torch.Tensor': return 'torch.randn(3, 3)';
                case 'np.ndarray': return 'np.array([1, 2, 3])';
                default: return 'None';
            }
        }
        
        // Guess based on name
        if (param.name.includes('size') || param.name.includes('dim')) {
            return '10';
        }
        if (param.name.includes('rate') || param.name.includes('prob')) {
            return '0.1';
        }
        if (param.name.includes('tensor') || param.name.includes('data')) {
            return 'torch.randn(3, 3)';
        }
        
        return 'None';
    }

    private generateBasicAssertions(code: TestableCode): string {
        if (code.returnType) {
            switch (code.returnType) {
                case 'torch.Tensor':
                    return 'assert isinstance(result, torch.Tensor)\n        assert result.shape is not None';
                case 'np.ndarray':
                    return 'assert isinstance(result, np.ndarray)\n        assert result.shape is not None';
                case 'int':
                    return 'assert isinstance(result, int)';
                case 'float':
                    return 'assert isinstance(result, float)';
                default:
                    return '# TODO: Add specific assertions based on return type';
            }
        }
        
        return '# TODO: Add appropriate assertions';
    }

    private async createTestFile(document: vscode.TextDocument, testCode: string, functionName: string) {
        const currentDir = path.dirname(document.uri.fsPath);
        const testFileName = `test_${path.basename(document.fileName, '.py')}.py`;
        const testFilePath = path.join(currentDir, testFileName);
        
        const testUri = vscode.Uri.file(testFilePath);
        
        try {
            // Check if test file already exists
            await vscode.workspace.fs.stat(testUri);
            
            // File exists, ask user what to do
            const choice = await vscode.window.showQuickPick(
                ['Append to existing file', 'Overwrite file', 'Cancel'],
                { placeHolder: 'Test file already exists. What would you like to do?' }
            );
            
            if (choice === 'Cancel') {
                return;
            }
            
            if (choice === 'Append to existing file') {
                const existingContent = Buffer.from(await vscode.workspace.fs.readFile(testUri)).toString();
                testCode = existingContent + '\n\n' + testCode;
            }
        } catch {
            // File doesn't exist, create new one
        }
        
        await vscode.workspace.fs.writeFile(testUri, Buffer.from(testCode, 'utf-8'));
        
        // Open the test file
        const testDocument = await vscode.workspace.openTextDocument(testUri);
        await vscode.window.showTextDocument(testDocument, vscode.ViewColumn.Beside);
        
        vscode.window.showInformationMessage(`🧪 Test file created: ${testFileName}`);
    }

    private capitalize(str: string): string {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

interface TestableCode {
    type: 'function' | 'class';
    name: string;
    parameters?: string;
    returnType?: string;
    body: string;
    startLine: number;
    endLine: number;
    isMLFunction: boolean;
}

interface Parameter {
    name: string;
    type?: string;
    defaultValue?: string;
}