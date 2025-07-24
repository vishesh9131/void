#!/usr/bin/env node

/**
 * ML Engineer Tools Extension Test Runner
 * 
 * This script helps automate the testing process for the ML Engineer Tools extension.
 * Run with: node test-runner.js
 */

const fs = require('fs');
const path = require('path');

console.log('🧪 ML Engineer Tools - Test Runner');
console.log('==================================\n');

// Check if test files exist
const testFiles = [
    'test-samples/test_model.py',
    'test-samples/colab_code.py',
    'TESTING.md'
];

console.log('📁 Checking test files...');
testFiles.forEach(file => {
    const filePath = path.join(__dirname, file);
    if (fs.existsSync(filePath)) {
        console.log(`✅ ${file} - Found`);
    } else {
        console.log(`❌ ${file} - Missing`);
    }
});

console.log('\n📋 Test Checklist:');
console.log('==================');

const testChecklist = [
    'Shape Inspector - Hover over tensor variables',
    'GPU Toggle - Click ⚡ icon or Ctrl+Shift+G',
    'Import Cleaner - Save file to trigger auto-cleaning',
    'Smart Paste - Copy from colab_code.py and paste',
    'Seed Synchronizer - Press Ctrl+Shift+S',
    'Loss Plotter - Click 📈 icon in editor',
    'NaN Detector - Look for wavy underlines',
    'Memory Monitor - Check 🧠 annotations',
    'Tensor Selector - Double-click on layers',
    'Hyperparameter Tweaker - Alt+click on numbers',
    'Code Colorizer - Verify syntax highlighting'
];

testChecklist.forEach((test, index) => {
    console.log(`${index + 1}. [ ] ${test}`);
});

console.log('\n🚀 Quick Start Instructions:');
console.log('============================');
console.log('1. Press F5 in VS Code to launch Extension Development Host');
console.log('2. Open test-samples/test_model.py in the new window');
console.log('3. Follow the testing guide in TESTING.md');
console.log('4. Test each feature systematically');

console.log('\n⚙️ Configuration Test:');
console.log('======================');
console.log('Add this to your VS Code settings.json to test configurations:');
console.log(`
{
  "mlTools.enableShapeInspection": true,
  "mlTools.enableGPUToggle": true,
  "mlTools.enableAutoImportClean": true,
  "mlTools.enableNaNDetection": true,
  "mlTools.enableMemoryMonitoring": true,
  "mlTools.defaultSeed": 42,
  "mlTools.enableSmartPaste": true,
  "mlTools.enableShapeMatching": true
}
`);

console.log('\n🐛 Common Issues:');
console.log('=================');
console.log('- If features don\'t work, ensure file is saved as .py');
console.log('- Check that extension is activated in Extensions panel');
console.log('- Reload VS Code window if needed (Ctrl+Shift+P → "Developer: Reload Window")');
console.log('- Verify configuration settings are correct');

console.log('\n📊 Performance Tests:');
console.log('=====================');
console.log('- Test with large files (1000+ lines)');
console.log('- Monitor VS Code memory usage');
console.log('- Check responsiveness of decorations');
console.log('- Verify webview panels don\'t leak memory');

console.log('\n✅ Success Criteria:');
console.log('====================');
console.log('- All hover tooltips show tensor information');
console.log('- GPU toggle successfully swaps device calls');
console.log('- Import cleaning removes unused imports only');
console.log('- Smart paste sanitizes Colab code and adds imports');
console.log('- Seed sync updates all random seed calls');
console.log('- Loss plotter shows interactive charts');
console.log('- NaN detector highlights risky operations');
console.log('- Memory monitor shows realistic estimates');
console.log('- All keyboard shortcuts work');
console.log('- Configuration options take effect');

console.log('\n🎉 Happy Testing!');
console.log('For detailed instructions, see TESTING.md');
console.log('Report issues and suggestions in the repository.\n');