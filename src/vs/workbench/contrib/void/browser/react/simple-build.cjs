#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Create output directory if it doesn't exist
const outDir = path.join(__dirname, 'out');
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

// List of component directories that need output files
const componentDirs = [
  'diff',
  'void-settings-tsx', 
  'void-editor-widgets-tsx',
  'void-onboarding',
  'void-tooltip',
  'sidebar-tsx',
  'quick-edit-tsx'
];

// Create placeholder index.js files for each component
componentDirs.forEach(componentDir => {
  const componentOutDir = path.join(outDir, componentDir);
  if (!fs.existsSync(componentOutDir)) {
    fs.mkdirSync(componentOutDir, { recursive: true });
  }
  
  const indexFile = path.join(componentOutDir, 'index.js');
  const placeholderContent = `// Placeholder module for ${componentDir}
export default function ${componentDir.replace(/-/g, '_')}Component() {
  return null;
}

export const render = () => null;
export const mount = () => null;
export const unmount = () => null;
`;
  
  fs.writeFileSync(indexFile, placeholderContent);
  console.log(`Created placeholder: ${indexFile}`);
});

console.log('✅ All placeholder React components created successfully!');