# How to Build and Run Void IDE Locally

Void IDE is a VS Code fork that includes AI-powered development features. This guide will walk you through setting up the development environment and building the project from source.

## 🚀 Quick Start

For the impatient, run these commands:

```bash
# Install NVM and Node.js 20.18.2
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install 20.18.2 && nvm use 20.18.2

# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential g++ libx11-dev libxkbfile-dev libsecret-1-dev libkrb5-dev python-is-python3 libgssapi-krb5-2

# Clone and build (replace with actual repository URL)
git clone https://github.com/your-username/void-ide.git
cd void-ide
npm install
cd src/vs/workbench/contrib/void/browser/react && node build.js && cd -
npm run compile
./scripts/code.sh
```

## 📋 Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu/Debian recommended), macOS, or Windows with WSL2
- **Node.js**: Version 20.18.2 (exact version required)
- **RAM**: At least 8GB (16GB recommended for smooth development)
- **Storage**: At least 10GB free space

### Required System Dependencies

The project requires several native dependencies for building:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  g++ \
  libx11-dev \
  libxkbfile-dev \
  libsecret-1-dev \
  libkrb5-dev \
  python-is-python3 \
  libgssapi-krb5-2
```

**macOS:**
```bash
# Ensure Xcode Command Line Tools are installed
xcode-select --install
```

**Windows (WSL2):**
Follow the Ubuntu/Debian instructions inside your WSL2 environment.

## 🔧 Setup Instructions

### 1. Install Node.js 20.18.2

Void IDE requires the exact Node.js version specified in `.nvmrc`. Using NVM is the recommended approach:

```bash
# Install NVM (if not already installed)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Reload your shell or run:
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install and use the required Node.js version
nvm install 20.18.2
nvm use 20.18.2

# Verify the version
node --version  # Should output v20.18.2
```

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/void-ide.git
cd void-ide
```

**Note**: Replace `your-username/void-ide` with the actual repository URL.

### 3. Install Dependencies

```bash
npm install
```

**Note**: If you encounter errors during `npm install`, make sure:
- You're using Node.js 20.18.2
- All system dependencies are installed
- You have sufficient RAM and storage space

### 4. Build React Components

Void IDE includes custom React components that need to be built before the main compilation:

```bash
cd src/vs/workbench/contrib/void/browser/react
node build.js
cd - # Return to project root
```

This step is **crucial** and must be completed before running the main compilation.

### 5. Compile the Project

```bash
npm run compile
```

This process takes several minutes and compiles:
- TypeScript source code
- Extensions
- Monaco Editor
- All Void-specific features

**Troubleshooting Compilation Issues:**
- If you see module resolution errors for React components, ensure step 4 was completed
- For memory issues, close other applications or increase available RAM
- For permission errors, ensure you have write access to the project directory

## 🏃‍♂️ Running Void IDE

### Development Mode

To run Void IDE in development mode:

```bash
./scripts/code.sh
```

**Additional options:**
```bash
# Run with specific flags
./scripts/code.sh --disable-gpu  # Useful for VMs or containers

# Open a specific folder
./scripts/code.sh /path/to/your/project

# Run with verbose logging
./scripts/code.sh --verbose
```

### Watch Mode (Live Development)

For active development with automatic recompilation:

```bash
npm run watch
```

This will:
- Watch for TypeScript changes and recompile automatically
- Monitor extension changes
- Provide faster iteration during development

**Note**: You'll need to restart the IDE to see changes take effect.

### Web Version

To run Void IDE in a web browser:

```bash
./scripts/code-web.sh
```

This starts a development server, accessible at `http://localhost:8000` by default.

**Custom port:**
```bash
./scripts/code-web.sh --port 3000
```

## 🧪 Testing the Build

After successful compilation, test that everything works:

1. **Launch the IDE**: `./scripts/code.sh`
2. **Check Void features**: Look for Void-specific UI elements and AI features
3. **Open a project**: Test basic functionality like file editing and search
4. **Verify extensions**: Ensure built-in extensions are working

## 🔍 Troubleshooting

### Common Issues

**Node.js Version Mismatch:**
```bash
# Error: "Unsupported engine"
nvm use 20.18.2
```

**Missing React Components:**
```bash
# Error: "Cannot find module './react/out/..."
cd src/vs/workbench/contrib/void/browser/react
node build.js
cd -
```

**Build Failures:**
```bash
# Clean and rebuild
rm -rf node_modules package-lock.json
npm install
npm run compile
```

**CLI Build Errors (Optional):**
The CLI component requires Rust and can be skipped for basic IDE functionality:
```bash
# CLI compilation error is normal if Rust is not installed
# The IDE works without CLI compilation
```

**Permission Issues on Linux:**
```bash
# If you get EACCES errors
sudo chown -R $(whoami) ~/.npm
```

### Performance Optimization

For better build performance:
- Use an SSD for the project directory
- Increase Node.js memory limit: `export NODE_OPTIONS="--max-old-space-size=8192"`
- Close unnecessary applications during compilation

### Development Tips

- Use `npm run watch` during active development
- The React components rarely change, so you typically only need to build them once
- If making changes to React components, rebuild them with `node build.js` in the react directory
- Use `./scripts/code.sh --disable-gpu` if running in a virtual machine

### Environment-Specific Notes

**Docker/Containers:**
```bash
./scripts/code.sh --disable-dev-shm-usage --disable-gpu
```

**WSL2:**
```bash
./scripts/code.sh --disable-gpu
```

**Headless Servers:**
- Use the web version: `./scripts/code-web.sh`
- The desktop version requires a display server

**CI/CD Environments:**
- Compilation works fine
- Running requires display server for desktop version
- Web version can be used for automated testing

## 📁 Project Structure

Key directories:
- `src/`: Main source code
- `src/vs/workbench/contrib/void/`: Void-specific features
- `src/vs/workbench/contrib/void/browser/react/`: React components
- `extensions/`: Built-in extensions
- `scripts/`: Launch scripts
- `build/`: Build configuration

## 🎯 Next Steps

After successfully building and running Void IDE:

1. Explore the Void-specific features
2. Check the contributing guidelines in `HOW_TO_CONTRIBUTE.md`
3. Set up your development environment for making changes
4. Review the codebase structure and architecture

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. Check the existing issues in the repository
2. Verify you've followed all steps exactly
3. Include your system information, Node.js version, and full error messages when reporting issues
4. Consider running in a clean environment (new terminal session with correct Node.js version)

---

**Happy coding with Void IDE! 🎉**