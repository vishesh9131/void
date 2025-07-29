# How to Build and Run Void IDE Locally

This guide provides step-by-step instructions for building and running the Void IDE on your local machine.

## Quick Start

If you're in a hurry, here's the minimal setup for Linux/macOS:

```bash
# Install nvm and use Node v20.18.2
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20.18.2
nvm use 20.18.2

# Clone and build
git clone https://github.com/voideditor/void.git
cd void
npm install
npm run buildreact
npm run compile
npm run download-builtin-extensions

# Run the IDE
./scripts/code.sh
```

For detailed instructions and troubleshooting, continue reading below.

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Node.js**: v20.18.2 (exact version required)
- **RAM**: Minimum 8GB recommended
- **Disk Space**: At least 2GB free space

### Required System Dependencies

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  g++ \
  make \
  python3 \
  python-is-python3 \
  libx11-dev \
  libxkbfile-dev \
  libsecret-1-dev \
  libkrb5-dev \
  libnss3 \
  libgtk-3-0 \
  libgbm1
```

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install dependencies via Homebrew
brew install python@3
```

#### Windows
- Install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) with:
  - Desktop development with C++
  - MSVC v143 - VS 2022 C++ x64/x86 build tools
  - Windows 10/11 SDK
- Install [Python 3.x](https://www.python.org/downloads/)

## Build Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/voideditor/void.git
   cd void
   ```

2. **Install Node.js v20.18.2**

   Using nvm (recommended):
   ```bash
   # Install nvm if you don't have it
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

   # Reload your shell configuration
   source ~/.bashrc  # or ~/.zshrc for zsh

   # Install and use the correct Node version
   nvm install 20.18.2
   nvm use 20.18.2
   ```

3. **Install dependencies**
   ```bash
   # Clean install to avoid conflicts
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **Build React components**
   ```bash
   npm run buildreact
   ```
   This step is crucial as it builds the React components that Void IDE depends on.

5. **Compile the project**
   ```bash
   npm run compile
   ```

6. **Download VS Code dependencies**
   ```bash
   npm run download-builtin-extensions
   ```

## Running Void IDE

### Running in Development Mode

There are several ways to run Void IDE:

1. **Desktop Application (Electron)**
   ```bash
   ./scripts/code.sh
   ```

2. **Web Version**
   ```bash
   ./scripts/code-web.sh
   ```
   Then open http://localhost:8080 in your browser

3. **Server Mode**
   ```bash
   ./scripts/code-server.sh
   ```

### Running with Watch Mode

For development, you can run the IDE with automatic recompilation:

```bash
# Start the watch mode
npm run watch

# In another terminal, run the IDE
./scripts/code.sh
```

## Building the Project

### Development Build
```bash
# Build the extension and web worker
npm run build:extension
npm run build:web

# Or build everything at once
npm run build:all
```

### Production Build
```bash
npm run build:all:prod
```

## Running the IDE

### Running in VS Code (Development)

1. Open the project in VS Code:
   ```bash
   code .
   ```

2. Press `F5` or go to Run → Start Debugging

3. A new VS Code window will open with the Void extension loaded

### Running Tests
```bash
# Run all tests
npm test

# Run specific test suites
npm run test:extension
npm run test:web
```

## Common Issues and Solutions

### Issue: Kerberos package build fails
**Error:** `node-gyp rebuild` fails for the `kerberos` package

**Solution:**
1. Ensure you're using Node.js v20.18.2 (not v22.x or other versions)
   ```bash
   node --version  # Should output: v20.18.2
   ```
2. Install required system dependencies:
   ```bash
   # Linux
   sudo apt-get install -y build-essential g++ libkrb5-dev

   # macOS
   brew install krb5
   ```

### Issue: Missing React components
**Error:** `Cannot find module '../react/out/*/index.js'`

**Solution:**
Build the React components before compiling:
```bash
npm run buildreact
npm run compile
```

### Issue: Port already in use
**Error:** When running web version, port 8080 is already in use

**Solution:**
```bash
# Kill the process using port 8080
lsof -ti:8080 | xargs kill -9

# Or run on a different port
PORT=3000 ./scripts/code-web.sh
```

### Issue: Compilation errors
**Solution:**
1. Clean build:
   ```bash
   rm -rf out/
   npm run compile
   ```
2. If errors persist, try a complete reinstall:
   ```bash
   rm -rf node_modules package-lock.json out/
   nvm use 20.18.2
   npm install
   npm run buildreact
   npm run compile
   ```

## Development Workflow

### 1. Making Changes
- The main source code is in the `src/` directory
- Extensions are in the `extensions/` directory
- Follow the existing code style and conventions

### 2. Running Tests
```bash
npm test
```

### 3. Debugging
- Use VS Code's built-in debugger
- Launch configurations are in `.vscode/launch.json`

## Complete Build Example

Here's what a successful build looks like:

```bash
$ node --version
v20.18.2

$ npm install
# ... npm will install all dependencies ...
added 2543 packages in 45s

$ npm run buildreact
> code-oss-dev@1.99.3 buildreact
> cd ./src/vs/workbench/contrib/void/browser/react/ && node build.js && cd ../../../../../../../

📦 Building...
✅ Successfully prefixified classNames
✅ Successfully prefixified css file
✅✅✅ All done! 🎉
ESM ⚡️ Build success in 3491ms
✅ Build complete!

$ NODE_OPTIONS="--max-old-space-size=8192" npm run compile
> code-oss-dev@1.99.3 compile
> node ./node_modules/gulp/bin/gulp.js compile

[10:15:37] Finished 'compile' after 3.25 min

$ npm run download-builtin-extensions
[10:46:56] Synchronizing built-in extensions...

$ ./scripts/code.sh
# Void IDE will launch!
```

## Project Structure
```
void/
├── src/
│   ├── extension/      # VS Code extension code
│   ├── vs/            # Web UI components
│   └── common/        # Shared utilities
├── out/               # Compiled output
├── assets/            # Static assets
├── scripts/           # Build scripts
└── test/             # Test files
```

## Additional Resources

- [Contributing Guide](./HOW_TO_CONTRIBUTE.md)
- [Architecture Overview](./README.md#architecture)
- [VS Code Extension API](https://code.visualstudio.com/api)

## Getting Help

If you encounter issues not covered in this guide:

1. Check existing [GitHub Issues](https://github.com/voideditor/void/issues)
2. Join the discussion on [Discord](https://discord.gg/RSvgxvwH7e)
3. Create a new issue with:
   - Your OS and version
   - Node.js version (`node --version`)
   - Error messages and logs
   - Steps to reproduce

## Tips for Successful Build

1. **Always use Node.js v20.18.2** - The project is sensitive to Node version
2. **Install system dependencies first** - Many build errors are due to missing system libraries
3. **Use clean installs** - When in doubt, delete `node_modules` and reinstall
4. **Check the logs** - Build errors usually have helpful messages
5. **Disable antivirus temporarily** - Some AV software can interfere with the build process

Happy coding with Void IDE! 🚀
