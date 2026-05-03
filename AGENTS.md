# Agent Instructions

## Cursor Cloud specific instructions

### Node version

This project requires Node.js v20.18.2 exactly. The version is pinned in `.nvmrc`.
nvm is installed at `$HOME/.nvm`. Source it before running any node commands:

```
export NVM_DIR="$HOME/.nvm" && source "$NVM_DIR/nvm.sh" && nvm use 20.18.2
```

### Build pipeline

The build has three stages that must run in order:

1. `npm run buildreact` -- builds the React UI components under `src/vs/workbench/contrib/void/browser/react/`
2. `npm run compile` -- runs the gulp-based TypeScript compilation (takes ~2-3 min, uses `--max-old-space-size=8192` internally)
3. The web dev server or Electron app can then be launched

If you only change React code, you only need to re-run step 1. For other TS changes, step 2 is needed.

### Running the app

- **Web version** (headless-friendly, no display needed): `node ./scripts/code-web.js --host 0.0.0.0 --port 8080 --browserType none`
- **Electron desktop** (needs X11/Xvfb): `./scripts/code.sh` -- the script auto-detects docker and adds `--disable-dev-shm-usage`
- **Watch mode** for incremental dev: `npm run watch` (runs both client and extensions watchers)

### Testing

- Unit tests: `npm run test-node` -- runs mocha-based Node tests (~4500 tests, ~11s)
- Browser tests: `npm run test-browser` -- needs playwright installed first
- See `scripts/test.sh` and `scripts/test-integration.sh` for integration tests

### Linting

- `npm run eslint` -- note: as of the current codebase, this emits a ts-node compilation error related to `moduleResolution` settings but still exits 0. This is a pre-existing issue.
- `npm run hygiene` -- runs the gulp hygiene task

### System deps (Linux)

The native modules (node-pty, kerberos, native-keymap) need these packages:
`build-essential g++ libx11-dev libxkbfile-dev libsecret-1-dev libkrb5-dev python-is-python3`

For running the Electron app: `libnss3 libgtk-3-0 libgbm1 libxss1 libasound2t64`

For headless Electron: install `xvfb` and run via `xvfb-run`.

### Gotchas

- npm is the only supported package manager. yarn is rejected by `build/npm/preinstall.js`.
- The `postinstall` script downloads the Electron binary; this can be slow on first run.
- The chrome-sandbox may need permissions fix: `sudo chown root:root .build/electron/chrome-sandbox && sudo chmod 4755 .build/electron/chrome-sandbox`
- If you see "Cannot find module '../react/out/..." errors, run `npm run buildreact` before `npm run compile`.
