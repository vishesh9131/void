#!/usr/bin/env bash

set -e

ROOT=$(dirname $(dirname $(realpath $0)))
echo "ROOT: $ROOT"

# Launch VS Aware with flags to disable sqlite3 dependent features
if [[ "$OSTYPE" == "darwin"* ]]; then
	CODE="./.build/electron/VS Aware.app/Contents/MacOS/Electron"
else
	CODE="./.build/electron/vscode"
fi

# Disable storage and other sqlite3 dependent features
exec "$CODE" . \
	--extensionDevelopmentPath="$ROOT/extensions/ml-engineer-tools" \
	--disable-extension=vscode.vscode-api-tests \
	--disable-dev-shm-usage \
	--disable-gpu-sandbox \
	--no-sandbox \
	--disable-features=VizDisplayCompositor \
	--disable-storage \
	--disable-workspace-storage \
	--disable-user-env-probe \
	--skip-getting-started \
	"$@"
