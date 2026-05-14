#!/bin/sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOK_DIR="$(git rev-parse --show-toplevel)/.git/hooks"
for hook in post-commit post-checkout; do
    cp "$SCRIPT_DIR/$hook" "$HOOK_DIR/$hook"
    chmod +x "$HOOK_DIR/$hook"
    echo "Installed $hook"
done
