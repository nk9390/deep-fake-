#!/bin/bash
# Install gstack (https://github.com/garrytan/gstack) in Claude Code cloud
# sessions, whose containers start without ~/.claude/skills/gstack.
set -euo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

GSTACK_DIR="$HOME/.claude/skills/gstack"
if [ ! -d "$GSTACK_DIR/bin" ]; then
  git clone --single-branch --depth 1 https://github.com/garrytan/gstack.git "$GSTACK_DIR" >&2
  # The cloud proxy blocks Playwright's Chromium download; use the preinstalled one.
  (cd "$GSTACK_DIR" && GSTACK_SKIP_PLAYWRIGHT=1 ./setup --team </dev/null >&2) || true
fi

if [ -n "${CLAUDE_ENV_FILE:-}" ] && [ -x /opt/pw-browsers/chromium-1194/chrome-linux/chrome ]; then
  echo "export GSTACK_CHROMIUM_PATH=/opt/pw-browsers/chromium-1194/chrome-linux/chrome" >> "$CLAUDE_ENV_FILE"
fi
