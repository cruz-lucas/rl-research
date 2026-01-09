#!/usr/bin/env bash

set -euo pipefail

# Sync mlruns from Compute Canada to local without deleting local files.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_LOCAL_DIR="$ROOT_DIR/mlruns"
DEFAULT_REMOTE_DIR="~/mlruns"

usage() {
  cat <<'EOF'
Usage:
  mlruns_sync_cc.sh <user@host> [REMOTE_DIR] [LOCAL_DIR]

Examples:
  mlruns_sync_cc.sh abc123@vulcan.alliancecan.ca ~/projects/mlruns

Defaults:
  REMOTE_DIR: ~/mlruns
  LOCAL_DIR:  <repo-root>/mlruns

Notes:
  - One-way pull only; remote deletions are NOT propagated locally.
  - You will be prompted for your Compute Canada password and 2FA code.
EOF
}

REMOTE_HOST="${1:-${CC_HOST:-}}"
REMOTE_DIR="${2:-${CC_MLRUNS_DIR:-$DEFAULT_REMOTE_DIR}}"
LOCAL_DIR="${3:-${LOCAL_MLRUNS_DIR:-$DEFAULT_LOCAL_DIR}}"

if [[ -z "$REMOTE_HOST" ]]; then
  echo "Missing remote host (e.g., user@vulcan.alliancecan.ca)" >&2
  usage
  exit 1
fi

mkdir -p "$LOCAL_DIR"

RSYNC_OPTS=(
  -avh
  --progress
  --partial
  --no-perms
  --no-owner
  --no-group
)

# rsync will prompt for password + OTP via SSH.
SSH_CMD=("ssh" "-o" "StrictHostKeyChecking=accept-new")

echo "Syncing mlruns from $REMOTE_HOST:${REMOTE_DIR%/}/ -> $LOCAL_DIR/"
rsync "${RSYNC_OPTS[@]}" -e "${SSH_CMD[*]}" "$REMOTE_HOST:${REMOTE_DIR%/}/" "$LOCAL_DIR/"
