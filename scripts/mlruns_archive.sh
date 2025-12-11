#!/usr/bin/env bash
# Compress or decompress the local mlruns directory.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ARCHIVE="$ROOT_DIR/mlruns.tar.gz"
DEFAULT_DIR="$ROOT_DIR/mlruns"

usage() {
  cat <<'EOF'
Usage:
  mlruns_archive.sh compress [ARCHIVE_PATH] [SOURCE_DIR]
  mlruns_archive.sh decompress [ARCHIVE_PATH] [DEST_DIR]

Defaults:
  ARCHIVE_PATH: <repo-root>/mlruns.tar.gz
  SOURCE_DIR:   <repo-root>/mlruns
  DEST_DIR:     <repo-root>/mlruns
EOF
}

cmd="${1:-}"
case "$cmd" in
  compress)
    archive_path="${2:-$DEFAULT_ARCHIVE}"
    source_dir="${3:-$DEFAULT_DIR}"

    if [ ! -d "$source_dir" ]; then
      echo "Source directory not found: $source_dir" >&2
      exit 1
    fi

    mkdir -p "$(dirname "$archive_path")"
    tar -czf "$archive_path" -C "$source_dir" .
    echo "Compressed $source_dir -> $archive_path"
    ;;
  decompress)
    archive_path="${2:-$DEFAULT_ARCHIVE}"
    dest_dir="${3:-$DEFAULT_DIR}"

    if [ ! -f "$archive_path" ]; then
      echo "Archive not found: $archive_path" >&2
      exit 1
    fi

    if [ -e "$dest_dir" ]; then
      echo "Destination already exists: $dest_dir" >&2
      echo "Move/remove it or choose a different destination." >&2
      exit 1
    fi

    mkdir -p "$dest_dir"
    tar -xzf "$archive_path" -C "$dest_dir"
    echo "Extracted $archive_path -> $dest_dir"
    ;;
  *)
    usage
    exit 1
    ;;
esac
