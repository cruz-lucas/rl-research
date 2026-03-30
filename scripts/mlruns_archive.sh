#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_ARCHIVE_DIR="$ROOT_DIR/mlruns_archive"
DEFAULT_DIR="$HOME/mlruns"
ARCHIVE_VERSION="mlruns-sharded-v1"

usage() {
  cat <<'EOF'
Usage:
  mlruns_archive.sh compress [ARCHIVE_DIR] [SOURCE_DIR]
  mlruns_archive.sh decompress [ARCHIVE_DIR] [DEST_DIR]
  mlruns_archive.sh verify [ARCHIVE_DIR]
  mlruns_archive.sh backup
  mlruns_archive.sh restore

Defaults:
  ARCHIVE_DIR: <repo-root>/mlruns_archive
  SOURCE_DIR:  $HOME/mlruns
  DEST_DIR:    $HOME/mlruns_restored

Environment:
  MLRUNS_ARCHIVE_CODEC=zst|gz          Default: zst when zstd is installed, else gz
  MLRUNS_ARCHIVE_LEVEL=<int>           Default: 3
  MLRUNS_ARCHIVE_JOBS=<int>            Default: min(cpu_count, 8)
  MLRUNS_ARCHIVE_THREADS_PER_JOB=<int> Default: max(cpu_count / jobs, 1)
  MLRUNS_ARCHIVE_VERBOSE=1             Print one log line per shard

Notes:
  - Compression is resumable. Re-running `compress` skips completed shards.
  - Interrupted work only loses the shard currently being written.
  - The archive is stored as a directory of `.tar.zst` or `.tar.gz` shards,
    which is much safer for walltime-limited jobs than one giant tarball.
EOF
}

log() {
  printf '[mlruns_archive] %s\n' "$*"
}

verbose_log() {
  if [[ "${MLRUNS_ARCHIVE_VERBOSE:-0}" != "0" ]]; then
    log "$@"
  fi
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

cpu_count() {
  local count
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi

  count="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
  if [[ -n "$count" ]]; then
    printf '%s\n' "$count"
    return
  fi

  printf '1\n'
}

default_codec() {
  if command -v zstd >/dev/null 2>&1; then
    printf 'zst\n'
  else
    printf 'gz\n'
  fi
}

archive_ext() {
  case "$1" in
    zst) printf 'tar.zst\n' ;;
    gz) printf 'tar.gz\n' ;;
    *) die "Unsupported codec: $1" ;;
  esac
}

compress_pipe() {
  local codec="$1"
  local level="$2"
  local threads="$3"

  case "$codec" in
    zst)
      command -v zstd >/dev/null 2>&1 || die "zstd is required for codec=zst"
      zstd -q -"${level}" -T"${threads}" -c
      ;;
    gz)
      if command -v pigz >/dev/null 2>&1; then
        pigz -q -"${level}" -p "${threads}" -c
      else
        gzip -n -"${level}" -c
      fi
      ;;
    *)
      die "Unsupported codec: $codec"
      ;;
  esac
}

decompress_pipe() {
  local codec="$1"

  case "$codec" in
    zst)
      command -v zstd >/dev/null 2>&1 || die "zstd is required to decompress .zst shards"
      zstd -q -d -c
      ;;
    gz)
      if command -v pigz >/dev/null 2>&1; then
        pigz -q -d -c
      else
        gzip -d -c
      fi
      ;;
    *)
      die "Unsupported codec: $codec"
      ;;
  esac
}

normalize_dir() {
  local target="$1"
  mkdir -p "$target"
  (
    cd "$target"
    pwd
  )
}

load_archive_info() {
  local archive_dir="$1"
  local info_path="$archive_dir/ARCHIVE_INFO"

  if [[ ! -f "$info_path" ]]; then
    return 1
  fi

  ARCHIVE_INFO_VERSION=""
  ARCHIVE_INFO_CODEC=""
  ARCHIVE_INFO_SOURCE_BASENAME=""

  while IFS='=' read -r key value; do
    case "$key" in
      version) ARCHIVE_INFO_VERSION="$value" ;;
      codec) ARCHIVE_INFO_CODEC="$value" ;;
      source_basename) ARCHIVE_INFO_SOURCE_BASENAME="$value" ;;
    esac
  done <"$info_path"

  [[ "$ARCHIVE_INFO_VERSION" == "$ARCHIVE_VERSION" ]] || die "Unsupported archive version in $info_path"
  [[ -n "$ARCHIVE_INFO_CODEC" ]] || die "Missing codec in $info_path"
  return 0
}

write_archive_info() {
  local archive_dir="$1"
  local codec="$2"
  local source_basename="$3"
  local info_path="$archive_dir/ARCHIVE_INFO"
  local tmp_path="$info_path.tmp.$$"

  cat >"$tmp_path" <<EOF
version=$ARCHIVE_VERSION
codec=$codec
source_basename=$source_basename
EOF
  mv "$tmp_path" "$info_path"
}

compress_items() {
  local source_dir="$1"
  local archive_path="$2"
  local codec="$3"
  local level="$4"
  local threads="$5"
  shift 5

  if [[ -f "$archive_path" ]]; then
    verbose_log "Skipping existing shard: ${archive_path#$PWD/}"
    return 0
  fi

  mkdir -p "$(dirname "$archive_path")"

  local tmp_path="$archive_path.tmp.$$"
  rm -f "$tmp_path"

  if tar -cf - -C "$source_dir" -- "$@" | compress_pipe "$codec" "$level" "$threads" >"$tmp_path"; then
    mv "$tmp_path" "$archive_path"
    verbose_log "Wrote shard: ${archive_path#$PWD/}"
    return 0
  fi

  rm -f "$tmp_path"
  return 1
}

compress_shard_command() {
  local source_dir="$1"
  local codec="$2"
  local level="$3"
  local threads="$4"
  local archive_path="$5"
  local rel_path="$6"

  compress_items "$source_dir" "$archive_path" "$codec" "$level" "$threads" "$rel_path"
}

write_manifest_header() {
  local manifest_path="$1"
  local tmp_path="$manifest_path.tmp.$$"

  cat >"$tmp_path" <<'EOF'
kind	relative_path	archive_path
EOF
  mv "$tmp_path" "$manifest_path"
}

append_manifest_line() {
  local manifest_path="$1"
  local kind="$2"
  local rel_path="$3"
  local archive_rel_path="$4"
  printf '%s\t%s\t%s\n' "$kind" "$rel_path" "$archive_rel_path" >>"$manifest_path"
}

collect_dir_entries() {
  local dir="$1"
  local prefix="$2"
  local child

  DIRECT_DIRS=()
  DIRECT_FILES=()

  shopt -s nullglob dotglob
  for child in "$dir"/*; do
    local rel_path="${child#$prefix/}"
    if [[ -d "$child" ]]; then
      DIRECT_DIRS+=("$rel_path")
    else
      DIRECT_FILES+=("$rel_path")
    fi
  done
  shopt -u nullglob dotglob
}

build_compress_job_file() {
  local archive_dir="$1"
  local source_dir="$2"
  local codec="$3"
  local level="$4"
  local threads_per_job="$5"
  local manifest_path="$6"
  local job_file="$7"
  local ext
  local root_archive
  local root_dirs=()
  local root_files=()
  local top_path
  local top_rel
  local top_archive_dir
  local top_dirs=()
  local top_files=()
  local meta_archive
  local child_rel
  local child_name
  local archive_path

  ext="$(archive_ext "$codec")"
  : >"$job_file"
  write_manifest_header "$manifest_path"

  collect_dir_entries "$source_dir" "$source_dir"
  set +u
  root_dirs=("${DIRECT_DIRS[@]}")
  root_files=("${DIRECT_FILES[@]}")
  set -u

  if [[ "${#root_files[@]}" -gt 0 ]]; then
    root_archive="$archive_dir/root_files.$ext"
    compress_items "$source_dir" "$root_archive" "$codec" "$level" "$threads_per_job" "${root_files[@]}"
    append_manifest_line "$manifest_path" "root_files" "." "${root_archive#$archive_dir/}"
  fi

  local top_index
  for ((top_index = 0; top_index < ${#root_dirs[@]}; top_index++)); do
    top_rel="${root_dirs[$top_index]}"
    top_path="$source_dir/$top_rel"
    top_archive_dir="$archive_dir/shards/$top_rel"

    collect_dir_entries "$top_path" "$source_dir"
    set +u
    top_dirs=("${DIRECT_DIRS[@]}")
    top_files=("${DIRECT_FILES[@]}")
    set -u

    if [[ "${#top_dirs[@]}" -eq 0 ]]; then
      archive_path="$top_archive_dir/__all__.$ext"
      printf '%s\0%s\0' "$archive_path" "$top_rel" >>"$job_file"
      append_manifest_line "$manifest_path" "tree" "$top_rel" "${archive_path#$archive_dir/}"
      continue
    fi

    if [[ "${#top_files[@]}" -gt 0 ]]; then
      meta_archive="$top_archive_dir/__meta__.$ext"
      compress_items "$source_dir" "$meta_archive" "$codec" "$level" "$threads_per_job" "${top_files[@]}"
      append_manifest_line "$manifest_path" "meta" "$top_rel" "${meta_archive#$archive_dir/}"
    fi

    local child_index
    for ((child_index = 0; child_index < ${#top_dirs[@]}; child_index++)); do
      child_rel="${top_dirs[$child_index]}"
      child_name="${child_rel##*/}"
      archive_path="$top_archive_dir/${child_name}.$ext"
      printf '%s\0%s\0' "$archive_path" "$child_rel" >>"$job_file"
      append_manifest_line "$manifest_path" "dir" "$child_rel" "${archive_path#$archive_dir/}"
    done
  done
}

run_parallel_compress_jobs() {
  local job_file="$1"
  local source_dir="$2"
  local codec="$3"
  local level="$4"
  local threads_per_job="$5"
  local jobs="$6"

  if [[ ! -s "$job_file" ]]; then
    return 0
  fi

  xargs -0 -n 2 -P "$jobs" bash "$SCRIPT_PATH" __compress_one "$source_dir" "$codec" "$level" "$threads_per_job" <"$job_file"
}

compress_archive() {
  local archive_dir="${1:-$DEFAULT_ARCHIVE_DIR}"
  local source_dir="${2:-$DEFAULT_DIR}"
  local total_threads
  local jobs
  local threads_per_job
  local codec
  local level
  local manifest_path
  local job_file

  [[ -d "$source_dir" ]] || die "Source directory not found: $source_dir"

  source_dir="$(
    cd "$source_dir"
    pwd
  )"
  archive_dir="$(normalize_dir "$archive_dir")"

  total_threads="$(cpu_count)"
  jobs="${MLRUNS_ARCHIVE_JOBS:-$total_threads}"
  if (( jobs > 8 )); then
    jobs=8
  fi
  if (( jobs < 1 )); then
    jobs=1
  fi

  threads_per_job="${MLRUNS_ARCHIVE_THREADS_PER_JOB:-$(( total_threads / jobs ))}"
  if (( threads_per_job < 1 )); then
    threads_per_job=1
  fi

  level="${MLRUNS_ARCHIVE_LEVEL:-3}"

  if load_archive_info "$archive_dir"; then
    codec="$ARCHIVE_INFO_CODEC"
  else
    codec="${MLRUNS_ARCHIVE_CODEC:-$(default_codec)}"
    write_archive_info "$archive_dir" "$codec" "$(basename "$source_dir")"
  fi

  manifest_path="$archive_dir/MANIFEST.tsv"
  job_file="$(mktemp "${TMPDIR:-/tmp}/mlruns_archive_jobs.XXXXXX")"
  trap 'rm -f "$job_file"' RETURN

  log "Compressing $source_dir into $archive_dir"
  log "codec=$codec level=$level jobs=$jobs threads_per_job=$threads_per_job"

  build_compress_job_file \
    "$archive_dir" \
    "$source_dir" \
    "$codec" \
    "$level" \
    "$threads_per_job" \
    "$manifest_path" \
    "$job_file"

  run_parallel_compress_jobs "$job_file" "$source_dir" "$codec" "$level" "$threads_per_job" "$jobs"
  log "Compression complete"
}

extract_archive_file() {
  local archive_file="$1"
  local codec="$2"
  local dest_dir="$3"

  decompress_pipe "$codec" <"$archive_file" | tar -xf - -C "$dest_dir"
}

decompress_archive() {
  local archive_dir="${1:-$DEFAULT_ARCHIVE_DIR}"
  local dest_dir="${2:-$HOME/mlruns_restored}"
  local archive_file
  local codec
  local root_archive

  [[ -d "$archive_dir" ]] || die "Archive directory not found: $archive_dir"
  archive_dir="$(
    cd "$archive_dir"
    pwd
  )"

  load_archive_info "$archive_dir" || die "Missing ARCHIVE_INFO in $archive_dir"
  codec="$ARCHIVE_INFO_CODEC"
  mkdir -p "$dest_dir"
  dest_dir="$(
    cd "$dest_dir"
    pwd
  )"

  log "Extracting $archive_dir into $dest_dir"

  root_archive="$archive_dir/root_files.$(archive_ext "$codec")"
  if [[ -f "$root_archive" ]]; then
    extract_archive_file "$root_archive" "$codec" "$dest_dir"
  fi

  if [[ -d "$archive_dir/shards" ]]; then
    while IFS= read -r archive_file; do
      extract_archive_file "$archive_file" "$codec" "$dest_dir"
    done < <(find "$archive_dir/shards" -type f -name "*.$(archive_ext "$codec")" | LC_ALL=C sort)
  fi

  log "Extraction complete"
}

verify_archive() {
  local archive_dir="${1:-$DEFAULT_ARCHIVE_DIR}"
  local codec
  local archive_file

  [[ -d "$archive_dir" ]] || die "Archive directory not found: $archive_dir"
  archive_dir="$(
    cd "$archive_dir"
    pwd
  )"

  load_archive_info "$archive_dir" || die "Missing ARCHIVE_INFO in $archive_dir"
  codec="$ARCHIVE_INFO_CODEC"

  while IFS= read -r archive_file; do
    decompress_pipe "$codec" <"$archive_file" | tar -tf - >/dev/null
  done < <(find "$archive_dir" -type f -name "*.$(archive_ext "$codec")" | LC_ALL=C sort)

  log "Verified all shards in $archive_dir"
}

backup_sqlite() {
  sqlite3 mlruns.db .dump > mlruns_snapshot.sql
}

restore_sqlite() {
  sqlite3 mlruns.db < mlruns_snapshot.sql
}

cmd="${1:-}"
case "$cmd" in
  compress)
    compress_archive "${2:-$DEFAULT_ARCHIVE_DIR}" "${3:-$DEFAULT_DIR}"
    ;;
  decompress)
    decompress_archive "${2:-$DEFAULT_ARCHIVE_DIR}" "${3:-$HOME/mlruns_restored}"
    ;;
  verify)
    verify_archive "${2:-$DEFAULT_ARCHIVE_DIR}"
    ;;
  backup)
    backup_sqlite
    ;;
  restore)
    restore_sqlite
    ;;
  __compress_one)
    shift
    compress_shard_command "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
