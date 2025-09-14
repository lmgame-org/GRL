#!/bin/bash
set -e

ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
CACHE_DIR="$ROOT_DIR/cache"
mkdir -p "$CACHE_DIR"
LOG_FILE="$CACHE_DIR/code_style_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

MODE="--diff"
BASE_REF=""
CHECK=0

for arg in "$@"; do
  case "$arg" in
    --diff) MODE="--diff";;
    --all-grl) MODE="--all-grl";;
    --check) CHECK=1;;
    *) if [[ -z "$BASE_REF" && "$MODE" == "--diff" ]]; then BASE_REF="$arg"; fi;;
  esac
done

if [[ $CHECK -eq 1 ]]; then
  CHECK_FLAGS="--check --diff --color"
else
  CHECK_FLAGS=""
fi

LINE_LENGTH=80
if [[ -f pylintrc ]]; then
  v=$(grep -E "^max-line-length=" pylintrc | cut -d '=' -f 2)
  if [[ -n "$v" ]]; then LINE_LENGTH="$v"; fi
fi

collect_diff_files() {
  git fetch --quiet || true
  if [[ -z "$BASE_REF" ]]; then
    if git rev-parse --verify --quiet origin/HEAD >/dev/null; then
      BASE_REF="origin/HEAD"
    elif git rev-parse --verify --quiet origin/main >/dev/null; then
      BASE_REF="origin/main"
    elif git rev-parse --verify --quiet origin/master >/dev/null; then
      BASE_REF="origin/master"
    else
      BASE_REF="HEAD"
    fi
  fi
  mapfile -d '' FILES < <( (git diff --name-only -z --diff-filter=d "$BASE_REF" -- "*.py" "*.ipynb"; git ls-files -z --others --exclude-standard -- "*.py" "*.ipynb") | sort -zu )
}

collect_grl_files() {
  local root
  root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
  local dir="$root/grl"
  if [[ ! -d "$dir" ]]; then
    echo "grl directory not found: $dir"
    exit 1
  fi
  mapfile -d '' FILES < <(find "$dir" -type f \( -name "*.py" -o -name "*.ipynb" \) -print0)
}

FILES=()
if [[ "$MODE" == "--all-grl" ]]; then
  collect_grl_files
else
  collect_diff_files
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No Python or Notebook files to format."
  exit 0
fi

echo "Formatting ${#FILES[@]} files ($MODE)"
pyink "${FILES[@]}" ${CHECK_FLAGS} --pyink-indentation=2 --line-length="${LINE_LENGTH}"

echo "Done."
