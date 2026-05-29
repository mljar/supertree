#!/bin/sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
JS_DIR="$ROOT_DIR/supertree/js"
OUTPUT_FILE="$JS_DIR/supertree.min.js"
TMP_FILE=$(mktemp)
TMP_D3_FILE=$(mktemp)
trap 'rm -f "$TMP_FILE" "$TMP_D3_FILE"' EXIT

if command -v terser >/dev/null 2>&1; then
  TERSER_CMD="terser"
elif command -v npx >/dev/null 2>&1; then
  TERSER_CMD="npx --yes terser"
else
  echo "Error: terser or npx is required to build supertree.min.js" >&2
  exit 1
fi

echo "Building $OUTPUT_FILE"

sh -c "$TERSER_CMD \"$JS_DIR/script.js\" -o \"$TMP_FILE\" -c -m toplevel=true"
sh -c "$TERSER_CMD \"$JS_DIR/dependencies/d3vs.js\" -o \"$TMP_D3_FILE\""

python3 - <<'PY' "$TMP_FILE"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
path.write_text(text.replace("debug", "error", 1), encoding="utf-8")
PY

cat "$TMP_FILE" "$TMP_D3_FILE" > "$OUTPUT_FILE"

echo "Wrote $OUTPUT_FILE"
