#!/bin/bash
# patch_openface.sh — fixes a bug in the pip-installed openface-test package:
# its STAR/conf/alignment.py hardcodes the author's Docker path ('/work/...'),
# which crashes on macOS's read-only root volume. Run once after pip install.
# Safe to run repeatedly. NOTE: macOS sed syntax (Linux would be: sed -i).
set -e

# Project root = parent of the folder this script lives in → works from any cwd
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$PROJECT_ROOT/.venv/lib/python3.11/site-packages/openface/STAR/conf/alignment.py"

if [ ! -f "$TARGET" ]; then
  echo "ERROR: $TARGET not found — set up the venv and pip install openface-test first."
  exit 1
fi

if grep -q "/work/jiewenh" "$TARGET"; then
  sed -i '' "s|self.ckpt_dir = '/work/jiewenh/openFace/OpenFace-3.0/STAR'|self.ckpt_dir = osp.expanduser('~/openface_star_workdir')|" "$TARGET"
  echo "Patched: ckpt_dir -> ~/openface_star_workdir"
else
  echo "Already patched — nothing to do."
fi

grep -n "ckpt_dir =" "$TARGET" | head -2   # show the result either way