#!/bin/bash
# download_weights.sh — fetch OpenFace 3.0 model weights from HuggingFace.
# (We skip `openface download`: it silently no-ops when weights/ already exists,
#  which it always does in a fresh clone. See README Gotchas.)
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$PROJECT_ROOT/OpenFace-3.0/weights"
BASE="https://huggingface.co/nutPace/openface_weights/resolve/main"

for f in Alignment_RetinaFace.pth Landmark_98.pkl Landmark_68.pkl MTL_backbone.pth; do
  if [ -s "$DEST/$f" ]; then echo "exists: $f"; else
    echo "downloading $f ..."; curl -L -o "$DEST/$f" "$BASE/$f"
  fi
done

# verify byte-exact sizes
check() { s=$(stat -f%z "$DEST/$1" 2>/dev/null); [ "$s" = "$2" ] && echo "OK   $1" || { echo "BAD  $1"; exit 1; }; }
check Alignment_RetinaFace.pth 1789735
check MTL_backbone.pth         101710914
check Landmark_98.pkl          178030153
check Landmark_68.pkl          176735623
echo "All weights verified."