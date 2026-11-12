#!/usr/bin/env bash
# Download the MOT17 labels and public detections (no images) into data/MOT17.
# MOT17 is licensed CC BY-NC-SA 3.0 (non-commercial). It is NOT part of this repo.
set -euo pipefail

URL="https://motchallenge.net/data/MOT17Labels.zip"
DEST="data/MOT17"

if [ -d "$DEST/train" ]; then
    echo "Already downloaded: $DEST"
    exit 0
fi

mkdir -p "$DEST"
curl -L --fail -o "$DEST/MOT17Labels.zip" "$URL"
unzip -q "$DEST/MOT17Labels.zip" -d "$DEST"
rm "$DEST/MOT17Labels.zip"
echo "Done: $(ls "$DEST/train" | wc -l) training folders in $DEST/train"
