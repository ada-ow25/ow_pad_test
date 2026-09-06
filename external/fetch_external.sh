#!/usr/bin/env bash
# Re-fetch the public replay-attack data used by external_validation.ipynb.
# Raw videos are ~1GB and gitignored; extracted frames and scores are committed.
#
#   AxonData/Display_replay_attacks           CC-BY-4.0
#   UniqueData/monitors-replay-attacks        CC-BY-NC-ND-4.0
set -euo pipefail
cd "$(dirname "$0")"

B="https://huggingface.co/datasets/AxonData/Display_replay_attacks/resolve/main/Axon%20Labs%20Replay%20Display%20sample"
mkdir -p axon_display/real axon_display/attack
for i in 1 2 3 4 5; do curl -sSL "$B/Real/$i.jpg" -o "axon_display/real/$i.jpg"; done
for f in 20240823_194131.mp4 20240823_194241.mp4 IMG_5915.MOV IMG_5987.MOV VID_20240823_171603.mp4; do
    curl -sSL "$B/Screen/$f" -o "axon_display/attack/$f"
done

curl -sSL "https://huggingface.co/datasets/UniqueData/monitors-replay-attacks-dataset/resolve/main/data/attacks.tar.gz" \
    -o monitors_attacks.tar.gz
mkdir -p monitors && tar xzf monitors_attacks.tar.gz -C monitors

echo "done -- now run external_validation.ipynb, which extracts frames and scores them"
