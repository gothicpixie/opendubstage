#!/usr/bin/env bash
# Install the Woosh-DVFlow local backend into vendor/woosh (code + ~4.6 GB of weights).
# Refuses to run if it would leave less than MIN_FREE_GB free on the home volume.
#   MIN_FREE_GB=40 scripts/install_woosh.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MIN_FREE_GB="${MIN_FREE_GB:-40}"
NEED_GB="${NEED_GB:-6}"   # zips + unzipped peak (one archive at a time) + Synchformer
free_gb=$(df -g "$HOME" | awk 'NR==2 {print $4}')
if (( free_gb - NEED_GB < MIN_FREE_GB )); then
  echo "refusing: ${free_gb} GB free, need ${NEED_GB} GB and must keep ${MIN_FREE_GB} GB" >&2
  exit 1
fi
PY=.venv/bin/python
[[ -d vendor/woosh ]] || git clone --depth 1 https://github.com/SonyResearch/Woosh vendor/woosh
uv pip install --python "$PY" -q --no-deps hear21passt==0.0.26
uv pip install --python "$PY" -q -e vendor/woosh torchvision
mkdir -p vendor/woosh/_zips
for z in Woosh-AE TextConditionerV Woosh-DVFlow-8s; do
  [[ -f "vendor/woosh/checkpoints/$z/.complete" ]] && continue
  [[ -f "vendor/woosh/_zips/$z.zip" ]] || gh release download v1.0.0 -R SonyResearch/Woosh -p "$z.zip" -D vendor/woosh/_zips
  unzip -q -o "vendor/woosh/_zips/$z.zip" -d vendor/woosh
  touch "vendor/woosh/checkpoints/$z/.complete"
  rm "vendor/woosh/_zips/$z.zip"          # keep only one copy on disk
done
# Synchformer weights (950 MB) are fetched by Woosh from hkchengrex/MMAudio on first use.
echo "woosh installed: $(du -sh vendor/woosh/checkpoints | cut -f1)"
