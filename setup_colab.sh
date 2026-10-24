#!/usr/bin/env bash
# One-time Colab setup. Usage: bash setup_colab.sh [/content]
# Clones See-through + tmp_upscaler next to this repo and installs only what inference needs.
# Colab already ships a CUDA build of torch, so torch is NOT reinstalled here
# (See-through pins torch 2.8 cu128; newer Colab torch has worked for diffusers-based inference,
#  but if you hit a torch/diffusers error, run:  SEETHROUGH_PIN_TORCH=1 bash setup_colab.sh).
set -euo pipefail
ROOT="${1:-/content}"
cd "$ROOT"

[ -d see-through ]   || git clone --depth 1 https://github.com/shitagaki-lab/see-through
[ -d tmp_upscaler ]  || git clone --depth 1 https://github.com/HisameOgasahara/tmp_upscaler

if [ "${SEETHROUGH_PIN_TORCH:-0}" = "1" ]; then
  pip install -q torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
fi

# See-through inference deps: drop torch (preinstalled), Qt UI, training-only and dev packages
grep -vE '^\s*(#|$)|^-e \./annotators|^(torch|torchvision|torchaudio|PyQt6|PyQt6-Qt6|qtpy|lpips|convnext_perceptual_loss|ipykernel|pytest|grad-cam|pycocotools)\b' \
  see-through/requirements.txt > /tmp/st_req.txt
sed -i 's#^-e \./common#-e '"$ROOT"'/see-through/common#' /tmp/st_req.txt
pip install -q -r /tmp/st_req.txt
pip install -q -r see-through/requirements-inference-bnb.txt

# upscaler + vectorizer
pip install -q spandrel vtracer cairosvg "psd-tools[composite]"
echo "setup done"
