#!/bin/bash

# Default checkpoint path from generate_wetok.py
CKPT_PATH="GrayShine/ImageNet/downsample8/WeTok.ckpt"
CKPT_URL="https://huggingface.co/GrayShine/WeTok/resolve/main/ImageNet/downsample8/WeTok.ckpt"

# Create directory
mkdir -p $(dirname "$CKPT_PATH")

# Download
if [ ! -f "$CKPT_PATH" ]; then
    echo "Downloading checkpoint to $CKPT_PATH..."
    curl -L "$CKPT_URL" -o "$CKPT_PATH"
else
    echo "Checkpoint already exists at $CKPT_PATH"
fi
