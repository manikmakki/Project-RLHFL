#!/bin/bash

set -e

MODELS_DIR="./volumes/models"
mkdir -p "$MODELS_DIR"

echo "=================================================="
echo "Project RLHFL - Model Download Script"
echo "=================================================="
echo ""

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Error: git-lfs is not installed."
    echo "Please install git-lfs first:"
    echo "  Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  macOS: brew install git-lfs"
    echo "  Then run: git lfs install"
    exit 1
fi

git lfs install

# Download HuggingFace model for training
echo "Step 1: Downloading Ministral-3-14B (HuggingFace format)..."
echo "This is the base model used for LoRA training (~60 GB)"
echo ""

if [ -d "$MODELS_DIR/Ministral-3-14B" ] && [ -f "$MODELS_DIR/Ministral-3-14B/config.json" ]; then
    echo "HuggingFace model already exists, skipping download."
else
    git clone 'https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16' "$MODELS_DIR/Ministral-3-14B"
    echo "✓ HuggingFace model downloaded"
fi

echo ""
echo "=================================================="
echo "Model download complete!"
echo "=================================================="
echo ""
echo "Models downloaded:"
echo "  - Training (HF): $MODELS_DIR/Ministral-3-14B"
echo "  - Inference: run llama-server on the host pointing at your GGUF"
echo ""
echo "Next steps:"
echo "  1. Review and customize volumes/config/system_config.yaml if needed"
echo "  2. Run: docker-compose build"
echo "  3. Run: docker-compose up -d"
echo ""
