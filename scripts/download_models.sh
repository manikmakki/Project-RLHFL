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

# Download GGUF model for inference
echo "Step 1: Downloading Mistral 7B Instruct GGUF (Q4_K_M quantization)..."
echo "This is the model used for inference (~4.4 GB)"
echo ""

if [ -f "$MODELS_DIR/mistral-7b-instruct-v0.2.Q4_K_M.gguf" ]; then
    echo "GGUF model already exists, skipping download."
else
    wget -O "$MODELS_DIR/mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
        "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    echo "✓ GGUF model downloaded"
fi

echo ""

# Download HuggingFace model for training
echo "Step 2: Downloading Mistral 7B Instruct (HuggingFace format)..."
echo "This is the base model used for training (~14 GB)"
echo ""

if [ -d "$MODELS_DIR/mistral-7b-instruct-base" ] && [ -f "$MODELS_DIR/mistral-7b-instruct-base/config.json" ]; then
    echo "HuggingFace model already exists, skipping download."
else
    git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 "$MODELS_DIR/mistral-7b-instruct-base"
    echo "✓ HuggingFace model downloaded"
fi

echo ""
echo "=================================================="
echo "Model download complete!"
echo "=================================================="
echo ""
echo "Models downloaded:"
echo "  - Inference (GGUF): $MODELS_DIR/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
echo "  - Training (HF):    $MODELS_DIR/mistral-7b-instruct-base"
echo ""
echo "Total size: ~18.4 GB (4.4 GB inference + 14 GB training base)"
echo ""
echo "Next steps:"
echo "  1. Review and customize volumes/config/system_config.yaml if needed"
echo "  2. Run: docker-compose build"
echo "  3. Run: docker-compose up -d"
echo ""
