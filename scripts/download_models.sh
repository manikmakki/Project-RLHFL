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
echo "Step 1: Downloading GPT-OSS-20B GGUF (Q4_K_M quantization)..."
echo "This is the model used for inference (~12 GB)"
echo ""

if [ -f "$MODELS_DIR/jinx-gpt-oss-20b-Q4_K_M.gguf" ]; then
    echo "GGUF model already exists, skipping download."
else
    wget -O "$MODELS_DIR/jinx-gpt-oss-20b-Q4_K_M.gguf" \
        "https://huggingface.co/jinx-org/jinx-gpt-oss-20b-GGUF/resolve/main/jinx-gpt-oss-20b-Q4_K_M.gguf"
    echo "✓ GGUF model downloaded"
fi

echo ""

# Download HuggingFace model for training
echo "Step 2: Downloading GPT-OSS-20B (HuggingFace format)..."
echo "This is the base model used for training (~40 GB)"
echo ""

if [ -d "$MODELS_DIR/jinx-gpt-oss-20b-base" ] && [ -f "$MODELS_DIR/jinx-gpt-oss-20b-base/config.json" ]; then
    echo "HuggingFace model already exists, skipping download."
else
    git clone https://huggingface.co/jinx-org/jinx-gpt-oss-20b "$MODELS_DIR/jinx-gpt-oss-20b-base"
    echo "✓ HuggingFace model downloaded"
fi

echo ""
echo "=================================================="
echo "Model download complete!"
echo "=================================================="
echo ""
echo "Models downloaded:"
echo "  - Inference (GGUF): $MODELS_DIR/jinx-gpt-oss-20b-Q4_K_M.gguf"
echo "  - Training (HF):    $MODELS_DIR/jinx-gpt-oss-20b-base"
echo ""
echo "Total size: ~52 GB (12 GB inference + 40 GB training base)"
echo ""
echo "Next steps:"
echo "  1. Review and customize volumes/config/system_config.yaml if needed"
echo "  2. Run: docker-compose build"
echo "  3. Run: docker-compose up -d"
echo ""
