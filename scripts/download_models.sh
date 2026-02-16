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

# Download GGUF model for inference (optional - can use Ollama instead)
echo "Step 1: Downloading GPT-OSS-20B GGUF (Q4_K_M quantization)..."
echo "This is optional if you're using Ollama for inference"
echo "Skip this if you have 'ollama pull gpt-oss:20b' already"
echo ""

if [ -f "$MODELS_DIR/gpt-oss-20b-Q4_K_M.gguf" ]; then
    echo "GGUF model already exists, skipping download."
else
    echo "Skipping GGUF download - use Ollama instead: ollama pull gpt-oss:20b"
    echo "If you need GGUF, download from: https://huggingface.co/openai/gpt-oss-20b-gguf"
fi

echo ""

# Download HuggingFace model for training
echo "Step 2: Downloading GPT-OSS-20B (HuggingFace format)..."
echo "This is the base model used for training (~40 GB)"
echo ""

if [ -d "$MODELS_DIR/gpt-oss-20b-base" ] && [ -f "$MODELS_DIR/gpt-oss-20b-base/config.json" ]; then
    echo "HuggingFace model already exists, skipping download."
else
    git clone https://huggingface.co/openai/gpt-oss-20b "$MODELS_DIR/gpt-oss-20b-base"
    echo "✓ HuggingFace model downloaded"
fi

echo ""
echo "=================================================="
echo "Model download complete!"
echo "=================================================="
echo ""
echo "Models downloaded:"
echo "  - Inference: Use Ollama (ollama pull gpt-oss:20b)"
echo "  - Training (HF):    $MODELS_DIR/gpt-oss-20b-base"
echo ""
echo "Total size: ~40 GB (training base model)"
echo ""
echo "Next steps:"
echo "  1. Review and customize volumes/config/system_config.yaml if needed"
echo "  2. Run: docker-compose build"
echo "  3. Run: docker-compose up -d"
echo ""
