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
echo "Step 1: GGUF model for Ollama inference..."
echo "This is optional if you already have the model in Ollama"
echo "Skip this if you have 'ollama pull dolphin3:8b' already"
echo ""

echo "Pull via Ollama:"
echo "  ollama pull dolphin3:8b"

echo ""

# Download HuggingFace model for training
echo "Step 2: Downloading Dolphin3.0-Llama3.1-8B (HuggingFace format)..."
echo "This is the base model used for LoRA training (~60 GB)"
echo ""

if [ -d "$MODELS_DIR/Dolphin3.0-Llama3.1-8B" ] && [ -f "$MODELS_DIR/Dolphin3.0-Llama3.1-8B/config.json" ]; then
    echo "HuggingFace model already exists, skipping download."
else
    git clone 'https://huggingface.co/dphn/Dolphin3.0-Llama3.1-8B' "$MODELS_DIR/Dolphin3.0-Llama3.1-8B"
    echo "✓ HuggingFace model downloaded"
fi

echo ""
echo "=================================================="
echo "Model download complete!"
echo "=================================================="
echo ""
echo "Models downloaded:"
echo "  - Inference: Use Ollama (ollama pull dolphin3:8b)"
echo "  - Training (HF):    $MODELS_DIR/Dolphin3.0-Llama3.1-8B"
echo ""
echo "Next steps:"
echo "  1. Review and customize volumes/config/system_config.yaml if needed"
echo "  2. Run: docker-compose build"
echo "  3. Run: docker-compose up -d"
echo ""
