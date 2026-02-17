# GGUF Conversion - Quick Reference

## Basic Usage

```bash
python3 convert_hf_to_gguf.py <MODEL_DIR> --outfile <OUTPUT.gguf> --outtype <TYPE>
```

## Example: Convert a Trained Checkpoint

### Step 1: Find Your Checkpoint

```bash
# List available checkpoints
ls -lh /opt/project-rlhfl/volumes/checkpoints/

# Example output:
# checkpoint_20260217_050631/
#   ├── adapter_sequential_final/  ← This is what we convert
#   ├── training_metadata.json
#   └── pass_1_layers_0_4/
```

### Step 2: Convert to F16 GGUF

```bash
# Inside trainer container:
docker exec -it llm-trainer python3 /app/scripts/convert_hf_to_gguf.py \
  /checkpoints/checkpoint_20260217_050631/adapter_sequential_final \
  --outfile /checkpoints/checkpoint_20260217_050631/converted_f16.gguf \
  --outtype f16
```

**Or from host (if script is in volumes):**

```bash
# From host machine:
cd /opt/project-rlhfl
docker exec -it llm-trainer python3 /app/scripts/convert_hf_to_gguf.py \
  /checkpoints/checkpoint_$(ls -t volumes/checkpoints/ | head -1)/adapter_sequential_final \
  --outfile /checkpoints/latest_f16.gguf \
  --outtype f16
```

### Step 3: Quantize to Q4_K_M (Production)

```bash
# Quantize for production use (reduces size by ~15%)
docker exec -it llm-trainer /app/bin/llama-quantize \
  /checkpoints/checkpoint_20260217_050631/converted_f16.gguf \
  /checkpoints/checkpoint_20260217_050631/converted_Q4_K_M.gguf \
  Q4_K_M
```

### Step 4: Deploy to Ollama

```bash
# Via API (recommended):
curl -X POST "http://localhost:8000/admin/api/ollama/deploy-checkpoint?checkpoint_id=checkpoint_20260217_050631"

# Or manually create Modelfile and deploy:
cat > /tmp/Modelfile <<EOF
FROM /checkpoints/checkpoint_20260217_050631/converted_Q4_K_M.gguf
TEMPLATE """<|start|>user<|message|>{{ .Prompt }}<|end|><|start|>assistant<|channel|>final<|message|>"""
EOF

ollama create gpt-oss:20b-ft -f /tmp/Modelfile
```

## Output Types Explained

| Type   | Size (20B model) | Quality | Speed | Use Case |
|--------|------------------|---------|-------|----------|
| `f32`  | ~80GB           | Best    | Slow  | Research/evaluation |
| `f16`  | ~40GB           | Excellent | Medium | Intermediate step |
| `q8_0` | ~20GB           | Very Good | Fast | High-quality production |
| `q4_k_m` | ~12GB         | Good    | Fastest | Production (recommended) |

## File Size Reference

For a 20B parameter model:
- **HuggingFace (FP32)**: ~80GB
- **F16 GGUF**: ~40GB
- **Q8_0 GGUF**: ~20GB
- **Q4_K_M GGUF**: ~12GB (production default)

## Common Issues

### "Model GptOssForCausalLM is not supported"
```bash
# Install required library:
docker exec -it llm-trainer pip install gguf==0.10.0
```

### "GGUF CONVERSION FAILED: Missing required library"
```bash
# Install gguf:
docker exec -it llm-trainer pip install gguf==0.10.0
```

### "File not found: /checkpoints/..."
```bash
# Make sure you're using the container path, not host path:
# ✓ Correct: /checkpoints/checkpoint_*/adapter_sequential_final
# ✗ Wrong: /opt/project-rlhfl/volumes/checkpoints/...

# Verify path inside container:
docker exec -it llm-trainer ls -la /checkpoints/
```

### Out of memory during conversion
```bash
# Check available RAM (needs ~50GB):
free -h

# If low on RAM, close other applications
# The conversion loads the entire model into memory
```

## Automated vs Manual

**Automatic (default):**
- Enabled in `system_config.yaml`: `enable_gguf_conversion: true`
- Runs after every successful training
- Creates Q4_K_M GGUF automatically
- Deploys to Ollama automatically
- No manual intervention needed

**Manual (this guide):**
- For testing different quantization levels
- For debugging conversion issues
- For custom deployments
- For educational purposes

## Quick Commands

```bash
# Latest checkpoint to F16:
docker exec -it llm-trainer python3 /app/scripts/convert_hf_to_gguf.py \
  /checkpoints/$(ls -t /opt/project-rlhfl/volumes/checkpoints/ | head -1)/adapter_sequential_final \
  --outfile /checkpoints/latest.gguf --outtype f16

# Then quantize:
docker exec -it llm-trainer /app/bin/llama-quantize \
  /checkpoints/latest.gguf \
  /checkpoints/latest_q4.gguf \
  Q4_K_M

# Deploy:
curl -X POST "http://localhost:8000/admin/api/ollama/deploy-checkpoint?checkpoint_id=$(ls -t /opt/project-rlhfl/volumes/checkpoints/ | head -1)"
```

## See Also

- [QUICKSTART.md](../QUICKSTART.md#manual-gguf-conversion) - Full setup guide
- [FOR_NERDS.md](../FOR_NERDS.md) - Technical architecture details
- Admin UI: http://localhost:8000/admin - Web-based model deployment
