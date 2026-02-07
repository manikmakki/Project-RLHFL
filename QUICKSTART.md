# Quick Start Guide

This guide walks you through setting up Project RLHFL from scratch on a fresh Ubuntu machine. Every step is spelled out -- no assumed knowledge beyond basic terminal usage.

**Time estimate:** 30-60 minutes (mostly model download time).

## Before You Begin

You need:

- An **NVIDIA GPU** with at least 6 GB of VRAM (16 GB recommended). Common compatible cards: GTX 980 Ti, GTX 1080, Tesla P40, Tesla P100, RTX 2080, RTX 3090, RTX 4090.
- **Ubuntu 22.04** (or similar Debian-based Linux). Other distros work but commands may differ.
- **30 GB of free disk space** for models and data.
- A working internet connection (for the initial setup only -- the system runs fully offline after that).

Not sure about your GPU? Run this:

```bash
lspci | grep -i nvidia
```

If you see your NVIDIA card listed, you're good.

---

## Step 1: Install NVIDIA Drivers

Skip this if `nvidia-smi` already works and shows your GPU.

```bash
# Check if drivers are already installed
nvidia-smi
```

If that command fails or isn't found:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
```

You should see a table with your GPU name, driver version, and CUDA version. The CUDA version shown should be **11.8 or higher**.

---

## Step 2: Install Docker

```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Let your user run Docker without sudo
sudo usermod -aG docker $USER

# Apply the group change (or log out and back in)
newgrp docker

# Verify Docker works
docker run hello-world
```

You should see "Hello from Docker!" in the output.

---

## Step 3: Install NVIDIA Container Toolkit

This lets Docker containers access your GPU.

```bash
# Add NVIDIA's package repository
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

You should see the same GPU table as before, but from inside the container. If this fails, Docker can't see your GPU -- don't proceed until this works.

---

## Step 4: Install Git LFS

The model files are large and stored with Git LFS.

```bash
sudo apt-get install -y git-lfs
git lfs install
```

---

## Step 5: Clone the Repository

```bash
git clone <your-repo-url> project-rlhfl
cd project-rlhfl
```

---

## Step 6: Download the Models

This downloads two copies of the Mistral 7B model:
- A **GGUF quantized version** (~4.4 GB) for fast inference
- A **HuggingFace format version** (~14 GB) for training

```bash
./scripts/download_models.sh
```

This will take **10-30 minutes** depending on your internet speed. The script will skip files that are already downloaded, so it's safe to re-run if interrupted.

When it's done, verify:

```bash
ls -lh volumes/models/
```

You should see:
```
mistral-7b-instruct-v0.2.Q4_K_M.gguf    (~4.4 GB)
mistral-7b-instruct-base/                 (directory, ~14 GB)
```

---

## Step 7: Review Configuration (Optional)

The default config works well for most setups. If you want to customize anything, the config lives at:

```
volumes/config/system_config.yaml
```

**Common things you might want to change:**

| Setting | Default | What it does | When to change |
|---------|---------|-------------|----------------|
| `model.n_gpu_layers` | `-1` (all) | How many model layers to put on GPU | Set to `32` if you're running out of VRAM |
| `model.n_threads` | `48` | CPU threads for inference | Set to your actual core count (`nproc`) |
| `training.min_interactions_threshold` | `50` | Interactions needed before auto-training | Lower to `10-20` for faster initial learning |
| `training.batch_size` | `2` | Training batch size | Set to `1` if you have less than 16 GB VRAM |
| `memory.rag_enabled` | `true` | Use past conversations as context | Disable if you prefer a clean-slate model |
| `memory.max_db_size_gb` | `10` | Max database size before auto-cleanup | Increase if you have plenty of disk space |

Edit with any text editor:

```bash
nano volumes/config/system_config.yaml
```

---

## Step 8: Build the Containers

```bash
docker-compose build
```

This builds two Docker images -- one for the API and one for the trainer. First build takes **5-15 minutes** as it installs PyTorch, llama.cpp, and other dependencies. Subsequent builds are cached and much faster.

---

## Step 9: Start the Services

```bash
docker-compose up -d
```

This starts both containers in the background. The API service starts first, and the trainer waits for the API to be healthy before starting.

Watch the startup logs:

```bash
docker-compose logs -f
```

You're looking for these lines in the logs:

```
llm-api     | INFO: LLM API service ready!
llm-trainer | INFO: Training scheduler started (checking every 3600s)
```

**Startup takes about 30-60 seconds** as the model loads into GPU memory.

Press `Ctrl+C` to stop watching logs (the services keep running).

---

## Step 10: Verify Everything Works

```bash
python3 scripts/health_check.py
```

Expected output:

```
================================================================
Project RLHFL - HEALTH CHECK
================================================================

1. Checking API health...
   ✓ API is healthy
   - Model loaded: True
   - Memory connected: True
   - GPU available: True

2. Checking models endpoint...
   ✓ Models endpoint working

3. Checking training stats...
   ✓ Training stats available

4. Testing completion generation...
   ✓ Completion generation working

================================================================
HEALTH CHECK PASSED - All systems operational
================================================================
```

If any check fails, see [Troubleshooting](#troubleshooting) below.

---

## Your First Conversation

### Using curl

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b-instruct",
    "messages": [{"role": "user", "content": "Hello! What can you help me with?"}]
  }' | python3 -m json.tool
```

### Using the example client

```bash
pip install openai requests    # if not already installed
python3 scripts/example_client.py
```

### Using the OpenAI Python library

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="mistral-7b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like on Mars?"}
    ]
)

print(response.choices[0].message.content)
```

### With streaming

```python
stream = client.chat.completions.create(
    model="mistral-7b-instruct",
    messages=[{"role": "user", "content": "Tell me a short story."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

---

## Teaching the Model

You don't need to do anything special. Just chat naturally and the system picks up on your cues:

**Give positive feedback** when you like a response:
> "That's exactly what I needed, thanks!"

**Give corrections** when something is wrong:
> "No, that's not right. The correct answer is X."

**Give instructions** for persistent preferences:
> "Always respond in bullet points."
> "Remember that I prefer concise answers."
> "Never use jargon -- explain things simply."

The model collects these signals and trains itself periodically. You can check progress at any time:

```bash
# How many interactions have been collected
curl -s http://localhost:8000/v1/training/stats | python3 -m json.tool

# Or open the admin dashboard in a browser
open http://localhost:8000/admin
```

To force a training run immediately:

```bash
curl -X POST http://localhost:8000/v1/training/trigger
```

---

## Stopping and Restarting

```bash
# Stop everything
docker-compose down

# Start again
docker-compose up -d

# Restart (picks up config changes)
docker-compose restart

# View logs
docker-compose logs -f api       # just the API
docker-compose logs -f trainer   # just the trainer
docker-compose logs -f           # both
```

All your data (conversations, checkpoints, config) is stored in the `volumes/` directory and persists across restarts.

---

## Troubleshooting

### "GPU not detected" or "No GPU available"

```bash
# 1. Check that your host can see the GPU
nvidia-smi

# 2. Check that Docker can see the GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 3. If step 2 fails, reinstall the NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### "Out of memory" errors

Your GPU doesn't have enough VRAM for the default settings. Edit `volumes/config/system_config.yaml`:

```yaml
model:
  n_gpu_layers: 32              # Offload some layers to CPU (was -1)

training:
  batch_size: 1                 # Smaller batches (was 2)
  gradient_accumulation_steps: 8  # Compensate with more accumulation (was 4)
```

Then restart: `docker-compose restart`

### Model download fails or is incomplete

```bash
# Make sure Git LFS is installed
git lfs install

# Re-run the download script (skips already-downloaded files)
./scripts/download_models.sh

# Or download the GGUF model directly
wget -O volumes/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

### Containers keep restarting

Check the logs for error details:

```bash
docker-compose logs --tail=50 api
docker-compose logs --tail=50 trainer
```

Common causes:
- Model file not found (check `volumes/models/`)
- Config file missing (check `volumes/config/system_config.yaml`)
- Port 8000 already in use (check with `ss -tlnp | grep 8000`)

### API is running but responses are slow

```bash
# Check GPU utilization -- should be high during inference
nvidia-smi

# Make sure all layers are on GPU
docker-compose logs api | grep "n_gpu_layers"
```

If `n_gpu_layers` isn't `-1`, the model is partially on CPU which is much slower. Set it to `-1` in config if your GPU has enough VRAM.

### "Connection refused" when calling the API

The API takes 30-60 seconds to start (model loading). Wait and retry:

```bash
# Watch for the "ready" message
docker-compose logs -f api | grep -i "ready"
```

---

## What's Next

- **Read the README** for a project overview: [README.md](README.md)
- **Read the technical docs** for architecture details: [FOR_NERDS.md](FOR_NERDS.md)
- **Open the admin dashboard** at `http://localhost:8000/admin` to see your interactions and training history
- **Integrate with your apps** -- any tool that supports the OpenAI API can point to `http://localhost:8000/v1`
- **Fine-tune the config** as you learn what works best for your use case
