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
mistral-7b-instruct-v0.3.Q4_K_M.gguf    (~4.4 GB)
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

This starts all three containers in the background:
- **llm-api** - FastAPI server (port 8000)
- **llm-trainer** - Background training worker
- **llm-ollama** - GPU-accelerated model serving (port 11434)

Watch the startup logs:

```bash
docker-compose logs -f
```

You're looking for these lines in the logs:

```
llm-api     | INFO: LLM API service ready!
llm-trainer | INFO: Training scheduler started (checking every 3600s)
llm-ollama  | Listening on [::]:11434 (version 0.15.4)
```

**Startup takes about 30-60 seconds** as Ollama pulls and loads the base model into GPU memory.

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
wget -O volumes/models/mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q4_K_M.gguf"
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

## Understanding Sentiment-Based Training

**RLHFL uses implicit sentiment feedback to automatically train your AI** - no manual intervention required!

### How It Works

The system tracks sentiment for every interaction:
- **Positive sentiment** (+0.3 to +1.0): Good responses you appreciate
- **Negative sentiment** (-1.0 to -0.3): Bad responses that need correction
- **Neutral sentiment** (-0.3 to +0.3): Normal conversations

When enough feedback accumulates, training triggers automatically:

| Trigger | Threshold | Training Mode | Purpose |
|---------|-----------|---------------|---------|
| **20+ positive/golden** | `sft_trigger_threshold` | **SFT** | Reinforce good behavior |
| **5+ negative sentiments** | `dpo_trigger_threshold` | **DPO** | Train out bad behavior |
| **8 hours inactivity** | `inactivity_threshold_hours` | **Auto** | Consolidate learnings |

### Training Modes

**SFT (Supervised Fine-Tuning)**
- Reinforces positive patterns
- Faster training (~2-4 hours on CPU)
- Used when you have good examples to learn from

**DPO (Direct Preference Optimization)**
- Corrects unwanted behaviors
- Generates synthetic "bad" responses for contrast
- Used when you have negative feedback to address

**The system automatically selects the right mode** based on your feedback!

### Training Schedule

To avoid impacting performance during peak hours, training runs **during off-peak times**:

```yaml
# In volumes/config/system_config.yaml
training:
  schedule_enabled: true              # Enable scheduled training
  schedule_time: "01:00"              # Run at 1:00 AM
  schedule_timezone: "America/New_York"
  schedule_window_minutes: 60         # Execute within ±60 min of scheduled time
```

**Manual triggers** (via Admin UI) bypass the schedule and run immediately.

### Example: Learning from Feedback

```
Day 1: User asks "Explain recursion"
       Model includes code example
       User gives negative feedback → negative sentiment

Day 2: 4 more similar interactions with negative feedback
       Total: 5 negative sentiments

Night 2: DPO training triggered at 1:00 AM
         Learns to avoid unsolicited code examples

Day 3: User asks "Explain recursion"
       Model provides clean explanation WITHOUT code ✓
```

**Your AI molds to your preferences automatically!**

### Configuration Tips

Adjust these in `volumes/config/system_config.yaml`:

```yaml
# Trigger thresholds
sft_trigger_threshold: 20    # How many positive examples before SFT training
dpo_trigger_threshold: 5     # How many negative examples before DPO training

# Schedule
schedule_time: "01:00"       # Change to fit your timezone/schedule
schedule_timezone: "America/New_York"  # Your timezone
```

**Lower thresholds** = more frequent training (faster learning, more CPU usage)
**Higher thresholds** = less frequent training (slower learning, less resource usage)

---

## Manual GGUF Conversion

If you need to manually convert a trained model to GGUF format (for testing, debugging, or custom deployment), you can use the conversion script directly.

### Prerequisites

Make sure the `gguf` library is installed in your trainer container:

```bash
docker exec -it llm-trainer pip install gguf==0.10.0
```

### Finding Your Trained Model

First, list your training checkpoints:

```bash
ls -lh volumes/checkpoints/
```

You'll see directories like `checkpoint_20260217_050631`. Each contains:
- `adapter_sequential_final/` - The merged model (HuggingFace format)
- `*.gguf` - The GGUF file (if conversion already ran)

### Converting a Merged Model to GGUF

Run the conversion script with the merged model:

```bash
# Example: Convert checkpoint from 2026-02-17 05:06
docker exec -it llm-trainer python3 /app/scripts/convert_hf_to_gguf.py \
  /checkpoints/checkpoint_20260217_050631/adapter_sequential_final \
  --outfile /checkpoints/checkpoint_20260217_050631/manual_convert.gguf \
  --outtype f16
```

**Parameters:**
- **First argument**: Path to the merged HuggingFace model directory
- `--outfile`: Where to save the GGUF file
- `--outtype`: Output data type:
  - `f16` - 16-bit float (recommended, ~14GB for 20B model)
  - `f32` - 32-bit float (larger, more precise)
  - `q8_0` - 8-bit quantized (smaller, faster)

### Quantizing to Q4_K_M (Production Format)

After creating the F16 GGUF, quantize it to Q4_K_M for production use:

```bash
# Quantize F16 → Q4_K_M (reduces ~14GB → ~12GB)
docker exec -it llm-trainer /app/bin/llama-quantize \
  /checkpoints/checkpoint_20260217_050631/manual_convert.gguf \
  /checkpoints/checkpoint_20260217_050631/manual_convert_Q4_K_M.gguf \
  Q4_K_M
```

### Deploying to Ollama

Once you have the GGUF file, deploy it to the containerized Ollama service:

**Via Admin UI (Recommended):**
1. Go to [http://localhost:8000/admin](http://localhost:8000/admin)
2. Find your checkpoint in the "Training Checkpoints" table
3. Click the "Deploy" button next to your checkpoint
4. The fine-tuned model will hot-reload into Ollama

**Via API:**
```bash
curl -X POST "http://localhost:8000/admin/api/ollama/deploy-checkpoint?checkpoint_id=checkpoint_20260217_050631"
```

**Via CLI (Advanced):**
```bash
# Create a Modelfile and deploy directly to the Ollama container
docker exec llm-ollama sh -c 'cat > /tmp/Modelfile <<EOF
FROM /checkpoints/checkpoint_20260217_050631/manual_convert_Q4_K_M.gguf
EOF
ollama create dolphin3:8b -f /tmp/Modelfile'
```

The model will be immediately available at `http://localhost:11434` and through your API at `http://localhost:8000/v1/chat/completions`.

### Verifying the Conversion

Check the file size and verify it's reasonable:

```bash
# F16 should be ~14GB for a 20B model
ls -lh volumes/checkpoints/checkpoint_*/manual_convert.gguf

# Q4_K_M should be ~12GB
ls -lh volumes/checkpoints/checkpoint_*/manual_convert_Q4_K_M.gguf
```

### Troubleshooting

**"Model GptOssForCausalLM is not supported"**
- The conversion script needs the `gguf` library. Install it:
  ```bash
  docker exec -it llm-trainer pip install gguf==0.10.0
  ```

**"No such file or directory"**
- Make sure you're using the full path inside the container (`/checkpoints/...`)
- List files first: `docker exec -it llm-trainer ls -la /checkpoints/`

**Conversion fails with memory error**
- The conversion needs ~50GB RAM
- Your system has 256GB, so this shouldn't happen
- If it does, check available RAM: `free -h`

### Automatic vs Manual Conversion

**Automatic (recommended):**
- Happens after every successful training
- Includes quantization
- Deploys to Ollama automatically
- Configured in `system_config.yaml`

**Manual (for testing/debugging):**
- Full control over output format
- Useful for experimenting with different quantization levels
- Good for troubleshooting deployment issues

## What's Next

- **Read the README** for a project overview: [README.md](README.md)
- **Read the technical docs** for architecture details: [FOR_NERDS.md](FOR_NERDS.md)
- **Open the admin dashboard** at `http://localhost:8000/admin` to see your interactions and training history
- **Integrate with your apps** -- any tool that supports the OpenAI API can point to `http://localhost:8000/v1`
- **Fine-tune the config** as you learn what works best for your use case
