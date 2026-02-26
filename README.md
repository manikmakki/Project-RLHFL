# Project RLHFL - Reinforcement Learning from Human Feedback Loop

A fully local, self-improving language model that learns from your conversations automatically. It runs entirely on your hardware, keeps your data private, and exposes an API that works as a drop-in replacement for OpenAI -- so any app or tool that speaks the OpenAI protocol can talk to it without changes.

## What It Does

You chat with the model through a standard API. Behind the scenes, the system watches how your conversations go: when you say "thanks, that's perfect" it takes note; when you say "no, that's wrong" it takes note of that too. Over time it collects enough signal to retrain itself using [LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) adapters, producing a version of the model that's more aligned with how you actually want it to behave.

No cloud services. No manual labeling. No thumbs-up buttons. Just natural conversation.

## Key Features

- **Fully local** -- Zero cloud dependencies. Your data never leaves your machine.
- **OpenAI-compatible API** -- Point any OpenAI client at `localhost:8000` and it just works.
- **Automatic learning** -- Infers feedback from conversation patterns (praise, corrections, instructions) without any manual labeling.
- **Safe training** -- Every training run is validated against the previous checkpoint. If performance degrades, the system rolls back automatically.
- **Hot model reload** -- After training, the fine-tuned model is automatically converted to GGUF and deployed to the containerized Ollama service with zero downtime. Automatic rollback on failure.
- **RAG-powered context** -- Retrieves relevant past interactions to inform responses using ChromaDB and semantic search.
- **Checkpoint management** -- Full version history of model adapters with rollback support.
- **Admin dashboard** -- Web UI at `/admin` for monitoring, triggering training, rolling back checkpoints, and managing the database.
- **Memory management** -- Automatic cleanup of low-value data, age-based pruning, and database size limits.

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 6GB+ VRAM (CUDA 6.1+) | 16GB+ VRAM |
| RAM | 16 GB | 32 GB |
| Storage | 30 GB free | 50 GB free |

Tested on GTX 980 Ti, Tesla P40, Tesla P100, and similar Maxwell/Pascal GPUs.

### Software

- Linux (Ubuntu 22.04 recommended)
- Docker and Docker Compose
- NVIDIA drivers with CUDA 11.8 support
- `nvidia-docker2` (NVIDIA Container Toolkit)
- Git with Git LFS

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a detailed, step-by-step walkthrough. The short version:

```bash
# Clone the repo
git clone <your-repo-url> project-rlhfl && cd project-rlhfl

# Download models (~18 GB)
./scripts/download_models.sh

# Build and launch
docker-compose build
docker-compose up -d

# Verify everything is running
python3 scripts/health_check.py
```

## Usage

### Talk to it with any OpenAI client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="ministral-3:14b",
    messages=[{"role": "user", "content": "Explain quicksort in plain English."}]
)

print(response.choices[0].message.content)
```

### Or with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ministral-3:14b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Streaming, system messages, temperature, top_p, and max_tokens are all supported. See [FOR_NERDS.md](FOR_NERDS.md) for the full API reference.

### How it learns

Just use it naturally. The system picks up on conversational cues:

| What you say | What the system infers |
|---|---|
| "Thanks!" / "Perfect!" / "Great answer" | Positive -- reinforce this behavior |
| "No, that's wrong" / "Try again" | Negative -- penalize this behavior |
| "Always format code with comments" / "Remember to be concise" | High-value instruction -- saved as a golden example |
| (continuing the conversation normally) | Mild positive -- the response was acceptable |

Training kicks in automatically when enough interactions accumulate (default: 50), after a period of inactivity (24 hours), or on a regular schedule (every 7 days). You can also trigger it manually:

```bash
curl -X POST http://localhost:8000/v1/training/trigger
```

### Check on things

```bash
# Training stats
curl http://localhost:8000/v1/training/stats

# System health
curl http://localhost:8000/health

# Admin dashboard (browser)
open http://localhost:8000/admin
```

## Configuration

All settings live in `volumes/config/system_config.yaml`. The most commonly adjusted values:

```yaml
training:
  min_interactions_threshold: 50    # How many interactions before auto-training
  learning_rate: 5.0e-05            # Training learning rate
  lora_rank: 16                     # LoRA adapter capacity (higher = more expressive, more VRAM)
  num_epochs: 3                     # Training epochs per run

memory:
  rag_enabled: true                 # Enable retrieval-augmented generation
  max_db_size_gb: 10                # Max ChromaDB size before auto-cleanup
  golden_examples_count: 20         # Number of high-value examples to always keep

model:
  n_gpu_layers: -1                  # GPU layers (-1 = all, reduce if out of VRAM)
  temperature: 0.7                  # Generation temperature
  context_length: 16384             # Context window size
```

Restart after changes: `docker-compose restart`

## Project Structure

```
project-rlhfl/
├── services/
│   ├── api/                  # Inference API (FastAPI + Ollama proxy)
│   │   ├── main.py           # API endpoints and request handling
│   │   ├── llm_engine.py     # Ollama client and streaming
│   │   ├── memory_manager.py # ChromaDB interaction storage and RAG
│   │   ├── sentiment_analyzer.py  # Conversation sentiment inference
│   │   └── admin_ui.py       # Admin dashboard endpoints
│   ├── trainer/              # Training pipeline
│   │   ├── trainer_worker.py # Training orchestrator (runs hourly checks)
│   │   ├── training_scheduler.py  # Trigger evaluation logic
│   │   ├── dataset_builder.py     # Builds weighted training datasets
│   │   ├── lora_trainer.py   # QLoRA fine-tuning with PEFT
│   │   └── model_evaluator.py    # Validation and deployment decisions
│   └── shared/               # Shared modules
│       ├── config.py         # System configuration
│       ├── gguf_converter.py # LoRA → GGUF conversion pipeline
│       └── ollama_deployer.py # GGUF → Ollama deployment
├── volumes/
│   ├── models/               # Base model weights (HuggingFace format)
│   ├── data/                 # ChromaDB vector database
│   ├── checkpoints/          # Training checkpoints, adapters, and GGUF files
│   └── config/               # system_config.yaml
├── scripts/                  # Utilities (health check, model download, GGUF conversion)
├── docker-compose.yml        # Service orchestration (API + Trainer + Ollama)
├── Dockerfile.api            # API container
└── Dockerfile.trainer        # Training container
```

### Docker Services

- **llm-api** (port 8000) - FastAPI server that proxies requests to Ollama
- **llm-trainer** - Background training worker with CPU-based LoRA fine-tuning
- **llm-ollama** (port 11434) - GPU-accelerated model serving via Ollama

## Monitoring

### Logs

```bash
docker-compose logs -f api       # FastAPI proxy service
docker-compose logs -f trainer   # Training service
docker-compose logs -f ollama    # Ollama model serving
docker stats                     # Resource usage
```

### Checkpoints

Each training run produces a checkpoint with metadata:

```bash
ls volumes/checkpoints/
```

Checkpoints track perplexity, sentiment alignment, training sample count, and lineage. The system automatically deploys improved checkpoints and rolls back degraded ones.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU not detected | Run `nvidia-smi` to verify drivers. Then `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi` to verify Docker GPU access. |
| Out of memory | Reduce `n_gpu_layers` in config, set training `batch_size: 1`, increase `gradient_accumulation_steps: 8`. |
| Model download fails | Install Git LFS (`git lfs install`), then retry. Or download the GGUF directly from HuggingFace. |
| Training not triggering | Check `curl http://localhost:8000/v1/training/stats` to see interaction count. Lower `min_interactions_threshold` or trigger manually. |
| Slow inference | Ensure `n_gpu_layers: -1` in config (all layers on GPU). Check `docker-compose logs api` for warnings. |

## Security

- All processing happens locally. No external API calls, no telemetry.
- All conversation data stays in `volumes/data/` on your machine.
- No authentication by default. For production use, put a reverse proxy (nginx) with auth in front of port 8000, or bind to `127.0.0.1` only.

## Technical Deep Dive

See [FOR_NERDS.md](FOR_NERDS.md) for architecture diagrams, data flow details, training pipeline internals, and the full API reference.

## Transparency statement from the author
I feel the need to add this statement: 
* Nobody asked for this, it is not sponsored, I am not being paid to develop this project.
* This is purely a "for fun" project that I didn't see any existing solution for that works for _my_ needs. I liked it enough to publish.
* The project was designed and intended to work within _my_ home lab environment, with _my_ hardware on hand. 
* If you want to modify anything to better fit your needs, please fork and submit a PR noting the Cuda Compute Capability level and reference hardware. This will help other folks to find something that works for them or serves as a starting point.

## License

[GNU Affero General Public License v3.0 (AGPL-3.0) ](LICENSE.md)

## Acknowledgments

Built on [Ollama](https://ollama.com/), [llama.cpp](https://github.com/ggerganov/llama.cpp), [PyTorch](https://pytorch.org/), [PEFT](https://github.com/huggingface/peft), [ChromaDB](https://www.trychroma.com/), and [Ministral-3-14B](https://huggingface.co/dphn/Ministral-3-14B) model.

