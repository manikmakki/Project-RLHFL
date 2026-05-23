# LoRA Adapter Deployment — Quick Reference

RLHFL trains LoRA adapters (not full models). After training, the adapter is
converted to GGUF LoRA format and hot-loaded into the running llama-server via
its `/lora` API — no model restart required.

## Automatic Flow (default)

When `enable_deployment: true` in `system_config.yaml`, the trainer does this
automatically after every successful training run:

1. PEFT adapter (`adapter_model.safetensors`) is written to the checkpoint dir
2. `scripts/convert_lora_to_gguf.py` converts it to `adapter.gguf`
3. `POST /lora` is called on the llama-server to hot-load the adapter

No manual steps needed.

## Manual Conversion

```bash
# Inside the running container:
docker exec -it rlhfl python3 /app/scripts/convert_lora_to_gguf.py \
  /checkpoints/checkpoint_20260217_050631/adapter_model.safetensors \
  --out /checkpoints/checkpoint_20260217_050631/adapter.gguf
```

## Manual Deployment via Admin UI

Open `http://localhost:8000/admin/` → Checkpoints → **Deploy to llama.cpp**

Or via API:

```bash
curl -X POST "http://localhost:8000/admin/api/model/deploy-lora?checkpoint_id=checkpoint_20260217_050631"
```

## Manual Deployment via llama-server API

```bash
# Load adapter (path must be accessible from the host running llama-server):
curl -X POST http://localhost:8080/lora \
  -H "Content-Type: application/json" \
  -d '[{"id": 0, "path": "/checkpoints/checkpoint_20260217_050631/adapter.gguf", "scale": 1.0}]'

# Check loaded adapters:
curl http://localhost:8080/lora

# Unload all adapters (revert to base model):
curl -X POST http://localhost:8080/lora \
  -H "Content-Type: application/json" \
  -d '[]'
```

## Path Note

The llama-server runs on the host, not in the container. The `adapter.gguf`
path sent to `/lora` must be the **host path**. If your host mounts the
checkpoints volume at a different path than `/checkpoints`, set
`host_checkpoints_path` in `system_config.yaml`:

```yaml
llama_cpp:
  host_checkpoints_path: /opt/project-rlhfl/volumes/checkpoints
```

## Checking Server Status

```bash
curl http://localhost:8000/admin/api/model/server-status
# {"reachable": true, "healthy": true, "loaded_adapters": [...]}
```

## Common Issues

### Adapter not loading

Check that the GGUF LoRA file exists and the path is correct for the host:

```bash
ls -lh /opt/project-rlhfl/volumes/checkpoints/checkpoint_*/adapter.gguf
```

### Conversion fails

The `gguf` package must be installed (it is in `requirements.txt`). Verify:

```bash
docker exec -it rlhfl python3 -c "import gguf; print(gguf.__version__)"
```

### llama-server not reachable

The server is expected at `http://host.docker.internal:8080` by default.
Check `base_url` in `system_config.yaml` and verify the server is running:

```bash
curl http://localhost:8080/health
```
