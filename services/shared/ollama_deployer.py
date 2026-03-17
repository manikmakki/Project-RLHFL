"""
Ollama Model Deployer

Deploys fine-tuned LoRA adapters to Ollama via adapter layering:
1. Uses Ollama HTTP API (POST /api/create) with 'from' + 'files' parameters
2. Layers the adapter safetensors on the existing base model
3. Verifies deployment via /api/tags
"""

import json
import logging
import shutil
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class OllamaDeployer:
    """Deploy fine-tuned LoRA adapters to Ollama via HTTP API."""

    def __init__(self, ollama_base_url: str = "http://llm-ollama:11434"):
        self.ollama_base_url = ollama_base_url
        self.ollama_api = f"{ollama_base_url}/api"

    def deploy_model(
        self,
        adapter_path: str,
        model_name: str = None,
        base_model_name: str = None,
        checkpoint_id: str = None,
    ) -> bool:
        """
        Deploy a LoRA adapter to Ollama by layering it on the base model.

        Uses Ollama's /api/create with 'from' (base model) and 'files'
        (adapter safetensors) to apply the adapter without a full model merge.

        Args:
            adapter_path: Path to the LoRA adapter directory (contains adapter_model.safetensors)
            model_name: Ollama model name to create/replace
            base_model_name: Base model already in Ollama (from parameter)
            checkpoint_id: Optional checkpoint ID for tracking
        """
        try:
            # Resolve adapter safetensors file
            adapter_dir = Path(adapter_path)
            adapter_file = adapter_dir / "adapter_model.safetensors"
            if not adapter_file.exists():
                logger.error(f"Adapter file not found: {adapter_file}")
                return False

            logger.info(
                f"Deploying {model_name} with adapter from {adapter_file} "
                f"(base: {base_model_name}, checkpoint: {checkpoint_id})"
            )

            # Unload existing model from memory
            self._unload_model(model_name)

            # Create model via HTTP API
            success = self._create_ollama_model(
                model_name=model_name,
                base_model_name=base_model_name,
                adapter_file=str(adapter_file),
            )

            if success:
                logger.info("=" * 60)
                logger.info(f"OLLAMA DEPLOYMENT SUCCESSFUL: {model_name}")
                logger.info(f"Checkpoint: {checkpoint_id}")
                logger.info("=" * 60)
                self._verify_model(model_name)
            else:
                logger.error(f"OLLAMA DEPLOYMENT FAILED for {model_name}")

            return success

        except Exception as e:
            logger.error(f"Deployment error: {e}", exc_info=True)
            return False

    def _create_ollama_model(
        self, model_name: str, base_model_name: str, adapter_file: str
    ) -> bool:
        """Create/replace Ollama model via HTTP API with adapter layering."""
        try:
            logger.info(f"Creating Ollama model via API: {model_name}")
            logger.info(f"  from: {base_model_name}")
            logger.info(f"  adapter: {adapter_file}")

            response = requests.post(
                f"{self.ollama_api}/create",
                json={
                    "model": model_name,
                    "from": base_model_name,
                    "files": {
                        "adapter": adapter_file,
                    },
                    "stream": True,
                },
                stream=True,
                timeout=1800,  # 30 minute timeout for adapter application
            )
            response.raise_for_status()

            # Stream progress from NDJSON response
            last_status = None
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if status and status != last_status:
                        logger.info(f"ollama create: {status}")
                        last_status = status
                    if data.get("error"):
                        logger.error(f"ollama create error: {data['error']}")
                        return False
                except json.JSONDecodeError:
                    logger.debug(f"ollama create non-JSON line: {line}")

            if last_status == "success":
                return True

            logger.error(f"ollama create did not report success (last status: {last_status})")
            return False

        except requests.exceptions.Timeout:
            logger.error("ollama create timed out (30 min)")
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"ollama create HTTP error: {e.response.status_code} {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model creation: {e}", exc_info=True)
            return False

    def _unload_model(self, model_name: str) -> None:
        """Force unload a model from Ollama memory."""
        try:
            response = requests.post(
                f"{self.ollama_api}/generate",
                json={"model": model_name, "keep_alive": "0"},
                timeout=30,
            )
            if response.status_code == 200:
                logger.info(f"Model {model_name} unloaded")
            else:
                logger.debug(f"Unload response: {response.status_code}")
        except requests.exceptions.RequestException:
            pass  # Best-effort

    def _verify_model(self, model_name: str) -> bool:
        """Verify the model is available in Ollama."""
        try:
            response = requests.get(f"{self.ollama_api}/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            if model_name in model_names:
                logger.info(f"Model {model_name} verified in Ollama")
                return True
            logger.warning(f"Model {model_name} not found in Ollama tags")
            return False
        except Exception as e:
            logger.warning(f"Failed to verify model: {e}")
            return False

    def update_config_pointer(self, model_name: str) -> bool:
        """Update system config to point to the new model."""
        try:
            import yaml
            config_path = Path("/config/system_config.yaml")
            if not config_path.exists():
                return False

            with open(config_path) as f:
                raw = f.read()

            import re
            old_model = None
            m = re.search(r"^\s*external_model_name:\s*(.+)$", raw, re.MULTILINE)
            if m:
                old_model = m.group(1).strip()
                raw = re.sub(
                    r"^(\s*external_model_name:\s*).*$",
                    rf"\g<1>{model_name}",
                    raw,
                    flags=re.MULTILINE,
                )
            else:
                # Key absent — append under llm_proxy block
                raw = re.sub(
                    r"(^llm_proxy:.*?)(\n\S|\Z)",
                    rf"\1\n  external_model_name: {model_name}\2",
                    raw,
                    flags=re.MULTILINE | re.DOTALL,
                )

            with open(config_path, "w") as f:
                f.write(raw)

            logger.info(f"Config updated: {old_model} -> {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False

    def rotate_checkpoints(self, keep_count: int = 3) -> None:
        """Keep only the N most recent checkpoint directories."""
        try:
            checkpoints_dir = Path("/checkpoints")
            if not checkpoints_dir.exists():
                return

            checkpoints = sorted(
                [d for d in checkpoints_dir.iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint_")],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            to_delete = checkpoints[keep_count:]
            for checkpoint in to_delete:
                logger.info(f"Removing old checkpoint: {checkpoint.name}")
                shutil.rmtree(checkpoint)

            if to_delete:
                logger.info(f"Cleaned up {len(to_delete)} old checkpoints")
        except Exception as e:
            logger.warning(f"Checkpoint rotation failed: {e}")

    def list_models(self) -> list:
        """List all models in Ollama."""
        try:
            response = requests.get(f"{self.ollama_api}/tags", timeout=10)
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
