"""
Ollama Model Deployer

Handles deployment of fine-tuned models to Ollama with hot-reload:
1. Takes merged HuggingFace model
2. Converts to GGUF format
3. Creates Modelfile
4. Deploys to Ollama (replacing existing model)
"""

import os
import logging
import subprocess
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class OllamaDeployer:
    """Deploy fine-tuned models to Ollama."""

    def __init__(self, ollama_base_url: str = "http://llm-ollama:11434"):
        self.ollama_base_url = ollama_base_url
        self.ollama_api = f"{ollama_base_url}/api"

    def deploy_model(
        self,
        gguf_path: str,
        model_name: str = "gpt-oss:20b",
        checkpoint_id: str = None
    ) -> bool:
        """
        Deploy a GGUF model to Ollama.

        Args:
            gguf_path: Path to GGUF model file (already converted)
            model_name: Ollama model name to create/replace
            checkpoint_id: Optional checkpoint ID for tracking

        Returns:
            bool: True if deployment successful
        """
        try:
            logger.info(f"Starting Ollama deployment for {model_name}")
            logger.info(f"Source GGUF: {gguf_path}")

            # Step 1: Create Modelfile
            modelfile_path = self._create_modelfile(gguf_path, model_name, checkpoint_id)

            # Step 2: Deploy to Ollama (replaces existing model)
            success = self._create_ollama_model(modelfile_path, model_name)

            if success:
                logger.info("=" * 80)
                logger.info("OLLAMA DEPLOYMENT SUCCESSFUL")
                logger.info("=" * 80)
                logger.info(f"Model name: {model_name}")
                logger.info(f"Checkpoint: {checkpoint_id}")
                logger.info(f"GGUF path: {gguf_path}")
                logger.info("Clients can continue using the same endpoint")
                logger.info("=" * 80)

                # Step 3: Verify model is loaded
                self._verify_model(model_name)
            else:
                logger.error("=" * 80)
                logger.error("OLLAMA DEPLOYMENT FAILED")
                logger.error("=" * 80)
                logger.error(f"Model creation returned failure")
                logger.error(f"Model name: {model_name}")
                logger.error(f"Check ollama create output above for details")
                logger.error("=" * 80)

            return success

        except Exception as e:
            logger.error("=" * 80)
            logger.error("OLLAMA DEPLOYMENT FAILED")
            logger.error("=" * 80)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {e}")
            logger.error(f"Model name: {model_name}")
            logger.error(f"GGUF path: {gguf_path}")
            logger.error(f"Checkpoint ID: {checkpoint_id}")
            logger.error("=" * 80)
            logger.error("Full traceback:", exc_info=True)
            logger.error("=" * 80)
            return False

    def _convert_to_gguf(
        self,
        hf_model_path: str,
        checkpoint_id: str
    ) -> Optional[str]:
        """
        Convert HuggingFace model to GGUF format using our GPT-OSS converter.
        """
        from shared.gguf_converter import GGUFConverter, GGUFConversionError

        logger.info(f"Converting {checkpoint_id} to GGUF format...")

        try:
            # Prepare paths
            merged_model_dir = f"/checkpoints/{checkpoint_id}/merged_model"
            gguf_output_path = f"/checkpoints/{checkpoint_id}/{checkpoint_id}.gguf"

            # Use existing GGUF converter (handles merge + convert + quantize)
            base_model_path = "/models/gpt-oss-20b-base"
            converter = GGUFConverter(base_model_path)

            # Note: hf_model_path is already merged from sequential training
            # The adapter_path is actually the merged model path
            gguf_path = converter.convert_adapter_to_gguf(
                adapter_path=hf_model_path,
                merged_model_dir=merged_model_dir,
                output_gguf_path=gguf_output_path,
                quantization_type="Q4_K_M",
                lora_trainer=None  # Already merged
            )

            logger.info(f"✓ GGUF conversion complete: {gguf_path}")
            return gguf_path

        except GGUFConversionError as e:
            logger.error(f"GGUF conversion failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during GGUF conversion: {e}", exc_info=True)
            return None

    def _create_modelfile(
        self,
        gguf_path: str,
        model_name: str,
        checkpoint_id: str
    ) -> str:
        """
        Create Ollama Modelfile for the fine-tuned model.

        The Modelfile preserves the original template and parameters
        while using the new GGUF weights.
        """
        modelfile_dir = Path("/checkpoints") / f"modelfile_{checkpoint_id}"
        modelfile_dir.mkdir(parents=True, exist_ok=True)
        modelfile_path = modelfile_dir / "Modelfile"

        # Create Modelfile (inherits template from base gpt-oss:20b)
        modelfile_content = f"""# Fine-tuned {model_name}
# Checkpoint: {checkpoint_id}
# Auto-generated by RLHFL trainer

FROM {gguf_path}

# Inherit template and parameters from base model
# (Temperature, system prompt, etc. remain the same)
"""

        modelfile_path.write_text(modelfile_content)
        logger.info(f"Created Modelfile: {modelfile_path}")

        return str(modelfile_path)

    def _create_ollama_model(
        self,
        modelfile_path: str,
        model_name: str
    ) -> bool:
        """
        Create/replace Ollama model using the Modelfile.

        This uses 'ollama create' which:
        - Creates the model if it doesn't exist
        - Replaces it if it does exist
        - Automatically handles blob storage
        - Triggers model reload
        """
        try:
            logger.info(f"Creating Ollama model: {model_name}")

            # Step 1: Force unload existing model if loaded
            self._unload_model(model_name)

            # Step 2: Create/replace model using ollama CLI via docker exec
            # (HTTP API has issues with modelfile parsing, CLI works reliably)
            cmd = [
                "docker", "exec", "llm-ollama",
                "ollama", "create", model_name,
                "-f", modelfile_path
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            logger.info(f"Ollama create output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Ollama create stderr: {result.stderr}")

            logger.info(f"✓ Ollama model {model_name} created/updated successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"ollama create failed: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("ollama create timed out")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model creation: {e}")
            return False

    def _unload_model(self, model_name: str) -> None:
        """Force unload a model from Ollama memory."""
        try:
            logger.info(f"Unloading model: {model_name}")

            # Use Ollama's delete endpoint to unload (doesn't delete the model)
            # Or use the unload API if available
            response = requests.post(
                f"{self.ollama_api}/unload",
                json={"name": model_name},
                timeout=30
            )

            # Unload endpoint might not exist in all versions, that's ok
            if response.status_code == 200:
                logger.info(f"✓ Model {model_name} unloaded")
            else:
                logger.debug(f"Unload response: {response.status_code} (model may not be loaded)")

        except requests.exceptions.RequestException as e:
            # Unload is best-effort, not critical
            logger.debug(f"Could not unload model (this is ok): {e}")

    def _verify_model(self, model_name: str) -> bool:
        """Verify the model is available in Ollama."""
        try:
            response = requests.get(
                f"{self.ollama_api}/tags",
                timeout=10
            )
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]

            if model_name in model_names:
                logger.info(f"✓ Model {model_name} verified in Ollama")
                return True
            else:
                logger.warning(f"Model {model_name} not found in Ollama")
                return False

        except Exception as e:
            logger.warning(f"Failed to verify model: {e}")
            return False

    def list_models(self) -> list:
        """List all models in Ollama."""
        try:
            response = requests.get(
                f"{self.ollama_api}/tags",
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


    def update_config_pointer(self, model_name: str) -> bool:
        """
        Update system config to point to the new model.

        This updates the external_model_name in the config so the API
        automatically routes to the fine-tuned model.
        """
        try:
            import yaml
            config_path = Path("/config/system_config.yaml")

            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                return False

            # Read config
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Update model pointer
            old_model = config.get("llm_proxy", {}).get("external_model_name")
            config.setdefault("llm_proxy", {})["external_model_name"] = model_name

            # Write back
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"✓ Config updated: {old_model} → {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to update config: {e}", exc_info=True)
            return False

    def rotate_checkpoints(self, keep_count: int = 2) -> None:
        """
        Keep only the N most recent checkpoints.

        Args:
            keep_count: Number of checkpoints to keep (default: 2 for current + previous)
        """
        try:
            checkpoints_dir = Path("/checkpoints")
            if not checkpoints_dir.exists():
                return

            # Find all checkpoint directories (sorted by modification time, newest first)
            checkpoints = sorted(
                [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Keep only the most recent N checkpoints
            to_delete = checkpoints[keep_count:]

            for checkpoint in to_delete:
                logger.info(f"Removing old checkpoint: {checkpoint.name}")
                shutil.rmtree(checkpoint)

            if to_delete:
                logger.info(f"✓ Cleaned up {len(to_delete)} old checkpoints")

        except Exception as e:
            logger.warning(f"Checkpoint rotation failed: {e}")


if __name__ == "__main__":
    # Test deployment
    logging.basicConfig(level=logging.INFO)
    deployer = OllamaDeployer()

    # List current models
    models = deployer.list_models()
    print(f"Current Ollama models: {[m['name'] for m in models]}")
