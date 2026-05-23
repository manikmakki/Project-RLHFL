"""
llama.cpp LoRA Deployer

After training completes:
1. Converts the PEFT adapter (safetensors) to GGUF LoRA format via scripts/convert_lora_to_gguf.py
2. Loads the GGUF adapter into the running llama-server via POST /lora
3. Verifies the adapter is active via GET /lora
"""

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests

from shared.config import LlamaCppConfig

logger = logging.getLogger(__name__)

# Path to the bundled conversion script (relative to project root, mounted at /app in container)
_CONVERT_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "convert_lora_to_gguf.py"


class LlamaCppDeployer:
    """Deploy LoRA adapters to an external llama-server via its HTTP /lora API."""

    def __init__(self, config: LlamaCppConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def deploy_lora(self, adapter_path: str, checkpoint_id: str) -> bool:
        """
        Full deployment pipeline: convert adapter → POST /lora → verify.

        adapter_path: directory containing adapter_model.safetensors
        checkpoint_id: used for logging only
        """
        logger.info(f"Deploying LoRA adapter for checkpoint {checkpoint_id}")
        logger.info(f"  Adapter dir:  {adapter_path}")
        logger.info(f"  Server:       {self.base_url}")

        gguf_path = self._convert_to_gguf(adapter_path)
        if not gguf_path:
            logger.error("Conversion failed — deployment aborted")
            return False

        host_path = self._to_host_path(gguf_path)
        logger.info(f"  GGUF path (host): {host_path}")

        success = self._load_lora(host_path)
        if success:
            logger.info("=" * 60)
            logger.info(f"LLAMA.CPP DEPLOYMENT SUCCESSFUL: {checkpoint_id}")
            logger.info("=" * 60)
            self._verify_lora(host_path)
        else:
            logger.error(f"LLAMA.CPP DEPLOYMENT FAILED for {checkpoint_id}")

        return success

    def _convert_to_gguf(self, adapter_path: str) -> Optional[str]:
        """Convert PEFT safetensors adapter to GGUF LoRA format."""
        adapter_dir = Path(adapter_path)
        st_file = adapter_dir / "adapter_model.safetensors"
        if not st_file.exists():
            logger.error(f"adapter_model.safetensors not found at {adapter_dir}")
            return None

        outfile = adapter_dir / "adapter.gguf"
        if outfile.exists():
            logger.info(f"Re-using existing GGUF: {outfile}")
            return str(outfile)

        script = _CONVERT_SCRIPT
        if not script.exists():
            logger.error(
                f"Conversion script not found at {script}. "
                "Place scripts/convert_lora_to_gguf.py from the llama.cpp repo."
            )
            return None

        logger.info(f"Converting {st_file} → {outfile}")
        try:
            result = subprocess.run(
                [sys.executable, str(script), "--adapter-dir", str(adapter_dir), "--outfile", str(outfile)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.stdout:
                for line in result.stdout.strip().splitlines():
                    logger.info(f"  convert: {line}")
            if result.returncode != 0:
                logger.error(f"Conversion failed (exit {result.returncode}): {result.stderr[:500]}")
                return None
            if not outfile.exists():
                logger.error("Conversion script reported success but output file missing")
                return None
            return str(outfile)
        except subprocess.TimeoutExpired:
            logger.error("GGUF conversion timed out (5 min)")
            return None
        except Exception as e:
            logger.error(f"Conversion error: {e}", exc_info=True)
            return None

    def _to_host_path(self, container_path: str) -> str:
        """Translate /checkpoints/... to the host-side path if configured."""
        if self.config.host_checkpoints_path:
            return container_path.replace("/checkpoints", self.config.host_checkpoints_path, 1)
        return container_path

    def _load_lora(self, gguf_path: str) -> bool:
        """POST /lora to load the adapter into the running llama-server."""
        try:
            payload = [{"id": 0, "path": gguf_path, "scale": self.config.lora_scale}]
            response = requests.post(
                f"{self.base_url}/lora",
                json=payload,
                timeout=60,
            )
            if response.status_code == 200:
                logger.info("llama-server accepted the adapter via POST /lora")
                return True
            logger.error(
                f"POST /lora returned {response.status_code}: {response.text[:300]}"
            )
            return False
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot reach llama-server at {self.base_url}. "
                "Ensure llama-server is running on the host with --lora-init-without-apply."
            )
            return False
        except Exception as e:
            logger.error(f"Error posting to /lora: {e}", exc_info=True)
            return False

    def _verify_lora(self, gguf_path: str) -> bool:
        """GET /lora to confirm the adapter is active."""
        try:
            response = requests.get(f"{self.base_url}/lora", timeout=10)
            response.raise_for_status()
            loaded = response.json()
            paths = [entry.get("path", "") for entry in (loaded if isinstance(loaded, list) else [])]
            if gguf_path in paths:
                logger.info(f"Verified: {gguf_path} is active in llama-server")
                return True
            logger.warning(
                f"Adapter not found in GET /lora response (may still be loading): {paths}"
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to verify adapter: {e}")
            return False

    def get_server_status(self) -> dict:
        """Query llama-server health and active adapters for the admin UI."""
        status = {"reachable": False, "healthy": False, "loaded_adapters": []}
        try:
            health = requests.get(f"{self.base_url}/health", timeout=5)
            status["reachable"] = True
            status["healthy"] = health.status_code == 200
            if health.status_code == 200:
                status["health_detail"] = health.json()
        except requests.exceptions.ConnectionError:
            return status
        except Exception as e:
            logger.warning(f"Health check error: {e}")
            return status

        try:
            lora_resp = requests.get(f"{self.base_url}/lora", timeout=5)
            if lora_resp.status_code == 200:
                status["loaded_adapters"] = lora_resp.json()
        except Exception:
            pass

        return status

    def update_config_pointer(self, model_id: str) -> bool:
        """Patch model_id: in system_config.yaml after a successful deployment."""
        try:
            from pathlib import Path as P
            from shared.config import settings
            config_path = P(settings.config_path)
            if not config_path.exists():
                return False

            raw = config_path.read_text()
            old_model = None
            m = re.search(r"^\s*model_id:\s*(.+)$", raw, re.MULTILINE)
            if m:
                old_model = m.group(1).strip()
                raw = re.sub(
                    r"^(\s*model_id:\s*).*$",
                    rf"\g<1>{model_id}",
                    raw,
                    flags=re.MULTILINE,
                )
            else:
                raw = re.sub(
                    r"(^model:.*?)(\n\S|\Z)",
                    rf"\1\n  model_id: {model_id}\2",
                    raw,
                    flags=re.MULTILINE | re.DOTALL,
                )

            config_path.write_text(raw)
            logger.info(f"Config updated: {old_model} → {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update config pointer: {e}")
            return False

    def rotate_checkpoints(self, keep_count: int = 3) -> None:
        """Keep only the N most recent checkpoint directories."""
        try:
            from shared.config import settings
            checkpoints_dir = Path(settings.checkpoints_path)
            if not checkpoints_dir.exists():
                return

            checkpoints = sorted(
                [d for d in checkpoints_dir.iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint_")],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            to_delete = checkpoints[keep_count:]
            for cp in to_delete:
                logger.info(f"Removing old checkpoint: {cp.name}")
                shutil.rmtree(cp)

            if to_delete:
                logger.info(f"Cleaned up {len(to_delete)} old checkpoints")
        except Exception as e:
            logger.warning(f"Checkpoint rotation failed: {e}")
