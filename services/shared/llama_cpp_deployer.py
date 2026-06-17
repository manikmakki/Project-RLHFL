"""
llama.cpp LoRA Deployer

After training completes:
1. Converts the PEFT adapter (safetensors) to GGUF LoRA format via scripts/convert_lora_to_gguf.py
2. Atomically repoints the current_adapter_path symlink at the new GGUF file
3. Restarts the llama-server container (via docker.sock) so it re-reads --lora
4. Verifies the new adapter is active via GET /lora-adapters

llama.cpp's server has no hot-reload for LoRA *content* — confirmed against
the real API (tools/server/README.md): POST /lora-adapters only accepts
{"id", "scale"} and toggles the scale of adapters already loaded into memory
at --lora <path> startup time. It cannot point at a new file at runtime. So
--lora is pinned to a stable symlink (current_adapter_path) and a restart is
what actually picks up new weights.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import docker
import requests

from shared.config import LlamaCppConfig

logger = logging.getLogger(__name__)

# Path to the bundled conversion script. The Dockerfile does `COPY services/ ./`,
# flattening services/shared/ straight into /app/shared/ (not /app/services/shared/),
# so this file's parent.parent is /app, where scripts/ actually lives.
_CONVERT_SCRIPT = Path(__file__).parent.parent / "scripts" / "convert_lora_to_gguf.py"


class LlamaCppDeployer:
    """Deploy LoRA adapters to the sibling llama-server container.

    Since llama.cpp can't hot-load new adapter content via its HTTP API,
    deployment = convert -> repoint the --lora symlink -> restart the
    container -> verify via GET /lora-adapters.
    """

    def __init__(self, config: LlamaCppConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def deploy_lora(self, adapter_path: str, checkpoint_id: str) -> bool:
        """Convert adapter to GGUF and repoint the stable --lora symlink.

        Does NOT restart the server — TrainerWorker stops the server before
        training and starts it after (in a finally block), so the restart
        happens regardless of training success/failure and picks up whichever
        adapter the symlink currently points to.
        """
        logger.info(f"Preparing LoRA adapter for checkpoint {checkpoint_id}")
        logger.info(f"  Adapter dir: {adapter_path}")

        gguf_path = self._convert_to_gguf(adapter_path)
        if not gguf_path:
            logger.error("Conversion failed — deployment aborted")
            return False

        if not self._update_symlink(gguf_path):
            logger.error("Symlink update failed — deployment aborted")
            return False

        logger.info(f"  {self.config.current_adapter_path} -> {gguf_path}")
        return True

    def stop_server(self) -> bool:
        """Stop the llama-server container to free GPU VRAM for training."""
        try:
            client = docker.from_env()
            container = client.containers.get(self.config.container_name)
            logger.info(f"Stopping {self.config.container_name} to free GPU VRAM...")
            container.stop(timeout=30)
            logger.info(f"{self.config.container_name} stopped")
            return True
        except docker.errors.NotFound:
            logger.warning(f"Container '{self.config.container_name}' not found — may already be stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop {self.config.container_name}: {e}", exc_info=True)
            return False

    def start_server(self) -> bool:
        """Start the llama-server container after training completes.

        Picks up whichever adapter the current_adapter_path symlink points to —
        new checkpoint on success, unchanged (rollback) on failure.
        """
        try:
            client = docker.from_env()
            container = client.containers.get(self.config.container_name)
            logger.info(f"Starting {self.config.container_name}...")
            container.start()
            return True
        except docker.errors.NotFound:
            logger.error(f"Container '{self.config.container_name}' not found — cannot start")
            return False
        except Exception as e:
            logger.error(f"Failed to start {self.config.container_name}: {e}", exc_info=True)
            return False

    def verify_deployment(self) -> bool:
        """Confirm the server loaded the current adapter after restart."""
        success = self._verify_lora()
        if success:
            logger.info("=" * 60)
            logger.info("LLAMA.CPP DEPLOYMENT VERIFIED")
            logger.info("=" * 60)
        else:
            logger.warning("Adapter verification failed — server may still be loading")
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
                "Expected this repo's scripts/convert_lora_to_gguf.py (custom script, not from llama.cpp)."
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

    def _update_symlink(self, gguf_path: str) -> bool:
        """Atomically repoint current_adapter_path at the new GGUF file.

        os.replace is atomic on POSIX even when overwriting an existing
        symlink, so there's no window where --lora's target is missing.
        """
        target = Path(self.config.current_adapter_path)
        tmp = target.with_name(target.name + f".tmp{os.getpid()}")
        try:
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
            tmp.symlink_to(gguf_path)
            os.replace(tmp, target)
            return True
        except Exception as e:
            logger.error(f"Failed to repoint {target} -> {gguf_path}: {e}", exc_info=True)
            return False

    def _wait_for_health(self, timeout: float = 120.0, poll_interval: float = 3.0) -> bool:
        """Poll GET /health until the restarted server is ready (model reload
        on a 26B/Q4 GGUF can take a while)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(poll_interval)
        return False

    def _verify_lora(self) -> bool:
        """GET /lora-adapters to confirm the adapter is loaded and active.

        Real API (tools/server/README.md): entries are {"id", "path", "scale"}.
        llama-server reports the path it was launched with via --lora — the
        stable current_adapter_path symlink — not the resolved per-checkpoint
        file, so that's what we compare against. scale 0.0 means loaded but
        inactive, not "this path isn't loaded".
        """
        try:
            response = requests.get(f"{self.base_url}/lora-adapters", timeout=10)
            response.raise_for_status()
            loaded = response.json()
            for entry in (loaded if isinstance(loaded, list) else []):
                if entry.get("path") == self.config.current_adapter_path:
                    scale = float(entry.get("scale", 0))
                    if scale > 0:
                        logger.info(f"Verified: adapter active at scale {scale}")
                        return True
                    logger.warning(f"Adapter loaded but scale={scale} (inactive)")
                    return False
            logger.warning(f"Adapter not found in GET /lora-adapters response: {loaded}")
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
            lora_resp = requests.get(f"{self.base_url}/lora-adapters", timeout=5)
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
