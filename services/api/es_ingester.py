import asyncio
import logging
from datetime import datetime, time as dt_time
from pathlib import Path

import pytz

from api.es_memory_manager import ElasticsearchMemoryManager
from api.llm_evaluator import LLMEvaluator
from shared.config import SystemConfig, EvaluationWindowConfig
from shared.config import settings as app_settings

logger = logging.getLogger(__name__)


def _within_window(window: EvaluationWindowConfig) -> bool:
    """Return True if the current local time is within the evaluation window."""
    try:
        tz = pytz.timezone(window.timezone)
        now = datetime.now(tz).time()

        start_h, start_m = map(int, window.start.split(":"))
        end_h, end_m = map(int, window.end.split(":"))
        start = dt_time(start_h, start_m)
        end = dt_time(end_h, end_m)

        if start <= end:
            return start <= now < end
        # Overnight window (e.g. 23:00 – 03:00)
        return now >= start or now < end
    except Exception as e:
        logger.warning(f"Could not determine evaluation window, defaulting to active: {e}")
        return True


class ElasticsearchIngester:
    """
    Background asyncio task that evaluates unevaluated TamAGI conversations
    during the configured off-hours window and writes rlhfl_* fields back.
    """

    def __init__(self, config: SystemConfig, memory_manager: ElasticsearchMemoryManager):
        self.config = config
        self.es_config = config.elasticsearch
        self.memory_manager = memory_manager
        self.evaluator = LLMEvaluator(config)
        self._task: asyncio.Task | None = None

    def start(self):
        """Schedule the ingester loop as a background asyncio task."""
        self._task = asyncio.create_task(self._loop(), name="es_ingester")
        logger.info("ES ingester started")

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("ES ingester stopped")

    def _training_active(self) -> bool:
        return (Path(app_settings.data_path) / "training.lock").exists()

    async def _loop(self):
        """Poll for unevaluated docs whenever inside the evaluation window."""
        check_interval_seconds = 300  # re-check window every 5 minutes when idle
        while True:
            try:
                if self._training_active():
                    logger.info("ES ingester: training in progress, pausing evaluation")
                    await asyncio.sleep(60)
                    continue

                if _within_window(self.es_config.evaluation_window):
                    processed = await asyncio.get_event_loop().run_in_executor(
                        None, self._run_batch
                    )
                    if processed == 0:
                        logger.info("ES ingester: backlog exhausted, sleeping until next window check")
                        await asyncio.sleep(check_interval_seconds)
                    else:
                        # More docs may remain — yield briefly then continue
                        await asyncio.sleep(1)
                else:
                    await asyncio.sleep(check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ES ingester loop error: {e}", exc_info=True)
                await asyncio.sleep(60)

    def _run_batch(self) -> int:
        """
        Fetch one batch of unevaluated docs, evaluate each, and write results back.
        Returns number of successfully evaluated docs.
        """
        hits = self.memory_manager.get_unevaluated_interactions(
            limit=self.es_config.batch_size
        )
        if not hits:
            return 0

        evaluated = 0
        skipped = 0

        for hit in hits:
            doc_id = hit["_id"]
            content = hit.get("_source", {}).get("data", {}).get("content", "")

            if not content:
                logger.debug(f"Skipping doc {doc_id}: empty content")
                skipped += 1
                continue

            evaluation = self.evaluator.evaluate(content)
            if evaluation is None:
                logger.warning(f"Evaluation failed for doc {doc_id}, will retry next window")
                skipped += 1
                continue

            if self.memory_manager.update_with_evaluation(doc_id, evaluation):
                evaluated += 1
                logger.debug(
                    f"Evaluated {doc_id}: quality={evaluation['rlhfl_quality']} "
                    f"sentiment={evaluation['rlhfl_sentiment']} "
                    f"golden={evaluation['rlhfl_is_golden']} "
                    f"noise={evaluation['rlhfl_is_noise']}"
                )
            else:
                skipped += 1

        logger.info(
            f"ES ingester batch complete: {evaluated} evaluated, {skipped} skipped "
            f"(of {len(hits)} fetched)"
        )
        return evaluated
