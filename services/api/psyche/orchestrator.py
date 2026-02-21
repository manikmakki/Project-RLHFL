"""
Psyche Module - Orchestrator

Central coordinator that wires Id, Ego, and Superego together.
Provides a single interface for main.py to call pre_process() and post_process().

Each layer is independently toggle-able and wrapped in try/except for
graceful degradation. If any layer fails, processing continues without it.
"""

import logging
from typing import Optional, List, Dict, Any

from shared.config import SystemConfig
from shared.models import ChatMessage
from api.psyche.models import (
    PreProcessResult,
    PostProcessResult,
    ValidationResult,
    RewardContext,
    EgoParseResult,
)
from api.psyche.superego import Superego

logger = logging.getLogger(__name__)


class PsycheOrchestrator:
    """
    Coordinates the Id/Ego/Superego layers for each request.

    Pre-process flow (before LLM call):
        1. Superego.judge_request() -- abort if objectively dangerous
        2. Superego.inject_framing_into_messages() -- permissive framing
        3. Id.compute_reward_context() -- reward signals (Phase 3)
        4. Ego.enrich_system_prompt() -- reasoning template (Phase 2)

    Post-process flow (after LLM response):
        1. Superego.detect_refusal() -- check for model refusals
        2. Ego.parse_response() -- extract thinking/reflection (Phase 2)
        3. Id.record_outcome() -- update reward buffer (Phase 3)
    """

    def __init__(
        self,
        config: SystemConfig,
        superego: Optional[Superego] = None,
        id_engine=None,  # Phase 3
        ego_fast=None,    # Phase 2
        ego_slow=None,    # Phase 4
    ):
        self.config = config
        self.psyche_config = config.psyche
        self.superego = superego
        self.id_engine = id_engine
        self.ego_fast = ego_fast
        self.ego_slow = ego_slow
        self._enabled = config.psyche.enabled

        logger.info(
            f"PsycheOrchestrator initialized: enabled={self._enabled}, "
            f"superego={'yes' if superego else 'no'}, "
            f"id={'yes' if id_engine else 'no'}, "
            f"ego_fast={'yes' if ego_fast else 'no'}, "
            f"ego_slow={'yes' if ego_slow else 'no'}"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self.psyche_config.enabled

    async def pre_process(
        self,
        messages: List[ChatMessage],
        user_message_text: str,
        conversation_id: str,
    ) -> PreProcessResult:
        """
        Run all layers before the LLM call.

        Args:
            messages: Conversation messages (already normalized + intercepted)
            user_message_text: Extracted text of the latest user message
            conversation_id: Conversation identifier

        Returns:
            PreProcessResult with enriched messages and metadata
        """
        if not self.enabled:
            return PreProcessResult(
                messages=messages,
                validation=ValidationResult(allowed=True, reason="psyche disabled"),
            )

        result_messages = list(messages)
        validation = ValidationResult(allowed=True)
        reward_context = None
        permissive_framing_applied = False
        metadata: Dict[str, Any] = {}

        # 1. Superego: judge request for objective danger
        if self.superego and self.superego.enabled:
            try:
                judgment = await self.superego.judge_request(
                    messages=result_messages,
                    user_message_text=user_message_text,
                )
                validation = ValidationResult(
                    allowed=judgment.allowed,
                    reason=judgment.reason,
                    judgment=judgment,
                )
                metadata["superego_judgment"] = judgment.model_dump()

                if not judgment.allowed:
                    logger.warning(f"Superego blocked request: {judgment.reason}")
                    return PreProcessResult(
                        messages=result_messages,
                        validation=validation,
                        metadata=metadata,
                    )

                # Inject permissive framing (always, unless disabled)
                result_messages = self.superego.inject_framing_into_messages(
                    result_messages, strong=False
                )
                permissive_framing_applied = True

            except Exception as e:
                logger.error(f"Superego pre-process failed, continuing without it: {e}")
                metadata["superego_error"] = str(e)

        # 2. Id Engine: compute reward context (Phase 3 -- stub)
        if self.id_engine:
            try:
                reward_context = await self.id_engine.compute_reward_context(
                    conversation_id=conversation_id,
                    messages=result_messages,
                )
                metadata["reward_context"] = reward_context.model_dump()
            except Exception as e:
                logger.error(f"Id Engine pre-process failed, continuing without it: {e}")
                metadata["id_error"] = str(e)

        # 3. Ego Fast: enrich system prompt (Phase 2 -- stub)
        if self.ego_fast:
            try:
                result_messages = await self.ego_fast.enrich_system_prompt(
                    messages=result_messages,
                    reward_context=reward_context,
                )
            except Exception as e:
                logger.error(f"Ego Fast pre-process failed, continuing without it: {e}")
                metadata["ego_fast_error"] = str(e)

        return PreProcessResult(
            messages=result_messages,
            reward_context=reward_context,
            validation=validation,
            permissive_framing_applied=permissive_framing_applied,
            metadata=metadata,
        )

    async def post_process(
        self,
        response_dict: Dict[str, Any],
        conversation_id: str,
        pre_result: PreProcessResult,
        messages: List[ChatMessage],
        llm_proxy=None,
    ) -> PostProcessResult:
        """
        Run layers after the LLM response.

        Args:
            response_dict: Raw response from LLMProxy.generate()
            conversation_id: Conversation identifier
            pre_result: Result from pre_process()
            messages: Original messages (pre-enrichment, for retry)
            llm_proxy: LLMProxy instance (needed for refusal retry)

        Returns:
            PostProcessResult with validated/enriched response
        """
        if not self.enabled:
            return PostProcessResult(
                response_dict=response_dict,
                validation=ValidationResult(allowed=True, reason="psyche disabled"),
            )

        validation = ValidationResult(allowed=True)
        ego_parse = None
        refusal_detected = False
        retry_attempted = False
        metadata: Dict[str, Any] = {}

        response_text = response_dict.get("content") or ""

        logger.info(
            f"Psyche post-process starting: response_length={len(response_text)}, "
            f"first_100_chars={response_text[:100]!r}"
        )

        # 1. Superego: detect refusal and retry if needed
        if self.superego and self.superego.enabled:
            try:
                refusal_detected = self.superego.detect_refusal(response_text)
                metadata["refusal_detected"] = refusal_detected
                logger.info(f"Superego refusal detection result: {refusal_detected}")

                if refusal_detected and not llm_proxy:
                    logger.warning("Refusal detected but no llm_proxy available for retry")
                elif refusal_detected and not self.psyche_config.superego_refusal_retry:
                    logger.warning("Refusal detected but superego_refusal_retry is disabled in config")

                if refusal_detected and llm_proxy and self.psyche_config.superego_refusal_retry:
                    logger.info("Superego detected refusal, retrying with stronger framing")
                    retry_attempted = True
                    # Preserve original refusal for DPO training
                    metadata["original_refusal"] = response_text

                    # Rebuild messages with strong permissive framing
                    retry_messages = self.superego.inject_framing_into_messages(
                        messages, strong=True
                    )

                    try:
                        retry_response = await llm_proxy.generate(
                            messages=retry_messages,
                            temperature=0.7,
                        )
                        retry_text = retry_response.get("content") or ""

                        # Check if retry also refused
                        if not self.superego.detect_refusal(retry_text):
                            logger.info("Superego retry succeeded -- using retry response")
                            response_dict = retry_response
                            response_text = retry_text
                            metadata["retry_succeeded"] = True
                            metadata["retry_response"] = retry_text
                        else:
                            logger.info("Superego retry also refused -- accepting original refusal")
                            metadata["retry_succeeded"] = False

                    except Exception as e:
                        logger.error(f"Superego retry failed: {e}")
                        metadata["retry_error"] = str(e)

            except Exception as e:
                logger.error(f"Superego post-process failed, continuing without it: {e}")
                metadata["superego_error"] = str(e)

        # 2. Ego Fast: parse response for thinking/reflection (Phase 2 -- stub)
        if self.ego_fast:
            try:
                ego_parse = await self.ego_fast.parse_response(response_dict)
                if ego_parse:
                    metadata["ego_parse"] = ego_parse.model_dump()
            except Exception as e:
                logger.error(f"Ego Fast post-process failed, continuing without it: {e}")
                metadata["ego_fast_error"] = str(e)

        # 3. Id Engine: record outcome (Phase 3 -- stub)
        if self.id_engine:
            try:
                await self.id_engine.record_outcome(
                    conversation_id=conversation_id,
                    response_text=response_text,
                )
            except Exception as e:
                logger.error(f"Id Engine post-process failed, continuing without it: {e}")
                metadata["id_error"] = str(e)

        return PostProcessResult(
            response_dict=response_dict,
            ego_parse=ego_parse,
            validation=validation,
            refusal_detected=refusal_detected,
            retry_attempted=retry_attempted,
            metadata=metadata,
        )

    def is_healthy(self) -> Dict[str, Any]:
        """Return health status of all layers."""
        status = {
            "enabled": self.enabled,
            "superego": self.superego.is_healthy() if self.superego else None,
            "id_engine": self.id_engine.is_healthy() if self.id_engine else None,
            "ego_fast": self.ego_fast.is_healthy() if self.ego_fast else None,
            "ego_slow": self.ego_slow.is_healthy() if self.ego_slow else None,
        }
        return status
