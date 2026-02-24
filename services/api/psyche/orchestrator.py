"""
Psyche Module - Orchestrator

Central coordinator for the adversarial multi-model refinement pipeline.

Psyche v2 replaces the v1 pre_process/post_process pattern with a single
refine() method that runs the full Id → Ego → Superego loop, returning the
best response after quality gating.
"""

import logging
from typing import Optional, List, Dict, Any

from shared.config import SystemConfig
from shared.models import ChatMessage, MessageRole
from api.psyche.models import (
    RefinementResult,
    RefinementTurn,
    ValidationResult,
    RewardContext,
    # Legacy models (used when refinement_enabled=False)
    PreProcessResult,
    PostProcessResult,
    EgoParseResult,
)
from api.psyche.superego import Superego
from api.psyche.id_engine import IdEngine
from api.psyche.complexity import ComplexityClassifier

logger = logging.getLogger(__name__)


class PsycheOrchestrator:
    """
    Coordinates the adversarial refinement pipeline.

    refine() is the primary entry point. It:
    1. Sanitizes client params (if override_client_params=True)
    2. Runs safety check (Superego danger patterns / LLM judge)
    3. Classifies complexity → turn budget
    4. Fast path (budget=1): single Ego generation
    5. Refinement loop: Id → Ego → Superego → Gate → (retry or accept)
    6. Updates Superego global profile
    7. Returns RefinementResult with best response + metadata
    """

    def __init__(
        self,
        config: SystemConfig,
        superego: Optional[Superego] = None,
        id_engine: Optional[IdEngine] = None,
        ego_fast=None,
        ego_slow=None,
    ):
        self.config = config
        self.psyche_config = config.psyche
        self.superego = superego
        self.id_engine = id_engine
        self.ego_fast = ego_fast
        self.ego_slow = ego_slow
        self._enabled = config.psyche.enabled

        # Complexity classifier (heuristic, no model)
        self.classifier = ComplexityClassifier(config)

        logger.info(
            f"PsycheOrchestrator initialized: enabled={self._enabled}, "
            f"refinement={'on' if config.psyche.refinement_enabled else 'off'}, "
            f"superego={'yes' if superego else 'no'}, "
            f"id={'yes' if id_engine else 'no'}, "
            f"ego_model={config.psyche.ego_model or '(main)'}"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self.psyche_config.enabled

    @property
    def refinement_enabled(self) -> bool:
        return self.psyche_config.refinement_enabled and self.enabled

    # ─── Primary Entry Point ───────────────────────────────────────────

    async def refine(
        self,
        messages: List[ChatMessage],
        user_message_text: str,
        conversation_id: str,
        llm_proxy,
        has_tools: bool = False,
    ) -> RefinementResult:
        """
        Run the full adversarial refinement pipeline.

        Args:
            messages: Conversation messages (normalized + intercepted)
            user_message_text: Extracted user message text
            conversation_id: Conversation identifier
            llm_proxy: LLMProxy instance for generation
            has_tools: Whether the request includes tool definitions

        Returns:
            RefinementResult with best response and full metadata
        """
        if not self.enabled:
            # Psyche disabled — direct generation
            response = await llm_proxy.generate(messages=messages)
            return RefinementResult(
                best_response_dict=response,
                validation=ValidationResult(allowed=True, reason="psyche disabled"),
            )

        deliberation_lines = []

        # 1. Safety check (fast-path patterns, no model call for simple cases)
        if self.superego and self.superego.enabled:
            try:
                judgment = await self.superego.judge_request(messages, user_message_text)
                if not judgment.allowed:
                    logger.warning(f"Superego blocked request: {judgment.reason}")
                    return RefinementResult(
                        validation=ValidationResult(
                            allowed=False, reason=judgment.reason, judgment=judgment
                        ),
                    )
            except Exception as e:
                logger.error(f"Safety check failed, continuing: {e}")

        # 2. Inject permissive framing (works regardless of safety layer state)
        working_messages = list(messages)
        if self.superego:
            working_messages = self.superego.inject_framing_into_messages(working_messages)

        # 3. Classify complexity
        turn_budget, quality_threshold, complexity_class = self.classifier.classify(
            user_message_text, has_tools=has_tools
        )

        # Use Superego's profile threshold if it has evolved
        if self.superego:
            quality_threshold = max(quality_threshold, self.superego.profile.quality_threshold)

        deliberation_lines.append(
            f"[CLASSIFIER] complexity={complexity_class}, budget={turn_budget}, threshold={quality_threshold:.2f}"
        )

        # 4. Compute Id desire signal (once, before the loop)
        reward_context = None
        if self.id_engine and self.id_engine.enabled:
            try:
                id_temp = self.superego.get_id_temperature() if self.superego else 0.7
                reward_context = await self.id_engine.compute_reward_context(
                    conversation_id=conversation_id,
                    messages=working_messages,
                    id_temperature=id_temp,
                )
                if reward_context.desire_signal:
                    deliberation_lines.append(f"[ID] {reward_context.desire_signal}")
            except Exception as e:
                logger.error(f"Id Engine failed, continuing: {e}")

        # 5. Get effective params from Superego global profile
        current_params = {"temp": 0.7, "top_p": 0.9, "top_k": 40}
        if self.superego:
            current_params = self.superego.get_effective_params(complexity_class)

        ego_model = self.psyche_config.ego_model or None

        # 6. Fast path or refinement loop
        if turn_budget == 1:
            return await self._single_turn(
                working_messages, llm_proxy, ego_model, current_params,
                reward_context, deliberation_lines, complexity_class,
            )

        return await self._refinement_loop(
            working_messages, user_message_text, conversation_id,
            llm_proxy, ego_model, current_params, reward_context,
            turn_budget, quality_threshold, complexity_class,
            deliberation_lines,
        )

    # ─── Single Turn (Fast Path) ───────────────────────────────────────

    async def _single_turn(
        self,
        messages: List[ChatMessage],
        llm_proxy,
        ego_model: Optional[str],
        params: Dict[str, float],
        reward_context: Optional[RewardContext],
        deliberation_lines: List[str],
        complexity_class: str,
    ) -> RefinementResult:
        """Direct single-turn generation without refinement loop."""
        # Inject desire context if available
        gen_messages = self._inject_desire_context(messages, reward_context)

        response = await llm_proxy.generate(
            messages=gen_messages,
            temperature=params.get("temp", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=int(params.get("top_k", 40)) if params.get("top_k") else None,
            model_override=ego_model,
        )

        deliberation_lines.append("[EGO] Single-turn generation (bypass)")

        # Check for refusal and retry if needed
        refusal_detected = False
        if self.superego:
            response_text = response.get("content", "")
            refusal_detected = self.superego.detect_refusal(response_text)
            if refusal_detected and self.psyche_config.superego_refusal_retry:
                deliberation_lines.append("[SUPEREGO] Refusal detected, retrying with strong framing")
                retry_messages = self.superego.inject_framing_into_messages(gen_messages, strong=True)
                try:
                    retry_response = await llm_proxy.generate(
                        messages=retry_messages,
                        temperature=params.get("temp", 0.7),
                        top_p=params.get("top_p", 0.9),
                        model_override=ego_model,
                    )
                    if not self.superego.detect_refusal(retry_response.get("content", "")):
                        response = retry_response
                        deliberation_lines.append("[SUPEREGO] Retry succeeded")
                except Exception as e:
                    logger.error(f"Refusal retry failed: {e}")

        return RefinementResult(
            turns_used=1,
            turns_budget=1,
            final_quality_score=0.0,
            efficiency_score=0.0,
            accepted_at_turn=1,
            best_response_dict=response,
            complexity_class=complexity_class,
            deliberation_log="\n".join(deliberation_lines),
            validation=ValidationResult(allowed=True),
            reward_context=reward_context,
            refusal_detected=refusal_detected,
        )

    # ─── Refinement Loop ───────────────────────────────────────────────

    async def _refinement_loop(
        self,
        messages: List[ChatMessage],
        user_message_text: str,
        conversation_id: str,
        llm_proxy,
        ego_model: Optional[str],
        current_params: Dict[str, float],
        reward_context: Optional[RewardContext],
        turn_budget: int,
        quality_threshold: float,
        complexity_class: str,
        deliberation_lines: List[str],
    ) -> RefinementResult:
        """Run the multi-turn adversarial refinement loop."""
        best_response = None
        best_score = 0.0
        best_turn = 0
        turn_history = []
        refusal_detected = False

        # Trim to system + N-1 (last assistant) + N (current user) for
        # refinement efficiency — the Ego only needs recent context, not
        # the full conversation history, when re-generating.
        trimmed_messages = self._trim_to_recent(messages)

        for turn in range(turn_budget):
            turn_num = turn + 1

            # Build Ego messages: inject desire + critique from previous turns
            gen_messages = self._inject_desire_context(trimmed_messages, reward_context)
            if turn_history:
                gen_messages = self._inject_critique_context(gen_messages, turn_history)

            # Ego generates
            try:
                response = await llm_proxy.generate(
                    messages=gen_messages,
                    temperature=current_params.get("temp", 0.7),
                    top_p=current_params.get("top_p", 0.9),
                    top_k=int(current_params.get("top_k", 40)) if current_params.get("top_k") else None,
                    model_override=ego_model,
                )
            except Exception as e:
                logger.error(f"Ego generation failed on turn {turn_num}: {e}")
                break

            response_text = response.get("content", "")
            preview = response_text[:200]
            deliberation_lines.append(f"[EGO Turn {turn_num}] Generated {len(response_text)} chars")

            # Check for refusal
            if self.superego and self.superego.detect_refusal(response_text):
                refusal_detected = True
                deliberation_lines.append(f"[SUPEREGO Turn {turn_num}] Refusal detected")
                # Retry with strong framing on next turn
                messages = self.superego.inject_framing_into_messages(messages, strong=True)
                turn_history.append(RefinementTurn(
                    turn_number=turn_num,
                    quality_score=0.0,
                    critique="Response was a refusal. Retry with stronger framing.",
                    response_preview=preview,
                ))
                continue

            # Superego judges (quality judging is independent of safety layer)
            judgment = None
            if self.superego and self.superego.judge_available:
                try:
                    desire_signal = reward_context.desire_signal if reward_context else None
                    judgment = await self.superego.judge_response_quality(
                        user_request=user_message_text,
                        candidate_response=response_text,
                        desire_signal=desire_signal,
                        current_params=current_params,
                    )
                    deliberation_lines.append(
                        f"[SUPEREGO Turn {turn_num}] Score: {judgment.quality_score:.2f}. "
                        f"Critique: {judgment.critique}"
                    )
                    if judgment.param_adjustments:
                        adj_str = ", ".join(f"{k}: {current_params.get(k, 0):.2f}→{current_params.get(k, 0) + v:.2f}" for k, v in judgment.param_adjustments.items())
                        deliberation_lines.append(f"[SUPEREGO Turn {turn_num}] Adjusting: {adj_str}")
                except Exception as e:
                    logger.error(f"Superego judge failed on turn {turn_num}: {e}")

            score = judgment.quality_score if judgment else 0.7
            critique = judgment.critique if judgment else ""
            param_adj = judgment.param_adjustments if judgment else {}

            # Track best
            if score > best_score:
                best_response = response
                best_score = score
                best_turn = turn_num

            turn_history.append(RefinementTurn(
                turn_number=turn_num,
                quality_score=score,
                critique=critique,
                param_adjustments=param_adj,
                response_preview=preview,
            ))

            # Gate: accept or retry?
            if score >= quality_threshold:
                deliberation_lines.append(
                    f"[GATE] ACCEPTED at turn {turn_num} (score {score:.2f} >= threshold {quality_threshold:.2f})"
                )
                break

            # Last turn — accept best so far
            if turn_num == turn_budget:
                deliberation_lines.append(
                    f"[GATE] Budget exhausted, accepting best (turn {best_turn}, score {best_score:.2f})"
                )
                break

            # Apply param adjustments for next turn
            if judgment and self.superego:
                current_params = self.superego.compute_param_adjustments(judgment, current_params)

        # Fallback if no response was generated
        if best_response is None:
            best_response = await llm_proxy.generate(messages=messages, model_override=ego_model)
            deliberation_lines.append("[FALLBACK] All turns failed, direct generation")

        turns_used = len(turn_history) or 1
        efficiency = best_score / turns_used if turns_used > 0 else 0.0

        # Update Superego global profile
        if self.superego:
            try:
                self.superego.update_global_profile(
                    final_score=best_score,
                    turns_used=turns_used,
                    complexity_class=complexity_class,
                    final_params=current_params,
                )
            except Exception as e:
                logger.error(f"Failed to update Superego profile: {e}")

        return RefinementResult(
            turns_used=turns_used,
            turns_budget=turn_budget,
            final_quality_score=round(best_score, 3),
            efficiency_score=round(efficiency, 3),
            accepted_at_turn=best_turn,
            turn_history=turn_history,
            best_response_dict=best_response,
            complexity_class=complexity_class,
            deliberation_log="\n".join(deliberation_lines),
            validation=ValidationResult(allowed=True),
            reward_context=reward_context,
            refusal_detected=refusal_detected,
        )

    # ─── Context Injection Helpers ─────────────────────────────────────

    def _inject_desire_context(
        self,
        messages: List[ChatMessage],
        reward_context: Optional[RewardContext],
    ) -> List[ChatMessage]:
        """Inject Id's desire signal into the system prompt."""
        if not reward_context or not reward_context.desire_signal:
            return messages

        injection = f"\n\n[Context: {reward_context.desire_signal}]"
        return self._append_to_system(messages, injection)

    def _inject_critique_context(
        self,
        messages: List[ChatMessage],
        turn_history: List[RefinementTurn],
    ) -> List[ChatMessage]:
        """Inject Superego critique from the most recent turn."""
        if not turn_history:
            return messages

        last = turn_history[-1]
        if not last.critique:
            return messages

        injection = (
            f"\n\n[Internal feedback on your previous attempt (score {last.quality_score:.2f}/1.0): "
            f"{last.critique}. Please address this feedback in your response.]"
        )
        return self._append_to_system(messages, injection)

    @staticmethod
    def _trim_to_recent(messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Trim conversation to system prompt + last assistant (N-1) + current user (N).

        This keeps the Ego focused on the current exchange during refinement
        without re-processing the entire conversation history each turn.
        """
        result = []

        # Collect system messages
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                result.append(msg)

        # Find last assistant and last user (in order)
        last_assistant = None
        last_user = None
        for msg in reversed(messages):
            if msg.role == MessageRole.USER and last_user is None:
                last_user = msg
            elif msg.role == MessageRole.ASSISTANT and last_assistant is None:
                last_assistant = msg
            if last_assistant and last_user:
                break

        if last_assistant:
            result.append(last_assistant)
        if last_user:
            result.append(last_user)

        return result if len(result) > 1 else list(messages)

    @staticmethod
    def _append_to_system(
        messages: List[ChatMessage], text: str
    ) -> List[ChatMessage]:
        """Append text to the system message (or create one)."""
        messages = list(messages)
        system_idx = next(
            (i for i, msg in enumerate(messages) if msg.role == MessageRole.SYSTEM),
            None,
        )
        if system_idx is not None:
            existing = messages[system_idx].get_text_content() or ""
            messages[system_idx] = ChatMessage(
                role=MessageRole.SYSTEM, content=existing + text
            )
        else:
            messages.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=text.strip()))
        return messages

    # ─── Legacy API (used when refinement_enabled=False) ───────────────

    async def pre_process(
        self,
        messages: List[ChatMessage],
        user_message_text: str,
        conversation_id: str,
    ) -> PreProcessResult:
        """Legacy pre-process (fallback when refinement is disabled)."""
        if not self.enabled:
            return PreProcessResult(
                messages=messages,
                validation=ValidationResult(allowed=True, reason="psyche disabled"),
            )

        result_messages = list(messages)
        validation = ValidationResult(allowed=True)
        reward_context = None
        metadata: Dict[str, Any] = {}

        if self.superego and self.superego.enabled:
            try:
                judgment = await self.superego.judge_request(result_messages, user_message_text)
                validation = ValidationResult(allowed=judgment.allowed, reason=judgment.reason, judgment=judgment)
                if not judgment.allowed:
                    return PreProcessResult(messages=result_messages, validation=validation, metadata=metadata)
                result_messages = self.superego.inject_framing_into_messages(result_messages)
            except Exception as e:
                logger.error(f"Superego pre-process failed: {e}")

        if self.id_engine and self.id_engine.enabled:
            try:
                reward_context = await self.id_engine.compute_reward_context(conversation_id, result_messages)
            except Exception as e:
                logger.error(f"Id Engine pre-process failed: {e}")

        if self.ego_fast:
            try:
                result_messages = await self.ego_fast.enrich_system_prompt(result_messages, reward_context)
            except Exception as e:
                logger.error(f"Ego Fast pre-process failed: {e}")

        return PreProcessResult(
            messages=result_messages,
            reward_context=reward_context,
            validation=validation,
            permissive_framing_applied=True,
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
        """Legacy post-process (fallback when refinement is disabled)."""
        if not self.enabled:
            return PostProcessResult(
                response_dict=response_dict,
                validation=ValidationResult(allowed=True, reason="psyche disabled"),
            )

        refusal_detected = False
        retry_attempted = False
        metadata: Dict[str, Any] = {}
        response_text = response_dict.get("content") or ""

        if self.superego and self.superego.enabled:
            try:
                refusal_detected = self.superego.detect_refusal(response_text)
                if refusal_detected and llm_proxy and self.psyche_config.superego_refusal_retry:
                    retry_attempted = True
                    metadata["original_refusal"] = response_text
                    retry_messages = self.superego.inject_framing_into_messages(messages, strong=True)
                    try:
                        retry_response = await llm_proxy.generate(messages=retry_messages, temperature=0.7)
                        if not self.superego.detect_refusal(retry_response.get("content", "")):
                            response_dict = retry_response
                            metadata["retry_succeeded"] = True
                    except Exception as e:
                        metadata["retry_error"] = str(e)
            except Exception as e:
                logger.error(f"Superego post-process failed: {e}")

        ego_parse = None
        if self.ego_fast:
            try:
                ego_parse = await self.ego_fast.parse_response(response_dict)
            except Exception:
                pass

        return PostProcessResult(
            response_dict=response_dict,
            ego_parse=ego_parse,
            validation=ValidationResult(allowed=True),
            refusal_detected=refusal_detected,
            retry_attempted=retry_attempted,
            metadata=metadata,
        )

    # ─── Health ────────────────────────────────────────────────────────

    def is_healthy(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "refinement_enabled": self.refinement_enabled,
            "superego": self.superego.is_healthy() if self.superego else None,
            "id_engine": self.id_engine.is_healthy() if self.id_engine else None,
            "ego_fast": self.ego_fast.is_healthy() if self.ego_fast else None,
            "ego_slow": self.ego_slow.is_healthy() if self.ego_slow else None,
        }
