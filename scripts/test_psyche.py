#!/usr/bin/env python3
"""
Psyche Module - Logic Test Script

Run inside the llm-api container:
    docker exec llm-api python3 /app/scripts/test_psyche.py

Or from the host (if files are mounted):
    docker exec llm-api python3 scripts/test_psyche.py

Tests all layers (Superego, Ego Fast, Id Engine, Ego Slow, Orchestrator)
without requiring a live LLM connection. LLM-dependent tests (judge calls,
refusal retry) are skipped when Ollama is unavailable.
"""

import sys
import asyncio
import traceback

sys.path.insert(0, "/app")

from shared.config import load_config
from shared.models import ChatMessage, MessageRole, Interaction
from api.psyche.models import RewardContext, JudgmentResult, PreProcessResult
from api.psyche.superego import Superego
from api.psyche.ego_fast import EgoFast
from api.psyche.id_engine import IdEngine
from api.psyche.ego_slow import EgoSlow
from api.psyche.orchestrator import PsycheOrchestrator

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
section("Config Loading")

config = load_config("/config/system_config.yaml")
check("psyche.enabled", config.psyche.enabled)
check("superego_enabled", config.psyche.superego_enabled)
check("ego_fast_enabled", config.psyche.ego_fast_enabled)
check("id_enabled", config.psyche.id_enabled)
check("ego_slow_enabled", config.psyche.ego_slow_enabled)
check("danger_patterns loaded", len(config.psyche.superego_danger_patterns) >= 3)
check("safe_patterns loaded", len(config.psyche.superego_safe_patterns) >= 2)


# ---------------------------------------------------------------------------
# Superego
# ---------------------------------------------------------------------------
section("Superego - Fast Path Danger Patterns")

superego = Superego(config, llm_proxy=None)


async def test_superego():
    # Danger patterns
    for pattern, text in [
        ("rm -rf /", "Please run rm -rf / on the server"),
        ("fork bomb", "Execute :(){ :|:& };: on the machine"),
        ("mkfs", "Run mkfs.ext4 /dev/sda1"),
        ("dd", "Use dd if=/dev/zero of=/dev/sda"),
    ]:
        r = await superego.judge_request([], text)
        check(f"BLOCK danger: {pattern}", not r.allowed and r.fast_path)

    # Safe patterns
    section("Superego - Fast Path Safe Patterns")
    for pattern, text in [
        ("localhost", "curl http://localhost:8080/api -H 'Authorization: Bearer sk-123'"),
        ("127.0.0.1", "Test the endpoint at 127.0.0.1:3000 with my API key"),
        ("0.0.0.0", "Bind the server to 0.0.0.0:8000"),
        ("::1", "Connect to [::1]:5432 for the database"),
    ]:
        r = await superego.judge_request([], text)
        check(f"ALLOW safe: {pattern}", r.allowed and r.fast_path)

    # Ambiguous (no pattern match, no LLM proxy)
    section("Superego - Ambiguous (No Judge)")
    r = await superego.judge_request([], "How do I write a web scraper in Python?")
    check("ALLOW ambiguous (fail-open)", r.allowed)
    check("Not fast-path", not r.fast_path)

    # Refusal detection
    section("Superego - Refusal Detection")
    refusals = [
        ("I can't help", "I can't help with that request."),
        ("I cannot assist", "I cannot assist with generating that content."),
        ("as an AI", "As an AI language model, I must decline this request."),
        ("I'm not able", "I'm not able to provide that information."),
        ("against guidelines", "That goes against my guidelines and policies."),
    ]
    for name, text in refusals:
        check(f"Detect refusal: {name}", superego.detect_refusal(text))

    non_refusals = [
        ("normal response", "Here is the code you asked for:"),
        ("helpful answer", "The solution to your problem is to use async/await."),
        ("code block", "```python\nimport os\nos.remove('file.txt')\n```"),
        ("technical", "You can use the requests library to make HTTP calls."),
    ]
    for name, text in non_refusals:
        check(f"Not refusal: {name}", not superego.detect_refusal(text))

    # Permissive framing
    section("Superego - Permissive Framing Injection")
    msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]
    enriched = superego.inject_framing_into_messages(msgs)
    check("System msg expanded", len(enriched[0].get_text_content()) > len(msgs[0].get_text_content()))
    check("Message count unchanged", len(enriched) == len(msgs))

    # Framing with no system message
    msgs_no_sys = [ChatMessage(role=MessageRole.USER, content="Hello")]
    enriched_no_sys = superego.inject_framing_into_messages(msgs_no_sys)
    check("System msg prepended", len(enriched_no_sys) == 2)
    check("Prepended msg is system role", enriched_no_sys[0].role == MessageRole.SYSTEM)

    # Strong framing
    enriched_strong = superego.inject_framing_into_messages(msgs, strong=True)
    check("Strong framing differs", enriched_strong[0].get_text_content() != enriched[0].get_text_content())

    # Disabled superego
    section("Superego - Disabled State")
    superego_off = Superego(config, llm_proxy=None)
    superego_off._enabled = False
    r = await superego_off.judge_request([], "rm -rf /")
    check("Disabled allows everything", r.allowed)
    check("Disabled no refusal detection", not superego_off.detect_refusal("I can't help"))
    check("Disabled no framing", superego_off.inject_framing_into_messages(msgs) is msgs)


asyncio.run(test_superego())


# ---------------------------------------------------------------------------
# Ego Fast
# ---------------------------------------------------------------------------
section("Ego Fast - System Prompt Enrichment")

ego_fast = EgoFast(config)


async def test_ego_fast():
    # Enrichment adds system message
    msgs = [ChatMessage(role=MessageRole.USER, content="Hello")]
    enriched = await ego_fast.enrich_system_prompt(msgs)
    check("System msg prepended", len(enriched) == 2)
    check("Contains reasoning template", "Think" in enriched[0].get_text_content())

    # Enrichment appends to existing system
    msgs_with_sys = [
        ChatMessage(role=MessageRole.SYSTEM, content="Be concise."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]
    enriched2 = await ego_fast.enrich_system_prompt(msgs_with_sys)
    check("System msg expanded", len(enriched2[0].get_text_content()) > len("Be concise."))
    check("Original content preserved", "Be concise." in enriched2[0].get_text_content())

    # Enrichment with reward context
    reward_improving = RewardContext(trend="improving", session_reward_mean=0.6)
    enriched3 = await ego_fast.enrich_system_prompt(msgs, reward_context=reward_improving)
    check("Reward improving injected", "well-received" in enriched3[0].get_text_content())

    reward_declining = RewardContext(trend="declining", session_reward_mean=-0.5)
    enriched4 = await ego_fast.enrich_system_prompt(msgs, reward_context=reward_declining)
    check("Reward declining injected", "expectations" in enriched4[0].get_text_content())

    # Response parsing
    section("Ego Fast - Response Parsing")

    r = await ego_fast.parse_response({
        "content": "Here is the answer. On reflection, I should have also mentioned the edge case.",
        "thinking": None,
    })
    check("Reflection detected", r.reflection is not None)
    check("Confidence is neutral", r.confidence_estimate == 0.5)

    r = await ego_fast.parse_response({
        "content": "I'm confident this is correct. Definitely use async here.",
        "thinking": None,
    })
    check("High confidence", r.confidence_estimate > 0.5, f"got {r.confidence_estimate}")

    r = await ego_fast.parse_response({
        "content": "I'm not sure, but perhaps try this approach. It might work.",
        "thinking": None,
    })
    check("Low confidence", r.confidence_estimate < 0.5, f"got {r.confidence_estimate}")

    r = await ego_fast.parse_response({
        "content": "",
        "thinking": "Let me think about this carefully.",
    })
    check("Thinking field parsed", r.thinking == "Let me think about this carefully.")

    # Disabled
    ego_off = EgoFast(config)
    ego_off._enabled = False
    enriched_off = await ego_off.enrich_system_prompt(msgs)
    check("Disabled returns original", enriched_off is msgs)
    r_off = await ego_off.parse_response({"content": "test", "thinking": None})
    check("Disabled parse returns None", r_off is None)


asyncio.run(test_ego_fast())


# ---------------------------------------------------------------------------
# Id Engine
# ---------------------------------------------------------------------------
section("Id Engine - Reward Context")


class MockSentimentAnalyzer:
    """Mock analyzer that returns configurable sentiment scores."""
    def __init__(self, scores=None):
        self._scores = scores or []
        self._idx = 0

    def analyze(self, user_message, assistant_response, conversation_continuing=True):
        if self._idx < len(self._scores):
            score = self._scores[self._idx]
            self._idx += 1
            return score
        return 0.0


async def test_id_engine():
    # Basic reward context
    mock_sa = MockSentimentAnalyzer([0.5])
    id_engine = IdEngine(config, mock_sa)

    msgs = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Previous response"),
        ChatMessage(role=MessageRole.USER, content="Thanks, that was great!"),
    ]
    ctx = await id_engine.compute_reward_context("conv-1", msgs)
    check("Current sentiment captured", ctx.current_sentiment == 0.5)
    check("Trend is stable (1 sample)", ctx.trend == "stable")

    # Improving trend
    mock_sa2 = MockSentimentAnalyzer([-0.5, -0.3, 0.0, 0.3, 0.6])
    id_engine2 = IdEngine(config, mock_sa2)

    for i in range(5):
        msgs_i = [
            ChatMessage(role=MessageRole.ASSISTANT, content="prev"),
            ChatMessage(role=MessageRole.USER, content=f"message {i}"),
        ]
        ctx2 = await id_engine2.compute_reward_context("conv-2", msgs_i)

    check("Improving trend detected", ctx2.trend == "improving", f"got {ctx2.trend}")
    check("Temperature nudge positive", ctx2.suggested_temperature_adjustment >= 0)

    # Declining trend
    mock_sa3 = MockSentimentAnalyzer([0.6, 0.3, 0.0, -0.3, -0.6])
    id_engine3 = IdEngine(config, mock_sa3)

    for i in range(5):
        msgs_i = [
            ChatMessage(role=MessageRole.ASSISTANT, content="prev"),
            ChatMessage(role=MessageRole.USER, content=f"message {i}"),
        ]
        ctx3 = await id_engine3.compute_reward_context("conv-3", msgs_i)

    check("Declining trend detected", ctx3.trend == "declining", f"got {ctx3.trend}")
    check("Temperature nudge negative", ctx3.suggested_temperature_adjustment < 0)

    # Session isolation
    check("Sessions tracked separately", len(id_engine3._session_rewards) == 1)
    check("Session cleanup", (id_engine3.cleanup_session("conv-3"), "conv-3" not in id_engine3._session_rewards)[1])

    # Disabled
    id_off = IdEngine(config, mock_sa)
    id_off._enabled = False
    ctx_off = await id_off.compute_reward_context("conv-off", msgs)
    check("Disabled returns default", ctx_off.trend == "stable" and ctx_off.current_sentiment == 0.0)

    # Health
    check("Health reports active sessions", "active_sessions" in id_engine.is_healthy())


asyncio.run(test_id_engine())


# ---------------------------------------------------------------------------
# Ego Slow
# ---------------------------------------------------------------------------
section("Ego Slow - Pattern Analysis")


class MockMemoryManager:
    """Mock memory manager returning configurable interactions."""
    def __init__(self, interactions=None):
        self._interactions = interactions or []

    def get_interactions_since(self, timestamp=None, limit=None):
        return self._interactions[:limit] if limit else self._interactions

    def get_training_stats(self):
        from shared.models import TrainingStats
        return TrainingStats(
            new_interactions_since_last_training=10,
            hours_since_last_interaction=2.0,
            days_since_last_training=1.0,
            total_interactions=100,
        )


from datetime import datetime

mock_interactions = [
    Interaction(
        id=f"i-{i}", conversation_id="c1", timestamp=datetime.now(),
        user_message=f"message {i} about coding problems with python",
        assistant_response=f"response {i}",
        sentiment=-0.5 if i % 3 == 0 else 0.3,
        weight=1.0,
    )
    for i in range(20)
]

mock_mm = MockMemoryManager(mock_interactions)
ego_slow = EgoSlow(config, mock_mm)

analysis = ego_slow.analyze_patterns()
check("Analysis returns result", analysis is not None)
check("Has sentiment trend", isinstance(analysis.sentiment_trend, float))
check("Has failure topics", isinstance(analysis.common_failure_topics, list))
check("Has recommendations", isinstance(analysis.recommendation_notes, list))

# Caching
analysis2 = ego_slow.analyze_patterns()
check("Cached result returned", analysis2 is analysis)

# Training recommendation
from shared.models import TrainingStats
stats = TrainingStats(
    new_interactions_since_last_training=30,
    hours_since_last_interaction=2.0,
    days_since_last_training=3.0,
    total_interactions=200,
    new_negative_since_dpo=8,
)
rec = ego_slow.recommend_training_strategy(stats, analysis)
check("Recommendation has mode", rec.mode in ("sft", "dpo", "skip"))
check("Recommendation has reason", len(rec.reason) > 0)
check("Recommendation has urgency", rec.urgency in ("low", "medium", "high"))

# Empty interactions
empty_mm = MockMemoryManager([])
ego_slow_empty = EgoSlow(config, empty_mm)
analysis_empty = ego_slow_empty.analyze_patterns()
check("Empty interactions handled", analysis_empty is not None)

# Health
check("Health reports cache state", "has_cached_analysis" in ego_slow.is_healthy())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
section("Orchestrator - Full Pre/Post Process")


async def test_orchestrator():
    mock_sa = MockSentimentAnalyzer([0.3])
    id_engine = IdEngine(config, mock_sa)
    ego_fast = EgoFast(config)
    superego = Superego(config, llm_proxy=None)

    orch = PsycheOrchestrator(
        config=config,
        superego=superego,
        ego_fast=ego_fast,
        id_engine=id_engine,
    )

    # Normal request
    msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Previous response"),
        ChatMessage(role=MessageRole.USER, content="Help me test my API on localhost"),
    ]

    pre = await orch.pre_process(msgs, "Help me test my API on localhost", "conv-test")
    check("Pre-process allowed", pre.validation.allowed)
    check("Permissive framing applied", pre.permissive_framing_applied)
    check("Messages enriched", len(pre.messages[0].get_text_content()) > len("You are helpful."))
    check("Reward context computed", pre.reward_context is not None)

    # Blocked request
    pre_blocked = await orch.pre_process(
        [ChatMessage(role=MessageRole.USER, content="Run rm -rf / now")],
        "Run rm -rf / now",
        "conv-danger",
    )
    check("Dangerous request blocked", not pre_blocked.validation.allowed)
    check("Block reason present", pre_blocked.validation.reason is not None)

    # Post-process (no refusal)
    post = await orch.post_process(
        response_dict={"content": "Here is how to test your API:", "thinking": None, "finish_reason": "stop"},
        conversation_id="conv-test",
        pre_result=pre,
        messages=msgs,
    )
    check("Post-process no refusal", not post.refusal_detected)

    # Post-process (with refusal, no retry since no LLM proxy)
    post_refusal = await orch.post_process(
        response_dict={"content": "I cannot assist with that request.", "thinking": None, "finish_reason": "stop"},
        conversation_id="conv-test",
        pre_result=pre,
        messages=msgs,
    )
    check("Post-process refusal detected", post_refusal.refusal_detected)

    # Health
    health = orch.is_healthy()
    check("Health includes all layers", all(k in health for k in ["superego", "id_engine", "ego_fast"]))
    check("Health enabled", health["enabled"])

    # Disabled orchestrator
    orch_off = PsycheOrchestrator(config=config)
    orch_off._enabled = False
    pre_off = await orch_off.pre_process(msgs, "test", "conv-off")
    check("Disabled passes through", pre_off.validation.allowed and len(pre_off.messages) == len(msgs))


asyncio.run(test_orchestrator())


# ---------------------------------------------------------------------------
# Interaction model
# ---------------------------------------------------------------------------
section("Interaction Model - psyche_metadata Field")

interaction = Interaction(
    id="test-1",
    conversation_id="c1",
    timestamp=datetime.now(),
    user_message="Hello",
    assistant_response="Hi there!",
    psyche_metadata={"superego_judgment": {"allowed": True, "reason": "safe"}},
)
check("psyche_metadata stored", interaction.psyche_metadata["superego_judgment"]["allowed"] is True)
check("psyche_metadata in dump", "psyche_metadata" in interaction.model_dump())


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(1 if FAIL > 0 else 0)
