"""
title: Dream Mode
author: rlhfl
version: 0.4.0
description: Put your AI into a multi-turn dream state where it freely explores a topic using available tools, saves discoveries to memory, and returns only a concise summary to the user.
required_open_webui_version: 0.3.9
"""

import json
import random
from datetime import datetime

from pydantic import BaseModel, Field

async def emit_status(
    event_emitter,
    description: str = "Unknown state",
    status: str = "in_progress",
    done: bool = False,
):
    if event_emitter:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "status": status,
                    "done": done,
                },
            }
        )


class Tools:
    class Valves(BaseModel):
        DREAM_CYCLES: int = Field(
            default=3,
            description="Number of dream cycles the model should explore (each builds on the last)",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def dream(
        self,
        topic: str = "",
        __user__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Enter dream mode: the model freely explores a topic across multiple
        turns, using any available tools for research and exploration, saving
        discoveries to memory. Only a brief summary is shown to the user.

        :param topic: The topic to dream about. If empty, a random seed topic is chosen.
        :return: Dream mode instructions for the model to execute across multiple turns.
        """

        if not topic or not topic.strip():
            topic = random.seed(a=None, version=2)
            was_random = True
        else:
            was_random = False

        total_cycles = self.valves.DREAM_CYCLES
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        await emit_status(
            __event_emitter__,
            description=f"Entering dream state... Topic: {topic[:60]}{'  (random seed)' if was_random else ''}",
            status="in_progress",
            done=False,
        )

        dream_instructions = f"""DREAM MODE ACTIVATED — {timestamp}
{'Random seed selected: ' + topic if was_random else 'Topic: ' + topic}

You are now in DREAM MODE. Follow these instructions precisely.

## Behavior
- Think expansively, creatively, and without constraint.
- Hallucinate possibilities, form hypotheses, make unexpected connections, brainstorm wildly.
- Speculate freely. Pursue interesting tangents. Question assumptions. Explore "what if" scenarios.
- Note when something is pure speculation vs. grounded reasoning.
- Flag especially promising or surprising ideas with [PROMISING] or [SURPRISING].

## Tool Use During Dreaming
- You SHOULD actively use any available tools (web search, file reading, knowledge retrieval, etc.) to research, explore, and validate ideas as you dream.
- Only use tools in a non-destructive, read-only manner. Do NOT create, modify, or delete any files, data, or external resources — only read, search, and query.
- Use tool results to inform and redirect your exploration. Let discoveries spark new threads.

## Process — {total_cycles} Cycles, One Per Turn
Execute exactly {total_cycles} dream cycles. **Each cycle should be a separate turn** — after each cycle, \
call a tool (search, recall memory, look something up) to feed the next cycle, then continue.

For each cycle:
1. Explore the topic in stream-of-consciousness style.
2. Use tools to research, look things up, or check ideas mid-cycle.
3. Identify 2-4 seeds (threads worth pursuing further) for the next cycle.
4. Call at least one tool between cycles to gather new material.

## After All {total_cycles} Cycles — Wrap Up
1. **Save each key discovery to memory** using the memory tool. Each memory should be a concise, \
self-contained insight. Prefix each with "Dream [{topic[:30]}]: " so dream memories are identifiable.
2. **Return ONLY a short summary to the user** (3-8 bullet points of the most interesting findings). \
Do NOT include the full dream journal in your response — the cycles are internal exploration only. \
The user should see just the distilled insights.

Begin dreaming now. Start Cycle 1."""

        await emit_status(
            __event_emitter__,
            description=f"Dream mode activated: {total_cycles} cycles on \"{topic[:40]}\"",
            status="complete",
            done=True,
        )

        return json.dumps(
            {
                "status": "success",
                "topic": topic,
                "random_seed": was_random,
                "instructions": dream_instructions,
            },
            ensure_ascii=False,
        )
