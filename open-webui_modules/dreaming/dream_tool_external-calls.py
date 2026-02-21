"""
title: Dream Mode
author: rlhfl
version: 0.1.0
description: Put your AI into a dream state where it freely hallucinates, explores, and researches a topic across multiple cycles, saving discoveries as Open WebUI notes.
required_open_webui_version: 0.3.9
"""

import json
import sys
from datetime import datetime
from typing import List, Optional

import aiohttp
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


DREAM_SYSTEM_PROMPT = """\
You are in DREAM MODE. You are free to think expansively, creatively, and without \
constraint. Your purpose is to explore a topic deeply — hallucinate possibilities, \
form hypotheses, make unexpected connections, brainstorm wildly, and surface ideas \
that a careful, conservative thinker would never reach.

Rules of dreaming:
- You SHOULD speculate, imagine, and invent freely.
- You SHOULD pursue tangents if they seem interesting.
- You SHOULD make connections across disparate domains.
- You SHOULD question assumptions and explore "what if" scenarios.
- You SHOULD note when something is pure speculation vs. grounded reasoning.
- You SHOULD flag anything that feels especially promising or surprising.
- Structure your output as a stream-of-consciousness dream journal entry.
- Use markdown formatting with headers for major threads of thought.
- End each dream cycle with a "## Seeds for Next Cycle" section listing \
  2-4 threads worth pursuing further.

You are on dream cycle {cycle} of {total_cycles}.\
"""

DREAM_CONTINUATION_PROMPT = """\
Continue dreaming. Here are the seeds from your previous cycle:

{seeds}

Pick up one or more of these threads and go deeper. Explore new angles, \
make new connections, challenge your earlier ideas, or branch into something \
entirely unexpected. Remember: you are dreaming — be bold.\
"""


class Tools:
    class Valves(BaseModel):
        OPENWEBUI_API_URL: str = Field(
            default="http://localhost:3000",
            description="Base URL of your Open WebUI instance (no trailing slash)",
        )
        OPENWEBUI_API_KEY: str = Field(
            default="",
            description="Open WebUI API key (from Settings > Account)",
        )
        LLM_API_URL: str = Field(
            default="http://localhost:11434/v1/chat/completions",
            description="OpenAI-compatible chat completions endpoint for the dreaming model",
        )
        LLM_API_KEY: str = Field(
            default="",
            description="API key for the LLM endpoint (leave empty if not required)",
        )
        LLM_MODEL: str = Field(
            default="llama3.2",
            description="Model to use for dreaming",
        )
        DREAM_CYCLES: int = Field(
            default=3,
            description="Number of dream cycles to run per session (each builds on the last)",
        )
        DREAM_TEMPERATURE: float = Field(
            default=1.3,
            description="Temperature for dream generation (higher = more creative, 0.0-2.0)",
        )
        DREAM_MAX_TOKENS: int = Field(
            default=2048,
            description="Maximum tokens per dream cycle",
        )
        NOTE_TITLE_PREFIX: str = Field(
            default="Dream Journal",
            description="Prefix for the note title saved in Open WebUI",
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable verbose debug logging",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _debug(self, message: str):
        if self.valves.DEBUG:
            print(f"[DREAM DEBUG] {message}", file=sys.stderr)

    async def _call_llm(
        self,
        messages: List[dict],
        event_emitter=None,
    ) -> Optional[str]:
        """Call the LLM API for a single dream cycle."""
        payload = {
            "model": self.valves.LLM_MODEL,
            "messages": messages,
            "temperature": self.valves.DREAM_TEMPERATURE,
            "max_tokens": self.valves.DREAM_MAX_TOKENS,
        }

        headers = {"Content-Type": "application/json"}
        if self.valves.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LLM_API_KEY}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.valves.LLM_API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"LLM API returned {response.status}: {error_text}")
                result = await response.json()
                return result["choices"][0]["message"]["content"].strip()

    async def _create_note(self, title: str, content: str) -> Optional[dict]:
        """Create a note in Open WebUI via the notes API."""
        url = f"{self.valves.OPENWEBUI_API_URL}/api/v1/notes/create"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}",
        }
        payload = {
            "title": title,
            "data": {"content": content},
            "meta": {
                "source": "dream_mode",
                "created_by_tool": True,
            },
        }

        self._debug(f"Creating note: {title}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._debug(f"Note creation failed: {response.status} {error_text}")
                    raise Exception(
                        f"Notes API returned {response.status}: {error_text}"
                    )
                return await response.json()

    async def _update_note(
        self, note_id: str, title: str, content: str
    ) -> Optional[dict]:
        """Update an existing note in Open WebUI."""
        url = f"{self.valves.OPENWEBUI_API_URL}/api/v1/notes/{note_id}/update"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}",
        }
        payload = {
            "title": title,
            "data": {"content": content},
            "meta": {
                "source": "dream_mode",
                "created_by_tool": True,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._debug(f"Note update failed: {response.status} {error_text}")
                    raise Exception(
                        f"Notes API returned {response.status}: {error_text}"
                    )
                return await response.json()

    async def _search_notes(self, query: str) -> Optional[dict]:
        """Search existing notes in Open WebUI."""
        url = f"{self.valves.OPENWEBUI_API_URL}/api/v1/notes/search"
        headers = {
            "Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}",
        }
        params = {"query": query}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    return None
                return await response.json()

    def _extract_seeds(self, dream_text: str) -> str:
        """Extract the 'Seeds for Next Cycle' section from a dream output."""
        markers = [
            "## Seeds for Next Cycle",
            "## Seeds for next cycle",
            "## Seeds",
            "**Seeds for Next Cycle**",
        ]
        for marker in markers:
            idx = dream_text.find(marker)
            if idx != -1:
                return dream_text[idx:]
        # Fallback: use the last 500 characters as context
        return dream_text[-500:]

    async def dream(
        self,
        topic: str,
        __user__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Enter dream mode: the AI freely explores a topic across multiple cycles,
        hallucinating, researching, and making connections. All discoveries are
        saved as a note in Open WebUI.

        :param topic: The topic, question, or prompt to dream about.
        :return: JSON result with dream summary and note ID.
        """

        if not topic or not topic.strip():
            return json.dumps(
                {
                    "status": "error",
                    "message": "Please provide a topic to dream about.",
                },
                ensure_ascii=False,
            )

        if not self.valves.OPENWEBUI_API_KEY:
            return json.dumps(
                {
                    "status": "error",
                    "message": "OPENWEBUI_API_KEY is required. Set it in the tool's Valves configuration.",
                },
                ensure_ascii=False,
            )

        total_cycles = self.valves.DREAM_CYCLES
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        note_title = f"{self.valves.NOTE_TITLE_PREFIX}: {topic[:80]} ({timestamp})"

        await emit_status(
            __event_emitter__,
            description=f"Entering dream state... Topic: {topic[:60]}",
            status="in_progress",
            done=False,
        )

        # Check for prior dreams on this topic
        prior_context = ""
        try:
            search_results = await self._search_notes(topic[:60])
            if search_results and search_results.get("items"):
                prior_items = search_results["items"][:3]
                prior_titles = [item.get("title", "Untitled") for item in prior_items]
                prior_context = (
                    f"\n\nYou have dreamed about related topics before. "
                    f"Prior dream notes: {', '.join(prior_titles)}. "
                    f"Try to build on or challenge those earlier explorations."
                )
                self._debug(f"Found {len(prior_items)} prior dream notes")
        except Exception as e:
            self._debug(f"Note search failed (non-critical): {e}")

        # Build the full dream journal as we go
        full_journal = f"# Dream Journal: {topic}\n"
        full_journal += f"*Started: {timestamp}*\n"
        full_journal += f"*Cycles: {total_cycles}*\n\n---\n\n"

        dream_outputs = []
        seeds = ""

        for cycle in range(1, total_cycles + 1):
            await emit_status(
                __event_emitter__,
                description=f"Dream cycle {cycle}/{total_cycles}...",
                status="in_progress",
                done=False,
            )

            system_prompt = DREAM_SYSTEM_PROMPT.format(
                cycle=cycle, total_cycles=total_cycles
            )

            messages = [{"role": "system", "content": system_prompt}]

            if cycle == 1:
                user_content = f"Dream about this topic:\n\n{topic}"
                if prior_context:
                    user_content += prior_context
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": DREAM_CONTINUATION_PROMPT.format(seeds=seeds),
                    }
                )

            try:
                dream_output = await self._call_llm(messages, __event_emitter__)
                if not dream_output:
                    raise Exception("LLM returned empty response")

                dream_outputs.append(dream_output)
                seeds = self._extract_seeds(dream_output)

                full_journal += f"## Cycle {cycle}\n\n{dream_output}\n\n---\n\n"

                self._debug(f"Cycle {cycle} complete: {len(dream_output)} chars")

            except Exception as e:
                error_msg = f"Dream cycle {cycle} failed: {e}"
                self._debug(error_msg)
                await emit_status(
                    __event_emitter__,
                    description=error_msg,
                    status="error",
                    done=False,
                )
                full_journal += (
                    f"## Cycle {cycle}\n\n*Dream interrupted: {e}*\n\n---\n\n"
                )
                break

        # Save the complete dream journal as a note
        await emit_status(
            __event_emitter__,
            description="Saving dream journal to notes...",
            status="in_progress",
            done=False,
        )

        note_id = None
        try:
            note_result = await self._create_note(note_title, full_journal)
            note_id = note_result.get("id") if note_result else None
            self._debug(f"Note created with ID: {note_id}")
        except Exception as e:
            self._debug(f"Failed to save note: {e}")
            await emit_status(
                __event_emitter__,
                description=f"Warning: Could not save note ({e})",
                status="warning",
                done=False,
            )

        # Build summary
        cycles_completed = len(dream_outputs)
        total_chars = sum(len(d) for d in dream_outputs)

        status_desc = (
            f"Dream complete: {cycles_completed}/{total_cycles} cycles, "
            f"~{total_chars // 4} tokens generated"
        )
        if note_id:
            status_desc += " | Saved to notes"

        await emit_status(
            __event_emitter__,
            description=status_desc,
            status="complete",
            done=True,
        )

        return json.dumps(
            {
                "status": "success",
                "topic": topic,
                "cycles_completed": cycles_completed,
                "total_cycles": total_cycles,
                "total_characters": total_chars,
                "note_id": note_id,
                "note_title": note_title,
                "final_seeds": seeds,
                "dream_journal": full_journal,
            },
            ensure_ascii=False,
        )

    async def recall_dreams(
        self,
        query: str,
        __user__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Search through previous dream journal notes for a topic or keyword.

        :param query: Search term to find in previous dream notes.
        :return: JSON result with matching dream notes.
        """

        if not self.valves.OPENWEBUI_API_KEY:
            return json.dumps(
                {
                    "status": "error",
                    "message": "OPENWEBUI_API_KEY is required.",
                },
                ensure_ascii=False,
            )

        await emit_status(
            __event_emitter__,
            description=f"Searching dream notes for: {query[:40]}...",
            status="in_progress",
            done=False,
        )

        try:
            results = await self._search_notes(query)
            if not results or not results.get("items"):
                await emit_status(
                    __event_emitter__,
                    description="No matching dream notes found",
                    status="complete",
                    done=True,
                )
                return json.dumps(
                    {
                        "status": "success",
                        "matches": [],
                        "message": "No dream notes found matching that query.",
                    },
                    ensure_ascii=False,
                )

            matches = []
            for item in results["items"]:
                matches.append(
                    {
                        "id": item.get("id"),
                        "title": item.get("title", "Untitled"),
                        "updated_at": item.get("updated_at"),
                        "created_at": item.get("created_at"),
                    }
                )

            await emit_status(
                __event_emitter__,
                description=f"Found {len(matches)} dream note(s)",
                status="complete",
                done=True,
            )

            return json.dumps(
                {
                    "status": "success",
                    "query": query,
                    "matches": matches,
                    "total": len(matches),
                },
                ensure_ascii=False,
            )

        except Exception as e:
            await emit_status(
                __event_emitter__,
                description=f"Search failed: {e}",
                status="error",
                done=True,
            )
            return json.dumps(
                {"status": "error", "message": str(e)},
                ensure_ascii=False,
            )

    async def continue_dream(
        self,
        note_id: str,
        additional_prompt: str = "",
        __user__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Continue dreaming from a previous dream note, appending new cycles
        to the existing journal.

        :param note_id: The ID of the dream note to continue from.
        :param additional_prompt: Optional extra guidance for the continuation.
        :return: JSON result with updated dream journal.
        """

        if not self.valves.OPENWEBUI_API_KEY:
            return json.dumps(
                {
                    "status": "error",
                    "message": "OPENWEBUI_API_KEY is required.",
                },
                ensure_ascii=False,
            )

        await emit_status(
            __event_emitter__,
            description="Loading previous dream...",
            status="in_progress",
            done=False,
        )

        # Fetch the existing note
        url = f"{self.valves.OPENWEBUI_API_URL}/api/v1/notes/{note_id}"
        headers = {"Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Could not fetch note {note_id}: {response.status} {error_text}"
                        )
                    note_data = await response.json()
        except Exception as e:
            await emit_status(
                __event_emitter__,
                description=f"Failed to load note: {e}",
                status="error",
                done=True,
            )
            return json.dumps(
                {"status": "error", "message": str(e)},
                ensure_ascii=False,
            )

        existing_content = ""
        if note_data.get("data") and isinstance(note_data["data"], dict):
            existing_content = note_data["data"].get("content", "")
        note_title = note_data.get("title", "Dream Journal")

        # Extract seeds from the existing content
        seeds = self._extract_seeds(existing_content)

        # Figure out what cycle number we're on
        cycle_count = existing_content.count("## Cycle ")
        total_cycles = self.valves.DREAM_CYCLES

        updated_journal = existing_content.rstrip()
        if not updated_journal.endswith("---"):
            updated_journal += "\n\n---\n\n"
        else:
            updated_journal += "\n\n"

        updated_journal += (
            f"*Continued: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        )

        dream_outputs = []

        for cycle in range(1, total_cycles + 1):
            cycle_num = cycle_count + cycle

            await emit_status(
                __event_emitter__,
                description=f"Continuation cycle {cycle}/{total_cycles} (overall #{cycle_num})...",
                status="in_progress",
                done=False,
            )

            system_prompt = DREAM_SYSTEM_PROMPT.format(
                cycle=cycle_num, total_cycles=cycle_count + total_cycles
            )

            messages = [{"role": "system", "content": system_prompt}]

            continuation_text = DREAM_CONTINUATION_PROMPT.format(seeds=seeds)
            if additional_prompt and cycle == 1:
                continuation_text += f"\n\nAdditional guidance: {additional_prompt}"

            messages.append({"role": "user", "content": continuation_text})

            try:
                dream_output = await self._call_llm(messages, __event_emitter__)
                if not dream_output:
                    raise Exception("LLM returned empty response")

                dream_outputs.append(dream_output)
                seeds = self._extract_seeds(dream_output)
                updated_journal += f"## Cycle {cycle_num}\n\n{dream_output}\n\n---\n\n"

            except Exception as e:
                self._debug(f"Continuation cycle {cycle} failed: {e}")
                updated_journal += (
                    f"## Cycle {cycle_num}\n\n*Dream interrupted: {e}*\n\n---\n\n"
                )
                break

        # Update the existing note
        await emit_status(
            __event_emitter__,
            description="Updating dream journal note...",
            status="in_progress",
            done=False,
        )

        try:
            await self._update_note(note_id, note_title, updated_journal)
        except Exception as e:
            self._debug(f"Failed to update note: {e}")
            await emit_status(
                __event_emitter__,
                description=f"Warning: Could not update note ({e})",
                status="warning",
                done=False,
            )

        cycles_completed = len(dream_outputs)
        total_chars = sum(len(d) for d in dream_outputs)

        await emit_status(
            __event_emitter__,
            description=f"Continuation complete: {cycles_completed} new cycles added",
            status="complete",
            done=True,
        )

        return json.dumps(
            {
                "status": "success",
                "note_id": note_id,
                "note_title": note_title,
                "new_cycles": cycles_completed,
                "total_cycles_now": cycle_count + cycles_completed,
                "new_characters": total_chars,
                "final_seeds": seeds,
                "dream_journal": updated_journal,
            },
            ensure_ascii=False,
        )
