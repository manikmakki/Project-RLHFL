"""
Prompt Regex Interceptor

Applies regex search-and-replace rules to incoming chat messages before
RAG or LLM processing. Rules are persisted as YAML for easy programmatic
and manual editing.
"""

import logging
import re
import uuid
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

from shared.models import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

# Map role strings to MessageRole enum values for matching
_ROLE_MAP = {
    "system": MessageRole.SYSTEM,
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
}


class PromptInterceptor:
    """Regex-based message interceptor with YAML-persisted rules."""

    def __init__(self, rules_path: str):
        self._rules_path = Path(rules_path)
        self._rules: List[Dict[str, Any]] = []
        self._compiled: Dict[str, re.Pattern] = {}  # id -> compiled pattern
        self._lock = threading.Lock()
        self._load_rules()

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def _load_rules(self):
        """Load rules from YAML file. Creates empty file if missing."""
        if not self._rules_path.exists():
            self._rules = []
            self._save_rules()
            logger.info(f"Created empty rules file: {self._rules_path}")
            return

        try:
            with open(self._rules_path, "r") as f:
                data = yaml.safe_load(f) or {}
            self._rules = data.get("rules", [])
            self._rules.sort(key=lambda r: r.get("priority", 100))
            self._compile_all()
            logger.info(f"Loaded {len(self._rules)} prompt interceptor rule(s)")
        except Exception as e:
            logger.error(f"Failed to load prompt rules: {e}")
            self._rules = []

    def _save_rules(self):
        """Persist current rules to YAML."""
        self._rules_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._rules_path, "w") as f:
            yaml.dump(
                {"rules": self._rules},
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    # ------------------------------------------------------------------ #
    #  Regex compilation
    # ------------------------------------------------------------------ #

    def _compile_all(self):
        """Compile all rule patterns. Skip rules with invalid regex."""
        self._compiled.clear()
        for rule in self._rules:
            try:
                self._compiled[rule["id"]] = re.compile(rule["pattern"])
            except re.error as e:
                logger.warning(
                    f"Invalid regex in rule {rule['id']} "
                    f"({rule.get('description', '')}): {e}"
                )

    @staticmethod
    def _validate_pattern(pattern: str):
        """Raise ValueError if pattern is not a valid regex."""
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    # ------------------------------------------------------------------ #
    #  Apply rules to messages
    # ------------------------------------------------------------------ #

    def apply(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Apply all enabled rules to a list of messages.

        Returns a new list; original messages are not mutated.
        Only string content is transformed; non-string content is passed through.
        """
        with self._lock:
            enabled = [
                r for r in self._rules
                if r.get("enabled", True) and r["id"] in self._compiled
            ]
        if not enabled:
            return messages

        result = []
        total_applied = 0
        for msg in messages:
            content = msg.content
            if not isinstance(content, str):
                result.append(msg)
                continue

            transformed = content
            for rule in enabled:
                targets = rule.get("target_roles", ["system"])
                role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                if role_str not in targets and "all" not in targets:
                    continue

                pat = self._compiled[rule["id"]]
                new_text = pat.sub(rule.get("replacement", ""), transformed)
                if new_text != transformed:
                    total_applied += 1
                    logger.debug(
                        f"Rule '{rule.get('description', rule['id'])}' matched "
                        f"on {role_str} message"
                    )
                    transformed = new_text

            if transformed != content:
                new_msg = ChatMessage(
                    role=msg.role,
                    content=transformed,
                    name=msg.name,
                    tool_calls=msg.tool_calls if hasattr(msg, "tool_calls") else None,
                    tool_call_id=msg.tool_call_id if hasattr(msg, "tool_call_id") else None,
                )
                result.append(new_msg)
            else:
                result.append(msg)

        if total_applied:
            logger.info(f"Prompt interceptor: {total_applied} rule match(es) applied")
        return result

    # ------------------------------------------------------------------ #
    #  Test helper
    # ------------------------------------------------------------------ #

    def test(self, text: str) -> Dict[str, Any]:
        """
        Test all enabled rules against a sample text.

        Returns dict with transformed output and per-rule match info.
        """
        with self._lock:
            enabled = [
                r for r in self._rules
                if r.get("enabled", True) and r["id"] in self._compiled
            ]

        output = text
        rules_applied = []
        for rule in enabled:
            pat = self._compiled[rule["id"]]
            new_text = pat.sub(rule.get("replacement", ""), output)
            matched = new_text != output
            rules_applied.append({
                "id": rule["id"],
                "description": rule.get("description", ""),
                "matched": matched,
            })
            output = new_text

        return {"output": output, "rules_applied": rules_applied}

    # ------------------------------------------------------------------ #
    #  CRUD
    # ------------------------------------------------------------------ #

    def get_rules(self) -> List[Dict[str, Any]]:
        """Return all rules sorted by priority."""
        with self._lock:
            return list(self._rules)

    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Return a single rule by ID, or None."""
        with self._lock:
            return next((r for r in self._rules if r["id"] == rule_id), None)

    def add_rule(
        self,
        pattern: str,
        replacement: str = "",
        description: str = "",
        target_roles: Optional[List[str]] = None,
        enabled: bool = True,
        priority: int = 50,
    ) -> Dict[str, Any]:
        """Add a new rule. Raises ValueError if regex is invalid."""
        self._validate_pattern(pattern)
        rule = {
            "id": uuid.uuid4().hex[:12],
            "description": description,
            "pattern": pattern,
            "replacement": replacement,
            "target_roles": target_roles or ["system"],
            "enabled": enabled,
            "priority": priority,
        }
        with self._lock:
            self._rules.append(rule)
            self._rules.sort(key=lambda r: r.get("priority", 100))
            self._compile_all()
            self._save_rules()
        logger.info(f"Added prompt rule: {rule['id']} ({description})")
        return rule

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing rule. Raises ValueError if new regex is invalid."""
        if "pattern" in updates:
            self._validate_pattern(updates["pattern"])

        with self._lock:
            rule = next((r for r in self._rules if r["id"] == rule_id), None)
            if rule is None:
                return None

            for key in ("description", "pattern", "replacement", "target_roles", "enabled", "priority"):
                if key in updates:
                    rule[key] = updates[key]

            self._rules.sort(key=lambda r: r.get("priority", 100))
            self._compile_all()
            self._save_rules()

        logger.info(f"Updated prompt rule: {rule_id}")
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID. Returns True if found and deleted."""
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r["id"] != rule_id]
            if len(self._rules) == before:
                return False
            self._compiled.pop(rule_id, None)
            self._save_rules()
        logger.info(f"Deleted prompt rule: {rule_id}")
        return True

    def toggle_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Toggle a rule's enabled state. Returns updated rule or None."""
        with self._lock:
            rule = next((r for r in self._rules if r["id"] == rule_id), None)
            if rule is None:
                return None
            rule["enabled"] = not rule.get("enabled", True)
            self._save_rules()
        logger.info(f"Toggled prompt rule {rule_id}: enabled={rule['enabled']}")
        return rule
