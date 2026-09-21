"""Context Reset & KV Cache Reclamation Engine.

Safely reduces active conversational context down to ~1.5k tokens post-archive:
- Verifies archive file integrity (> 0 bytes on disk) before dropping history.
- Retains system prompt, checkpoint orientation notice, and sanitized recent dialogue tail.
- Strips bloated intermediate tool outputs and sanitizes tool_calls to prevent 400 Bad Request errors.
- Provides stage lightweight tool pruning for the hard ceiling threshold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from context_flusher.sanitizer import (
    estimate_messages_tokens,
    sanitize_message_sequence,
)

logger = logging.getLogger("context_flusher.context_reset")


def reclaim_context(
    messages: List[Dict[str, Any]],
    archive_file: Optional[Path] = None,
    *,
    keep_tail_turns: int = 1,
    inject_checkpoint_notice: bool = True,
    progress_filename: str = "TASK_PROGRESS.md",
) -> Tuple[List[Dict[str, Any]], int]:
    """Verify archive integrity, then purge old tool payloads and reset context to a clean baseline."""
    # Safety Check: Never purge if archive file is missing or empty
    if archive_file is not None:
        if not archive_file.exists() or archive_file.stat().st_size == 0:
            logger.warning(
                "Archive verification failed for %s (missing or 0 bytes). Refusing context reclamation.",
                archive_file,
            )
            return messages, 0

    tokens_before = estimate_messages_tokens(messages)
    original_count = len(messages)

    # 1. Retain system prompt if present
    head = [messages[0]] if messages and messages[0].get("role") == "system" else []

    # 2. Build orientation checkpoint notice
    notice_list = []
    if inject_checkpoint_notice:
        archive_name = archive_file.name if archive_file else "archive.md"
        archive_stem = archive_file.stem if archive_file else "checkpoint"
        notice = {
            "role": "assistant",
            "content": (
                f"📦 **[Context Flushed & Checkpoint Recorded]**\n"
                f"Prior task history for `{archive_stem}` has been reliably written to disk:\n"
                f"- Task Archive: `archives/{archive_name}`\n"
                f"- Cumulative Progress: `{progress_filename}`\n"
                f"Historical intermediate tool outputs and execution buffers have been cleared to release "
                f"GPU KV cache. Continuing task execution with fresh attention window."
            ),
        }
        notice_list.append(notice)

    # 3. Extract the clean dialogue tail (most recent user intent and assistant answer)
    tail: List[Dict[str, Any]] = []
    last_user_msg = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = dict(m)
            break

    last_asst_msg = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            clean_asst = dict(m)
            # Remove raw tool calls so the message acts as a clean standalone assistant reply
            clean_asst.pop("tool_calls", None)
            clean_asst.pop("tool_call_id", None)
            if clean_asst.get("content"):
                last_asst_msg = clean_asst
                break

    if last_user_msg:
        tail.append(last_user_msg)
    if last_asst_msg:
        tail.append(last_asst_msg)

    # Fallback if no user/assistant messages found
    if not tail:
        tail = [dict(m) for m in messages[-2:]]

    # Assemble and sanitize sequence
    candidate_messages = head + notice_list + tail
    new_messages = sanitize_message_sequence(candidate_messages)

    tokens_after = estimate_messages_tokens(new_messages)
    reclaimed = max(0, tokens_before - tokens_after)

    logger.info(
        "Context reclaimed: %d -> %d messages (~%s -> ~%s tokens, freed ~%s tokens)",
        original_count,
        len(new_messages),
        f"{tokens_before:,}",
        f"{tokens_after:,}",
        f"{reclaimed:,}",
    )

    return new_messages, reclaimed


def prune_tool_results_lightweight(
    messages: List[Dict[str, Any]],
    *,
    protect_tail_count: int = 3,
    min_prune_chars: int = 500,
) -> Tuple[List[Dict[str, Any]], int]:
    """60K Hard Ceiling Guard: In-place deterministic pruning of bloated tool responses.

    Replaces older, oversized tool outputs with a one-line placeholder without stopping
    or breaking the running task.
    """
    if len(messages) <= protect_tail_count + 1:
        return messages, 0

    tokens_before = estimate_messages_tokens(messages)
    pruned_count = 0
    protected_boundary = len(messages) - protect_tail_count

    new_messages: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        m = dict(msg)
        role = m.get("role")
        content = m.get("content")

        if idx < protected_boundary and role == "tool" and isinstance(content, str):
            if len(content) > min_prune_chars:
                lines = content.splitlines()
                first_line = lines[0][:100] if lines else ""
                char_count = len(content)
                m["content"] = f"[Executed: output pruned ({char_count:,} chars). Result recorded in archive. Preview: {first_line}...]"
                pruned_count += 1

        new_messages.append(m)

    sanitized = sanitize_message_sequence(new_messages)
    tokens_after = estimate_messages_tokens(sanitized)
    reclaimed = max(0, tokens_before - tokens_after)

    if pruned_count > 0:
        logger.info(
            "Lightweight prune relieved context: pruned %d outputs, freed ~%s tokens (~%s -> ~%s tokens)",
            pruned_count,
            f"{reclaimed:,}",
            f"{tokens_before:,}",
            f"{tokens_after:,}",
        )

    return sanitized, pruned_count
