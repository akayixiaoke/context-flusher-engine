"""OpenAI Client Wrapper for Context-Flusher.

Provides a one-line, non-intrusive transparent proxy around any OpenAI / llama-server / Ollama
client. Automatically handles preflight checks, task-aware deferral, and post-turn KV cache flushing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from context_flusher.engine import ContextFlusher

logger = logging.getLogger("context_flusher.wrapper")


class WrappedCompletions:
    """Transparent proxy around client.chat.completions."""

    def __init__(self, original_completions: Any, flusher: ContextFlusher) -> None:
        self._orig = original_completions
        self.flusher = flusher
        self._api_call_count = 0

    def create(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages")
        
        # Tools Schema Pinning for Prefix Cache protection (llama.cpp / vLLM)
        tools = kwargs.get("tools")
        if isinstance(tools, list):
            # Sort tools deterministically by function name so the prefix never breaks across runs
            kwargs["tools"] = sorted(
                tools,
                key=lambda t: t.get("function", {}).get("name", "") if isinstance(t, dict) else ""
            )

        # 1. Intercept with Preflight Check
        # Attempt model-specific server n_ctx probe if base_url is known but context length is not yet resolved
        if self.flusher.server_n_ctx is None and self.flusher.base_url:
            model = kwargs.get("model")
            if model:
                self.flusher.probe_and_clamp(model=str(model))

        if isinstance(messages, list):
            updated_messages, action = self.flusher.preflight_check(
                messages, api_call_count=self._api_call_count
            )
            kwargs["messages"] = updated_messages
            messages = updated_messages
            self._api_call_count += 1


        # 2. Call underlying provider (llama-server, Ollama, vLLM, OpenAI)
        response = self._orig.create(*args, **kwargs)

        # 3. Intercept response to detect step completion
        if isinstance(messages, list) and hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            msg = getattr(choice, "message", None)
            tool_calls = getattr(msg, "tool_calls", None) if msg else None

            # If the model produced a response without further tool_calls, turn is concluding
            if not tool_calls:
                content = getattr(msg, "content", "") or ""
                # Build snapshot including this final response
                turn_snapshot = list(messages)
                if msg is not None:
                    if hasattr(msg, "model_dump"):
                        turn_snapshot.append(msg.model_dump())
                    elif isinstance(msg, dict):
                        turn_snapshot.append(msg)
                    else:
                        turn_snapshot.append({
                            "role": "assistant",
                            "content": content,
                        })

                # Automatically flush & reclaim context if threshold reached
                reclaimed_messages, stats = self.flusher.finalize_turn(
                    turn_snapshot,
                    final_response=content,
                )

                if stats.get("archived"):
                    # Mutate the caller's message list in place so the caller immediately enjoys the flushed context
                    messages[:] = reclaimed_messages
                    # Attach flush stats to the response object for visibility
                    try:
                        setattr(response, "context_flusher_stats", stats)
                    except Exception:
                        pass

                # Reset multi-step counter for next user turn
                self._api_call_count = 0

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


class WrappedChat:
    """Transparent proxy around client.chat."""

    def __init__(self, original_chat: Any, flusher: ContextFlusher) -> None:
        self._orig = original_chat
        self.flusher = flusher
        self.completions = WrappedCompletions(original_chat.completions, flusher)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


class WrappedOpenAIClient:
    """Transparent proxy wrapper for any OpenAI SDK client instance."""

    def __init__(self, client: Any, flusher: ContextFlusher) -> None:
        self._client = client
        self.flusher = flusher
        self.chat = WrappedChat(client.chat, flusher)

    @property
    def retriever(self) -> Any:
        """Access the underlying task archive retriever."""
        return self.flusher.retriever

    def recall_by_file(self, file_path: str) -> Any:
        """Precision recall past task history and decisions anchored to a specific file."""
        return self.flusher.recall_by_file(file_path)

    def recall_by_dir(self, dir_path: str) -> List[Dict[str, Any]]:
        """Precision recall all task archives that touched files inside a directory."""
        return self.flusher.recall_by_dir(dir_path)

    def set_thresholds(self, active_threshold: int, hard_ceiling: int) -> None:
        """Dynamically adjust soft threshold and hard ceiling at runtime."""
        self.flusher.set_thresholds(active_threshold, hard_ceiling)

    def freeze_tools(self) -> None:
        """Explicit contract declaration: tools schema is frozen and static for prefix KV caching."""
        logger.info("Tools schema declared frozen. Stable prefix KV caching active.")

    @property
    def server_n_ctx(self) -> Optional[int]:
        """Detected or configured server context window length."""
        return self.flusher.server_n_ctx

    def format_flush_diff(self, stats: Dict[str, Any]) -> str:
        """Format an explicit human-readable reconciliation report of what was preserved vs flushed."""
        return self.flusher.format_flush_diff(stats)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)




def wrap_openai(
    client: Any,
    flusher: Optional[ContextFlusher] = None,
    **flusher_kwargs: Any,
) -> WrappedOpenAIClient:
    """Wrap any OpenAI / OpenAI-compatible client with one line of code.

    Example:
        >>> from openai import OpenAI
        >>> from context_flusher import wrap_openai
        >>> client = wrap_openai(OpenAI(base_url="http://localhost:8080/v1"), workspace_dir="./workspace")
        >>> # Every client.chat.completions.create call is now automatically managed and flushed!
    """
    base_url = getattr(client, "base_url", None)
    base_url_str = str(base_url) if base_url is not None else None

    if flusher is None:
        if base_url_str and "base_url" not in flusher_kwargs:
            flusher_kwargs["base_url"] = base_url_str
        flusher = ContextFlusher(**flusher_kwargs)
    elif base_url_str:
        flusher.probe_and_clamp(base_url=base_url_str)

    return WrappedOpenAIClient(client, flusher)
