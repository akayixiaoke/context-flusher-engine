"""Context-Flusher: Task-Aware Context Management & KV Cache Flushing Middleware.

High-performance, zero-LLM context truncation and structured Markdown archiving
tailored for local LLM agents (llama.cpp, Ollama, vLLM).
"""

from context_flusher.archiver import (
    create_task_archive,
    extract_heuristic_decisions,
    extract_touched_files,
)
from context_flusher.context_reset import (
    prune_tool_results_lightweight,
    reclaim_context,
)
from context_flusher.engine import (
    PRESETS,
    ContextFlusher,
    clamp_thresholds_to_n_ctx,
    clear_server_n_ctx_cache,
    probe_server_n_ctx,
)
from context_flusher.retriever import TaskRetriever
from context_flusher.sanitizer import (
    ProtocolSanitizer,
    estimate_messages_tokens,
    estimate_tokens_rough,
    sanitize_message_sequence,
)
from context_flusher.wrapper import WrappedOpenAIClient, wrap_openai

__version__ = "0.1.0"
__all__ = [
    "ContextFlusher",
    "PRESETS",
    "clamp_thresholds_to_n_ctx",
    "probe_server_n_ctx",
    "clear_server_n_ctx_cache",
    "TaskRetriever",
    "create_task_archive",
    "extract_heuristic_decisions",
    "extract_touched_files",
    "reclaim_context",
    "prune_tool_results_lightweight",
    "estimate_messages_tokens",
    "estimate_tokens_rough",
    "sanitize_message_sequence",
    "ProtocolSanitizer",
    "WrappedOpenAIClient",
    "wrap_openai",
]

