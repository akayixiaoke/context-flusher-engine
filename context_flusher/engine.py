"""ContextFlusher: Core Orchestrator & Middleware Engine.

Coordinates the task-aware context lifecycle:
- Manages soft threshold (default 32k) and hard ceiling (default 60k).
- Defers compaction during continuous multi-step tool loops.
- Executes zero-LLM structured archiving and KV cache reset at turn boundaries.
- Provides seamless wrapper integration with OpenAI SDK / llama-server / Ollama.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import yaml

from context_flusher.archiver import create_task_archive
from context_flusher.context_reset import (
    prune_tool_results_lightweight,
    reclaim_context,
)
from context_flusher.retriever import TaskRetriever
from context_flusher.sanitizer import (
    estimate_messages_tokens,
    sanitize_message_sequence,
)

logger = logging.getLogger("context_flusher")

# Presets strictly categorized by token context window size (not hardware vendor assumptions)
PRESETS: Dict[str, Dict[str, int]] = {
    "8k": {"active_threshold": 8000, "hard_ceiling": 14000},
    "16k": {"active_threshold": 16000, "hard_ceiling": 28000},
    "32k": {"active_threshold": 32000, "hard_ceiling": 60000},
    "64k": {"active_threshold": 64000, "hard_ceiling": 120000},
}

_SERVER_N_CTX_CACHE: Dict[str, Optional[int]] = {}


def clear_server_n_ctx_cache() -> None:
    """Clear in-memory server n_ctx probe cache (useful for testing)."""
    _SERVER_N_CTX_CACHE.clear()


def probe_server_n_ctx(
    base_url: Optional[str],
    model: Optional[str] = None,
    timeout: float = 2.0,
) -> Optional[int]:
    """Probe server context window size (n_ctx) from llama-server (/props) or Ollama (/api/show).
    
    Returns detected n_ctx integer if successful, or None if unreachable or unsupported.
    Cached per base_url/model for the lifetime of the process with a strict 2.0s timeout.
    """
    if not base_url:
        return None

    url_str = str(base_url).strip().rstrip("/")
    cache_key = f"{url_str}@{model}" if model else url_str
    if cache_key in _SERVER_N_CTX_CACHE:
        return _SERVER_N_CTX_CACHE[cache_key]
    if url_str in _SERVER_N_CTX_CACHE and _SERVER_N_CTX_CACHE[url_str] is not None:
        return _SERVER_N_CTX_CACHE[url_str]

    root_url = url_str
    if root_url.endswith("/v1"):
        root_url = root_url[:-3].rstrip("/")

    # 1. Try llama.cpp /props endpoint (global server setting, no model required)
    try:
        props_url = f"{root_url}/props"
        req = urllib.request.Request(props_url, headers={"User-Agent": "ContextFlusher/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                n_ctx = data.get("default_generation_settings", {}).get("n_ctx")
                if n_ctx and isinstance(n_ctx, (int, float)) and n_ctx > 0:
                    val = int(n_ctx)
                    _SERVER_N_CTX_CACHE[cache_key] = val
                    _SERVER_N_CTX_CACHE[url_str] = val
                    return val
    except Exception:
        pass

    # 2. Try Ollama /api/show if model name is provided
    if model:
        try:
            show_url = f"{root_url}/api/show"
            body = json.dumps({"name": model}).encode("utf-8")
            req = urllib.request.Request(
                show_url,
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": "ContextFlusher/0.1"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    model_info = data.get("model_info", {})
                    for k, v in model_info.items():
                        if "context_length" in k and isinstance(v, (int, float)) and v > 0:
                            val = int(v)
                            _SERVER_N_CTX_CACHE[cache_key] = val
                            return val
        except Exception:
            pass

    _SERVER_N_CTX_CACHE[cache_key] = None
    return None


def clamp_thresholds_to_n_ctx(
    active_threshold: int,
    hard_ceiling: int,
    server_n_ctx: Optional[int],
) -> Tuple[int, int, bool]:
    """Clamps ceiling to 90% of server_n_ctx and scales active_threshold proportionally.
    
    Preserves user-configured ratio (active / ceiling) without arbitrarily distorting shape.
    Emits an explicit yellow warning with actionable remediation advice.
    
    Returns:
        (clamped_active, clamped_ceiling, was_clamped)
    """
    if not server_n_ctx or server_n_ctx <= 0:
        return active_threshold, hard_ceiling, False

    if hard_ceiling >= server_n_ctx:
        clamped_ceiling = max(100, int(server_n_ctx * 0.90))
        scale_ratio = clamped_ceiling / max(1, hard_ceiling)
        clamped_active = max(50, int(active_threshold * scale_ratio))

        msg = (
            f"[ContextFlusher WARNING] Detected server n_ctx ({server_n_ctx}) <= hard_ceiling ({hard_ceiling}). "
            f"Thresholds automatically clamped with equal-ratio scaling: active={clamped_active}, ceiling={clamped_ceiling}. "
            "建议检查服务端启动参数 --ctx-size 或 --n-ctx (Suggest checking server args: --ctx-size or --n-ctx)"
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return clamped_active, clamped_ceiling, True

    return active_threshold, hard_ceiling, False


class ContextFlusher:
    """Task-aware context flushing and KV cache management middleware."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        *,
        preset: Optional[str] = None,
        workspace_dir: Optional[str | Path] = None,
        active_threshold: Optional[int] = None,
        hard_ceiling: Optional[int] = None,
        server_n_ctx: Optional[int] = None,
        base_url: Optional[str] = None,
        archives_subdir: str = "archives",
        progress_filename: str = "TASK_PROGRESS.md",
        keep_tail_turns: int = 1,
        inject_checkpoint_notice: bool = True,
        max_tool_history_steps: int = 15,
        min_prune_chars: int = 500,
        print_diff_by_default: bool = True,
    ) -> None:
        # Load from config file if provided
        cfg = {}
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Failed to load config from %s: %s", config_path, e)

        # 1. Determine preset base (arg -> env -> config -> default 32k)
        selected_preset = preset or os.getenv("CONTEXT_FLUSHER_PRESET") or cfg.get("preset")
        base_thresholds = PRESETS.get(str(selected_preset).lower(), PRESETS["32k"])

        # 2. Strict 4-Tier Priority: explicit argument -> env var -> config file -> preset base
        env_ws = os.getenv("CONTEXT_FLUSHER_WORKSPACE")
        self.workspace_dir = Path(
            workspace_dir or env_ws or cfg.get("workspace_dir", "./workspace")
        ).resolve()

        cfg_thresholds = cfg.get("thresholds", {})
        env_active = os.getenv("CONTEXT_FLUSHER_ACTIVE_THRESHOLD")
        env_hard = os.getenv("CONTEXT_FLUSHER_HARD_CEILING")

        if active_threshold is not None:
            raw_active = int(active_threshold)
        elif env_active:
            raw_active = int(env_active)
        elif "active_threshold" in cfg_thresholds:
            raw_active = int(cfg_thresholds["active_threshold"])
        else:
            raw_active = base_thresholds["active_threshold"]

        if hard_ceiling is not None:
            raw_hard = int(hard_ceiling)
        elif env_hard:
            raw_hard = int(env_hard)
        elif "hard_ceiling" in cfg_thresholds:
            raw_hard = int(cfg_thresholds["hard_ceiling"])
        else:
            raw_hard = base_thresholds["hard_ceiling"]

        # 3. Server context detection & clamping
        self.base_url = str(base_url) if base_url else None
        self.server_n_ctx = int(server_n_ctx) if server_n_ctx is not None else None

        if self.base_url and self.server_n_ctx is None:
            self.server_n_ctx = probe_server_n_ctx(self.base_url, timeout=2.0)

        act, ceil, _ = clamp_thresholds_to_n_ctx(raw_active, raw_hard, self.server_n_ctx)
        self.active_threshold = act
        self.hard_ceiling = ceil

        self.print_diff_by_default = print_diff_by_default

        arch_cfg = cfg.get("archiver", {})
        self.archives_subdir = str(arch_cfg.get("archives_subdir", archives_subdir))
        self.progress_filename = str(arch_cfg.get("progress_filename", progress_filename))
        self.max_tool_history_steps = int(
            arch_cfg.get("max_tool_history_steps", max_tool_history_steps)
        )

        reset_cfg = cfg.get("context_reset", {})
        self.keep_tail_turns = int(reset_cfg.get("keep_tail_turns", keep_tail_turns))
        self.inject_checkpoint_notice = bool(
            reset_cfg.get("inject_checkpoint_notice", inject_checkpoint_notice)
        )
        self.min_prune_chars = int(reset_cfg.get("min_prune_chars", min_prune_chars))

        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.retriever = TaskRetriever(
            workspace_dir=self.workspace_dir,
            archives_subdir=self.archives_subdir,
            progress_filename=self.progress_filename,
        )

        logger.info(
            "ContextFlusher initialized (workspace=%s, active_threshold=%s, hard_ceiling=%s, preset=%s, server_n_ctx=%s)",
            self.workspace_dir,
            f"{self.active_threshold:,}",
            f"{self.hard_ceiling:,}",
            selected_preset or "custom",
            f"{self.server_n_ctx:,}" if self.server_n_ctx else "unknown",
        )

    def set_thresholds(self, active_threshold: int, hard_ceiling: int) -> None:
        """Dynamically adjust soft threshold and hard ceiling at runtime with server n_ctx clamping."""
        act, ceil, clamped = clamp_thresholds_to_n_ctx(
            active_threshold=int(active_threshold),
            hard_ceiling=int(hard_ceiling),
            server_n_ctx=self.server_n_ctx,
        )
        self.active_threshold = act
        self.hard_ceiling = ceil
        logger.info(
            "Context thresholds dynamically adjusted: active=%s, ceiling=%s%s",
            f"{self.active_threshold:,}",
            f"{self.hard_ceiling:,}",
            " (clamped to server n_ctx)" if clamped else "",
        )

    def probe_and_clamp(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 2.0,
    ) -> bool:
        """Probe server context window and clamp thresholds if necessary. Returns True if clamped."""
        target_url = base_url or self.base_url
        if not target_url:
            return False
        self.base_url = target_url
        detected = probe_server_n_ctx(target_url, model=model, timeout=timeout)
        if detected:
            self.server_n_ctx = detected
            act, ceil, clamped = clamp_thresholds_to_n_ctx(
                self.active_threshold, self.hard_ceiling, detected
            )
            self.active_threshold = act
            self.hard_ceiling = ceil
            return clamped
        return False

    @staticmethod
    def _get_pending_tool_calls(messages: List[Dict[str, Any]]) -> Set[str]:
        """Collect all assistant tool_call_ids that have not received a matching tool response."""
        pending: Set[str] = set()
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if isinstance(tc, dict) and tc.get("id"):
                        pending.add(tc["id"])
            elif m.get("role") == "tool" and m.get("tool_call_id"):
                pending.discard(m["tool_call_id"])
        return pending

    @staticmethod
    def _extract_tool_call_info(
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        """Extract tool name and arguments from assistant tool_calls."""
        info: Dict[str, Dict[str, str]] = {}
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if isinstance(tc, dict) and tc.get("id"):
                        fn = tc.get("function", {})
                        if isinstance(fn, dict):
                            name = fn.get("name", "unknown_tool")
                            args = fn.get("arguments", "{}")
                        else:
                            name = "unknown_tool"
                            args = "{}"
                        info[tc["id"]] = {"name": str(name), "arguments": str(args)}
        return info

    def _apply_emergency_escape_hatch(
        self, messages: List[Dict[str, Any]], pending_tools: Set[str]
    ) -> List[Dict[str, Any]]:
        """Hard ceiling emergency escape hatch: inject identified emergency stubs for hung tool calls."""
        escaped: List[Dict[str, Any]] = [dict(m) for m in messages]
        tool_info = self._extract_tool_call_info(messages)
        archive_ref = f"{self.archives_subdir}/TASK_*.md"

        for call_id in sorted(pending_tools):
            meta = tool_info.get(call_id, {"name": "unknown_tool", "arguments": "{}"})
            fn_name = meta["name"]
            raw_args = meta["arguments"]
            args_summary = (raw_args[:200] + "...") if len(raw_args) > 200 else raw_args

            escaped.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": (
                    f"[System Alert: Tool '{fn_name}' (args: {args_summary}) execution status UNKNOWN - "
                    f"timed out or truncated before response. "
                    f"DO NOT assume side-effects completed without verification. "
                    f"Full call archived at: {archive_ref}]"
                ),
            })
        return escaped

    def preflight_check(
        self,
        messages: List[Dict[str, Any]],
        *,
        api_call_count: int = 0,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Pre-API call evaluation gate with layered iron law and escape hatch.

        Returns (updated_messages, action):
        - action "ok": continue with call normally.
        - action "pruned": hard ceiling hit, messages lightweight pruned (YELLOW).
        - action "emergency_escaped": hard ceiling hit with unclosed tool calls; escape hatch activated (RED).
        - action "deferred": >= active_threshold but pending tools or active loop; compaction deferred (Iron Law).
        """
        current_tokens = estimate_messages_tokens(messages)
        pending_tools = self._get_pending_tool_calls(messages)

        # 1. Hard Ceiling Guard with Emergency Escape Hatch
        if current_tokens >= self.hard_ceiling:
            if pending_tools:
                logger.warning(
                    "EMERGENCY ESCAPE HATCH: ~%s tokens >= hard ceiling (%s) with unclosed tool calls %s. "
                    "Injecting emergency stubs to prevent context window overflow.",
                    f"{current_tokens:,}",
                    f"{self.hard_ceiling:,}",
                    pending_tools,
                )
                escaped_msgs = self._apply_emergency_escape_hatch(messages, pending_tools)
                return escaped_msgs, "emergency_escaped"

            pruned_msgs, pruned_count = prune_tool_results_lightweight(
                messages,
                protect_tail_count=3,
                min_prune_chars=self.min_prune_chars,
            )
            if pruned_count > 0:
                logger.info(
                    "Hard Ceiling hit (~%s tokens). Lightweight pruned %d outputs.",
                    f"{current_tokens:,}",
                    pruned_count,
                )
                return pruned_msgs, "pruned"

        # 2. Layered Iron Law: Soft Threshold Check
        if current_tokens >= self.active_threshold:
            # If inside active tool loop or pending tools remain unclosed, STRICTLY DEFER!
            if api_call_count > 0 or pending_tools:
                logger.debug(
                    "Layered Iron Law: ~%s tokens >= %s threshold with pending tools=%s (step #%d). "
                    "Compaction strictly deferred until tool responses close.",
                    f"{current_tokens:,}",
                    f"{self.active_threshold:,}",
                    bool(pending_tools),
                    api_call_count,
                )
                return messages, "deferred"

        return messages, "ok"

    def finalize_turn(
        self,
        messages: List[Dict[str, Any]],
        *,
        final_response: Optional[str] = None,
        user_message: Optional[str] = None,
        task_name: Optional[str] = None,
        force_archive: bool = False,
        is_emergency: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Invoked when a multi-step tool sequence finishes or turn completes."""
        current_tokens = estimate_messages_tokens(messages)
        pending_tools = self._get_pending_tool_calls(messages)

        stats: Dict[str, Any] = {
            "archived": False,
            "tokens_before": current_tokens,
            "tokens_after": current_tokens,
            "tokens_reclaimed": 0,
            "archive_path": None,
            "lossiness_status": "GREEN",
        }

        # Iron law at turn finalization: if tool calls are pending and not emergency, defer
        if pending_tools and not is_emergency and current_tokens < self.hard_ceiling:
            logger.debug("Turn finalization deferred: tool calls %s still pending.", pending_tools)
            return messages, stats

        if current_tokens < self.active_threshold and not force_archive and not is_emergency:
            return sanitize_message_sequence(messages), stats

        logger.info(
            "Turn completing with ~%s tokens >= threshold (%s). Initiating task archive & context reset.",
            f"{current_tokens:,}",
            f"{self.active_threshold:,}",
        )

        # Determine 3-color status
        if is_emergency or (pending_tools and current_tokens >= self.hard_ceiling):
            status = "RED"
        elif any("[Tool output pruned" in str(m.get("content")) or "[Executed: output pruned" in str(m.get("content")) for m in messages):
            status = "YELLOW"
        else:
            status = "GREEN"

        # 1. Structured Markdown Archiving (2ms Python native)
        archive_path, summary = create_task_archive(
            messages,
            workspace_dir=self.workspace_dir,
            archives_subdir=self.archives_subdir,
            progress_filename=self.progress_filename,
            max_tool_history_steps=self.max_tool_history_steps,
            task_name=task_name,
            user_message=user_message,
            final_response=final_response,
            context_tokens=current_tokens,
        )

        if not archive_path:
            logger.error("Archiving failed. Aborting context reset to protect data.")
            return sanitize_message_sequence(messages), stats

        # 2. Context Reset & KV Cache Reclamation
        new_messages, reclaimed = reclaim_context(
            messages,
            archive_file=archive_path,
            keep_tail_turns=self.keep_tail_turns,
            inject_checkpoint_notice=self.inject_checkpoint_notice,
            progress_filename=self.progress_filename,
        )

        tokens_after = estimate_messages_tokens(new_messages)
        stats.update({
            "archived": True,
            "tokens_after": tokens_after,
            "tokens_reclaimed": reclaimed,
            "archive_path": str(archive_path),
            "reclaim_ratio": f"{((current_tokens - tokens_after) / max(1, current_tokens)) * 100:.1f}%",
            "lossiness_status": status,
        })

        if self.print_diff_by_default:
            print("\n" + self.format_flush_diff(stats))

        return new_messages, stats

    def recall_by_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Precision recall: find the most recent task and decisions that touched a specific file."""
        return self.retriever.recall_by_file(file_path)

    def recall_by_dir(self, dir_path: str) -> List[Dict[str, Any]]:
        """Recall all past task archives that touched files under a specific directory."""
        return self.retriever.recall_by_dir(dir_path)

    @staticmethod
    def format_flush_diff(stats: Dict[str, Any]) -> str:
        """Format an explicit human-readable reconciliation report with 3-color lossiness status."""
        if not stats.get("archived"):
            return "No flush occurred (context within operational threshold)."

        before = stats.get("tokens_before", 0)
        after = stats.get("tokens_after", 0)
        freed = stats.get("tokens_reclaimed", 0)
        ratio = stats.get("reclaim_ratio", "N/A")
        archive = stats.get("archive_path", "N/A")
        status = stats.get("lossiness_status", "GREEN")

        badge = {
            "GREEN": "[STATUS: [GREEN] CLEAN FLUSH (Lossless Split: Full context archived)]",
            "YELLOW": "[STATUS: [YELLOW] LIGHTWEIGHT PRUNED (Light Prune: Old tool outputs folded)]",
            "RED": "[STATUS: [RED] EMERGENCY ESCAPE HATCH (Emergency Escape: Hung calls truncated)]",
        }.get(status, f"[STATUS: [{status}]]")


        lines = [
            "========================================================================",
            "                 CONTEXT-FLUSHER RECONCILIATION DIFF                    ",
            f"  {badge}",
            "========================================================================",
            f" [Context Shift]   ~{before:,} -> ~{after:,} tokens (Freed {freed:,} tokens, -{ratio})",
            f" [Disk Guarantee]  100% full logs, diffs & tools written to: {archive}",
            " [Preserved]       System prompt (Prefix Cache safe), latest dialogue tail, checkpoint marker",
            " [Flushed]         Intermediate raw tool execution buffers (attention compute & context budget reclaimed)",
            "========================================================================",
        ]
        return "\n".join(lines)


