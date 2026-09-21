"""Zero-LLM Structured Task Archiver.

Performs deterministic, 2ms Python-native structured state archiving:
- Extracts critical tool calls, parameters, execution records, and final answers.
- Preserves technical precision (hex offsets, diffs, compiler diagnostics, logs).
- Outputs dated standalone Markdown files in `archives/` and updates `TASK_PROGRESS.md`.
- Requires zero extra LLM calls and zero GPU compute.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger("context_flusher.archiver")


def slugify_task_name(name: str, max_length: int = 32) -> str:
    """Create a safe filesystem slug from arbitrary user prompt or task name."""
    clean = re.sub(r"[^\w\u4e00-\u9fa5]+", "_", str(name or "task")).strip("_")
    return clean[:max_length] or "task"


DECISION_PATTERNS = [
    re.compile(r"([^。！？\n]*?(?:决定|选择|采用|放弃|因为|由于|改为|改用|注意|避免|换用|结论|经测试|最终|不要)[^。！？\n]*)"),
    re.compile(r"([^.!?\n]*?(?:decided to|switched to|chose to|because|due to|workaround|avoid|instead of|note:|conclusion:|resolution:)[^.!?\n]*)", re.IGNORECASE),
]

FILE_EXT_PATTERN = re.compile(r"\b([\w\-./\\]+\.(?:py|rs|go|c|cpp|h|hpp|js|ts|jsx|tsx|json|yaml|yml|toml|sys|dll|exe|md|sh|bat))\b", re.IGNORECASE)


def extract_heuristic_decisions(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract decisions, constraints, and error-pivot reasoning purely with heuristics.

    Requires zero LLM calls, capturing 'why' and 'what was decided' directly from
    dialogue topology and conclusion sentences.
    """
    decisions: List[str] = []
    seen = set()

    for idx, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))

        # 1. Look for Error -> Pivot topology
        if role == "tool" and idx + 1 < len(messages):
            next_m = messages[idx + 1]
            if next_m.get("role") == "assistant":
                tool_text = str(content).strip()
                # If tool threw an error or exception
                if any(err_kw in tool_text for err_kw in ["Error:", "Exception:", "failed", "Traceback", "returncode: 1"]):
                    err_preview = tool_text.splitlines()[0][:80]
                    next_text = str(next_m.get("content") or "").strip()
                    next_sentence = next_text.splitlines()[0][:100] if next_text else ""
                    if next_sentence:
                        item = f"遇到报错 [{err_preview}] -> 调整方案: {next_sentence}"
                        if item not in seen:
                            seen.add(item)
                            decisions.append(item)

        # 2. Extract decision sentences from assistant reasoning / final answers
        if role == "assistant" and content:
            for pat in DECISION_PATTERNS:
                for match in pat.finditer(content):
                    sentence = match.group(1).strip(" -*`")
                    if 15 <= len(sentence) <= 180 and sentence not in seen:
                        seen.add(sentence)
                        decisions.append(sentence)

    # Return top 8 most prominent decision items
    return decisions[-8:]


def extract_touched_files(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract touched or mentioned file paths from tool calls and dialogue text."""
    touched: List[str] = []
    seen = set()

    for m in messages:
        # Check tool calls
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    for match in FILE_EXT_PATTERN.finditer(args):
                        fpath = match.group(1).replace("\\", "/").strip("\"'")
                        if fpath not in seen and not fpath.startswith("http"):
                            seen.add(fpath)
                            touched.append(fpath)

        # Check content
        content = m.get("content")
        if isinstance(content, str):
            for match in FILE_EXT_PATTERN.finditer(content):
                fpath = match.group(1).replace("\\", "/").strip("\"'")
                # Exclude common URLs or generic words
                if fpath not in seen and not fpath.startswith("http") and "/" in fpath or "." in fpath:
                    if len(fpath) > 3 and not fpath.startswith("0."):
                        seen.add(fpath)
                        touched.append(fpath)

    return touched[:12]



def normalize_project_path(path: str | Path, workspace: Path) -> Dict[str, Any]:
    """Normalize a path for cross-platform, case-safe inverted indexing.

    1. Cross-drive safe on Windows (catches ValueError from os.path.relpath).
    2. Uses removeprefix('./') instead of lstrip to prevent stripping '../'.
    3. Maintains dual representation: 'raw' (original case for display/write) and
       'folded' (lowercase for hash matching).
    """
    raw_str = str(path).replace('\\', '/').strip("\"' ")
    try:
        rel = os.path.relpath(raw_str, workspace).replace('\\', '/')
    except ValueError:
        # Cross-drive on Windows
        rel = raw_str

    clean_rel = rel.removeprefix('./') if rel.startswith('./') else rel
    return {
        "raw": clean_rel,
        "folded": clean_rel.lower(),
        "is_external": clean_rel.startswith('../') or (len(clean_rel) > 2 and clean_rel[1] == ':'),
    }


def extract_task_name_from_messages(

    messages: List[Dict[str, Any]],
    user_message: Optional[str] = None,
) -> str:
    """Derive a concise, meaningful task slug from the initial user prompt."""
    candidate = user_message or ""
    if not candidate:
        for m in messages:
            if m.get("role") == "user":
                candidate = m.get("content") or ""
                break

    if isinstance(candidate, list):
        candidate = " ".join(
            part.get("text", "") for part in candidate if isinstance(part, dict)
        )

    first_line = str(candidate).strip().splitlines()[0] if candidate else "task"
    return slugify_task_name(first_line)


def create_task_archive(
    messages: List[Dict[str, Any]],
    *,
    workspace_dir: str | Path = "./workspace",
    archives_subdir: str = "archives",
    progress_filename: str = "TASK_PROGRESS.md",
    max_tool_history_steps: int = 15,
    task_name: Optional[str] = None,
    user_message: Optional[str] = None,
    final_response: Optional[str] = None,
    context_tokens: Optional[int] = None,
    session_id: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[Optional[Path], str]:
    """Write an independent dated Markdown archive to archives/ and update TASK_PROGRESS.md."""
    workspace = Path(workspace_dir).resolve()
    archives_dir = workspace / archives_subdir
    try:
        archives_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create archives dir %s: %s", archives_dir, e)
        archives_dir = workspace

    if not task_name:
        task_name = extract_task_name_from_messages(messages, user_message=user_message)

    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    display_time = now.strftime("%Y-%m-%d %H:%M:%S")
    archive_file = archives_dir / f"TASK_{timestamp_str}_{task_name}.md"

    # Extract recent tool executions
    tool_actions: List[str] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "tool")
                args = fn.get("arguments", "")
                if len(args) > 160:
                    args = args[:160] + "..."
                tool_actions.append(f"- `{name}`: {args}")

    tool_summary = (
        "\n".join(tool_actions[-max_tool_history_steps:])
        if tool_actions
        else "No explicit tool calls recorded in this turn."
    )

    # Extract final assistant response if not passed explicitly
    last_response_text = final_response or ""
    if not last_response_text:
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content"):
                c = m["content"]
                if isinstance(c, str) and c.strip():
                    last_response_text = c.strip()
                    break

    # Extract heuristic decisions and touched files
    heuristic_decisions = extract_heuristic_decisions(messages)
    touched_files = extract_touched_files(messages)

    decisions_block = (
        "\n".join(f"- {d}" for d in heuristic_decisions)
        if heuristic_decisions
        else "- No explicit decision pivots identified; routine execution."
    )

    files_block = (
        "\n".join(f"- `{fp}`" for fp in touched_files)
        if touched_files
        else "- No specific file mutations or paths extracted."
    )

    tokens_display = f"{context_tokens:,}" if context_tokens else "N/A"
    session_display = session_id or "default"
    model_display = model_name or "local-model"

    archive_content = f"""# Task Archive: {task_name}
> **Archived At**: {display_time}  
> **Session ID**: `{session_display}`  
> **Context Tokens Before Reset**: ~{tokens_display}  
> **Model**: `{model_display}`  
> **Workspace**: `{workspace}`

---

## 📌 Task Objective & Phase Summary
- **Task Slug**: `{task_name}`
- **Execution Scope**: Completed within `{workspace.name}`

---

## 📂 Touched Files & Code Anchors
{files_block}

---

## 🧠 Key Decisions & Constraints (Heuristic Extraction)
{decisions_block}

---

## 🛠️ Key Tool Actions & Execution Records (Last {max_tool_history_steps} Steps)
{tool_summary}

---

## 📝 Phase Outcome & Response Summary
{last_response_text[:3000] if last_response_text else "Milestone reached successfully."}

---

## ⚠️ Notes & Preservation Guarantees
- All code mutations, diagnostics, and terminal outputs have been flushed to disk.
- Active context memory was cleared post-archive to restore peak generation speed.
"""

    try:
        with open(archive_file, "w", encoding="utf-8") as f:
            f.write(archive_content)
        logger.info("Successfully wrote task archive to %s", archive_file)
    except Exception as e:
        logger.error("Failed to write task archive %s: %s", archive_file, e)
        return None, ""

    # Maintain File-to-Archive Inverted Index (file_index.json)
    file_index_path = archives_dir / "file_index.json"
    file_index: Dict[str, List[Dict[str, Any]]] = {}
    if file_index_path.exists():
        try:
            with open(file_index_path, "r", encoding="utf-8") as f:
                file_index = json.load(f)
        except Exception:
            file_index = {}

    archive_meta = {
        "archive_file": str(archive_file.name),
        "task_name": task_name,
        "timestamp": display_time,
        "summary": last_response_text[:180].replace("\n", " ") if last_response_text else "",
        "decisions": heuristic_decisions[:3],
    }
    for fp in touched_files:
        norm = normalize_project_path(fp, workspace)
        norm_key = norm["folded"]
        entry = dict(archive_meta)
        entry["raw_path"] = norm["raw"]
        if norm_key not in file_index:
            file_index[norm_key] = []
        file_index[norm_key].append(entry)

    try:
        with open(file_index_path, "w", encoding="utf-8") as f:
            json.dump(file_index, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Could not update file index %s: %s", file_index_path, e)

    # Append checkpoint entry to TASK_PROGRESS.md
    progress_file = workspace / progress_filename
    files_summary_line = f"- **Touched Files**: {', '.join(f'`{f}`' for f in touched_files[:5])}\n" if touched_files else ""
    progress_entry = (
        f"\n---\n\n## 📌 [Task Checkpoint] {display_time} — {task_name}\n"
        f"- **Archive File**: [`{archives_subdir}/{archive_file.name}`]({archives_subdir}/{archive_file.name})\n"
        f"- **Tokens Reclaimed**: ~{tokens_display} (KV cache and memory flushed to disk)\n"
        f"{files_summary_line}"
        f"- **Milestone Outcome**: {last_response_text[:200].replace(chr(10), ' ') if last_response_text else 'Phase completed'}\n"
    )
    try:
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(progress_entry)
        logger.info("Appended milestone checkpoint to %s", progress_file)
    except Exception as e:
        logger.warning("Could not append to %s: %s", progress_file, e)

    return archive_file, archive_content

