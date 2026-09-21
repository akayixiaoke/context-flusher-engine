"""Lightweight Archive & Progress Retriever.

Allows agents and users to search, query, and restore past task state from
`archives/` and `TASK_PROGRESS.md` on demand without cluttering the active context window.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class TaskRetriever:
    """Zero-overhead file-based retriever for task archives and progress roadmaps."""

    def __init__(
        self,
        workspace_dir: str | Path = "./workspace",
        archives_subdir: str = "archives",
        progress_filename: str = "TASK_PROGRESS.md",
    ) -> None:
        self.workspace = Path(workspace_dir).resolve()
        self.archives_dir = self.workspace / archives_subdir
        self.progress_file = self.workspace / progress_filename

    def read_progress_summary(self, max_lines: int = 100) -> str:
        """Read the cumulative progress file (or the latest lines)."""
        if not self.progress_file.exists():
            return "No TASK_PROGRESS.md found in workspace."
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return "".join(lines[-max_lines:])
        except Exception as e:
            return f"Error reading {self.progress_file.name}: {e}"

    def list_archives(self) -> List[Dict[str, Any]]:
        """List all available task archives sorted by timestamp (newest first)."""
        if not self.archives_dir.exists():
            return []

        results = []
        for file in sorted(self.archives_dir.glob("TASK_*.md"), reverse=True):
            stat = file.stat()
            results.append({
                "filename": file.name,
                "path": str(file),
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
            })
        return results

    def search_archives(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search across all archived markdown files for matching keywords or phrases."""
        if not self.archives_dir.exists():
            return []

        query_lower = query.lower()
        matches = []

        for file in sorted(self.archives_dir.glob("TASK_*.md"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                if query_lower in content.lower():
                    # Extract surrounding context snippet
                    match_pos = content.lower().find(query_lower)
                    start = max(0, match_pos - 100)
                    end = min(len(content), match_pos + 200)
                    snippet = content[start:end].strip()
                    matches.append({
                        "filename": file.name,
                        "path": str(file),
                        "snippet": f"...{snippet}...",
                    })
                    if len(matches) >= max_results:
                        break
            except Exception:
                continue

        return matches

    def recall_by_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Precision recall: find the most recent task and decisions that touched a specific file.

        Eliminates the synonym mismatch problem of natural language search (e.g. 'login' vs 'auth')
        by anchoring directly to deterministic file paths and symbols.
        """
        if not file_path:
            return None

        raw_str = str(file_path).replace("\\", "/").strip("\"' ")
        clean_target_basename = Path(raw_str).name.lower()
        clean_target_full = raw_str.removeprefix("./").lower()

        # 1. Try reading the precomputed inverted index (file_index.json)
        index_path = self.archives_dir / "file_index.json"
        if index_path.exists():
            try:
                import json
                with open(index_path, "r", encoding="utf-8") as f:
                    file_index: Dict[str, List[Dict[str, Any]]] = json.load(f)

                # Match by folded full path or basename
                for key, entries in file_index.items():
                    key_norm = key.replace("\\", "/").removeprefix("./").lower().strip()
                    if key_norm == clean_target_full or Path(key_norm).name == clean_target_basename:
                        if entries:
                            latest_entry = entries[-1]
                            archive_path = self.archives_dir / latest_entry.get("archive_file", "")
                            full_text = ""
                            if archive_path.exists():
                                with open(archive_path, "r", encoding="utf-8") as af:
                                    full_text = af.read()
                            return {
                                "file": file_path,
                                "raw_path": latest_entry.get("raw_path", file_path),
                                "matched_key": key,
                                "task_name": latest_entry.get("task_name"),
                                "timestamp": latest_entry.get("timestamp"),
                                "summary": latest_entry.get("summary"),
                                "decisions": latest_entry.get("decisions", []),
                                "archive_file": str(archive_path),
                                "content": full_text,
                            }
            except Exception:
                pass

        # 2. Fallback: Scan archive Markdown files directly for filename anchor
        for file in sorted(self.archives_dir.glob("TASK_*.md"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                if clean_target_basename in content.lower():
                    return {
                        "file": file_path,
                        "raw_path": file_path,
                        "matched_key": clean_target_basename,
                        "task_name": file.stem,
                        "archive_file": str(file),
                        "summary": "Matched in archive markdown",
                        "decisions": [],
                        "content": content,
                    }
            except Exception:
                continue

        return None

    def recall_by_dir(self, dir_path: str) -> List[Dict[str, Any]]:
        """Precision recall: find all task archives that touched files inside a directory."""
        if not dir_path:
            return []

        clean_dir = str(dir_path).replace("\\", "/").strip("\"' /").removeprefix("./").lower() + "/"
        index_path = self.archives_dir / "file_index.json"
        if not index_path.exists():
            return []

        matching_tasks: Dict[str, Dict[str, Any]] = {}
        try:
            import json
            with open(index_path, "r", encoding="utf-8") as f:
                file_index: Dict[str, List[Dict[str, Any]]] = json.load(f)

            for key, entries in file_index.items():
                key_norm = key.replace("\\", "/").removeprefix("./").lower()
                if key_norm.startswith(clean_dir) or clean_dir in key_norm:
                    for entry in entries:
                        task_name = entry.get("task_name", "")
                        if task_name not in matching_tasks:
                            matching_tasks[task_name] = {
                                "dir": dir_path,
                                "matched_file": entry.get("raw_path", key),
                                "task_name": task_name,
                                "timestamp": entry.get("timestamp"),
                                "summary": entry.get("summary"),
                                "archive_file": str(self.archives_dir / entry.get("archive_file", "")),
                            }
        except Exception:
            pass

        return list(matching_tasks.values())


    def list_files_indexed(self) -> List[str]:
        """List all touched files tracked across past task archives."""
        index_path = self.archives_dir / "file_index.json"
        if not index_path.exists():
            return []
        try:
            import json
            with open(index_path, "r", encoding="utf-8") as f:
                file_index = json.load(f)
            return sorted(list(file_index.keys()))
        except Exception:
            return []

    def get_latest_archive(self) -> Optional[str]:
        """Fetch the full content of the most recent task archive."""
        archives = self.list_archives()
        if not archives:
            return None
        latest_file = Path(archives[0]["path"])
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

