"""Unit & Integration Tests for Context-Flusher Engine."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path

# Add package root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_flusher import (
    PRESETS,
    ContextFlusher,
    TaskRetriever,
    clamp_thresholds_to_n_ctx,
    clear_server_n_ctx_cache,
    create_task_archive,
    estimate_messages_tokens,
    probe_server_n_ctx,
    prune_tool_results_lightweight,
    reclaim_context,
    sanitize_message_sequence,
)


def test_sanitizer_orphaned_tool_calls():
    """Test that assistant messages with orphaned tool_calls are sanitized to prevent 400 Bad Request."""
    messages = [
        {"role": "user", "content": "Run tests"},
        {
            "role": "assistant",
            "content": "Running...",
            "tool_calls": [{"id": "call_123", "function": {"name": "bash", "arguments": "{}"}}],
        },
        # Notice: No corresponding tool message with id call_123 (e.g. pruned)
        {"role": "assistant", "content": "Done!"},
    ]

    sanitized = sanitize_message_sequence(messages)
    # The tool_calls key should be stripped since call_123 has no matching tool response
    assert "tool_calls" not in sanitized[1]
    assert sanitized[1]["content"] == "Running..."


def test_archiver_and_reclamation():
    """Test full cycle of archiving and context reclamation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir)

        messages = [
            {"role": "system", "content": "You are a helpful coding agent."},
            {"role": "user", "content": "Disassemble function at 0x401000"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "objdump", "arguments": "-d target.bin"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "0x401000: push rbp\n0x401001: mov rbp, rsp\n" * 500,  # ~10k chars
            },
            {
                "role": "assistant",
                "content": "Identified critical function preamble at offset 0x401000.",
            },
        ]

        tokens_before = estimate_messages_tokens(messages)
        assert tokens_before > 2000

        # 1. Archive
        archive_file, content = create_task_archive(
            messages,
            workspace_dir=workspace,
            task_name="reverse_step_1",
            context_tokens=tokens_before,
        )

        assert archive_file is not None
        assert archive_file.exists()
        assert archive_file.stat().st_size > 0
        assert "TASK_PROGRESS.md" in [f.name for f in workspace.iterdir()]

        # 2. Reclaim Context
        new_msgs, reclaimed = reclaim_context(
            messages,
            archive_file=archive_file,
            progress_filename="TASK_PROGRESS.md",
        )

        tokens_after = estimate_messages_tokens(new_msgs)
        assert len(new_msgs) < len(messages)
        assert reclaimed > 0
        assert tokens_after < tokens_before
        assert any("Context Flushed" in m.get("content", "") for m in new_msgs)

        # 3. Test Retriever
        retriever = TaskRetriever(workspace_dir=workspace)
        archives = retriever.list_archives()
        assert len(archives) == 1
        assert "reverse_step_1" in archives[0]["filename"]

        search_results = retriever.search_archives("0x401000")
        assert len(search_results) >= 1


def test_flusher_engine_lifecycle():
    """Test ContextFlusher middleware preflight and turn finalization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        flusher = ContextFlusher(
            workspace_dir=tmp_dir,
            active_threshold=1000,  # Low threshold for test
            hard_ceiling=5000,
        )

        messages = [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "Task 1: generate big data"},
            {
                "role": "assistant",
                "content": "Executing...",
                "tool_calls": [{"id": "tc1", "function": {"name": "gen", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "sample data output line\n" * 400},
            {"role": "assistant", "content": "Task 1 finished."},
        ]

        tokens = estimate_messages_tokens(messages)
        assert tokens > 1000

        # Preflight while in continuous execution (api_call_count > 0) -> should defer
        msgs, action = flusher.preflight_check(messages, api_call_count=1)
        assert action == "deferred"

        # Finalize turn -> should trigger archive & context reset
        final_msgs, stats = flusher.finalize_turn(
            messages,
            final_response="Task 1 finished.",
            user_message="Task 1: generate big data",
        )

        assert stats["archived"] is True
        assert stats["tokens_reclaimed"] > 0
        assert len(final_msgs) < len(messages)


def test_protocol_sanitizer_validation():
    """Test ProtocolSanitizer validation and cleanup of orphaned calls."""
    from context_flusher import ProtocolSanitizer

    # Invalid sequence: assistant has tool_calls but user responds next without tool outputs
    corrupted = [
        {"role": "system", "content": "Prompt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "orphan_call_1", "function": {"name": "test", "arguments": "{}"}}],
        },
        {"role": "user", "content": "Interruption"},
    ]

    issues = ProtocolSanitizer.validate_sequence(corrupted)
    assert len(issues) > 0

    sanitized = ProtocolSanitizer.clean_and_validate(corrupted)
    # The empty assistant message with dangling tool_calls should be safely purged
    clean_issues = ProtocolSanitizer.validate_sequence(sanitized)
    assert len(clean_issues) == 0


def test_wrap_openai_mutation():
    """Test wrap_openai proxy and in-place context mutation."""
    from context_flusher import wrap_openai

    class MockResp:
        def __init__(self, content, tool_calls=None):
            self.choices = [
                type("Choice", (), {
                    "message": type("Msg", (), {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                        "model_dump": lambda self: {"role": "assistant", "content": content}
                    })()
                })()
            ]

    class MockClient:
        def __init__(self):
            self.chat = type("Chat", (), {
                "completions": type("Completions", (), {
                    "create": lambda *a, **kw: MockResp(content="Finished task", tool_calls=None)
                })()
            })()

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_client = MockClient()
        wrapped = wrap_openai(raw_client, workspace_dir=tmp_dir, active_threshold=500)

        # Call with large context exceeding threshold
        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Process big file"},
            {"role": "assistant", "content": "Calling tool", "tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "huge output line\n" * 500},
            {"role": "assistant", "content": "Calling tool 2", "tool_calls": [{"id": "t2", "function": {"name": "f2", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t2", "content": "another huge output\n" * 500},
        ]
        tokens_orig = estimate_messages_tokens(messages)
        resp = wrapped.chat.completions.create(model="mock", messages=messages)

        tokens_new = estimate_messages_tokens(messages)
        # Context must be substantially reduced in place
        assert tokens_new < tokens_orig
        assert len(messages) < 6
        assert hasattr(resp, "context_flusher_stats")
        assert resp.context_flusher_stats["archived"] is True


def test_heuristic_decision_extraction():
    """Test zero-LLM heuristic extraction of decision and error-pivot reasoning."""
    from context_flusher import extract_heuristic_decisions, extract_touched_files

    messages = [
        {"role": "user", "content": "Update auth module"},
        {
            "role": "assistant",
            "content": "Running test on auth.py",
            "tool_calls": [{"id": "tc1", "function": {"name": "run", "arguments": '{"file": "src/auth/session.py"}'}}],
        },
        {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": "Error: Traceback (most recent call last): InvalidTokenException in session.py line 42",
        },
        {
            "role": "assistant",
            "content": "经测试发现原方案有死锁隐患。因此决定采用双重检查锁（DCL）重构 session.py。注意不要直接修改全局锁字典。",
        },
    ]

    decisions = extract_heuristic_decisions(messages)
    assert len(decisions) >= 1
    # Check that error-pivot and decision sentences were captured
    assert any("调整方案" in d or "死锁" in d or "决定" in d or "注意" in d for d in decisions)

    files = extract_touched_files(messages)
    assert "src/auth/session.py" in files or "auth.py" in files or "session.py" in files


def test_recall_by_file():
    """Test file-centric clustering and recall_by_file functionality."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir)

        messages = [
            {"role": "user", "content": "Refactor driver dispatch in target_driver.sys"},
            {
                "role": "assistant",
                "content": "Patched IRP dispatch table in target_driver.sys. 决定放弃hook方案，改为直接修改分发函数指针。",
                "tool_calls": [{"id": "t1", "function": {"name": "patch", "arguments": '{"target": "target_driver.sys"}'}}],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "Patch written successfully."},
            {"role": "assistant", "content": "Verification passed on target_driver.sys."},
        ]

        # 1. Create Archive
        create_task_archive(
            messages,
            workspace_dir=workspace,
            task_name="patch_driver",
            context_tokens=3000,
        )

        retriever = TaskRetriever(workspace_dir=workspace)
        # Recall by exact file name
        recalled = retriever.recall_by_file("target_driver.sys")
        assert recalled is not None
        assert recalled["task_name"] == "patch_driver"
        assert "target_driver.sys" in recalled["file"]

        # Recall non-existent file returns None
        assert retriever.recall_by_file("non_existent.py") is None


def test_tool_stubbing_intent_preservation():
    """Test that stub_pruned_tools preserves tool_calls and tool matching while saving tokens."""
    from context_flusher import ProtocolSanitizer

    messages = [
        {"role": "system", "content": "Sys"},
        {
            "role": "assistant",
            "content": "Calling db migration",
            "tool_calls": [{"id": "call_db_01", "function": {"name": "migrate_db", "arguments": '{"drop": true}'}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_db_01",
            "content": "Database migration executed: table users dropped, recreated, indexed.\n" * 50,  # ~3,000 chars
        },
        {"role": "assistant", "content": "Next action...", "tool_calls": [{"id": "call_recent", "function": {"name": "status", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_recent", "content": "recent short output"},
    ]

    # Stub tool results
    stubbed = ProtocolSanitizer.stub_pruned_tools(messages, min_chars_to_stub=200, protect_tail_tools=1)

    # 1. tool_calls in assistant MUST be preserved
    assert "tool_calls" in stubbed[1]
    assert stubbed[1]["tool_calls"][0]["id"] == "call_db_01"

    # 2. matching tool message MUST be preserved with stub text
    assert stubbed[2]["role"] == "tool"
    assert stubbed[2]["tool_call_id"] == "call_db_01"
    assert "Result recorded" in stubbed[2]["content"]
    assert "DO NOT repeat" in stubbed[2]["content"]

    # 3. Protocol validation must pass with zero issues
    issues = ProtocolSanitizer.validate_sequence(stubbed)
    assert len(issues) == 0


def test_format_flush_diff():
    """Test format_flush_diff human-readable terminal report with 3-color status."""
    stats = {
        "archived": True,
        "tokens_before": 35000,
        "tokens_after": 1500,
        "tokens_reclaimed": 33500,
        "reclaim_ratio": "95.7%",
        "archive_path": "archives/TASK_test.md",
        "lossiness_status": "GREEN",
    }
    report = ContextFlusher.format_flush_diff(stats)
    assert "CONTEXT-FLUSHER RECONCILIATION DIFF" in report
    assert "GREEN" in report
    assert "35,000" in report
    assert "1,500" in report
    assert "33,500" in report


def test_path_normalization_safety():
    """Test path normalization: removeprefix('./') does NOT strip '../', handles dual-case."""
    from context_flusher.archiver import normalize_project_path

    ws = Path("C:/project/workspace")
    # 1. Test ../ is preserved and not stripped to x
    norm_external = normalize_project_path("../external/util.py", ws)
    assert norm_external["raw"].startswith("../")
    assert norm_external["is_external"] is True

    # 2. Test ./ prefix is removed cleanly
    norm_local = normalize_project_path("./src/auth/session.py", ws)
    assert not norm_local["raw"].startswith("./")
    assert "src/auth/session.py" in norm_local["raw"]
    assert norm_local["folded"] == norm_local["raw"].lower()


def test_threshold_tuning_and_presets():
    """Test preset loading (8k/14k, 32k/60k) and dynamic runtime adjustment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Preset 8k for 8G/12G GPUs
        flusher_8k = ContextFlusher(workspace_dir=tmp_dir, preset="8k")
        assert flusher_8k.active_threshold == 8000
        assert flusher_8k.hard_ceiling == 14000

        # 2. Dynamic runtime adjustment
        flusher_8k.set_thresholds(active_threshold=5000, hard_ceiling=9000)
        assert flusher_8k.active_threshold == 5000
        assert flusher_8k.hard_ceiling == 9000

        # 3. Explicit override beats preset
        flusher_custom = ContextFlusher(
            workspace_dir=tmp_dir,
            preset="8k",
            active_threshold=10000,
            hard_ceiling=20000,
        )
        assert flusher_custom.active_threshold == 10000
        assert flusher_custom.hard_ceiling == 20000


def test_layered_iron_law_and_escape_hatch():
    """Test layered iron law: soft threshold defers on pending calls; hard ceiling triggers emergency escape hatch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        flusher = ContextFlusher(
            workspace_dir=tmp_dir,
            active_threshold=2000,
            hard_ceiling=8000,
            print_diff_by_default=False,
        )

        # Sequence with pending tool call (tool has not returned yet)
        messages = [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "Run hung task"},
            {
                "role": "assistant",
                "content": "Executing hung tool...",
                "tool_calls": [{"id": "call_hung_001", "function": {"name": "sleep_forever", "arguments": "{}"}}],
            },
            # Notice: tool message for call_hung_001 is missing!
        ]

        # Artificially expand messages to exceed soft threshold (~3,000 tokens)
        messages[1]["content"] = "Long user prompt " * 600

        # 1. Soft threshold check: MUST defer (Iron Law)
        msgs, action = flusher.preflight_check(messages, api_call_count=1)
        assert action == "deferred"

        # 2. Hard ceiling check: simulate context reaching 8,500 tokens with hung tool
        messages[1]["content"] = "Gigantic prompt " * 2000
        escaped_msgs, emergency_action = flusher.preflight_check(messages, api_call_count=2)
        assert emergency_action == "emergency_escaped"
        # Emergency escape hatch injected stub for call_hung_001
        stub = next((m for m in escaped_msgs if m.get("role") == "tool" and m.get("tool_call_id") == "call_hung_001"), None)
        assert stub is not None
        # Must contain tool identity, arguments summary, warning, and archive ref
        assert "sleep_forever" in stub["content"]
        assert "{}" in stub["content"]
        assert "DO NOT assume side-effects completed without verification" in stub["content"]
        assert "Full call archived at:" in stub["content"]


def test_defer_starvation_recovery():
    """Verify that a deferred flush during a multi-step tool loop fires immediately upon clean turn completion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        flusher = ContextFlusher(
            workspace_dir=tmp_dir,
            active_threshold=800,
            hard_ceiling=5000,
            print_diff_by_default=False,
        )

        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Run multi-step task"},
            {
                "role": "assistant",
                "content": "Step 1: calling tool...",
                "tool_calls": [{"id": "tc_step1", "function": {"name": "fetch", "arguments": "{}"}}],
            },
            # Tool has returned heavy payload
            {"role": "tool", "tool_call_id": "tc_step1", "content": "data block\n" * 300},
        ]

        # 1. During multi-step execution (step count > 0), soft threshold triggers deferral
        updated_msgs, action = flusher.preflight_check(messages, api_call_count=1)
        assert action == "deferred"

        # 2. Assistant now completes turn with final text answer (no more tool calls)
        final_response = "All multi-step operations completed successfully."
        messages.append({"role": "assistant", "content": final_response})

        # 3. Finalize turn triggers immediately - starvation recovered!
        reclaimed_msgs, stats = flusher.finalize_turn(
            messages,
            final_response=final_response,
            user_message="Run multi-step task",
        )

        assert stats["archived"] is True
        assert stats["tokens_reclaimed"] > 0
        assert stats["lossiness_status"] == "GREEN"
        assert len(reclaimed_msgs) < len(messages)


def test_server_n_ctx_clamping():
    """Verify equal-ratio scaling, 2s timeout probe, and runtime set_thresholds clamping."""
    # 1. Standalone equal-ratio clamping logic
    # Ratio: 8000 / 14000 = 0.5714...
    # server_n_ctx = 8192 <= 14000 -> clamp!
    # clamped_ceiling = int(8192 * 0.90) = 7372
    # clamped_active = int(8000 * (7372 / 14000)) = 4212
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        act, ceil, clamped = clamp_thresholds_to_n_ctx(
            active_threshold=8000,
            hard_ceiling=14000,
            server_n_ctx=8192,
        )
        assert clamped is True
        assert ceil == 7372
        assert act == 4212
        assert len(w) == 1
        assert "建议检查服务端启动参数 --ctx-size 或 --n-ctx" in str(w[0].message)

    # When server_n_ctx is sufficiently large (32768 > 14000) -> no clamp
    act_ok, ceil_ok, clamped_ok = clamp_thresholds_to_n_ctx(
        active_threshold=8000,
        hard_ceiling=14000,
        server_n_ctx=32768,
    )
    assert clamped_ok is False
    assert act_ok == 8000
    assert ceil_ok == 14000

    # 2. Test ContextFlusher initialization clamping & set_thresholds runtime clamping
    with tempfile.TemporaryDirectory() as tmp_dir:
        flusher = ContextFlusher(
            workspace_dir=tmp_dir,
            server_n_ctx=8192,
            active_threshold=8000,
            hard_ceiling=14000,
            print_diff_by_default=False,
        )
        assert flusher.hard_ceiling == 7372
        assert flusher.active_threshold == 4212

        # Runtime adjustment via set_thresholds must also be clamped against server_n_ctx!
        flusher.set_thresholds(active_threshold=10000, hard_ceiling=16000)
        # 16000 >= 8192 -> clamped_ceiling = 7372
        # clamped_active = int(10000 * (7372 / 16000)) = 4607
        assert flusher.hard_ceiling == 7372
        assert flusher.active_threshold == 4607

    # 3. Test probe_server_n_ctx 2.0s timeout and in-memory cache
    clear_server_n_ctx_cache()
    # Unreachable port returns None safely
    res = probe_server_n_ctx("http://127.0.0.1:59998", timeout=0.1)
    assert res is None
    # Second call returns None instantly from in-memory cache
    res2 = probe_server_n_ctx("http://127.0.0.1:59998", timeout=0.1)
    assert res2 is None


def test_threshold_priority_order():
    """Verify 4-tier parameter priority: Explicit Argument > Env Var > Config File > Preset/Default."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Tier 4: Default fallback (no args, no env, no config, no preset)
        flusher_default = ContextFlusher(workspace_dir=tmp_dir, print_diff_by_default=False)
        assert flusher_default.active_threshold == 32000
        assert flusher_default.hard_ceiling == 60000

        # Tier 4b: Preset specified
        flusher_preset = ContextFlusher(workspace_dir=tmp_dir, preset="16k", print_diff_by_default=False)
        assert flusher_preset.active_threshold == 16000
        assert flusher_preset.hard_ceiling == 28000

        # Tier 3: Config file overrides preset
        cfg_file = Path(tmp_dir) / "test_config.yaml"
        cfg_file.write_text(
            "preset: 16k\nthresholds:\n  active_threshold: 12345\n  hard_ceiling: 23456\n",
            encoding="utf-8",
        )
        flusher_cfg = ContextFlusher(config_path=cfg_file, workspace_dir=tmp_dir, print_diff_by_default=False)
        assert flusher_cfg.active_threshold == 12345
        assert flusher_cfg.hard_ceiling == 23456

        # Tier 2: Environment variable overrides config file and preset
        os.environ["CONTEXT_FLUSHER_ACTIVE_THRESHOLD"] = "11111"
        os.environ["CONTEXT_FLUSHER_HARD_CEILING"] = "22222"
        try:
            flusher_env = ContextFlusher(config_path=cfg_file, workspace_dir=tmp_dir, print_diff_by_default=False)
            assert flusher_env.active_threshold == 11111
            assert flusher_env.hard_ceiling == 22222

            # Tier 1: Explicit argument overrides everything
            flusher_explicit = ContextFlusher(
                config_path=cfg_file,
                workspace_dir=tmp_dir,
                active_threshold=99999,
                hard_ceiling=199999,
                print_diff_by_default=False,
            )
            assert flusher_explicit.active_threshold == 99999
            assert flusher_explicit.hard_ceiling == 199999
        finally:
            os.environ.pop("CONTEXT_FLUSHER_ACTIVE_THRESHOLD", None)
            os.environ.pop("CONTEXT_FLUSHER_HARD_CEILING", None)


def test_recall_by_dir():
    """Test recall_by_dir finding all task archives that touched files in a directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir)

        messages = [
            {"role": "user", "content": "Update auth session"},
            {
                "role": "assistant",
                "content": "Updated src/auth/session.py and src/auth/token.py.",
                "tool_calls": [{"id": "t1", "function": {"name": "write", "arguments": '{"path": "src/auth/session.py"}'}}],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "Saved"},
            {"role": "assistant", "content": "Done with auth session."},
        ]

        create_task_archive(messages, workspace_dir=workspace, task_name="auth_setup", context_tokens=2500)

        retriever = TaskRetriever(workspace_dir=workspace)
        # Recall by directory
        results = retriever.recall_by_dir("src/auth")
        assert len(results) >= 1
        assert results[0]["task_name"] == "auth_setup"


def test_parallel_tool_calls_sanitizer():
    """Test sanitizer handling parallel tool_calls where only some tool responses survive."""
    from context_flusher import ProtocolSanitizer

    messages = [
        {"role": "system", "content": "Assistant prompt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_p1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                {"id": "call_p2", "function": {"name": "read_file", "arguments": '{"path": "b.py"}'}},
                {"id": "call_p3", "function": {"name": "read_file", "arguments": '{"path": "c.py"}'}},
            ],
        },
        # call_p1 and call_p3 answered, but call_p2 was pruned/dropped
        {"role": "tool", "tool_call_id": "call_p1", "content": "content a"},
        {"role": "tool", "tool_call_id": "call_p3", "content": "content c"},
        {"role": "assistant", "content": "Read a and c."},
    ]
    sanitized = sanitize_message_sequence(messages)
    # The assistant message at index 1 should now only contain call_p1 and call_p3
    remaining_ids = [tc["id"] for tc in sanitized[1]["tool_calls"]]
    assert remaining_ids == ["call_p1", "call_p3"]
    assert "call_p2" not in remaining_ids
    # Sequence validation should be clean
    assert len(ProtocolSanitizer.validate_sequence(sanitized)) == 0


def test_utf8_chinese_path_and_content():
    """Test full cycle with Chinese paths, UTF-8 filenames, and CJK text."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir)
        messages = [
            {"role": "user", "content": "修改认证模块并测试"},
            {
                "role": "assistant",
                "content": "修改了 src/认证/session.py，决定放弃全局单例模式。",
                "tool_calls": [{"id": "tc_cjk", "function": {"name": "write", "arguments": '{"path": "src/认证/session.py"}'}}],
            },
            {"role": "tool", "tool_call_id": "tc_cjk", "content": "写入成功，耗时 12ms。"},
            {"role": "assistant", "content": "重构完成，验证通过。"},
        ]
        archive_path, _ = create_task_archive(
            messages,
            workspace_dir=workspace,
            task_name="重构认证模块",
            context_tokens=1500,
        )
        assert archive_path is not None
        assert archive_path.exists()

        retriever = TaskRetriever(workspace_dir=workspace)
        recalled = retriever.recall_by_file("src/认证/session.py")
        assert recalled is not None
        assert "重构认证模块" in recalled["task_name"]
        assert any("放弃全局单例模式" in d for d in recalled["decisions"])


class ContextFlusherTestCase(unittest.TestCase):
    """Standard unittest wrapper for CI/CD discovery."""

    def test_01_sanitizer_orphaned_tool_calls(self):
        test_sanitizer_orphaned_tool_calls()

    def test_02_protocol_sanitizer_validation(self):
        test_protocol_sanitizer_validation()

    def test_03_archiver_and_reclamation(self):
        test_archiver_and_reclamation()

    def test_04_flusher_engine_lifecycle(self):
        test_flusher_engine_lifecycle()

    def test_05_wrap_openai_mutation(self):
        test_wrap_openai_mutation()

    def test_06_heuristic_decision_extraction(self):
        test_heuristic_decision_extraction()

    def test_07_recall_by_file(self):
        test_recall_by_file()

    def test_08_tool_stubbing_intent_preservation(self):
        test_tool_stubbing_intent_preservation()

    def test_09_format_flush_diff(self):
        test_format_flush_diff()

    def test_10_path_normalization_safety(self):
        test_path_normalization_safety()

    def test_11_threshold_tuning_and_presets(self):
        test_threshold_tuning_and_presets()

    def test_12_layered_iron_law_and_escape_hatch(self):
        test_layered_iron_law_and_escape_hatch()

    def test_13_defer_starvation_recovery(self):
        test_defer_starvation_recovery()

    def test_14_server_n_ctx_clamping(self):
        test_server_n_ctx_clamping()

    def test_15_threshold_priority_order(self):
        test_threshold_priority_order()

    def test_16_recall_by_dir(self):
        test_recall_by_dir()

    def test_17_parallel_tool_calls_sanitizer(self):
        test_parallel_tool_calls_sanitizer()

    def test_18_utf8_chinese_path_and_content(self):
        test_utf8_chinese_path_and_content()


ALL_TESTS = [
    ("test_sanitizer_orphaned_tool_calls", test_sanitizer_orphaned_tool_calls),
    ("test_protocol_sanitizer_validation", test_protocol_sanitizer_validation),
    ("test_archiver_and_reclamation", test_archiver_and_reclamation),
    ("test_flusher_engine_lifecycle", test_flusher_engine_lifecycle),
    ("test_wrap_openai_mutation", test_wrap_openai_mutation),
    ("test_heuristic_decision_extraction", test_heuristic_decision_extraction),
    ("test_recall_by_file", test_recall_by_file),
    ("test_tool_stubbing_intent_preservation", test_tool_stubbing_intent_preservation),
    ("test_format_flush_diff", test_format_flush_diff),
    ("test_path_normalization_safety", test_path_normalization_safety),
    ("test_threshold_tuning_and_presets", test_threshold_tuning_and_presets),
    ("test_layered_iron_law_and_escape_hatch", test_layered_iron_law_and_escape_hatch),
    ("test_defer_starvation_recovery", test_defer_starvation_recovery),
    ("test_server_n_ctx_clamping", test_server_n_ctx_clamping),
    ("test_threshold_priority_order", test_threshold_priority_order),
    ("test_recall_by_dir", test_recall_by_dir),
    ("test_parallel_tool_calls_sanitizer", test_parallel_tool_calls_sanitizer),
    ("test_utf8_chinese_path_and_content", test_utf8_chinese_path_and_content),
]


def run_all_tests_verbose() -> bool:
    """Run all test functions with detailed timing and status output."""
    total = len(ALL_TESTS)
    passed = 0
    start_total = time.perf_counter()

    print("=" * 76)
    print("      CONTEXT-FLUSHER ENGINE: FULL AUTOMATED TEST SUITE EXECUTION")
    print(f"      Total Tests to Run: {total} | Python: {sys.version.split()[0]}")
    print("=" * 76)

    for idx, (name, test_fn) in enumerate(ALL_TESTS, 1):
        t0 = time.perf_counter()
        try:
            test_fn()
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[{idx:02d}/{total:02d}] [PASS] {name:<46} ({elapsed:6.1f} ms)")
            passed += 1
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[{idx:02d}/{total:02d}] [FAIL] {name:<46} ({elapsed:6.1f} ms)")
            print(f"       Error: {type(e).__name__}: {e}")
            raise

    total_time = time.perf_counter() - start_total
    print("=" * 76)
    print(f"RESULT: ALL {passed}/{total} TESTS PASSED (100% SUCCESS) IN {total_time:.3f}s")
    print("=" * 76)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        unittest.main()
    else:
        run_all_tests_verbose()




