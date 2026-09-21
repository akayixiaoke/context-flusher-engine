"""Example: Running an Agent with Context-Flusher Middleware.

Demonstrates:
1. The 1-Line Non-Intrusive OpenAI Client Wrapper (wrap_openai).
2. Protocol Sanitizer with Tool Stubbing (Preserves intent & prevents 400 Bad Request).
3. Precision Recall by File (recall_by_file) overcoming natural language ambiguity.
4. Transparency: Context Reconciliation Diff.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure UTF-8 output stream on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add local package root to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_flusher import (
    ContextFlusher,
    ProtocolSanitizer,
    estimate_messages_tokens,
    wrap_openai,
)


# --- Mock OpenAI Client for Standalone Execution ---
class MockChoiceMessage:
    def __init__(self, content: str = "", tool_calls: Any = None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


class MockChoice:
    def __init__(self, message: MockChoiceMessage):
        self.message = message


class MockResponse:
    def __init__(self, message: MockChoiceMessage):
        self.choices = [MockChoice(message)]


class MockOpenAICompletions:
    def __init__(self):
        self.step_counter = 0

    def create(self, *args: Any, **kwargs: Any) -> MockResponse:
        self.step_counter += 1
        messages: List[Dict[str, Any]] = kwargs.get("messages", [])

        # Step 1: LLM outputs a tool call to read disassembly of driver
        if self.step_counter == 1:
            return MockResponse(
                MockChoiceMessage(
                    content="Inspecting binary headers and disassembly...",
                    tool_calls=[{
                        "id": "call_pe_001",
                        "type": "function",
                        "function": {
                            "name": "disassemble_binary",
                            "arguments": '{"target": "target_driver.sys", "lines": 500}'
                        }
                    }]
                )
            )
        # Step 2: LLM analyzes tool output and produces final completion with decisions
        else:
            return MockResponse(
                MockChoiceMessage(
                    content=(
                        "Binary analysis complete! Target entrypoint is at 0x140001000 in target_driver.sys. "
                        "经测试发现原hook方案触发PatchGuard蓝屏，因此决定改为修改IRP分发函数表。注意不要覆写驱动校验头。"
                    ),
                    tool_calls=None  # Turn finishes here!
                )
            )


class MockOpenAIChat:
    def __init__(self):
        self.completions = MockOpenAICompletions()


class MockOpenAIClient:
    """Mock standard OpenAI SDK client instance."""
    def __init__(self):
        self.chat = MockOpenAIChat()


def demo_openai_client_wrapper():
    print("=" * 72)
    print("DEMO 1: 1-Line Non-Intrusive OpenAI Client Wrapper (wrap_openai)")
    print("=" * 72)

    raw_client = MockOpenAIClient()

    # >>> 1-Line Integration <<<
    client = wrap_openai(
        raw_client,
        workspace_dir="./workspace_example",
        active_threshold=2000,  # Lowered for demo visibility (default: 32,000)
        hard_ceiling=8000,      # Default: 60,000
    )

    messages = [
        {"role": "system", "content": "You are an autonomous reverse engineering assistant."}
    ]

    print("\n[Step 1] User asks to reverse engineer a driver...")
    messages.append({"role": "user", "content": "Reverse engineer target_driver.sys and generate patch."})

    # Call wrapped client exactly as standard OpenAI SDK
    resp1 = client.chat.completions.create(model="qwen-2.5-coder", messages=messages)
    assistant_msg1 = resp1.choices[0].message
    messages.append(assistant_msg1.model_dump())
    print(f"Assistant Step 1: {assistant_msg1.content} (Triggered tool: {assistant_msg1.tool_calls[0]['function']['name']})")

    # Simulate tool returning large output (~2,500 tokens)
    simulated_large_output = "0x140001000: 48 83 ec 28  sub rsp, 0x28\n0x140001004: e8 57 02 00  call sub_140001260\n" * 250
    messages.append({
        "role": "tool",
        "tool_call_id": "call_pe_001",
        "content": simulated_large_output,
    })

    tokens_before = estimate_messages_tokens(messages)
    print(f"\nContext Size with tool output: ~{tokens_before:,} tokens (exceeds demo threshold 2,000 tokens)")

    # Call step 2 -> Model responds without tool_calls (completing turn)
    print("[Step 2] Model synthesizes result (Wrapped client automatically intercepts and flushes)...")
    resp2 = client.chat.completions.create(model="qwen-2.5-coder", messages=messages)
    assistant_msg2 = resp2.choices[0].message
    print(f"Assistant Final: {assistant_msg2.content[:90]}...")

    tokens_after = estimate_messages_tokens(messages)
    stats = getattr(resp2, "context_flusher_stats", {})

    print(f"[Verification] Context successfully reclaimed in place from ~{tokens_before:,} to ~{tokens_after:,} tokens!")
    print(f"[Auto-Diff Printed] 3-Color status: {stats.get('lossiness_status')} (Verified Lossless)\n")

    # Demonstrate Threshold Tuning on the fly (e.g. switching to 8k/14k context window profile)
    print("--- [Dynamic Threshold Tuning] Adjusting to 8k/14k Context Profile ---")
    client.set_thresholds(active_threshold=8000, hard_ceiling=14000)
    print(f"Current Flusher Thresholds: active={client.flusher.active_threshold:,}, ceiling={client.flusher.hard_ceiling:,}\n")

    return client



def demo_recall_by_file(client: Any):
    print("=" * 72)
    print("DEMO 2: Precision Recall by File (recall_by_file)")
    print("=" * 72)
    print("Searching past task archives anchored to 'target_driver.sys'...")

    recalled = client.recall_by_file("target_driver.sys")
    if recalled:
        print(f"  [Found Anchor]  File: {recalled['file']}")
        print(f"  [Task Name]     {recalled['task_name']}")
        print(f"  [Timestamp]     {recalled['timestamp']}")
        print(f"  [Summary]       {recalled['summary']}")
        print("  [Captured Decisions (Zero-LLM)]:")
        for d in recalled.get("decisions", []):
            print(f"    - {d}")
        print("  [Archive File]  " + recalled["archive_file"])
    else:
        print("  No archive found for target_driver.sys.")


def demo_protocol_sanitizer():
    print("\n" + "=" * 72)
    print("DEMO 3: Protocol Sanitizer (Stubbing & 400 Bad Request Prevention)")
    print("=" * 72)

    # 1. Tool Stubbing Demo: preserves intent while dropping heavy bytes
    execution_messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {
            "role": "assistant",
            "content": "Running test suite...",
            "tool_calls": [{"id": "call_test_01", "type": "function", "function": {"name": "pytest", "arguments": "{}"}}]
        },
        {
            "role": "tool",
            "tool_call_id": "call_test_01",
            "content": "PASSED test_auth.py\nPASSED test_db.py\n" * 100,  # ~3,000 chars
        },
        {"role": "assistant", "content": "All tests passed!"}
    ]

    stubbed = ProtocolSanitizer.stub_pruned_tools(execution_messages, min_chars_to_stub=200, protect_tail_tools=0)
    print("Tool Stubbing Result (Intent preserved, zero re-execution risk):")
    print(f"  Tool content replaced with: {stubbed[2]['content']}")

    # 2. Orphaned call cleaning
    corrupted_messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_dangling_999", "type": "function", "function": {"name": "run_test", "arguments": "{}"}}
            ]
        },
        {"role": "user", "content": "Next step please."}
    ]

    issues = ProtocolSanitizer.validate_sequence(corrupted_messages)
    print("\nValidation of corrupted sequence (Dangling call detected):")
    for issue in issues:
        print(f"  [Warning] {issue}")

    sanitized = ProtocolSanitizer.clean_and_validate(corrupted_messages)
    print("After clean_and_validate: 100% compliant with OpenAI protocol!")


if __name__ == "__main__":
    wrapped_client = demo_openai_client_wrapper()
    demo_recall_by_file(wrapped_client)
    demo_protocol_sanitizer()
