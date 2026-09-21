# 🚀 Context-Flusher

> **Task-Aware Context Compaction & Zero-LLM Checkpointing Middleware for Local AI Agents**  
> *专为本地大模型（llama.cpp / Ollama / vLLM，全平台支持 NVIDIA CUDA / AMD ROCm / Apple Metal / CPU）手写 Agent 循环打造的实用任务检查点、解码延迟治理与文件级精准记忆检索中间件。*

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://python.org)
[![Zero-LLM-Cost](https://img.shields.io/badge/extra%20LLM%20cost-0%20tokens-orange.svg)](#)
[![Hardware-Agnostic](https://img.shields.io/badge/hardware-NVIDIA%20%7C%20AMD%20%7C%20Apple%20%7C%20CPU-blueviolet.svg)](#)
[![Dependencies](https://img.shields.io/badge/heavy%20dependencies-Zero%20(PyYAML%20only)-success.svg)](#)

---

## 🎯 Positioning & Core Philosophy / 项目定位与工程哲学

**不搞虚假颠覆，诚实解决真实工程痛点。**

在本地显卡（NVIDIA RTX 30/40系列、AMD Radeon / ROCm、Apple Silicon Mac 等）或 CPU 上运行自主编程与长程多轮 Agent 时：
1. **上下文膨胀与解码算力瓶颈**：随着工具日志和长代码累积，活动上下文膨胀到 30k~60k 时，因 Attention 掩码与显存带宽压力，Token 生成速度通常可能出现大幅下降，在长文本与多步复杂推理下甚至可能触发客户端超时；
2. **模型自总结的代价**：若让本地 27B/32B 模型在中途执行对话自压缩总结，往往会阻塞单卡推理 1~2 分钟，且极易意外抹掉关键的十六进制偏移、汇编代码和报错堆栈；
3. **记忆丢失的隐性陷阱**：盲目截断上下文会让 Agent 丢失前因后果（例如“为什么放弃方案 A 换用方案 B”），导致后续轮次反复重试已失败方案。

`Context-Flusher` 是一个轻量级（纯标准库 + PyYAML）、确定性（毫秒级执行、0 额外模型开销）的本地中间件：
- **步骤收尾自动落盘**：多轮工具循环中不打断执行，步骤收尾时将全量事实落盘到 Markdown；
- **文件级倒排索引（`recall_by_file`）**：以修改的代码文件路径为锚点精准召回历史，解决自然语言同义词（“登录” vs “auth”）匹配漂移难题；
- **启发式决策捕获（实验性）**：从回答尾部和“报错-重试”拓扑中提炼关键决策与避坑候选，零 LLM 也能留住核心意图；
- **工具调用桩降级（Tool Stubbing）**：将旧工具输出替换为执行桩而非物理删除，杜绝模型因“失忆”而重复触发写文件或调接口等副作用。

---

## 💡 Engineering Comparison / 核心机制对比

| Dimension / 维度 | ❌ Naive Context Truncation / 暴力截断 | ❌ Local LLM Summarization / 本地模型自压缩 | ✅ Context-Flusher / 本实用中间件 |
| :--- | :--- | :--- | :--- |
| **Compute Overhead / 算力开销** | 0 tokens | 消耗数千 tokens，单卡卡死 1~2 分钟 | **0 extra LLM tokens; 毫秒级 Python 确定性执行** |
| **Hardware Support / 硬件兼容** | 通用 | 依赖高端显卡算力 | **全平台通用（NVIDIA / AMD / Apple / CPU）** |
| **Why & Decision / 决策记忆** | 彻底丢失决策依据 | 理解好但耗时耗卡，易丢失关键数据 | **启发式抽取决策句 + 错误转向拓扑 [Experimental]** |
| **Code Anchoring / 代码定位** | 无法定位 | 依赖模糊自然语言回忆 | **`recall_by_file` 按触碰文件倒排索引** |
| **Side-effect Safety / 副作用安全** | 删除 tool_call 导致模型重复重试 | 无意图保护 | **Tool Stubbing: 保留调用意图，杜绝重复副作用** |
| **Protocol Safety / 协议合规** | 易引发 `400 Bad Request` | 需自己处理校验 | **ProtocolSanitizer 从机制上消除修剪引入的 400 报错** |
| **Integration / 集成方式** | 大改代码 | 大改 Prompt 与控制流 | **`wrap_openai` 一行代码无侵入拦截包装** |

---

## 📈 The Sawtooth Performance Curve / 锯齿波性能趋势示意

*(以下为长程多轮 Agent 任务中的典型性能趋势示意，具体生成速率与耗时取决于模型尺寸、量化格式与本地硬件规格)*

```text
  Context Tokens in Memory (活动上下文窗口占用)
  60k |                               [60K Hard Ceiling: Lightweight Pruning]
      |                                      /\
  32k |         [Task End: Flush]           /  \           [Task End: Flush]
      |                /\                  /    \                 /\
      |               /  \                /      \               /  \
  1.5k|______________/____\______________/________\_____________/____\__________ Baseline Window
      +------------------------------------------------------------------------> Turns (对话轮次)
                         (Instant drop to ~1.5k post-archive)

  Inference Speed on Local Hardware (解码生成速度 tok/s 走势示意)
  Steady |======================================================================== (Context-Flusher: 维持初始高速解码)
         |
  Drop   |                \_______ (Without Flusher: 随上下文拉长算力带宽见顶，速度可能显著下滑)
         +------------------------------------------------------------------------>
```

> 💡 **核心工程常识（关于 KV Cache 与显存）**：  
> `llama.cpp` 与 `Ollama` 等本地推理框架的 KV Cache 往往在启动时依据 `--ctx-size` / `n_ctx` 一次性预分配，单纯从聊天数组中裁剪旧消息并**不会**将物理显存从操作系统或 `nvidia-smi` / `rocm-smi` 中减小。  
> `Context-Flusher` 真正节省的是**每一步 Token 生成时的 Attention 掩码计算量与显存带宽读写压力**，并将上下文严控在模型容量预算之内，从根源上消除超出 `n_ctx` 导致的上下文溢出截断与推理停滞。

---

## 🏗️ Workflow / 任务感知调度时序

```mermaid
flowchart TD
    Start([Turn Begins: User Prompt]) --> ToolLoop[Agent Executes Multi-Step Tools]
    
    ToolLoop --> PreflightCheck{Preflight: Check Token Pressure}
    
    PreflightCheck -->|>= 60K Hard Ceiling| StagePrune[60K Guard: Lightweight Prune Old Tool Outputs<br/>Continues without breaking in-flight task]
    StagePrune --> ToolLoop
    
    PreflightCheck -->|32K <= Tokens < 60K & In Tool Loop| DeferCompaction[Task-Aware Deferral:<br/>Bypass compaction during continuous execution]
    DeferCompaction --> ModelCall[Proceed with Model Call]
    ModelCall --> ToolLoop
    
    ToolLoop --> StepCompletion([Step Completion / Turn Finalize])
    
    StepCompletion --> CheckThreshold{Tokens >= 32K?}
    CheckThreshold -->|No| NextTurn([Ready for Next Turn])
    
    CheckThreshold -->|Yes: Turn Boundary Reached| WriteArchive["1. 毫秒级 Python 结构化归档:<br/>提取启发式决策与触碰文件<br/>写入 archives/TASK_*.md 与 file_index.json"]
    WriteArchive --> VerifyDisk{2. Verify File on Disk > 0 bytes}
    
    VerifyDisk -->|Confirmed| ContextReset["3. Reset Context (32k -> ~1.5k):<br/>Keep System + Clean Tail<br/>Stub tool results & sanitize orphaned calls"]
    ContextReset --> FlushKV["4. 上下文开销即刻释放:<br/>解除 Attention 计算负担，下一轮高效起跑！"]
    FlushKV --> NextTurn
```

---

## 📦 Quickstart / 快速上手

### 1. Installation / 安装
零重依赖，仅需 Python 3.9+ 与 PyYAML：

```bash
git clone https://github.com/akayixiaoke/context-flusher-engine.git
cd context-flusher-engine
pip install -e .
```

### 2. Method 1: 1-Line Non-Intrusive Wrapper (Recommended) / 一行代码无侵入接入
适用于手写 Agent 循环的开发者。使用 `wrap_openai` 包装原生客户端，后续所有的 `create` 调用自动享受预检拦截与原地上下文释放：

```python
from openai import OpenAI
from context_flusher import wrap_openai

# 1. 一行代码包装客户端（支持 llama-server / Ollama / vLLM / 官方 OpenAI）
client = wrap_openai(
    OpenAI(base_url="http://localhost:8080/v1", api_key="sk-local"),
    workspace_dir="./my_project",
    preset="32k",             # 可选：8k, 16k, 32k (默认), 64k
    active_threshold=32000,   # 活跃门限：任务进行中延迟，步骤收尾自动清理
    hard_ceiling=60000,       # 兜底上限：仅对旧工具日志轻量折叠，绝不中断任务
)

messages = [{"role": "system", "content": "You are an autonomous engineering assistant."}]
messages.append({"role": "user", "content": "Reverse engineer target_driver.sys and generate patch"})

# 2. 原有调用完全不用改动！
# 当工具调用收尾且上下文超过门限时，messages 数组在原地自动回收至 ~1.5k！
response = client.chat.completions.create(model="qwen-2.5-coder", messages=messages)

# 3. 打印透明留存与损益对账单
stats = getattr(response, "context_flusher_stats", {})
print(client.format_flush_diff(stats))
```

### 3. Precision Recall by File / 文件级记忆精准召回（核心亮点）
解决自然语言模糊搜索的同义词漂移问题（例如“登录模块”与“auth”）：

```python
# 当 Agent 再次需要修改 target_driver.sys 时，直接根据文件路径调取精准记忆：
memory = client.recall_by_file("target_driver.sys")
if memory:
    print(f"上次修改任务: {memory['task_name']}")
    print(f"核心决策与避坑: {memory['decisions']}")
```

---

## 🛡️ Protocol Sanitizer & Tool Stubbing / 协议安全与意图桩

### 1. Tool Stubbing（避免重复副作用）
许多框架在裁剪历史时直接将 `role: tool` 消息物理删除。这会导致模型误以为自己“从未调用过该工具”，从而重复执行可能具有破坏性副作用的操作（如反复重写文件、重复跑数据库迁移）。

Context-Flusher 采用 **Tool Stubbing** 机制：
```json
// 原始 10,000 字符的工具日志：
{"role": "tool", "tool_call_id": "call_01", "content": "...10,000 characters..."}

// 降级为保留意图的极简执行桩（以实际代码输出为准）：
{"role": "tool", "tool_call_id": "call_01", "content": "[System: Action 'write_file' executed. Result recorded (10,000 chars). DO NOT repeat execution. Full output archived to disk. Preview: ... ]"}
```
- 模型明确知晓动作已成功完成，无需重复重试；
- 避免历史日志占用活动上下文预算；
- 从机制上消除因修剪引入的 `400 Bad Request` 协议报错。

### 2. 悬挂调用检测与修复
对真正因异常中断产生的孤儿调用执行自动化修复：
```python
from context_flusher import ProtocolSanitizer

# 校验并清洗序列
safe_messages = ProtocolSanitizer.clean_and_validate(messages)
```

---

## 🎛️ Context Window Presets & Auto-Clamping / 上下文预设与服务端自适应

### 1. 严格以 Token 容量命名的预设（全硬件通用）
预设取决于模型 `n_ctx` 与注意力预算，完全独立于显卡品牌（**NVIDIA / AMD / Apple Silicon / CPU 通用**）：
- **`"8k"`**: `active_threshold: 8000`, `hard_ceiling: 14000`（适合小窗口或紧凑预算）
- **`"16k"`**: `active_threshold: 16000`, `hard_ceiling: 28000`（均衡型编程 Agent）
- **`"32k"`** (Default): `active_threshold: 32000`, `hard_ceiling: 60000`（主流推荐标准）
- **`"64k"`**: `active_threshold: 64000`, `hard_ceiling: 120000`（长文本深度规划）

### 2. 服务端 `n_ctx` 探测与等比自适应缩放（Server Clamping）
当用户配置的 `hard_ceiling >= server_n_ctx` 时，底层服务端会先报溢出或截断，导致清理器无机会执行。
`Context-Flusher` 具备自适应硬件防线：
- **轻量无感探测**：包装客户端时以 2.0s 严格超时探测 llama.cpp（`/props`）或 Ollama（`/api/show`），结果进程内持久缓存，失败静默放行，绝不阻断主链路；
- **等比保形缩放**：`hard_ceiling` 截断至 `n_ctx * 0.90`，`active_threshold` 按原有的 `active / ceiling` 比例等比下调，尊重用户自定义配置形状；
- **明确指引告警**：终端输出明确提示 `建议检查服务端启动参数 --ctx-size 或 --n-ctx`；
- **双重兜底生效**：初始化与运行时 `set_thresholds` 均自动通过 `clamp_thresholds_to_n_ctx` 校验。

### 3. 严格四级参数生效优先级
```text
显式调用参数 (Explicit Args) > 环境变量 (Env Vars) > 配置文件 (config.yaml) > 预设与默认 (Presets/32k)
```

---

## 🚨 Emergency Escape Hatch / 紧急逃生舱机制

当发生死循环、第三方工具挂死或网络中断导致工具输出迟迟未返回，且上下文逼近 `hard_ceiling` 时：
1. **注入身份明确的紧急桩**：
   ```json
   {
     "role": "tool",
     "tool_call_id": "call_make_01",
     "content": "[System Alert: Tool 'execute_command' (args: {\"cmd\": \"make build\"}) execution status UNKNOWN - timed out or truncated before response. DO NOT assume side-effects completed without verification. Full call archived at: archives/TASK_*.md]"
   }
   ```
2. **拒绝盲猜**：工具函数名、参数预览、存盘指引三要素完整，模型下一轮可清晰判定哪一步受挫并采取补救，而非面对匿名报错。
3. **诚实审计**：终端以 `[RED] EMERGENCY ESCAPE HATCH` 醒目对账单告警，全量事实已安全存盘。

---

## ⚠️ Limitations & Honest Disclosures / 局限性与诚实声明

作为坚持“不搞虚假颠覆、诚实解决工程痛点”的实用工具，在集成使用前请充分知悉以下技术边界：

1. **无语义理解 (No Semantic Understanding)**：归档决策抽取与文件定位完全基于确定性的正则表达式、字符拓扑与状态机规则，不消耗额外模型推理，因此无法理解深层语义泛化。
2. **关键词与路径级索引 (Path & Inverted Indexing)**：`recall_by_file` 与 `recall_by_dir` 基于轻量级 JSON 倒排索引文件（`file_index.json`），专为代码文件路径精准回查设计，不替代具备向量 Embedding 的复杂 RAG 知识库。
3. **启发式决策抽取仍属实验性 (Heuristic Extraction is Experimental)**：从助手推理尾部提炼的“避坑决策”目前定义为“候选建议（Candidate Rules）”，在极其复杂或非常规提示词工程下仍需持续验证与调优。
4. **显存预分配机制说明 (VRAM Allocation Note)**：如前所述，本地推理框架的 KV Cache 往往按 `n_ctx` 预占显存，本工具优化的是**计算开销、生成延迟与上下文可用预算**，不会改变底层 `nvidia-smi` 或 `rocm-smi` 的物理静态显存分配数值。

---

## 📂 Project Structure / 项目结构

```text
context-flusher-engine/
├── context_flusher/
│   ├── __init__.py          # 导出 wrap_openai, ContextFlusher, clamp_thresholds_to_n_ctx 等
│   ├── engine.py            # 核心调度引擎、n_ctx 探测与等比缩放、对账单格式化
│   ├── wrapper.py           # 一行代码无侵入 OpenAI Client 包装器（支持 Prefix Cache 工具排序）
│   ├── sanitizer.py         # ProtocolSanitizer（协议防400、Tool Stubbing、Token估算）
│   ├── archiver.py          # 启发式决策抽取[Experimental]、触碰文件追踪与毫秒级 Markdown 归档
│   ├── context_reset.py     # 上下文修剪与稳定前缀保护
│   └── retriever.py         # JSON 倒排索引（recall_by_file / recall_by_dir）
├── tests/
│   └── test_engine.py       # 18 项全自动化测试套件（全部 PASS）
├── example_run.py           # 立即可跑的演示脚本（内置 MockClient，无需 API Key）
├── config.example.yaml      # 标准配置文件模板（纯 Token 预设划分）
├── pyproject.toml           # PEP 517/621 规范打包配置（MIT 协议）
├── requirements.txt         # 仅 pyyaml>=6.0
├── .gitignore               # Python 忽略规则
├── LICENSE                  # MIT License
└── README.md                # 诚实导向的中英双语说明文档
```

---

## 🧪 Running Tests & Demo / 运行验证

```bash
# 运行开箱即用的完整实测案例（包含 Wrapper、recall_by_file、Tool Stubbing 演示）：
python example_run.py

# 运行 18 项全量单元与集成测试：
python tests/test_engine.py
```

---

## 📄 License / 开源协议

Released under the [MIT License](LICENSE). Copyright (c) 2026 Context-Flusher Contributors.  
遵循 MIT 开源许可协议，允许个人、商业及科研自由使用与集成。
