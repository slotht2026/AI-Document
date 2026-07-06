# 从问题到进化：Hermes Agent 如何重新定义 AI Agent

> **作者按**：这不是一本 API 参考手册，而是一部"问题解决史"。每一章都从一个真实的痛点出发，带你走过"发现问题→分析根因→设计方案→实现突破"的完整路径。如果你曾好奇一个 AI Agent 是如何从"能用"进化到"好用"再到"自己变聪明"的——这本书就是为你写的。

---

## 目录

- [第一章：起点——为什么需要一个"会成长"的 AI Agent？](#第一章起点为什么需要一个会成长的-ai-agent)
- [第二章：第一块基石——工具调用（Function Calling）与 Agent 循环](#第二章第一块基石工具调用function-calling与-agent-循环)
- [第三章：万能插头——工具注册与工具集系统的设计哲学](#第三章万能插头工具注册与工具集系统的设计哲学)
- [第四章：记忆的难题——上下文管理与压缩引擎](#第四章记忆的难题上下文管理与压缩引擎)
- [第五章：越用越聪明——自进化学习闭环（技能系统 + Curator）](#第五章越用越聪明自进化学习闭环技能系统--curator)
- [第六章：走出终端——消息网关与跨平台统一](#第六章走出终端消息网关与跨平台统一)
- [第七章：一个人不够——多 Agent 协作（Kanban 系统）](#第七章一个人不够多-agent-协作kanban-系统)
- [第八章：韧性设计——错误分类、故障转移与安全防御](#第八章韧性设计错误分类故障转移与安全防御)
- [第九章：供应链之战——从 Mini Shai-Hulud 蠕虫看依赖安全](#第九章供应链之战从-mini-shai-hulud-蠕虫看依赖安全)
- [第十章：面向未来——可插拔架构与 RL 训练集成](#第十章面向未来可插拔架构与-rl-训练集成)
- [结语：进化的意义](#结语进化的意义)

---

## 第一章：起点——为什么需要一个"会成长"的 AI Agent？

### 1.1 问题引入：聊天机器人的天花板

2023 年，ChatGPT 横空出世，全世界都在和 AI 对话。但很快，一个尴尬的现实浮出水面：

**你让 AI 帮你写一段代码，它写得很漂亮。你接着说"把这个部署到 AWS"，它只能回复一段文字说明，而不是真的帮你执行。**

这不是 AI 不够聪明——它有足够的知识。问题在于：**它没有手。**

传统的聊天 AI 是一个封闭的文本生成系统。它能理解你的意图，能生成高质量的回复，但它无法：
- 读取你的文件系统
- 执行命令行操作
- 调用外部 API
- 操作浏览器
- 与其他 Agent 协作

更关键的是，即使你给它接上了"手"（工具调用），它仍然面临一个更深层的问题：**每次对话都是从零开始的。** 你今天教会它你的项目结构，明天它就忘了。你告诉它一个解决 bug 的方法，下次遇到同样的问题，它还是会走弯路。

**人类之所以能成长，是因为我们有记忆、有经验、有学习能力。** AI Agent 要想真正有用，也必须具备这些能力。

这就是 Hermes Agent 诞生的起点。

### 1.2 从 RAG 到 Agent：技术演进的三个阶段

回顾 AI 应用的发展，我们可以清晰地看到三个阶段：

**2023 年——提示词工程年**：人们发现，给 AI 一个好的提示词，它就能输出好的结果。各种 Prompt 技巧层出不穷，"提示词工程师"成为新职业。

**2024 年——RAG 年**：人们发现，光靠提示词不够，AI 需要"参考资料"。检索增强生成（Retrieval-Augmented Generation）让 AI 能够查阅外部知识库，回答更准确。

**2025 年——Agent 年**：人们发现，光能查阅资料还不够，AI 需要"动手做事"。Agent 范式应运而生——AI 不再是被动回答问题的机器，而是能主动规划、执行、反馈的自主体。

```
传统 AI：      你问 → 它答（单轮）
RAG：          你问 → 它查资料 → 它答（增强单轮）
Agent：        你问 → 它思考 → 它行动 → 它观察 → 它再思考...（循环）
```

Hermes Agent 正是 Agent 时代的产物，但它走得更远——**它要让 Agent 自己也能成长。**

### 1.3 Hermes Agent 的核心愿景

Hermes Agent 是 Nous Research 开发的开源自我进化 AI Agent 框架。截至 v0.13.0 版本，它包含：

- **3433 个源文件**，构成一个庞大的工程体系
- **约 17000 个测试**，保证代码质量
- **20 个消息平台适配**，从 CLI 到 Discord、Telegram、WhatsApp
- **40+ 核心工具**，覆盖文件操作、终端执行、网页浏览、图像生成等
- **MIT 开源许可证**，完全开放

但数字不是重点。重点是它解决的核心问题：**如何让 AI Agent 越用越聪明？**

这个问题的答案，不是一个单一的技术突破，而是一系列环环相扣的设计决策。从工具调用的基础设施，到上下文管理的记忆系统，到技能自进化的学习闭环——每一个环节都是对一个真实问题的回应。

让我们从最基础的问题开始。

---

## 第二章：第一块基石——工具调用（Function Calling）与 Agent 循环

### 2.1 问题引入：AI 说了"我来帮你执行"，然后呢？

假设你对 AI 说："帮我看看当前目录下有什么文件。"

一个纯聊天 AI 会回答："你可以使用 `ls` 命令来查看当前目录下的文件。" 然后呢？没有然后了。它不能真的运行 `ls`。

2023 年，OpenAI 引入了 **Function Calling** 机制，让 AI 能够"声明"它想调用什么函数、传什么参数。但这只是半个解决方案——AI 说了"我想调用 `run_terminal_command(command='ls')`"，谁来真正执行这个调用？

**Agent 框架的核心职责，就是把这个"谁"做出来。**

### 2.2 ReAct 循环：思考-行动-观察

Hermes Agent 的核心对话循环基于 **ReAct（Reasoning + Acting）** 范式。这个范式的名字很直白：先推理，再行动，然后观察结果，循环往复。

让我们看看 `run_agent.py` 中 `AIAgent` 类的 `run_conversation()` 方法是如何实现这个循环的（简化版）：

```python
# run_agent.py — AIAgent.run_conversation() 核心循环（简化）

class AIAgent:
    def run_conversation(self, user_message: str) -> str:
        """核心对话循环：ReAct 模式"""
        # 将用户消息加入上下文
        self.messages.append({"role": "user", "content": user_message})
        
        while True:
            # ① 调用 LLM，获取回复
            response = self.call_llm(self.messages)
            
            # ② 检查是否有工具调用
            if not response.tool_calls:
                # 没有工具调用 → 这是最终回复，循环结束
                self.messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                return response.content
            
            # ③ 有工具调用 → 将 assistant 消息加入上下文
            self.messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls
            })
            
            # ④ 执行所有工具调用
            for tool_call in response.tool_calls:
                result = self.execute_tool(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments
                )
                # ⑤ 将工具结果加入上下文
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            # ⑥ 回到步骤 ①，继续循环
```

这个循环看起来简单，但它是整个 Agent 系统的心脏。每一跳都包含三个关键动作：

1. **Reasoning（推理）**：LLM 分析当前上下文，决定下一步做什么
2. **Acting（行动）**：Agent 执行 LLM 指定的工具调用
3. **Observing（观察）**：工具返回结果，被追加到上下文中

```
┌─────────────────────────────────────────┐
│            Agent 对话循环               │
│                                         │
│   ┌──────┐    ┌──────┐    ┌──────┐    │
│   │ 思考 │───→│ 行动 │───→│ 观察 │    │
│   │(LLM) │    │(工具) │    │(结果) │    │
│   └──────┘    └──────┘    └──────┘    │
│       ↑                            │    │
│       └────────────────────────────┘    │
│       （循环直到 LLM 不再调用工具）       │
└─────────────────────────────────────────┘
```

### 2.3 迭代预算：防止 Agent 陷入死循环

**新问题出现了：** 如果 LLM 一直调用工具怎么办？

想象这样一个场景：Agent 试图读取一个不存在的文件，失败后决定再试一次，又失败，又重试……无限循环。

Hermes Agent 引入了 `IterationBudget` 来解决这个问题：

```python
# 迭代预算：线程安全的迭代计数器

class IterationBudget:
    """限制 Agent 的最大迭代次数"""
    
    def __init__(self, max_iterations: int = 50, parent: 'IterationBudget' = None):
        self._count = 0
        self._max = max_iterations
        self._lock = threading.Lock()
        self._parent = parent  # 支持父子独立预算
    
    def increment(self) -> bool:
        """增加迭代计数。返回 False 表示超出预算。"""
        with self._lock:
            self._count += 1
            if self._count >= self._max:
                return False  # 超出预算，必须停止
            return True
    
    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self._max - self._count)
```

`IterationBudget` 的设计有几个精妙之处：

- **线程安全**：使用 `threading.Lock` 保护计数器，因为并行工具执行可能在多线程中调用
- **父子独立预算**：当一个 Agent 创建子 Agent 时，子 Agent 有自己的预算，不会互相干扰
- **单调递增**：计数器只增不减，确保即使在并发场景下也不会超过上限

在主循环中，每次迭代前都会检查预算：

```python
while True:
    if not self.iteration_budget.increment():
        # 超出预算，强制生成最终回复
        self.messages.append({
            "role": "user",
            "content": "你已达到最大迭代次数。请立即总结当前状态并给出最终回复。"
        })
        final_response = self.call_llm(self.messages)
        return final_response.content
    
    # ... 正常的 ReAct 循环 ...
```

### 2.4 并行工具执行：不浪费每一秒

**又一个新问题：** 如果 Agent 需要同时读取三个文件，串行执行就太慢了。

但并非所有工具都能并行执行。比如"写入文件 A"和"读取文件 A"就有冲突。

Hermes Agent 的解决方案是维护一个 `_PARALLEL_SAFE_TOOLS` 集合，并加入路径冲突检测：

```python
# 并行工具执行策略

_PARALLEL_SAFE_TOOLS = {
    "read_file", "search_files", "web_search", 
    "read_url", "image_describe", ...
}

def _has_path_conflict(tool_a: ToolCall, tool_b: ToolCall) -> bool:
    """检测两个工具调用是否存在文件路径冲突"""
    path_a = _extract_file_path(tool_a)
    path_b = _extract_file_path(tool_b)
    if path_a and path_b:
        return os.path.normpath(path_a) == os.path.normpath(path_b)
    return False

async def execute_tools_parallel(self, tool_calls: list[ToolCall]) -> list[str]:
    """并行执行无冲突的工具调用"""
    # 将工具调用分组：可并行 vs 必须串行
    parallel_group = []
    serial_group = []
    
    for call in tool_calls:
        if call.function.name in _PARALLEL_SAFE_TOOLS:
            # 检查与已加入并行组的工具是否有路径冲突
            conflict = any(
                _has_path_conflict(call, existing) 
                for existing in parallel_group
            )
            if conflict:
                serial_group.append(call)
            else:
                parallel_group.append(call)
        else:
            serial_group.append(call)
    
    # 并行执行无冲突的工具
    results = {}
    if parallel_group:
        tasks = [
            self.execute_tool_async(call) 
            for call in parallel_group
        ]
        parallel_results = await asyncio.gather(*tasks)
        for call, result in zip(parallel_group, parallel_results):
            results[call.id] = result
    
    # 串行执行有冲突的工具
    for call in serial_group:
        results[call.id] = await self.execute_tool_async(call)
    
    return [results[call.id] for call in tool_calls]
```

这个设计的核心思想是：**安全第一，性能第二。** 只有确定安全的工具才并行执行，有任何疑虑就串行。

### 2.5 兼容性噩梦：Surrogate 清理与参数修复

在理想世界里，所有 LLM 都会返回格式完美的 JSON。但在现实世界里……

**问题一：推理模型的 Surrogate 字符。** MiMo、Kimi、GLM 等推理模型在生成过程中可能产生"lone surrogates"——一种非法的 Unicode 字符，会导致 JSON 解析失败。

```python
import re

def _clean_surrogates(text: str) -> str:
    """清理推理模型产生的 lone surrogates"""
    # 移除未配对的代理对字符（U+D800 到 U+DFFF）
    return re.sub(r'[\ud800-\udfff]', '', text)
```

**问题二：不同模型的参数格式差异。** 比如 GLM-5.1 模型返回的工具调用参数可能是字符串化的 JSON（双重编码），而不是直接的 JSON 对象。

```python
def _repair_tool_call_arguments(tool_calls: list) -> list:
    """修复不同模型返回的工具调用参数格式问题"""
    for call in tool_calls:
        args = call.function.arguments
        if isinstance(args, str):
            try:
                # 尝试解析字符串化的 JSON
                call.function.arguments = json.loads(args)
            except json.JSONDecodeError:
                # 如果解析失败，尝试清理后重试
                cleaned = _clean_surrogates(args)
                call.function.arguments = json.loads(cleaned)
    return tool_calls
```

这些看起来是"脏活累活"，但它们是让 Agent 在真实环境中可靠运行的关键。**工程的伟大，往往不在于优雅的设计，而在于对混乱现实的容忍与修复。**

### 2.6 本章小结

Agent 循环是整个系统的基石。它解决的核心问题是：让 AI 从"只能说"变成"能动手做"。但仅仅是"能做"还不够——我们需要控制它（迭代预算），加速它（并行执行），容忍它（兼容性修复）。

下一章，我们将看到一个更深层的问题：**当工具越来越多，如何管理它们？**

---

## 第三章：万能插头——工具注册与工具集系统的设计哲学

### 3.1 问题引入：40 个工具，怎么管？

当 Hermes Agent 只有 5 个工具的时候，管理很简单——写 5 个函数，手动注册，完事。

但当工具数量增长到 40+，问题就来了：

1. **发现**：怎么知道有哪些工具可用？手动维护一个列表？每次添加新工具都要改？
2. **注册**：每个工具的 schema（参数定义）、handler（执行函数）、check_fn（可用性检查）散落在不同文件里，怎么统一管理？
3. **组合**：不同场景需要不同工具集。CLI 需要终端工具，Telegram 需要消息工具，调试需要终端+网页+文件——怎么灵活组合？
4. **缓存**：工具的 schema 定义是不变的吗？如果工具依赖外部服务（Docker、Playwright），可用性会动态变化——怎么处理？

### 3.2 工具注册中心：registry.py

所有工具最终都汇聚到一个中心：`tools/registry.py`。它的核心是 `ToolEntry` 数据类：

```python
# tools/registry.py — 工具注册中心

from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class ToolEntry:
    """工具注册条目"""
    name: str              # 工具名称，如 "read_file"
    toolset: str           # 所属工具集，如 "file", "terminal"
    schema: dict           # JSON Schema，描述参数格式
    handler: Callable      # 执行函数
    check_fn: Callable = None  # 可用性检查函数
    emoji: str = "🔧"     # 显示用的 emoji

class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}
        self._generation: int = 0  # 单调递增的版本计数器
        self._check_cache: dict[str, tuple[float, Any]] = {}  # check_fn TTL 缓存
        self._CHECK_TTL: float = 30.0  # 缓存有效期（秒）
    
    def register(self, name: str, toolset: str, schema: dict, 
                 handler: Callable, check_fn: Callable = None, emoji: str = "🔧"):
        """注册一个工具"""
        # 防影子注册：内置工具不允许被 MCP 工具覆盖
        if name in self._tools:
            existing = self._tools[name]
            if existing.toolset != "mcp" and toolset == "mcp":
                raise ValueError(
                    f"内置工具 '{name}' 不允许被 MCP 工具覆盖"
                )
        
        self._tools[name] = ToolEntry(
            name=name, toolset=toolset, schema=schema,
            handler=handler, check_fn=check_fn, emoji=emoji
        )
        self._generation += 1  # 版本号递增
    
    def get_available_tools(self) -> list[ToolEntry]:
        """获取当前可用的工具列表（带 check_fn 缓存）"""
        available = []
        import time
        now = time.monotonic()
        
        for entry in self._tools.values():
            if entry.check_fn is None:
                available.append(entry)
                continue
            
            # 检查缓存
            cache_key = entry.name
            if cache_key in self._check_cache:
                cached_time, cached_result = self._check_cache[cache_key]
                if now - cached_time < self._CHECK_TTL:
                    if cached_result:
                        available.append(entry)
                    continue
            
            # 缓存过期，重新检查
            try:
                result = entry.check_fn()
                self._check_cache[cache_key] = (now, result)
                if result:
                    available.append(entry)
            except Exception:
                # 检查失败，视为不可用
                self._check_cache[cache_key] = (now, False)
        
        return available
```

几个关键设计决策值得深入分析：

**check_fn TTL 缓存（30 秒）：** 为什么要缓存？因为某些工具的可用性检查涉及外部调用——比如检查 Docker 是否运行、Playwright 是否安装。每次调用 LLM 前都需要获取可用工具列表，如果每次都重新检查，延迟会很高。30 秒的 TTL 是一个折中：足够新鲜，足够快。

**防影子注册：** MCP（Model Context Protocol）是外部工具的接入协议。如果一个 MCP 工具和一个内置工具同名怎么办？Hermes Agent 的策略是：**内置工具优先，MCP 工具不能覆盖。** 但 MCP-to-MCP 之间允许覆盖——这支持了工具的热更新。

**generation 计数器：** 这是一个单调递增的版本号。每次注册或注销工具时递增。外部系统（如缓存层）可以通过比较 generation 值来判断工具列表是否发生了变化，实现增量更新。

### 3.3 自动发现：AST 扫描的魔法

**新问题：** 工具写在 `tools/*.py` 的各个文件里，怎么让注册中心知道它们的存在？

最笨的办法是手动导入每个文件。但 Hermes Agent 选择了一个更聪明的方案：**AST（抽象语法树）扫描。**

```python
# model_tools.py — 通过 AST 扫描自动发现工具

import ast
import importlib
from pathlib import Path

def discover_builtin_tools(tools_dir: str = "tools") -> list[str]:
    """扫描 tools/ 目录，自动发现所有注册了工具的模块"""
    modules = []
    tools_path = Path(tools_dir)
    
    for py_file in tools_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue  # 跳过私有文件
        
        # 读取文件内容，解析 AST
        source = py_file.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        
        # 在 AST 中查找 registry.register() 调用
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # 检查是否是 registry.register(...) 调用
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr == "register" and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == "registry"):
                    modules.append(py_file.stem)
                    break  # 一个文件只需要发现一次
    
    return modules

def load_tools(modules: list[str]):
    """动态导入发现的工具模块"""
    for module_name in modules:
        importlib.import_module(f"tools.{module_name}")
```

为什么用 AST 扫描而不是简单地 `import` 所有文件？因为：

1. **性能**：AST 扫描只解析语法，不执行代码。如果一个文件有副作用（如启动网络连接），AST 扫描不会触发它们
2. **选择性**：只导入真正注册了工具的文件，跳过工具函数库、常量定义等辅助文件
3. **增量更新**：配合 generation 计数器，可以只重新扫描变化的文件

### 3.4 工具集系统：组合的力量

40+ 个工具不可能全部同时暴露给 LLM——太多工具会让 LLM 困惑，也会浪费 token。Hermes Agent 引入了**工具集（Toolset）**的概念：

```python
# toolsets.py — 工具集定义

# 核心工具列表：所有平台共享的 40+ 工具
_HERMES_CORE_TOOLS = [
    "read_file", "write_file", "edit_file", "search_files",
    "run_terminal_command", "read_url", "web_search",
    "image_generate", "image_describe", "memory_store",
    "memory_search", "skill_create", "skill_search",
    # ... 共 40+ 个
]

# 工具集可以组合其他工具集
TOOLSETS = {
    # 基础工具集
    "file": ["read_file", "write_file", "edit_file", "search_files"],
    "terminal": ["run_terminal_command", "run_background_command"],
    "web": ["read_url", "web_search", "web_screenshot"],
    
    # 组合工具集
    "debugging": ["terminal", "web", "file"],  # 引用其他工具集
    "creative": ["file", "web", "image_generate", "image_describe"],
    
    # 平台特定工具集
    "hermes-cli": ["_HERMES_CORE_TOOLS"],  # CLI 拥有全部核心工具
    "hermes-telegram": ["_HERMES_CORE_TOOLS", "telegram_send", "telegram_media"],
    "hermes-discord": ["_HERMES_CORE_TOOLS", "discord_send", "discord_embed"],
    "hermes-acp": ["_HERMES_CORE_TOOLS", "acp_protocol"],
}

def resolve_toolset(name: str) -> set[str]:
    """解析工具集，递归展开组合引用"""
    if name not in TOOLSETS:
        return {name}  # 单个工具名
    
    resolved = set()
    for item in TOOLSETS[name]:
        if item.startswith("_"):
            # 全局常量引用
            resolved.update(globals().get(item, []))
        elif item in TOOLSETS:
            # 递归展开子工具集
            resolved.update(resolve_toolset(item))
        else:
            resolved.add(item)
    return resolved
```

这个设计的精髓在于**分层组合**：

```
hermes-discord
  └── _HERMES_CORE_TOOLS（40+ 核心工具）
  └── discord_send（平台特定）
  └── discord_embed（平台特定）

debugging
  └── terminal
       └── run_terminal_command
       └── run_background_command
  └── web
       └── read_url
       └── web_search
  └── file
       └── read_file
       └── write_file
```

新增一个平台？只需定义该平台的特定工具，然后组合核心工具集。新增一个工具？只需在对应的工具集注册，所有使用该工具集的平台自动获得。

### 3.5 异步桥接：sync 与 async 的鸿沟

Python 的异步生态和同步生态之间有一道鸿沟。很多工具的 handler 是同步函数（因为简单好写），但 Agent 循环需要异步执行（为了并行）。

```python
# model_tools.py — 同步到异步的桥接

import asyncio
import threading

# 持久化 event loop，避免 "Event loop is closed" 错误
_loop: asyncio.AbstractEventLoop = None
_loop_thread: threading.Thread = None

def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """获取或创建持久化的 event loop"""
    global _loop, _loop_thread
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(
            target=_loop.run_forever, 
            daemon=True
        )
        _loop_thread.start()
    return _loop

def _run_async(coro) -> any:
    """在持久化 event loop 中运行异步协程"""
    loop = _get_or_create_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=300)  # 5 分钟超时
```

为什么要"持久化 event loop"？因为 Python 的 `asyncio.run()` 每次都会创建一个新的 event loop 并在结束时关闭它。如果在循环关闭后还有未完成的协程，就会报 "Event loop is closed" 错误。持久化 loop 避免了这个问题。

### 3.6 本章小结

工具系统是 Agent 的"手"。Hermes Agent 的工具系统解决了四个问题：

1. **自动发现**：AST 扫描，零手动配置
2. **统一注册**：中心化的 registry，防冲突、带缓存
3. **灵活组合**：工具集分层，平台特定 + 核心共享
4. **异步桥接**：sync/async 无缝转换

但"手"只是能力的一半。另一半是"记忆"——下一章，我们来聊 Agent 的上下文管理难题。

---

## 第四章：记忆的难题——上下文管理与压缩引擎

### 4.1 问题引入：当对话变长，一切开始崩塌

你和 Agent 聊了两个小时。前一个小时讨论了项目架构，后一个小时在调试一个 bug。

突然，Agent 忘了你在前一个小时讨论的内容。

这不是 Agent 变笨了——**是它的"短期记忆"（上下文窗口）满了。**

所有 LLM 都有上下文窗口的限制。GPT-4 是 128K token，Claude 是 200K token，听起来很多？在一个涉及多轮工具调用的 Agent 对话中，每轮可能消耗数千 token（用户消息 + LLM 回复 + 工具调用 + 工具结果）。128K token 可能只够聊 30 分钟。

更糟糕的是，当上下文接近满时，LLM 的表现会急剧下降——它开始"遗忘"早期的对话内容，做出前后矛盾的决策，甚至忘记自己正在做什么。

**Agent 需要记忆管理。就像人类需要遗忘和概括的能力一样。**

### 4.2 上下文引擎：可插拔的记忆管理

Hermes Agent 的解决方案是 `ContextEngine`——一个可插拔的上下文管理引擎。

```python
# agent/context_engine.py — 上下文引擎抽象基类

from abc import ABC, abstractmethod
from typing import Optional

class ContextEngine(ABC):
    """上下文管理引擎的抽象基类"""
    
    @abstractmethod
    def on_session_start(self, messages: list[dict]) -> list[dict]:
        """会话开始时的初始化钩子"""
        ...
    
    @abstractmethod
    def update_from_response(self, messages: list[dict], response: dict):
        """从 LLM 响应中更新上下文状态"""
        ...
    
    @abstractmethod
    def should_compress(self, messages: list[dict]) -> bool:
        """判断是否需要压缩"""
        ...
    
    @abstractmethod
    def compress(self, messages: list[dict], llm_caller: callable) -> list[dict]:
        """执行压缩，返回压缩后的消息列表"""
        ...
    
    @abstractmethod
    def on_session_end(self, messages: list[dict]):
        """会话结束时的清理钩子"""
        ...
```

这个抽象基类定义了上下文管理的完整生命周期：

```
on_session_start
      │
      ▼
  ┌──────────┐     ┌───────────────────┐
  │ 正常对话 │────→│ update_from_response │
  └──────────┘     └───────────────────┘
      │                    │
      │              should_compress?
      │              ┌─────┴─────┐
      │              │ No        │ Yes
      │              │           │
      │              │     ┌─────▼─────┐
      │              │     │ compress  │
      │              │     └───────────┘
      │              │           │
      └──────────────┴───────────┘
      │
      ▼
 on_session_end
```

### 4.3 保护策略：哪些记忆不能丢？

压缩意味着丢弃信息。但有些信息绝对不能丢——比如系统提示词（它定义了 Agent 的身份和规则），比如最近的对话（它包含当前任务的上下文）。

Hermes Agent 的默认压缩器 `ContextCompressor` 使用了**保护策略**：

```python
# agent/context_engine.py — 默认压缩器

class ContextCompressor(ContextEngine):
    """默认上下文压缩器：通过摘要压缩旧消息"""
    
    def __init__(self, 
                 protect_first_n: int = 3,    # 保护前 N 条消息
                 protect_last_n: int = 6,     # 保护后 N 条消息
                 compress_threshold: float = 0.75):  # 压缩阈值（上下文使用的百分比）
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.compress_threshold = compress_threshold
    
    def should_compress(self, messages: list[dict]) -> bool:
        """当上下文使用超过 75% 时触发压缩"""
        total_tokens = self._count_tokens(messages)
        max_tokens = self._get_max_context_tokens()
        return total_tokens > max_tokens * self.compress_threshold
    
    def compress(self, messages: list[dict], llm_caller: callable) -> list[dict]:
        """将中间消息压缩为摘要"""
        # ① 分离：保护区 + 可压缩区
        protected_start = messages[:self.protect_first_n]       # 系统提示、初始上下文
        protected_end = messages[-self.protect_last_n:]          # 最近的对话
        compressible = messages[self.protect_first_n:-self.protect_last_n]
        
        if not compressible:
            return messages  # 没有可压缩的内容
        
        # ② 生成摘要
        summary_prompt = f"""请将以下对话历史压缩为简洁的摘要，保留关键信息：
        
对话历史：
{self._format_messages(compressible)}

要求：
- 保留所有重要的决策和结论
- 保留工具调用的关键结果
- 保留未完成的任务
- 压缩到原文的 20% 以内"""
        
        summary = llm_caller([
            {"role": "system", "content": "你是一个对话摘要专家。"},
            {"role": "user", "content": summary_prompt}
        ])
        
        # ③ 重组：保护区 + 摘要 + 保护区
        compressed_messages = protected_start + [
            {"role": "system", "content": f"[对话摘要]\n{summary}"}
        ] + protected_end
        
        return compressed_messages
```

保护策略的数字不是随便选的：

- **protect_first_n=3**：通常包含系统提示词、用户身份信息、初始任务描述——这些是 Agent 的"身份"，丢了就迷失了
- **protect_last_n=6**：最近 6 条消息通常包含当前任务的最新状态——丢了就断片了

```
消息列表：
[0] 系统提示词 ──────────────┐
[1] 用户身份信息              │ protect_first_n = 3
[2] 初始任务描述 ─────────────┘
[3] 历史对话 A ──┐
[4] 历史对话 B   │
[5] 历史对话 C   │ 可压缩区域
[6] 历史对话 D   │
[7] 历史对话 E ──┘
[8] 最近对话 1 ──┐
[9] 最近对话 2   │
[10] 最近对话 3  │ protect_last_n = 6
[11] 最近对话 4  │
[12] 最近对话 5  │
[13] 最近对话 6 ─┘
```

### 4.4 压缩的时机：75% 阈值

为什么要设 75% 而不是 100%？

因为当上下文接近 100% 满时，LLM 的表现已经严重退化了。在 75% 时就触发压缩，可以：

1. **留出缓冲**：压缩本身也需要调用 LLM（生成摘要），需要额外的 token 空间
2. **保持质量**：在 LLM 还没开始"遗忘"之前就压缩，摘要质量更高
3. **平滑体验**：用户不会感受到突然的"记忆断裂"

### 4.5 本章小结

上下文管理是 Agent 的"记忆术"。Hermes Agent 的方案是：

1. **可插拔**：ContextEngine 是抽象基类，可以替换实现
2. **保护关键信息**：首尾保护，中间压缩
3. **自动触发**：75% 阈值，无需手动干预
4. **LLM 辅助压缩**：用 AI 来压缩 AI 的对话，比简单截断更智能

但压缩只是"遗忘"的艺术。下一章，我们来看"记忆"的艺术——如何让 Agent 越用越聪明。

---

## 第五章：越用越聪明——自进化学习闭环（技能系统 + Curator）

### 5.1 问题引入：为什么 Agent 总是犯同样的错误？

你昨天教 Agent 用 `git rebase` 而不是 `git merge` 来整理提交历史。今天，它又开始用 `git merge` 了。

为什么？因为它没有"长期记忆"。上下文压缩保留的是"摘要"，不是"技能"。当对话结束后，Agent 学到的一切都消失了。

**人类之所以能成长，是因为我们把经验转化为技能。** 你第一次骑自行车需要刻意思考每个动作，但熟练后就变成了"肌肉记忆"。Agent 也需要类似的能力：把反复使用的方法固化为"技能"，下次直接调用。

### 5.2 技能系统：从经验到代码

Hermes Agent 的技能系统是一个完整的知识管理闭环：

```
技能创建（手动或自动）
      │
      ▼
技能存储（Markdown 文件 + FTS5 索引）
      │
      ▼
技能使用（Agent 在对话中搜索并应用）
      │
      ▼
技能改进（在使用中发现不足，自动更新）
      │
      ▼
技能维护（Curator 后台管理生命周期）
      │
      ▼
技能回忆（FTS5 全文搜索，快速定位）
```

一个技能本质上是一个 Markdown 文件，包含：

```markdown
# 技能名称：Git Rebase 最佳实践

## 触发条件
当用户提到"整理提交历史"、"合并分支"、"rebase"时

## 执行步骤
1. 先 `git fetch origin` 获取最新代码
2. 使用 `git rebase -i HEAD~N` 进入交互式 rebase
3. 对于 squash 操作，保留第一个 commit 的消息
4. rebase 完成后使用 `git push --force-with-lease` 而不是 `--force`

## 注意事项
- 不要对已推送到远程的公共分支进行 rebase
- 如果遇到冲突，先解决冲突再 `git rebase --continue`
```

技能不是代码——它是**自然语言描述的操作指南**。Agent 在执行任务时，可以通过全文搜索（FTS5）找到相关技能，然后按照指南操作。

### 5.3 Curator：技能的生命周期管理

**新问题：** 技能会过时。一个关于旧版 API 的技能，如果一直保留，反而会误导 Agent。

Hermes Agent 引入了 `Curator`——一个后台技能维护器，负责管理技能的生命周期：

```python
# agent/curator.py — 技能维护器

from datetime import datetime, timedelta
from enum import Enum

class SkillStatus(Enum):
    ACTIVE = "active"       # 活跃状态，正常使用
    STALE = "stale"         # 过时状态，30 天未使用
    ARCHIVED = "archived"   # 归档状态，90 天未使用

class Curator:
    """技能生命周期管理器"""
    
    STALE_THRESHOLD = timedelta(days=30)    # 30 天未使用 → 过时
    ARCHIVE_THRESHOLD = timedelta(days=90)  # 90 天未使用 → 归档
    CHECK_INTERVAL_HOURS = 24               # 每 24 小时检查一次
    MIN_IDLE_HOURS = 2                      # 至少空闲 2 小时才运行
    
    def __init__(self, skill_store, agent):
        self.skill_store = skill_store
        self.agent = agent
    
    def should_run_now(self) -> bool:
        """判断是否应该现在运行维护任务"""
        # 不是定时触发，而是空闲触发
        if not self._is_agent_idle():
            return False
        
        last_run = self.skill_store.get_last_curator_run()
        if last_run is None:
            return True
        
        hours_since_last = (datetime.now() - last_run).total_seconds() / 3600
        return hours_since_last >= self.CHECK_INTERVAL_HOURS
    
    def run_maintenance(self):
        """执行维护任务"""
        skills = self.skill_store.list_all_skills()
        
        for skill in skills:
            days_since_use = (datetime.now() - skill.last_used).days
            
            if days_since_use >= 90 and skill.status != SkillStatus.ARCHIVED:
                # 90 天未使用 → 归档
                skill.status = SkillStatus.ARCHIVED
                self.skill_store.update_skill(skill)
                
            elif days_since_use >= 30 and skill.status == SkillStatus.ACTIVE:
                # 30 天未使用 → 标记为过时
                skill.status = SkillStatus.STALE
                self.skill_store.update_skill(skill)
    
    def _is_agent_idle(self) -> bool:
        """检查 Agent 是否处于空闲状态"""
        return self.agent.current_task is None
```

Curator 的设计哲学是**温和管理**：

- **永不自动删除**：技能只被归档，不会被删除。因为被归档的技能可能在将来某个时刻又变得相关
- **空闲触发，非定时**：不在 Agent 忙碌时运行维护，避免干扰正常任务
- **渐进式状态转换**：active → stale → archived，给每个阶段足够的观察期

### 5.4 FTS5 搜索：快速技能回忆

当 Agent 面对一个任务时，它需要快速找到相关技能。Hermes Agent 使用 SQLite 的 FTS5（Full-Text Search 5）扩展来实现高效的全文搜索：

```python
# 技能搜索实现

import sqlite3

def create_skill_index(db_path: str):
    """创建技能的 FTS5 索引"""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts 
        USING fts5(
            name, 
            description, 
            content, 
            tags,
            tokenize='unicode61'
        )
    """)
    conn.commit()
    return conn

def search_skills(conn, query: str, limit: int = 5) -> list[dict]:
    """搜索相关技能"""
    cursor = conn.execute("""
        SELECT name, description, content, tags,
               rank  -- FTS5 内置的相关性评分
        FROM skills_fts
        WHERE skills_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (query, limit))
    
    return [
        {
            "name": row[0],
            "description": row[1],
            "content": row[2],
            "tags": row[3],
            "relevance": row[4]
        }
        for row in cursor.fetchall()
    ]
```

FTS5 的优势在于：
- **速度**：基于倒排索引，百万级文档也能毫秒级返回
- **相关性排序**：自动按匹配度排序
- **轻量**：SQLite 内置，无需额外服务

### 5.5 完整闭环：技能如何越用越好

让我们走一遍完整的自进化闭环：

1. **创建**：Agent 在调试一个 Docker 问题时，发现了一个有效的解决方法。它自动创建一个技能文件。

2. **存储**：技能被写入 `skills/` 目录，同时更新 FTS5 索引。

3. **使用**：下次遇到类似问题，Agent 搜索技能库，找到这个技能，按照指南操作。

4. **改进**：在使用过程中，Agent 发现指南中的某一步可以优化。它更新技能文件。

5. **维护**：如果这个技能 30 天没被用到，Curator 将它标记为 stale。90 天后归档。

6. **回忆**：即使归档了，技能仍然在 FTS5 索引中。如果将来又遇到相关问题，Agent 仍然可以搜索到它。

```
┌─────────────────────────────────────────────┐
│           自进化学习闭环                     │
│                                             │
│   经验 ──→ 技能创建 ──→ FTS5 索引           │
│    ↑                           │             │
│    │                           ▼             │
│    │                     搜索匹配           │
│    │                           │             │
│    │                           ▼             │
│    └──── 改进更新 ←── 使用验证              │
│                                │             │
│                                ▼             │
│                          Curator 维护        │
│                          (stale → archived)  │
└─────────────────────────────────────────────┘
```

### 5.6 本章小结

自进化是 Hermes Agent 最核心的创新。它解决了一个根本问题：**如何让 AI Agent 不是消耗品，而是成长品？**

答案是三个系统的协同：
1. **技能系统**：把经验固化为可检索的知识
2. **Curator**：管理技能的生命周期，防止知识腐化
3. **FTS5 搜索**：快速定位相关技能，让知识真正"可用"

下一章，我们来看一个完全不同的问题：**当 Agent 走出终端，进入消息平台，会发生什么？**

---

## 第六章：走出终端——消息网关与跨平台统一

### 6.1 问题引入：Agent 不应该只活在终端里

一个只能在终端里运行的 Agent，就像一个只能在实验室里工作的科学家——能力很强，但触达有限。

现实世界的用户在 Discord 里聊天、在 Telegram 里发消息、在 WhatsApp 里分享图片。如果 Agent 只能通过 CLI 访问，那 99% 的用户永远不会用到它。

**问题不只是"让 Agent 接入消息平台"——而是"让同一个 Agent 在 20 个不同的平台上表现一致"。**

### 6.2 消息网关：GatewayRunner

Hermes Agent 的消息网关（`gateway/run.py`）是所有平台适配器的统一管理器：

```python
# gateway/run.py — 消息网关（简化）

class GatewayRunner:
    """消息网关：管理所有平台适配器的生命周期"""
    
    _AGENT_CACHE_MAX_SIZE = 128       # 最大会话缓存数
    _AGENT_CACHE_IDLE_TTL_SECS = 3600 # 空闲会话 TTL（1 小时）
    
    def __init__(self, config):
        self.config = config
        self.adapters: dict[str, PlatformAdapter] = {}
        self._agent_cache: dict[str, AIAgent] = {}  # 会话缓存
        self._cache_access_times: dict[str, float] = {}
    
    async def start(self):
        """启动网关，注册所有平台适配器"""
        # 根据配置启动对应的平台适配器
        for platform in self.config.enabled_platforms:
            adapter = self._create_adapter(platform)
            self.adapters[platform] = adapter
            await adapter.start()
        
        # 启动会话缓存清理任务
        asyncio.create_task(self._cleanup_idle_sessions())
    
    def get_or_create_agent(self, session_id: str) -> AIAgent:
        """获取或创建 Agent 实例（LRU 缓存）"""
        if session_id in self._agent_cache:
            self._cache_access_times[session_id] = time.monotonic()
            return self._agent_cache[session_id]
        
        # 缓存满时，淘汰最久未访问的会话
        if len(self._agent_cache) >= self._AGENT_CACHE_MAX_SIZE:
            self._evict_oldest()
        
        # 创建新的 Agent 实例
        agent = AIAgent(config=self.config)
        self._agent_cache[session_id] = agent
        self._cache_access_times[session_id] = time.monotonic()
        return agent
    
    async def _cleanup_idle_sessions(self):
        """定期清理空闲会话"""
        while True:
            await asyncio.sleep(300)  # 每 5 分钟检查一次
            now = time.monotonic()
            expired = [
                sid for sid, last_access in self._cache_access_times.items()
                if now - last_access > self._AGENT_CACHE_IDLE_TTL_SECS
            ]
            for sid in expired:
                self._evict_session(sid)
```

网关的设计考虑了几个关键问题：

**会话缓存**：每个用户会话对应一个 Agent 实例。缓存避免了每次都重新创建 Agent 的开销。LRU + 空闲 TTL 的双重淘汰策略确保内存不会无限增长。

**自动恢复**：网关重启后，会尝试恢复中断的会话。这对于消息平台尤其重要——用户不会因为你的服务重启就消失。

### 6.3 平台适配器：统一接口，多样实现

每个消息平台都有自己的 API、消息格式、媒体处理方式。Hermes Agent 通过适配器模式统一了这些差异：

```python
# 平台适配器抽象

from abc import ABC, abstractmethod

class PlatformAdapter(ABC):
    """平台适配器抽象基类"""
    
    @abstractmethod
    async def start(self):
        """启动适配器，开始监听消息"""
        ...
    
    @abstractmethod
    async def stop(self):
        """停止适配器"""
        ...
    
    @abstractmethod
    async def send_message(self, session_id: str, content: str, **kwargs):
        """发送文本消息"""
        ...
    
    @abstractmethod
    async def send_media(self, session_id: str, media_path: str, **kwargs):
        """发送媒体文件"""
        ...
    
    @abstractmethod
    def parse_incoming(self, raw_event: dict) -> IncomingMessage:
        """将平台原始事件解析为统一格式"""
        ...

# Discord 适配器示例
class DiscordAdapter(PlatformAdapter):
    async def send_message(self, session_id: str, content: str, **kwargs):
        channel = self.bot.get_channel(int(session_id))
        # Discord 有 2000 字符限制，需要分段发送
        for chunk in self._split_message(content, max_length=2000):
            await channel.send(chunk)
    
    def parse_incoming(self, raw_event: dict) -> IncomingMessage:
        return IncomingMessage(
            platform="discord",
            session_id=str(raw_event["channel_id"]),
            user_id=str(raw_event["author"]["id"]),
            content=raw_event["content"],
            media=self._extract_media(raw_event)
        )
```

### 6.4 20 个平台的挑战

截至 v0.13.0，Hermes Agent 支持 20 个消息平台：

| 平台 | 特殊挑战 |
|------|----------|
| Discord | 2000 字符限制、Embed 支持、线程/话题 |
| Telegram | Markdown 格式差异、媒体组、内联键盘 |
| WhatsApp | 默认拒绝陌生人、端到端加密限制 |
| Slack | Block Kit 格式、应用权限 |
| CLI | 本地终端、TUI 界面 |
| ACP | Agent Communication Protocol，Agent-to-Agent |
| ... | ... |

每个平台都有自己的"坑"。比如 Discord 的 2000 字符限制意味着 Agent 的长回复必须分段发送；WhatsApp 的安全策略意味着默认拒绝陌生人的消息；Telegram 的 Markdown 解析和标准 Markdown 有微妙差异。

### 6.5 安全考量：平台特定的访问控制

不同平台有不同的安全模型。Hermes Agent 针对每个平台实现了特定的访问控制：

```python
# Discord：基于角色的访问控制（按 guild 作用域）

class DiscordAccessControl:
    def __init__(self):
        self.allowed_roles: dict[str, set[str]] = {}  # guild_id → allowed_role_ids
    
    def check_access(self, guild_id: str, user_roles: list[str]) -> bool:
        """检查用户是否有权使用 Agent"""
        if guild_id not in self.allowed_roles:
            return True  # 未配置限制，默认允许
        return bool(set(user_roles) & self.allowed_roles[guild_id])

# WhatsApp：默认拒绝陌生人
class WhatsAppAccessControl:
    def check_access(self, sender: str, contacts: list[str]) -> bool:
        """默认拒绝不在通讯录中的发送者"""
        return sender in contacts
```

### 6.6 本章小结

消息网关解决了 Agent 的"触达"问题。通过统一的适配器模式和会话管理，Hermes Agent 可以在 20 个不同的平台上提供一致的体验，同时针对每个平台的安全模型进行定制化防护。

但一个 Agent 的力量终究有限。下一章，我们来看当"一个人不够"时，多个 Agent 如何协作。

---

## 第七章：一个人不够——多 Agent 协作（Kanban 系统）

### 7.1 问题引入：复杂任务需要团队

你让 Agent "重构整个项目的数据库层，同时更新所有相关的 API 接口，并确保测试通过"。

一个 Agent 面对这个任务，要么顾此失彼，要么进度缓慢。就像一个人试图同时做五件事——不如五个人各做一件事。

**多 Agent 协作不是简单的"多个 Agent 各干各的"。它需要一个协调机制来分配任务、监控进度、处理失败。**

### 7.2 Kanban 看板：可视化的任务管理

Hermes Agent v0.13.0 引入了 Kanban（看板）系统来实现多 Agent 协作：

```python
# Kanban 任务管理

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional
import asyncio

class TaskStatus(Enum):
    BACKLOG = "backlog"         # 待办
    IN_PROGRESS = "in_progress" # 进行中
    REVIEW = "review"           # 待审核
    DONE = "done"               # 完成
    BLOCKED = "blocked"         # 阻塞

class TaskPriority(Enum):
    P0 = 0  # 最高优先级
    P1 = 1
    P2 = 2
    P3 = 3  # 最低优先级

@dataclass
class Task:
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.BACKLOG
    priority: TaskPriority = TaskPriority.P2
    assigned_to: Optional[str] = None  # Agent ID
    dependencies: list[str] = field(default_factory=list)  # 依赖的任务 ID
    created_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None
    result: Optional[str] = None

class KanbanBoard:
    """Kanban 看板：多 Agent 任务协调中心"""
    
    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.agents: dict[str, AgentInfo] = {}
        self._lock = asyncio.Lock()
    
    async def create_task(self, title: str, description: str, 
                          priority: TaskPriority = TaskPriority.P2,
                          dependencies: list[str] = None) -> Task:
        """创建新任务"""
        async with self._lock:
            task = Task(
                id=self._generate_id(),
                title=title,
                description=description,
                priority=priority,
                dependencies=dependencies or []
            )
            self.tasks[task.id] = task
            return task
    
    async def claim_task(self, agent_id: str) -> Optional[Task]:
        """Agent 认领一个可执行的任务"""
        async with self._lock:
            # 找到优先级最高的、无阻塞的、未分配的任务
            candidates = [
                t for t in self.tasks.values()
                if t.status == TaskStatus.BACKLOG
                and t.assigned_to is None
                and all(
                    self.tasks[dep].status == TaskStatus.DONE 
                    for dep in t.dependencies
                )
            ]
            
            if not candidates:
                return None
            
            # 按优先级排序
            candidates.sort(key=lambda t: t.priority.value)
            task = candidates[0]
            
            task.assigned_to = agent_id
            task.status = TaskStatus.IN_PROGRESS
            task.last_heartbeat = datetime.now()
            return task
    
    async def update_heartbeat(self, task_id: str):
        """更新任务心跳"""
        async with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].last_heartbeat = datetime.now()
    
    async def complete_task(self, task_id: str, result: str):
        """标记任务完成"""
        async with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.DONE
                self.tasks[task_id].result = result
```

### 7.3 心跳与僵尸检测

**新问题：** Agent 领了任务，然后崩溃了。任务卡在 "in_progress"，永远不会完成。

Kanban 系统通过心跳机制解决这个问题：

```python
# 僵尸任务检测与回收

class ZombieDetector:
    """僵尸任务检测器"""
    
    HEARTBEAT_TIMEOUT = 300  # 5 分钟无心跳视为僵尸
    
    def __init__(self, board: KanbanBoard):
        self.board = board
    
    async def detect_and_recover(self):
        """检测僵尸任务并回收"""
        while True:
            await asyncio.sleep(60)  # 每分钟检查一次
            now = datetime.now()
            
            for task in self.board.tasks.values():
                if task.status != TaskStatus.IN_PROGRESS:
                    continue
                
                if task.last_heartbeat is None:
                    continue
                
                elapsed = (now - task.last_heartbeat).total_seconds()
                if elapsed > self.HEARTBEAT_TIMEOUT:
                    # 僵尸任务！回收并重新分配
                    await self._recover_task(task)
    
    async def _recover_task(self, task: Task):
        """回收僵尸任务"""
        print(f"⚠️ 检测到僵尸任务: {task.id} ({task.title})")
        print(f"   分配给: {task.assigned_to}")
        print(f"   最后心跳: {task.last_heartbeat}")
        
        # 重置任务状态
        task.assigned_to = None
        task.status = TaskStatus.BACKLOG
        task.last_heartbeat = None
        
        # 提升优先级（被中断的任务应该优先处理）
        if task.priority.value > 0:
            task.priority = TaskPriority(task.priority.value - 1)
```

### 7.4 /goal 命令：目标锁定

v0.13.0 引入了 `/goal` 命令，让用户可以锁定一个明确的目标：

```python
# /goal 命令实现

class GoalManager:
    """目标管理器"""
    
    def __init__(self, kanban: KanbanBoard):
        self.kanban = kanban
        self.active_goals: dict[str, Goal] = {}
    
    async def set_goal(self, user_id: str, goal_description: str) -> str:
        """设置一个目标，并分解为子任务"""
        # ① 创建目标
        goal = Goal(
            id=self._generate_id(),
            user_id=user_id,
            description=goal_description,
            status="active"
        )
        self.active_goals[goal.id] = goal
        
        # ② 使用 LLM 将目标分解为子任务
        subtasks = await self._decompose_goal(goal_description)
        
        # ③ 创建任务并设置依赖关系
        task_ids = []
        for i, subtask in enumerate(subtasks):
            task = await self.kanban.create_task(
                title=subtask["title"],
                description=subtask["description"],
                priority=TaskPriority(subtask.get("priority", 2)),
                dependencies=task_ids[-1:] if i > 0 else []  # 串行依赖
            )
            task_ids.append(task.id)
        
        goal.task_ids = task_ids
        return f"目标已锁定，分解为 {len(task_ids)} 个子任务。"
```

### 7.5 Checkpoints v2：进度检查点

长时间运行的任务需要检查点机制，以便在失败时能够恢复：

```python
# Checkpoints v2

@dataclass
class Checkpoint:
    task_id: str
    step_index: int
    state: dict  # 任务在该步骤的状态快照
    timestamp: datetime
    can_resume: bool = True

class CheckpointManager:
    """检查点管理器"""
    
    def save_checkpoint(self, task_id: str, step_index: int, state: dict):
        """保存检查点"""
        checkpoint = Checkpoint(
            task_id=task_id,
            step_index=step_index,
            state=state,
            timestamp=datetime.now()
        )
        self._persist(checkpoint)
    
    def get_latest_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
        """获取最新的检查点"""
        return self._load_latest(task_id)
    
    async def resume_from_checkpoint(self, task_id: str, agent: 'AIAgent'):
        """从检查点恢复任务"""
        checkpoint = self.get_latest_checkpoint(task_id)
        if checkpoint and checkpoint.can_resume:
            # 恢复 Agent 状态
            agent.restore_state(checkpoint.state)
            # 从断点继续执行
            return checkpoint.step_index
        return 0  # 从头开始
```

### 7.6 本章小结

多 Agent 协作是 Hermes Agent 处理复杂任务的关键能力。通过 Kanban 看板、心跳检测、僵尸回收、目标锁定和检查点机制，多个 Agent 可以像一个高效的团队一样协作——分工明确、进度可控、失败可恢复。

下一章，我们来聊一个更"阴暗"的话题：当事情出错时，Agent 该怎么办？

---

## 第八章：韧性设计——错误分类、故障转移与安全防御

### 8.1 问题引入：错误不是异常，是常态

在生产环境中，错误不是"可能发生"，而是"一定会发生"。API 会超时，服务会宕机，配额会用尽，网络会中断。

对于一个 AI Agent 来说，错误的种类更加复杂：

- **LLM 服务错误**：API key 无效、配额耗尽、服务过载、上下文溢出
- **工具执行错误**：文件不存在、命令执行失败、网络超时
- **安全威胁**：提示词注入、越权操作、数据泄露
- **模型兼容性**：不同模型返回格式不一致、参数编码差异

**一个不能优雅处理错误的 Agent，不是生产级的 Agent。**

### 8.2 错误分类器：error_classifier.py

Hermes Agent 的错误分类器（`error_classifier.py`）是一个优先级分类管道，能够从错误消息中识别出具体的错误类型，并给出相应的恢复策略：

```python
# agent/error_classifier.py — 错误分类器

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import re

class FailoverReason(Enum):
    """故障转移原因枚举"""
    AUTH = "auth"                       # 认证失败
    BILLING = "billing"                 # 账单/配额问题
    RATE_LIMIT = "rate_limit"          # 速率限制
    OVERLOADED = "overloaded"          # 服务过载
    CONTEXT_OVERFLOW = "context_overflow"  # 上下文溢出
    MODEL_NOT_FOUND = "model_not_found"    # 模型不存在
    NETWORK = "network"                # 网络错误
    UNKNOWN = "unknown"                # 未知错误

@dataclass
class ErrorClassification:
    """错误分类结果"""
    reason: FailoverReason
    retryable: bool                    # 是否可重试
    should_compress: bool              # 是否应该压缩上下文
    should_rotate_credential: bool     # 是否应该轮换凭证
    should_fallback: bool              # 是否应该切换到备用模型
    message: str                       # 人类可读的描述

# 错误模式匹配规则（支持中英文）
ERROR_PATTERNS = [
    # 认证错误
    {
        "patterns": [
            r"invalid.?api.?key", r"authentication.?failed",
            r"无效.?api.?key", r"认证.?失败", r"unauthorized"
        ],
        "reason": FailoverReason.AUTH,
        "retryable": False,
        "should_rotate_credential": True,
        "should_fallback": True
    },
    # 配额/账单错误
    {
        "patterns": [
            r"quota.?exceeded", r"billing", r"insufficient.?credits",
            r"配额.?超出", r"余额.?不足", r"payment.?required"
        ],
        "reason": FailoverReason.BILLING,
        "retryable": False,
        "should_rotate_credential": True,
        "should_fallback": True
    },
    # 速率限制
    {
        "patterns": [
            r"rate.?limit", r"too.?many.?requests", r"429",
            r"速率.?限制", r"请求.?过多"
        ],
        "reason": FailoverReason.RATE_LIMIT,
        "retryable": True,
        "should_fallback": False
    },
    # 上下文溢出
    {
        "patterns": [
            r"context.?length.?exceeded", r"token.?limit",
            r"maximum.?context", r"上下文.?溢出", r"token.?超出"
        ],
        "reason": FailoverReason.CONTEXT_OVERFLOW,
        "retryable": True,
        "should_compress": True,
        "should_fallback": False
    },
    # 服务过载
    {
        "patterns": [
            r"overloaded", r"capacity", r"503", r"502",
            r"服务.?过载", r"服务器.?繁忙"
        ],
        "reason": FailoverReason.OVERLOADED,
        "retryable": True,
        "should_fallback": True
    },
]

class ErrorClassifier:
    """错误分类器"""
    
    def classify(self, error: Exception) -> ErrorClassification:
        """将错误分类并返回恢复策略"""
        error_msg = str(error).lower()
        
        for rule in ERROR_PATTERNS:
            for pattern in rule["patterns"]:
                if re.search(pattern, error_msg, re.IGNORECASE):
                    return ErrorClassification(
                        reason=rule["reason"],
                        retryable=rule.get("retryable", False),
                        should_compress=rule.get("should_compress", False),
                        should_rotate_credential=rule.get("should_rotate_credential", False),
                        should_fallback=rule.get("should_fallback", False),
                        message=f"错误类型: {rule['reason'].value}"
                    )
        
        # 未知错误：保守策略
        return ErrorClassification(
            reason=FailoverReason.UNKNOWN,
            retryable=True,
            should_compress=False,
            should_rotate_credential=False,
            should_fallback=False,
            message="未知错误，将尝试重试"
        )
```

错误分类器的核心设计是**优先级管道**：从最具体的错误模式开始匹配，一旦命中就立即返回。这避免了模糊匹配导致的误分类。

### 8.3 恢复策略：分类之后怎么做？

分类只是第一步。关键是根据分类结果采取相应的恢复策略：

```python
# 基于错误分类的恢复策略

class ErrorRecovery:
    """错误恢复策略执行器"""
    
    def __init__(self, agent: 'AIAgent'):
        self.agent = agent
    
    async def handle_error(self, error: Exception) -> bool:
        """处理错误，返回是否可以继续"""
        classification = ErrorClassifier().classify(error)
        
        # ① 轮换凭证
        if classification.should_rotate_credential:
            rotated = await self.agent.rotate_credential()
            if rotated:
                return True  # 凭证轮换成功，重试
        
        # ② 压缩上下文
        if classification.should_compress:
            await self.agent.compress_context()
            return True  # 压缩后重试
        
        # ③ 切换到备用模型
        if classification.should_fallback:
            fallback_available = await self.agent.switch_to_fallback_model()
            if fallback_available:
                return True  # 切换成功，重试
        
        # ④ 可重试的错误：指数退避重试
        if classification.retryable:
            await self._exponential_backoff_retry(error)
            return True
        
        # ⑤ 不可恢复的错误
        return False
    
    async def _exponential_backoff_retry(self, error: Exception, max_retries: int = 3):
        """指数退避重试"""
        import asyncio
        for attempt in range(max_retries):
            wait_time = min(2 ** attempt, 30)  # 最大等待 30 秒
            print(f"⏳ 重试中... ({attempt + 1}/{max_retries})，等待 {wait_time} 秒")
            await asyncio.sleep(wait_time)
            try:
                # 重试操作
                return True
            except Exception as retry_error:
                if attempt == max_retries - 1:
                    raise retry_error
```

### 8.4 工具调用护栏：安全防御

除了处理"意外"错误，Hermes Agent 还需要防范"故意"的安全威胁。`ToolCallGuardrailController` 是工具调用的安全护栏：

```python
# 工具调用护栏

class ToolCallGuardrailController:
    """工具调用安全护栏"""
    
    def __init__(self):
        self._sensitive_patterns = [
            r"rm\s+-rf\s+/",           # 危险删除
            r"curl.*\|.*sh",            # 远程代码执行
            r"wget.*\|.*bash",          # 远程代码执行
            r"chmod\s+777",             # 过度权限
            r"eval\s*\(",               # 代码注入
        ]
    
    def check_tool_call(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """检查工具调用是否安全，返回 (通过, 原因)"""
        
        # ① 检查敏感命令模式
        if tool_name == "run_terminal_command":
            command = arguments.get("command", "")
            for pattern in self._sensitive_patterns:
                if re.search(pattern, command):
                    return False, f"危险命令被拦截: 匹配模式 {pattern}"
        
        # ② 脱敏检查：确保不泄露敏感信息
        if tool_name in ("send_message", "write_file"):
            content = arguments.get("content", "") or arguments.get("text", "")
            if self._contains_sensitive_data(content):
                return False, "内容包含敏感数据（API key、密码等）"
        
        # ③ TOCTOU 窗口关闭：文件操作的原子性检查
        if tool_name in ("write_file", "edit_file"):
            path = arguments.get("path", "")
            if self._is_protected_path(path):
                return False, f"受保护路径: {path}"
        
        return True, "通过"
    
    def _contains_sensitive_data(self, content: str) -> bool:
        """检查内容是否包含敏感数据"""
        patterns = [
            r"sk-[a-zA-Z0-9]{20,}",           # OpenAI API key
            r"AKIA[0-9A-Z]{16}",               # AWS Access Key
            r"ghp_[a-zA-Z0-9]{36}",            # GitHub Token
            r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",  # 私钥
        ]
        return any(re.search(p, content) for p in patterns)
```

### 8.5 本章小结

韧性设计是 Hermes Agent 能够在生产环境中可靠运行的关键。通过错误分类、恢复策略和安全护栏三个层次，Agent 能够：

1. **识别错误**：快速分类，知道发生了什么
2. **恢复运行**：自动重试、轮换凭证、压缩上下文、切换模型
3. **防范威胁**：拦截危险命令、检测敏感数据泄露、保护受控路径

下一章，我们来看一个更加隐蔽但致命的威胁：供应链攻击。

---

## 第九章：供应链之战——从 Mini Shai-Hulud 蠕虫看依赖安全

### 9.1 问题引入：你信任你的依赖吗？

2026 年 5 月，一个名为 **Mini Shai-Hulud** 的蠕虫在 npm 生态中爆发。它通过被感染的包传播，一旦安装就会窃取环境变量中的 API key 和 token，然后感染该用户发布的其他包。

这不是假设场景——这是真实事件。

对于一个 AI Agent 框架来说，供应链攻击尤其危险，因为：
1. Agent 通常需要访问敏感资源（API key、数据库密码、云服务凭证）
2. Agent 有能力执行任意代码（工具调用）
3. Agent 的上下文中可能包含用户的私人信息

**一个被供应链攻击的 Agent，就是一个被劫持的 Agent。**

### 9.2 Hermes Agent 的依赖安全策略

Hermes Agent 采取了多层防御策略：

**第一层：精确版本锁定**

```toml
# pyproject.toml — 精确版本锁定（无范围约束）

[dependencies]
openai = "1.82.0"           # 不是 "^1.82.0" 或 ">=1.82.0"
anthropic = "0.52.0"        # 精确到补丁版本
pydantic = "2.11.1"         # 无范围，无通配
httpx = "0.28.1"
```

为什么不用范围约束（如 `^1.82.0`）？因为范围约束意味着"在一定范围内自动升级"。如果 1.82.1 版本被注入了恶意代码，范围约束会自动引入它。

**精确版本锁定意味着每次升级都是人工审查后的决策。**

**第二层：懒加载依赖**

```python
# 懒加载：只在真正需要时才导入

def get_playwright():
    """懒加载 Playwright"""
    try:
        import playwright
        return playwright
    except ImportError:
        raise RuntimeError(
            "Playwright 未安装。请运行: pip install playwright"
        )

# 使用时才加载
async def take_screenshot(url: str):
    playwright = get_playwright()  # 这里才真正导入
    async with playwright.async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        await page.screenshot(path="screenshot.png")
```

懒加载的好处是：即使一个可选依赖被攻击了，如果用户从未使用该功能，恶意代码也不会被执行。

**第三层：TOCTOU 窗口关闭**

TOCTOU（Time-of-Check to Time-of-Use）是一种经典的安全漏洞。在 Agent 的上下文中，一个 TOCTOU 攻击可能是：

1. Agent 检查一个文件是否安全（Time-of-Check）
2. 攻击者在检查和使用之间修改了文件（Time-of-Use）
3. Agent 使用了被修改的文件

Hermes Agent 通过原子性操作来关闭这个窗口：

```python
# 原子性文件操作：关闭 TOCTOU 窗口

import os
import tempfile

def atomic_write(path: str, content: str):
    """原子性写入文件"""
    # 写入临时文件
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        mode='w', dir=dir_name, delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    # 原子性重命名（在 POSIX 系统上是原子操作）
    os.rename(tmp_path, path)
```

### 9.3 安全波次：v0.13.0 的安全加固

v0.13.0 版本包含了一个专门的"安全波次"——8 个 P0（最高优先级）安全修复：

1. **凭证脱敏默认开启**：所有输出默认过滤敏感信息
2. **Discord 角色允许列表**：按 guild 作用域控制访问
3. **WhatsApp 默认拒绝陌生人**：不在通讯录中的发送者被默认拒绝
4. **工具调用参数验证**：严格校验所有工具参数
5. **路径遍历防护**：防止通过 `../` 访问受保护目录
6. **命令注入防护**：终端命令参数的转义处理
7. **SSRF 防护**：限制 Agent 可以访问的 URL 范围
8. **日志脱敏**：确保日志中不包含敏感数据

### 9.4 本章小结

供应链安全不是一个可以"一次性解决"的问题。它需要持续的警惕和多层防御。Hermes Agent 的策略是：

1. **精确锁定**：每个依赖版本都是人工审查的结果
2. **最小权限**：懒加载确保未使用的功能不会引入风险
3. **原子操作**：关闭 TOCTOU 窗口
4. **安全波次**：定期进行安全加固

最后一章，让我们展望 Hermes Agent 的未来。

---

## 第十章：面向未来——可插拔架构与 RL 训练集成

### 10.1 问题引入：今天的架构能支撑明天的需求吗？

AI 领域的变化速度是惊人的。今天最好的模型，明天可能就被超越。今天的工具调用标准，明天可能就被新标准取代。

**一个 Agent 框架如果不能适应变化，它就会在变化中消亡。**

Hermes Agent 从一开始就将"可插拔"作为核心设计原则。让我们看看它是如何在各个层面实现这一原则的。

### 10.2 可插拔的推理后端

不同的任务可能需要不同的模型。代码生成用 Claude，数据分析用 GPT-4，本地推理用 Llama。Hermes Agent 通过 `ProviderProfile` 抽象基类实现了推理后端的可插拔：

```python
# 可插拔推理后端

from abc import ABC, abstractmethod
from typing import AsyncIterator

class ProviderProfile(ABC):
    """推理后端抽象基类"""
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: list[dict], 
        tools: list[dict] = None,
        **kwargs
    ) -> dict:
        """聊天补全"""
        ...
    
    @abstractmethod
    async def stream_completion(
        self, 
        messages: list[dict], 
        tools: list[dict] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """流式聊天补全"""
        ...
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """获取模型信息（上下文窗口大小、支持的特性等）"""
        ...

# OpenAI 实现
class OpenAIProvider(ProviderProfile):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def chat_completion(self, messages, tools=None, **kwargs):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            **kwargs
        )
        return response.model_dump()

# 本地模型实现（如 Ollama）
class OllamaProvider(ProviderProfile):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def chat_completion(self, messages, tools=None, **kwargs):
        # Ollama 的 API 格式与 OpenAI 类似但不完全相同
        # 这里进行适配
        ...
```

### 10.3 插件系统：从功能到生态

Hermes Agent 的插件系统覆盖了多个维度：

```
plugins/
├── memory/              # 记忆插件
│   ├── short_term/      # 短期记忆（上下文）
│   └── long_term/       # 长期记忆（持久化）
├── context_engine/      # 上下文管理插件
│   ├── compressor/      # 压缩器
│   └── truncator/       # 截断器
├── model_providers/     # 模型提供商插件
│   ├── openai/
│   ├── anthropic/
│   ├── ollama/
│   └── custom/
├── kanban/              # 协作插件
├── observability/       # 可观测性插件
│   ├── logging/
│   ├── metrics/
│   └── tracing/
└── image_gen/           # 图像生成插件
```

MCP（Model Context Protocol）集成进一步扩展了插件能力：

```python
# MCP 集成：SSE 传输 + OAuth 转发

class MCPClient:
    """MCP 客户端：连接外部工具服务"""
    
    def __init__(self, server_url: str, auth_token: str = None):
        self.server_url = server_url
        self.auth_token = auth_token
        self._tools: dict[str, dict] = {}
    
    async def connect(self):
        """通过 SSE 连接 MCP 服务器"""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "GET", f"{self.server_url}/sse", headers=headers
            ) as response:
                async for event in response.aiter_lines():
                    await self._handle_sse_event(event)
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用 MCP 服务器上的工具"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/tools/{tool_name}",
                json=arguments,
                headers=self._get_auth_headers()
            )
            return response.json()["result"]
```

### 10.4 Nous Tool Gateway：订阅制工具访问

v0.10.0 引入了 Nous Tool Gateway——一个订阅制的工具访问服务。这意味着：

- 用户不需要自己部署所有工具
- 工具提供者可以通过 Gateway 分发他们的工具
- 使用量按订阅计费

```python
# Nous Tool Gateway 集成

class NousToolGateway:
    """Nous Tool Gateway 客户端"""
    
    def __init__(self, subscription_key: str):
        self.subscription_key = subscription_key
        self.base_url = "https://gateway.nousresearch.com"
    
    async def list_available_tools(self) -> list[dict]:
        """列出当前订阅可用的工具"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/tools",
                headers={"X-Subscription-Key": self.subscription_key}
            )
            return response.json()["tools"]
    
    async def invoke_tool(self, tool_name: str, arguments: dict) -> dict:
        """通过 Gateway 调用工具"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/invoke/{tool_name}",
                json=arguments,
                headers={"X-Subscription-Key": self.subscription_key}
            )
            return response.json()
```

### 10.5 RL 训练集成：让 Agent 从行为中学习

Hermes Agent 的架构天然适合与强化学习（RL）集成。Agent 的每次对话都可以被视为一个"episode"，其中：

- **状态（State）**：当前的消息上下文
- **动作（Action）**：选择调用哪个工具、传什么参数
- **奖励（Reward）**：任务是否成功完成、用户满意度

```python
# RL 训练数据收集

@dataclass
class AgentEpisode:
    """Agent 的一次对话 episode"""
    session_id: str
    messages: list[dict]           # 完整的消息历史
    tool_calls: list[dict]         # 所有工具调用
    tool_results: list[dict]       # 所有工具结果
    final_outcome: str             # 最终结果
    user_satisfaction: float       # 用户满意度评分（0-1）
    task_completed: bool           # 任务是否完成
    total_tokens: int              # 总 token 消耗
    total_time: float              # 总耗时

class RLDataCollector:
    """RL 训练数据收集器"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
    
    def collect_episode(self, episode: AgentEpisode):
        """收集一个 episode 的数据"""
        # 计算奖励信号
        reward = self._compute_reward(episode)
        
        # 保存为训练数据格式
        training_sample = {
            "state": self._extract_state(episode.messages),
            "actions": episode.tool_calls,
            "reward": reward,
            "metadata": {
                "tokens": episode.total_tokens,
                "time": episode.total_time,
                "completed": episode.task_completed
            }
        }
        
        self._persist(training_sample)
    
    def _compute_reward(self, episode: AgentEpisode) -> float:
        """计算复合奖励"""
        reward = 0.0
        
        # 任务完成奖励
        if episode.task_completed:
            reward += 1.0
        
        # 用户满意度奖励
        reward += episode.user_satisfaction * 0.5
        
        # 效率惩罚（token 越多，惩罚越大）
        efficiency_penalty = min(episode.total_tokens / 100000, 0.3)
        reward -= efficiency_penalty
        
        return reward
```

这种 RL 集成的潜力是巨大的：通过分析大量 Agent 行为数据，可以训练出更好的工具选择策略、更高效的上下文管理策略、更准确的错误恢复策略。

### 10.6 国际化：7 种语言

v0.13.0 支持 7 种语言的界面和错误消息：

```python
# 国际化支持

I18N_MESSAGES = {
    "en": {
        "error.auth_failed": "Authentication failed. Please check your API key.",
        "error.rate_limit": "Rate limit exceeded. Retrying in {seconds} seconds.",
        "task.completed": "Task completed successfully.",
    },
    "zh": {
        "error.auth_failed": "认证失败。请检查您的 API key。",
        "error.rate_limit": "已超出速率限制。将在 {seconds} 秒后重试。",
        "task.completed": "任务已成功完成。",
    },
    "ja": {
        "error.auth_failed": "認証に失敗しました。APIキーを確認してください。",
        "error.rate_limit": "レート制限を超過しました。{seconds}秒後に再試行します。",
        "task.completed": "タスクが正常に完了しました。",
    },
    # ... 共 7 种语言
}

def get_message(key: str, lang: str = "en", **kwargs) -> str:
    """获取国际化消息"""
    message = I18N_MESSAGES.get(lang, I18N_MESSAGES["en"]).get(key, key)
    return message.format(**kwargs)
```

### 10.7 本章小结

Hermes Agent 的可插拔架构确保了它能够适应未来的各种变化：

1. **推理后端可插拔**：随时切换到新模型、新提供商
2. **插件系统**：功能模块化，按需组合
3. **MCP 集成**：标准化的外部工具接入
4. **RL 训练集成**：从行为中学习，持续优化
5. **国际化**：面向全球用户

---

## 结语：进化的意义

回顾整本书，我们走过了 Hermes Agent 的十个关键设计领域：

| 章节 | 核心问题 | 解决方案 |
|------|----------|----------|
| 第一章 | AI 不能"动手" | Agent 范式的诞生 |
| 第二章 | 如何让 AI 执行操作 | ReAct 循环 + 工具调用 |
| 第三章 | 工具太多怎么管 | 注册中心 + 工具集系统 |
| 第四章 | 上下文会溢出 | 可插拔压缩引擎 |
| 第五章 | Agent 不会成长 | 技能系统 + Curator |
| 第六章 | Agent 只能在终端用 | 消息网关 + 20 平台适配 |
| 第七章 | 复杂任务一个人不够 | Kanban 多 Agent 协作 |
| 第八章 | 错误无处不在 | 错误分类 + 恢复策略 |
| 第九章 | 供应链会投毒 | 精确锁定 + 多层防御 |
| 第十章 | 架构要面向未来 | 可插拔 + RL 集成 |

每一个解决方案都不是凭空设计的——它们都是对真实问题的回应。迭代预算来自"Agent 陷入死循环"的教训，check_fn 缓存来自"工具可用性检查太慢"的痛点，Curator 来自"技能会过时"的现实。

**这就是"从问题到进化"的含义：不是先有完美的架构，而是先有问题，然后在解决问题的过程中，架构自然地进化出来。**

Hermes Agent 还在进化。v0.13.0 不是终点，而是又一个起点。随着 AI 模型能力的提升、应用场景的扩展、安全威胁的演变，它会继续遇到新的问题，继续寻找新的解决方案，继续进化。

这，就是一个开源项目的生命力所在。

---

> **附录 A：快速开始**
>
> ```bash
> # 克隆项目
> git clone https://github.com/NousResearch/hermes-agent.git
> cd hermes-agent
>
> # 安装依赖（Python 3.11+）
> pip install -e .
>
> # 运行测试
> pytest tests/ -x -q
>
> # 启动 CLI
> python -m hermes.cli
> ```

> **附录 B：项目数据**
>
> - 版本：v0.13.0
> - 许可证：MIT
> - 文件数：3433
> - 测试数：~17000
> - 支持平台：20 个消息平台
> - 核心工具：40+
> - 支持语言：7 种
> - 开发周期：2026.2 ~ 2026.5
> - Commits：864
> - PRs：588

> **附录 C：术语表**
>
> - **ReAct**：Reasoning + Acting，推理与行动交替的 Agent 范式
> - **Function Calling**：LLM 声明要调用外部函数的机制
> - **FTS5**：SQLite 的全文搜索扩展第 5 版
> - **MCP**：Model Context Protocol，模型上下文协议
> - **TOCTOU**：Time-of-Check to Time-of-Use，检查时间到使用时间的竞争条件
> - **Kanban**：看板式任务管理方法
> - **Curator**：策展人/维护者，在此指技能生命周期管理器
> - **Surrogate**：Unicode 代理对字符，推理模型可能产生的非法字符
> - **AST**：Abstract Syntax Tree，抽象语法树

---

*本书基于 Hermes Agent v0.13.0 源码分析撰写。项目持续更新中，请以最新版本为准。*
