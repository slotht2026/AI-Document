# 《CLI-Anything：让所有软件成为AI Agent的原生工具》

**一本关于AI Agent与专业桌面软件交互的系统性著作**

---

## 书籍信息

- **书名**: CLI-Anything：让所有软件成为AI Agent的原生工具
- **副标题**: 从GUI自动化脆弱性到结构化命令行接口的范式革命
- **字数**: 约10万字
- **基于**: HKUDS/CLI-Anything 开源项目深度解析
- **版本**: v1.0 (2026年5月)

## 目录结构

本书分为 **八大部分、三十二章**，从问题定义到架构解析，从实现细节到生态展望，系统性地阐述了CLI-Anything项目的理论基础、技术实现和工程实践。

```
CLI-Anything-book/
├── README.md                    # 本文件
├── 00-前言与导读.md
├── PART-1-问题空间.md           # 第一部分：AI Agent与软件交互的困境
├── PART-2-解决方案.md           # 第二部分：CLI-Anything的范式革命
├── PART-3-架构深度解析.md       # 第三部分：七阶段流水线与核心架构
├── PART-4-实现细节.md           # 第四部分：代码实现与工程模式
├── PART-5-领域案例.md           # 第五部分：50+软件的领域实践
├── PART-6-生态系统.md           # 第六部分：CLI-Hub、Preview协议与生态
├── PART-7-前沿与未来.md         # 第七部分：Agent-Native软件的未来
└── PART-8-附录.md               # 第八部分：参考、术语与资源
```

## 阅读建议

- **快速了解**: 阅读前言 + 第1章 + 第2章，约1小时
- **技术深度**: 阅读第三部分 + 第四部分，约4小时
- **完整阅读**: 按顺序阅读全部八部分，约15-20小时
- **按需查阅**: 各部分相对独立，可按兴趣跳读

## 核心数据

| 指标 | 数值 |
|------|------|
| 全书总字数 | 约10万字 |
| 总章节数 | 41章 + 10个附录 |
| 覆盖软件数 | 50+ |
| 分析代码行数 | 1518个文件 |
| 引用测试数 | 2,280+ |
| 八大部分 | 问题空间、解决方案、架构、实现、案例、生态、未来、附录 |

## 关键词

AI Agent、CLI-Anything、命令行接口、GUI自动化、Agent-Native、工具使用、HKUDS、
七阶段流水线、Preview协议、CLI-Hub、SKILL.md、ReplSkin、Click框架、命名空间包

## 版权声明

本书基于CLI-Anything项目（Apache License 2.0）的公开代码和文档撰写。
CLI-Anything由香港大学数据科学实验室（HKUDS）创建和维护。
GitHub: https://github.com/HKUDS/CLI-Anything

# 前言与导读

## 写在前面

2026年春天，一个名为CLI-Anything的开源项目在GitHub上悄然出现。它来自香港大学数据科学实验室（HKUDS），承诺用一条命令将任何专业桌面软件变成AI Agent可以稳定调用的工具。这个承诺听起来像是又一个自动化领域的过度营销——直到你亲眼看到AI Agent用它生成的命令行接口，真的在Blender里渲染出一个完整的3D场景，在FreeCAD里组装出一辆火星车模型，在Shotcut里完成一段多轨视频剪辑，在LibreOffice里生成一份格式精良的PDF报告。

这不是演示，不是玩具，不是PPT上的概念图。2280个测试全部通过，50多个专业软件被成功"驯化"，从图像编辑到3D建模，从音频处理到游戏引擎，从GIS地理信息系统到GPU帧捕获分析——每一个CLI接口都调用真实的软件后端，生成真实的项目文件，产出真实的渲染结果。

本书是对CLI-Anything项目的全面解剖。我们不仅要说清楚"它是什么"和"它怎么工作"，更要深入探讨"为什么需要它"以及"它将把我们带向何方"。

## 为什么写这本书

在过去的两年里，AI领域发生了两次重大转变。

第一次转变是大语言模型（LLM）从"聊天机器人"进化为"推理引擎"。GPT-4、Claude、Gemini、MiMo等模型不仅能回答问题，还能制定计划、分解任务、调用工具。AI Agent的概念从论文走向现实。

第二次转变是AI Agent从"玩具演示"走向"真实工作"。早期的Agent只能做一些简单的API调用和文件操作。如今的Agent需要能够操作Photoshop、Blender、LibreOffice、OBS Studio这些人类用了几十年的专业软件——不是通过截屏点击的脆弱方式，而是通过稳定、可靠、可验证的结构化接口。

CLI-Anything正是这两次转变交汇处的产物。它回答了一个迫切的问题：**当AI Agent需要使用为人类设计的专业软件时，我们应该怎么做？**

这个问题的答案，决定了AI Agent能否真正进入专业工作流，能否真正替代（或增强）人类在创意、工程、科研等领域的生产力。

## 本书的读者

本书面向以下读者：

**AI Agent开发者**：如果你正在构建能够操作外部软件的Agent系统，CLI-Anything提供了一种比GUI自动化更可靠的范式。你需要理解它的架构、它的限制、以及如何将它集成到你的Agent框架中。

**桌面软件开发者**：如果你维护着一个开源的桌面应用，CLI-Anything提供了一种方法论，让你的软件无需编写额外的API就能被AI Agent操控。你需要理解这个方法论的每一步，以及它对你的软件架构意味着什么。

**AI研究人员**：如果你研究的是Agent与环境的交互、工具使用（Tool Use）、或具身智能（Embodied AI），CLI-Anything代表了一种独特的思路——它不试图让Agent"看懂"GUI，而是为Agent创造一个全新的交互层。

**技术决策者**：如果你需要评估AI自动化技术的成熟度和适用场景，本书提供了大量的实证数据和工程经验，帮助你做出判断。

**对技术趋势感兴趣的读者**：如果你关心AI将如何改变软件行业，CLI-Anything的故事提供了一个有趣的视角——软件的"用户"正在从人类变成AI，这将如何改变软件的设计、分发和使用方式。

## 如何阅读本书

本书分为八个部分，建议按以下方式阅读：

**第一部分（问题空间）** 建立背景认知。如果你已经熟悉AI Agent和GUI自动化的问题，可以快速浏览。

**第二部分（解决方案）** 理解CLI-Anything的核心思路。这是全书的理论基础，建议仔细阅读。

**第三部分（架构深度解析）** 技术核心。七阶段流水线、状态管理、会话模型——如果你想深度理解这个项目，这部分是必读的。

**第四部分（实现细节）** 代码级解析。适合有编程经验的读者，包含大量的代码片段和架构图。

**第五部分（领域案例）** 50+软件的实践总结。可以按兴趣跳读，每个领域相对独立。

**第六部分（生态系统）** CLI-Hub、Preview协议、SKILL.md规范。适合需要在生产环境中使用CLI-Anything的读者。

**第七部分（前沿与未来）** 趋势分析和展望。适合所有读者。

**第八部分（附录）** 术语表、参考文献、资源链接。

## 致谢

感谢香港大学数据科学实验室（HKUDS）的CLI-Anything团队创建了这个出色的开源项目。感谢所有为CLI-Anything贡献代码、文档和测试的社区开发者。感谢Apache 2.0许可证让这个项目能够被自由地研究和使用。

感谢所有在AI Agent、CLI设计、桌面软件自动化领域做出贡献的研究者和工程师。CLI-Anything不是凭空出现的，它站在无数前辈的肩膀上。

---

*本书基于CLI-Anything项目的公开代码和文档撰写，力求准确反映项目的技术细节和设计理念。由于项目持续更新，书中描述的代码细节可能与最新版本存在差异。建议读者结合GitHub仓库的最新代码阅读。*

## 全书结构概览

本书共分为八大部分，涵盖从理论到实践的完整知识体系：

| 部分 | 主题 | 章节数 | 核心内容 |
|------|------|--------|----------|
| 第一部分 | 问题空间 | 4章 | AI Agent与软件交互的困境、GUI自动化的脆弱性、现有方案的局限 |
| 第二部分 | 解决方案 | 4章 | CLI-Anything的核心理念、七阶段流水线概览、关键架构模式 |
| 第三部分 | 架构深度解析 | 5章 | 代码分析、CLI设计、实现模式、测试架构、Preview协议 |
| 第四部分 | 实现细节 | 6章 | Click框架、后端包装器、ReplSkin、会话管理、测试实现、SKILL.md |
| 第五部分 | 领域案例 | 10章 | 50+软件在创意、视频、办公、3D、游戏、开发、网络、AI等领域的实践 |
| 第六部分 | 生态系统 | 4章 | CLI-Hub、Preview协议、多平台集成、社区贡献 |
| 第七部分 | 前沿与未来 | 7章 | Agent-Native软件、技术趋势、行业影响、哲学思考 |
| 第八部分 | 附录 | 10节 | 术语表、软件列表、命令参考、FAQ、局限性、致谢 |

**全书总计：约10万字，39章，10个附录**

# 第一部分：AI Agent与软件交互的困境

> "AI agents are great at reasoning but terrible at using real professional software."
> — CLI-Anything README

---

## 第1章 AI Agent的崛起与工具使用需求

### 1.1 从聊天到行动：AI Agent的进化

2022年末，ChatGPT的发布标志着大语言模型（LLM）进入公众视野。在最初的阶段，LLM主要被视为一个高级的聊天机器人——它能回答问题、生成文本、翻译语言，但它的能力边界止于对话窗口。

然而，研究者们很快意识到，LLM的真正潜力不在于"说"，而在于"做"。2023年，一系列关于AI Agent的研究开始涌现。Agent的核心思想是：让LLM不仅能够推理，还能够感知环境、制定计划、采取行动、并根据结果调整策略。

这个思想并不新。早在1990年代，强化学习社区就在研究Agent与环境的交互。但LLM带来了一个根本性的变化：Agent不再需要针对每个任务从头训练。一个足够强大的LLM可以零样本（zero-shot）地理解新任务，制定合理的计划，并调用从未见过的工具。

到2025年，AI Agent的概念已经从论文走向产品。Claude Code、Cursor、Copilot CLI、OpenClaw等工具让开发者可以直接在终端中与AI Agent协作。这些Agent不仅能编写代码，还能执行命令、操作文件、调用API——它们正在成为真正的"数字工人"。

### 1.2 工具使用：Agent的"手"

如果推理是Agent的"脑"，那么工具使用就是Agent的"手"。一个不能使用工具的Agent，充其量是一个高级的搜索引擎。

工具使用（Tool Use）的核心是：Agent需要知道有哪些工具可用，理解每个工具的功能和参数，然后在正确的时机调用正确的工具。这个过程看似简单，但在实践中充满了挑战。

**挑战一：工具发现**。Agent如何知道一个系统中有哪些工具可用？传统的方法是人工编写工具描述，但这种方法不可扩展。CLI-Anything的解决方案是利用`--help`标志——每个CLI命令都自带文档，Agent可以通过`which`命令发现工具，通过`--help`理解工具。

**挑战二：参数推理**。给定一个任务，Agent需要推理出正确的工具调用序列和参数。这要求工具的接口设计得清晰、一致、可预测。CLI-Anything通过统一的命令结构和`--json`输出模式，大幅降低了Agent的推理负担。

**挑战三：错误恢复**。工具调用可能失败——参数错误、依赖缺失、权限不足。Agent需要能够理解错误信息并自我修正。CLI-Anything通过"失败大声且清晰"的设计原则，确保Agent能够获得足够的信息来诊断和修复问题。

**挑战四：状态管理**。许多专业操作需要多步骤完成，每一步都依赖前一步的结果。Agent需要一种可靠的方式来维护和查询状态。CLI-Anything通过持久化的会话状态、撤销/重做机制和JSON序列化，提供了可靠的状态管理基础。

### 1.3 Agent工具使用的光谱

AI Agent使用工具的方式可以排列成一个光谱：

**光谱的一端是纯API调用**。Agent直接调用REST API、SDK函数或RPC接口。这种方式精确、可靠、高效，但需要为每个软件编写专门的集成代码。大多数软件没有提供API，或者API只覆盖了软件功能的一小部分。

**光谱的另一端是GUI自动化**。Agent通过截屏、OCR、鼠标点击和键盘输入来操作软件的图形界面。这种方式理论上可以操作任何GUI软件，但实际上极其脆弱——UI布局变化、分辨率差异、动画延迟、弹窗干扰都会导致自动化失败。

**CLI-Anything占据了光谱的中间位置**。它不直接调用API（因为大多数软件没有API），也不模拟GUI操作（因为太脆弱）。它的做法是：为软件创建一个结构化的命令行接口，这个接口既不是API也不是GUI，而是一种专门为Agent设计的"第三种交互方式"。

这个思路的优雅之处在于：命令行是人类已经使用了50多年的交互范式，它天然具有结构化（命令+参数+输出）、可组合（管道和重定向）、自文档化（`--help`）的特性。CLI-Anything所做的，只是将这些已有的、经过时间验证的特性，与AI Agent的需求对齐。

### 1.4 从"能用"到"好用"：Agent体验的层次

一个Agent可以使用工具，不等于它能高效地使用工具。CLI-Anything在设计中考虑了Agent体验（Agent Experience, AX）的多个层次：

**第一层：可发现性（Discoverability）**。Agent能否找到工具？CLI-Anything通过`which`命令和标准的命名约定（`cli-anything-<software>`）确保Agent可以发现工具。

**第二层：可理解性（Understandability）**。Agent能否理解工具的功能？CLI-Anything通过`--help`输出和SKILL.md文件提供全面的文档。

**第三层：可操作性（Operability）**。Agent能否正确地调用工具？CLI-Anything通过一致的命令结构、清晰的参数定义和`--json`输出模式，确保Agent可以正确操作。

**第四层：可恢复性（Recoverability）**。Agent能否从错误中恢复？CLI-Anything通过清晰的错误消息、幂等操作和撤销/重做机制，确保Agent可以从失败中恢复。

**第五层：可观测性（Observability）**。Agent能否理解操作的结果？CLI-Anything通过结构化的JSON输出、进度指示和状态查询命令，确保Agent可以观测操作结果。

这五个层次共同构成了CLI-Anything的"Agent体验"框架，也是本书后续章节的核心分析维度。

---

## 第2章 GUI自动化的脆弱性

### 2.1 人类的界面，Agent的噩梦

图形用户界面（GUI）是为人类设计的。它依赖人类的视觉感知、空间推理和手眼协调能力。当AI Agent试图操作GUI时，它面临的是一个为另一种智能体设计的界面——就像让一个盲人使用一个依赖颜色编码的控制面板。

GUI自动化的技术演进可以分为三代：

**第一代：坐标点击**。最原始的GUI自动化是直接操作屏幕坐标。`pyautogui.click(100, 200)`在屏幕位置(100, 200)执行一次点击。这种方式简单直接，但极度脆弱——窗口移动一像素，自动化就失败了。

**第二代：元素定位**。通过可访问性树（Accessibility Tree）或DOM结构定位UI元素。Selenium、Playwright等工具使用这种方式。比坐标点击稳定，但仍然依赖UI的具体结构——一个按钮改了ID，自动化就断了。

**第三代：视觉理解**。通过多模态大模型"看懂"屏幕截图，然后决定点击哪里。CogAgent、SeeClick、OS-Copilot等研究使用这种方式。这种方式更接近人类的操作方式，但准确率和速度都远低于前两代。

### 2.2 GUI自动化的六大脆弱性

无论采用哪一代技术，GUI自动化都存在以下根本性问题：

**脆弱性一：布局依赖**。GUI的布局是为人类视觉设计的，不是为机器解析设计的。窗口大小变化、面板折叠展开、主题切换都会改变元素的位置和可见性。一个在1920x1080分辨率下工作的自动化脚本，在1366x768分辨率下可能完全失败。

**脆弱性二：时序依赖**。GUI操作有固有的时间延迟——动画、加载、渲染都需要时间。自动化脚本需要在正确的时间点等待和检查。等待太短会操作未就绪的元素，等待太长会浪费时间。更糟糕的是，正确的等待时间取决于系统负载、网络速度、数据大小等不可预测的因素。

**脆弱性三：状态不可见**。GUI的许多状态对自动化脚本是不可见的。一个按钮是否可点击？一个菜单是否已展开？一个对话框是否已关闭？这些状态可能需要复杂的逻辑来推断，而GUI本身并不提供可靠的查询接口。

**脆弱性四：错误不可恢复**。当GUI操作失败时，自动化脚本通常无法获得足够的信息来诊断和修复问题。一个点击没有反应——是因为元素不可见？被其他窗口遮挡？还是应用状态不对？GUI不会告诉你。

**脆弱性五：跨平台差异**。同一个软件在Windows、macOS和Linux上的GUI可能完全不同。菜单栏的位置、对话框的样式、键盘快捷键的映射——这些差异使得跨平台的GUI自动化成为一场噩梦。

**脆弱性六：版本脆弱**。软件更新经常会改变GUI的布局、样式和行为。一个在v2.0下工作的GUI自动化脚本，在v2.1下可能就失效了。GUI不是API，没有向后兼容的承诺。

### 2.3 RPA的困境：自动化自动化

机器人流程自动化（RPA）是GUI自动化在企业场景中的商业化形态。UiPath、Automation Anywhere、Blue Prism等RPA平台投入了数十亿美元来解决GUI自动化的脆弱性问题。

RPA行业的数据令人深思：根据行业报告，RPA项目的维护成本通常占总成本的40-60%。一个RPA机器人平均每年需要修改3-5次，每次修改需要数天的开发和测试时间。更令人沮丧的是，许多RPA项目在部署后6个月内就因为"脆弱性疲劳"而被放弃。

RPA的困境反映了一个根本性的问题：**试图用自动化来弥补界面设计的缺陷，是一种治标不治本的方案**。如果界面本身不适合自动化，那么再多的自动化技术也只能延缓而不能消除脆弱性。

### 2.4 截屏+点击：Agent的"眼睛"和"手指"

2024-2025年，一种新的GUI自动化范式开始流行：让AI Agent通过截屏"看"屏幕，然后决定在哪里点击。这种方法被称为"视觉GUI Agent"或"Screenshot-based Agent"。

这种方法的典型流程是：
1. 截取当前屏幕的截图
2. 将截图发送给多模态大模型
3. 大模型分析截图内容，决定要执行的操作（点击、输入、滚动等）
4. 执行操作
5. 截取新的截图，重复上述过程

这种方法的优势是通用性——理论上可以操作任何GUI软件，无需任何预定义的集成代码。但它的劣势也很明显：

**速度慢**。每次操作都需要一次LLM推理调用，延迟在0.5-2秒之间。一个简单的任务可能需要数十次操作，总耗时在分钟级别。

**准确率低**。即使是最先进的多模态大模型，在GUI元素识别和坐标定位上的准确率也远低于100%。特别是在密集的UI区域（如工具栏、菜单），误点的概率很高。

**不可靠**。由于依赖视觉理解，这种方法对屏幕分辨率、字体大小、颜色主题、窗口布局等因素高度敏感。一个在英文界面下工作的Agent，在中文界面下可能完全失败。

**不可验证**。视觉GUI Agent很难验证自己的操作是否成功。它可能点击了错误的按钮而不知道，或者认为操作成功了但实际上失败了。

### 2.5 一个真实的案例：让AI Agent使用GIMP

让我们用一个具体的例子来说明GUI自动化的脆弱性。

GIMP（GNU Image Manipulation Program）是一个功能强大的开源图像编辑软件。假设我们想让AI Agent执行以下任务："创建一个新的1920x1080画布，添加一个纯色背景层，然后应用高斯模糊滤镜。"

**使用GUI自动化**：
1. 截屏 → 识别菜单栏 → 点击"File" → 等待菜单展开
2. 截屏 → 识别"New..."菜单项 → 点击 → 等待对话框
3. 截屏 → 识别宽度输入框 → 点击 → 输入"1920"
4. 截屏 → 识别高度输入框 → 点击 → 输入"1080"
5. 截屏 → 识别"OK"按钮 → 点击 → 等待画布创建
6. 截屏 → 识别图层面板 → 右键 → 等待上下文菜单
7. ...（至少还需要10-15步）

这个流程至少需要15-20次截屏-推理-操作循环，每次循环需要1-3秒，总耗时30-60秒。而且，如果GIMP的版本不同、语言不同、主题不同，任何一步都可能失败。

**使用CLI-Anything生成的CLI**：
```bash
cli-anything-gimp project new --width 1920 --height 1080 -o poster.json
cli-anything-gimp --json layer add -n "Background" --type solid --color "#1a1a2e"
cli-anything-gimp --project poster.json filter apply-gaussian-blur --radius 5.0
```

三条命令，5秒内完成，100%可靠。输出是结构化的JSON，Agent可以立即验证结果。

这就是CLI-Anything的力量：它不是让Agent去适应人类的界面，而是为Agent创造了一个适合Agent的界面。

---

## 第3章 现有解决方案的局限

### 3.1 API优先的方法：理想与现实

面对GUI自动化的脆弱性，一个自然的想法是：让软件提供API，Agent直接调用API。这就是"API优先"（API-first）的方法。

这个方法在理论上是完美的。API是结构化的、有文档的、版本化的、可测试的。如果每个软件都提供完整的API，Agent就可以通过API高效地操作软件，无需任何GUI自动化。

但现实是：**绝大多数专业桌面软件没有提供完整的API**。

**原因一：历史包袱**。许多专业软件有20-30年的历史。在它们最初设计的时代，API不是优先考虑的需求。软件的核心逻辑与GUI代码深度耦合，要将其分离为独立的API层，需要大量的重构工作。

**原因二：用户需求**。桌面软件的主要用户是人类，不是Agent。软件开发者投入大量资源优化GUI体验，但很少有用户要求API。在商业软件的优先级排序中，API通常排在功能开发、性能优化和bug修复之后。

**原因三：复杂性**。专业软件的功能极其丰富。以Blender为例，它包含3D建模、材质系统、灯光系统、动画系统、渲染引擎、合成器、视频编辑器等数十个子系统。为每个子系统提供完整的API，工作量可能比软件本身还大。

**原因四：维护成本**。API一旦发布，就需要维护向后兼容性。对于快速迭代的软件来说，这是一个沉重的负担。许多软件选择不发布API，就是因为不想承担这个负担。

### 3.2 SDK和脚本接口：半成品

一些软件提供了SDK或脚本接口，允许用户通过代码扩展软件的功能。例如：

- Blender提供了Python API（bpy）
- GIMP提供了Script-Fu和Python-Fu
- Inkscape提供了命令行接口和扩展API
- LibreOffice提供了UNO API

这些SDK和脚本接口是CLI-Anything的重要构建块。但它们本身并不足以满足Agent的需求，原因如下：

**接口不一致**。每个软件的SDK都有自己的API设计风格、参数约定和错误处理方式。Agent需要为每个软件学习一套全新的接口。

**文档不充分**。许多SDK的文档是为高级用户编写的，假设读者已经熟悉软件的内部架构。Agent需要的是面向操作的文档，告诉它"做什么"而不是"怎么实现"。

**缺少高层抽象**。SDK通常暴露的是底层原语（如Blender的bpy.ops.mesh.primitive_cube_add），而不是高层任务（如"创建一个红色的立方体"）。Agent需要自行将高层任务分解为底层操作。

**状态管理缺失**。SDK通常不提供项目状态的持久化、撤销/重做、会话管理等功能。Agent需要自行实现这些基础设施。

### 3.3 MCP（Model Context Protocol）：连接而非替代

2024年，Anthropic提出了Model Context Protocol（MCP），旨在为AI模型提供一种标准化的方式来访问外部工具和数据源。MCP定义了一个客户端-服务器架构，其中MCP服务器暴露工具（tools）、资源（resources）和提示（prompts），MCP客户端（如Claude Desktop）可以发现和调用这些能力。

MCP是一个重要的进步，但它解决的是不同层次的问题。MCP关注的是"Agent如何发现和调用工具"，而不是"如何为没有API的软件创建工具"。CLI-Anything可以被看作是MCP的一个上游——它为那些没有API的软件创建了结构化接口，这些接口可以被MCP服务器包装后暴露给Agent。

实际上，CLI-Anything项目中的一些软件（如浏览器CLI的DOMShell后端）就是通过MCP服务器与软件交互的。CLI-Anything的HARNESS.md文档中专门有一节讨论"MCP Backend Pattern"，指导如何为那些提供MCP服务器的软件创建CLI接口。

### 3.4 Computer Use：Anthropic的另一种尝试

2024年末，Anthropic发布了"Computer Use"功能，允许Claude直接操作计算机的屏幕——移动鼠标、点击按钮、输入文字、截取屏幕。这是GUI自动化的Agent版本。

Computer Use的优势是通用性——它可以操作任何GUI软件，无需任何集成代码。但它的劣势与前述的GUI自动化问题完全一致：脆弱、缓慢、不可靠。

更深层次的问题是：Computer Use本质上是在用Agent的"眼睛"和"手指"来弥补软件界面设计的不足。这就像让一个天才数学家用手指在沙地上做计算——他的能力被界面限制了。

CLI-Anything代表了一种完全不同的思路：**不要让Agent去适应人类的界面，而是为Agent创造一个适合Agent的界面**。这不是GUI自动化的改进，而是GUI自动化的替代。

### 3.5 各种方案的对比

| 方案 | 通用性 | 可靠性 | 速度 | 维护成本 | Agent体验 |
|------|--------|--------|------|----------|-----------|
| GUI自动化/RPA | 高 | 低 | 慢 | 高 | 差 |
| 视觉GUI Agent | 很高 | 很低 | 很慢 | 低 | 很差 |
| 软件原生API | 低 | 高 | 快 | 中 | 好 |
| MCP | 中 | 高 | 快 | 中 | 好 |
| **CLI-Anything** | **高** | **高** | **快** | **低** | **很好** |

CLI-Anything的独特优势在于：它不需要软件提供API（通用性高），也不依赖GUI自动化（可靠性高），同时保持了结构化接口的高效性（速度快）和可维护性（维护成本低）。

---

## 第4章 问题的本质：两种智能体之间的界面错配

### 4.1 设计给人类的软件

现代桌面软件的GUI设计基于人类的认知特性：

**视觉优先**。人类是视觉动物，GUI通过图标、颜色、布局来传递信息。一个红色的"X"按钮意味着"关闭"，一个绿色的勾号意味着"确认"。这些视觉线索对人类是直觉的，但对Agent是无意义的。

**空间推理**。GUI利用人类的空间推理能力。菜单在顶部，工具栏在侧边，画布在中间。这种空间布局帮助人类建立心理模型（mental model），但对Agent来说，空间位置只是需要额外解析的噪声。

**渐进式探索**。人类通过"看看有什么"来发现功能。他们点击菜单、探索选项、试错学习。Agent不能（也不应该）这样工作——它需要明确的、可预测的接口。

**容错性**。人类可以容忍GUI中的模糊性和不一致性。一个按钮的图标不太直观？用户可以悬停查看工具提示。一个操作的结果不太确定？用户可以撤销重试。Agent需要的是确定性的、可验证的接口。

### 4.2 设计给Agent的接口

Agent需要的接口具有完全不同的特性：

**结构化**。Agent需要的是"命令 + 参数 + 输出"的结构化交互，而不是"截屏 + 理解 + 点击"的视觉交互。

**自文档化**。Agent需要能够通过查询（如`--help`）了解接口的功能和用法，而不需要预先学习。

**确定性**。相同的输入应该产生相同的输出。Agent需要能够预测操作的结果。

**可验证**。Agent需要能够验证操作是否成功，结果是否正确。

**可组合**。Agent需要能够将多个操作组合成复杂的工作流。

**可恢复**。当操作失败时，Agent需要能够理解失败原因并采取修正措施。

### 4.3 CLI：两种智能体的共同语言

命令行接口（CLI）恰好满足了上述所有需求：

- **结构化**：`命令 子命令 --参数 值`是天然的结构化格式
- **自文档化**：`--help`是所有CLI的标准功能
- **确定性**：相同的命令通常产生相同的结果
- **可验证**：退出码（exit code）和结构化输出（JSON）提供了可靠的验证机制
- **可组合**：管道（pipe）和重定向（redirect）是CLI的核心特性
- **可恢复**：错误消息、撤销命令、状态查询提供了恢复能力

但更重要的是，CLI不仅适合Agent，也适合人类。命令行是人类使用了50多年的交互范式，有成熟的工具链、丰富的最佳实践和庞大的用户群体。

CLI-Anything的核心洞察是：**CLI是人类和Agent的共同语言**。通过为软件创建CLI接口，我们不仅让Agent可以使用软件，也让人类可以用新的方式使用软件。

### 4.4 范式转换：从"让人用Agent的界面"到"让Agent用人的软件"

CLI-Anything代表了一种范式转换：

**旧范式**：AI Agent试图通过模拟人类的GUI操作来使用软件。Agent需要"看懂"屏幕，"找到"按钮，"点击"正确的像素。这种方式将Agent限制在了为人类设计的交互范式中。

**新范式**：AI Agent通过结构化的CLI接口来使用软件。Agent不需要"看"屏幕，不需要"找"按钮，不需要"猜"操作的结果。它直接发出命令，接收结构化的响应。

这个转换的深层意义是：**软件的"用户"正在从人类变成Agent**。当软件的用户是人类时，GUI是最佳选择。当软件的用户是Agent时，CLI是最佳选择。CLI-Anything不要求软件开发者二选一——它在GUI之上叠加了一个CLI层，让软件同时服务于两种用户。

这就是CLI-Anything的愿景：**让所有软件成为Agent-Native（Agent原生）的软件**。不是取代GUI，而是增加一个CLI层。不是为Agent重写软件，而是让现有软件对Agent友好。

---

## 第一部分小结

本部分建立了CLI-Anything的问题空间：

1. **AI Agent需要使用工具**，而最强大的工具是为人类设计的专业桌面软件。
2. **GUI自动化是脆弱的**，它依赖于为人类视觉设计的界面，对Agent来说充满了不确定性。
3. **现有解决方案都有局限**：API不是普遍存在的，SDK不够高层，MCP解决的是不同层次的问题，Computer Use本质上是GUI自动化。
4. **CLI是人类和Agent的共同语言**，CLI-Anything通过为软件创建CLI接口，提供了一种比GUI自动化更可靠的Agent-软件交互范式。

在下一部分，我们将深入解析CLI-Anything的解决方案——它是如何将这个理论洞察转化为一个可工作的系统的。
# 第二部分：CLI-Anything的范式革命

> "One Command Line: Make any software agent-ready."
> — CLI-Anything

---

## 第5章 CLI-Anything的核心理念

### 5.1 一句话概括

CLI-Anything的核心理念可以用一句话概括：**用一条命令，将任何有代码库的桌面软件，自动转化为AI Agent可以稳定调用的结构化命令行接口。**

这句话里每个词都很重要：

- **一条命令**：不是手动编写，而是自动生成。输入是软件的源代码，输出是一个完整的CLI包。
- **任何有代码库的软件**：不限于特定领域。图像编辑、3D建模、音频处理、视频剪辑、办公软件、游戏引擎——只要它有代码库，CLI-Anything就可以为它生成CLI。
- **自动转化**：由AI Agent（如Claude Code）执行七阶段流水线，自动分析代码、设计架构、实现接口、编写测试、生成文档。
- **结构化命令行接口**：不是简单的脚本包装，而是设计良好的、有状态管理的、支持JSON输出的CLI。

### 5.2 六个设计原则

CLI-Anything的HARNESS.md文档中定义了六个不可违反的设计原则：

**原则一：使用真实软件**。CLI必须调用真实的软件后端来渲染和导出。不允许用Python重新实现软件的功能。不允许用Pillow替代GIMP，不允许用自定义渲染器替代Blender。这是CLI-Anything的第一原则，也是最重要的原则。

这个原则的深层原因是：专业软件的价值不仅在于它的功能，还在于它的实现。GIMP的色彩管理、Blender的物理渲染、LibreOffice的排版引擎——这些是数十年研发的成果，不可能用几行Python代码替代。CLI-Anything的目标是为这些软件创建一个"遥控器"，而不是用一个"玩具版"来替代它们。

**原则二：直接操作原生格式**。CLI解析和修改软件的原生项目文件（MLT XML、ODF、SVG等），而不是发明新的中间格式。这确保了CLI生成的项目文件可以在原始软件中打开和编辑。

**原则三：利用现有CLI工具**。使用软件自带的命令行工具（如`libreoffice --headless`、`blender --background`、`melt`、`inkscape --actions`）作为后端。这些工具是软件开发者维护的，比任何第三方包装都更可靠。

**原则四：失败大声且清晰**。错误消息必须明确、具体、可操作。Agent需要足够的信息来自我修正，模糊的错误消息会导致Agent陷入重试循环。

**原则五：可幂等**。尽可能使操作幂等——执行两次应该和执行一次有相同的效果。这对于Agent的重试逻辑至关重要。

**原则六：可内省**。提供`info`、`list`、`status`等探查命令，让Agent在修改之前先了解当前状态。

### 5.3 Agent-Native设计：`--json`的核心地位

CLI-Anything的一个关键设计决策是：**每个命令都必须支持`--json`标志**。

这不是一个简单的输出格式选项。`--json`是CLI-Anything的Agent-Native设计的核心。当Agent调用一个CLI命令时，它需要的不是人类可读的表格或彩色文本，而是结构化的、可解析的JSON数据。

例如，当Agent创建一个新的Blender场景时：

```bash
# 人类模式（默认）
$ cli-anything-blender scene new --name "ProductShot"
Created scene: ProductShot

# Agent模式（--json）
$ cli-anything-blender --json scene new --name "ProductShot"
{
  "name": "ProductShot",
  "resolution_x": 1920,
  "resolution_y": 1080,
  "engine": "CYCLES",
  "samples": 128,
  "fps": 24,
  "objects": 0,
  "materials": 0,
  "modified": true
}
```

JSON输出让Agent可以：
- 确认操作是否成功
- 获取操作的详细结果
- 将结果传递给下一步操作
- 验证输出是否符合预期

### 5.4 双模式交互：CLI + REPL

CLI-Anything的每个CLI都支持两种交互模式：

**子命令模式**（One-shot）：适合脚本和自动化。
```bash
cli-anything-blender scene new --name "MyScene"
cli-anything-blender object add cube --name "MyCube"
cli-anything-blender render execute output.png
```

**REPL模式**（Interactive）：适合交互式探索和调试。
```
$ cli-anything-blender
╔══════════════════════════════════════════╗
║       cli-anything-blender v1.0.0       ║
║     Blender CLI for AI Agents           ║
╚══════════════════════════════════════════╝

blender> scene new --name ProductShot
✓ Created scene: ProductShot

blender[ProductShot]> object add-mesh --type cube --location 0 0 1
✓ Added mesh: Cube at (0, 0, 1)

blender[ProductShot]> exit
Goodbye! 👋
```

REPL模式通过`ReplSkin`类提供统一的品牌化界面，包括：
- 彩色的启动横幅（显示CLI名称、版本、SKILL.md路径）
- 上下文感知的提示符（显示当前项目名称和修改状态）
- 命令历史记录（持久化到`~/.cli-anything-<software>/history`）
- 自动补全和语法高亮
- 统一的成功/错误/警告/信息消息格式

REPL是CLI-Anything的默认行为——当用户不带任何参数运行`cli-anything-blender`时，自动进入REPL模式。这个设计通过Click框架的`invoke_without_command=True`实现。

---

## 第6章 七阶段流水线概览

### 6.1 流水线的全景

CLI-Anything的核心是一个七阶段的自动化流水线。这个流水线由AI Agent（如Claude Code）执行，从分析软件的源代码开始，到发布可安装的CLI包结束。

```
阶段1: 代码分析 (Codebase Analysis)
    ↓
阶段2: CLI架构设计 (CLI Architecture Design)
    ↓
阶段3: 实现 (Implementation)
    ↓
阶段4: 测试计划 (Test Planning)
    ↓
阶段5: 测试实现 (Test Implementation)
    ↓
阶段6: 测试文档 (Test Documentation)
    ↓
阶段6.5: SKILL.md生成 (SKILL.md Generation)
    ↓
阶段7: PyPI发布 (PyPI Publishing)
```

每个阶段都有明确的输入、输出和验收标准。整个流水线的设计遵循"渐进式披露"（Progressive Disclosure）原则——每个阶段的结果为下一个阶段提供上下文。

### 6.2 阶段1：代码分析

**目标**：理解软件的架构，找到CLI接口的切入点。

**步骤**：
1. **识别后端引擎**。大多数GUI应用将展示层和逻辑层分离。找到核心库/框架（如Shotcut的MLT、GIMP的GEGL、Blender的bpy）。
2. **映射GUI动作到API调用**。每个按钮点击、拖拽和菜单项都对应一个函数调用。编目这些映射。
3. **识别数据模型**。软件使用什么文件格式？项目状态如何表示？（XML、JSON、二进制、数据库？）
4. **找到现有CLI工具**。许多后端自带CLI（`melt`、`ffmpeg`、`convert`）。这些是构建块。
5. **编目命令/撤销系统**。如果应用有撤销/重做功能，它可能使用命令模式。这些命令就是CLI操作。

**输出**：一份`<SOFTWARE>.md`文档，描述软件的架构、后端引擎、数据模型和CLI切入点。

### 6.3 阶段2：CLI架构设计

**目标**：设计CLI的命令结构、状态模型和输出格式。

**步骤**：
1. **选择交互模型**：状态化REPL + 子命令CLI（推荐两者都支持）。
2. **定义命令组**：匹配软件的逻辑领域——项目管理、核心操作、导入导出、配置、会话管理。
3. **设计状态模型**：什么需要在命令之间持久化？状态存储在哪里？如何序列化？
4. **规划输出格式**：人类可读（表格、颜色）+ 机器可读（JSON），通过`--json`标志切换。

**输出**：CLI的命令树设计文档，包含命令组、子命令、参数定义和状态模型。

### 6.4 阶段3：实现

**目标**：编写CLI的核心代码。

**步骤**：
1. **数据层**：XML/JSON操作，解析和修改项目文件。
2. **探查命令**：`info`、`list`、`status`，让Agent先了解再操作。
3. **变更命令**：每个逻辑操作一个命令。
4. **后端集成**：`utils/<software>_backend.py`模块，包装真实的软件CLI。
5. **渲染/导出**：生成有效的中间文件，然后调用真实软件进行转换。
6. **会话管理**：状态持久化、撤销/重做。
7. **REPL**：使用统一的`ReplSkin`包装子命令。

**输出**：完整的CLI代码，包含核心模块、后端包装器和REPL界面。

### 6.5 阶段4-6：测试

**阶段4（测试计划）**：在写任何测试代码之前，创建TEST.md文件，列出计划的测试文件、测试数量、测试场景。

**阶段5（测试实现）**：编写四个层次的测试：
- 单元测试（test_core.py）：合成数据，无外部依赖
- E2E测试-原生（test_full_e2e.py）：验证项目文件生成
- E2E测试-真实后端（test_full_e2e.py）：调用真实软件
- CLI子进程测试（test_full_e2e.py）：测试安装后的命令

**阶段6（测试文档）**：运行所有测试，将结果追加到TEST.md。

### 6.6 阶段6.5：SKILL.md生成

**目标**：生成AI Agent可发现的技能定义文件。

SKILL.md是一个标准化的文件，包含：
- YAML frontmatter（名称和描述，用于Agent技能发现）
- 命令组文档
- 使用示例
- Agent特定指导（JSON输出、错误处理、程序化使用）

SKILL.md让CLI-Anything生成的CLI可以被各种AI Agent框架（Claude Code、OpenClaw、Pi、Codex等）自动发现和使用。

### 6.7 阶段7：PyPI发布

**目标**：创建可安装的Python包。

使用PEP 420命名空间包，所有CLI都在`cli_anything.*`命名空间下。`cli_anything/`目录没有`__init__.py`（命名空间包的关键），每个子包（`gimp/`、`blender/`等）有自己的`__init__.py`。

安装后，CLI命令直接出现在PATH中：
```bash
pip install -e .
which cli-anything-gimp  # → /usr/local/bin/cli-anything-gimp
```

---

## 第7章 关键架构模式

### 7.1 后端包装器模式

CLI-Anything的每个CLI都包含一个后端包装器模块（`utils/<software>_backend.py`），负责与真实软件交互。

后端包装器的核心职责是：
1. **找到软件的可执行文件**（使用`shutil.which()`）
2. **构建正确的命令行参数**
3. **调用软件**（使用`subprocess.run()`）
4. **处理错误**，提供清晰的安装指导
5. **验证输出**，确保结果正确

以LibreOffice后端为例：

```python
def convert_odf_to(odf_path, output_format, output_path=None, overwrite=False):
    """Convert ODF file to specified format using real LibreOffice."""
    lo = find_libreoffice()  # Raises RuntimeError with install instructions
    subprocess.run([
        lo, "--headless",
        "--convert-to", output_format,
        "--outdir", output_dir,
        odf_path,
    ])
    return {"output": final_path, "format": output_format, "method": "libreoffice-headless"}
```

这个模式的精髓是：**CLI不自己渲染，它只是软件的遥控器**。

### 7.2 渲染鸿沟（The Rendering Gap）

HARNESS.md文档中特别强调了"渲染鸿沟"问题：大多数GUI应用在渲染时应用效果。当CLI操作项目文件时，必须同时处理渲染——如果使用简单的导出工具，效果会被静默丢弃。

例如，视频编辑软件的滤镜是在渲染时应用的。如果CLI只是修改了项目文件中的滤镜参数，然后用`ffmpeg`直接拼接原始视频片段，滤镜就不会被应用。用户看到的输出和输入完全一样，但CLI报告"操作成功"。

**解决方案的优先级**：
1. **最佳**：使用应用的原生渲染器（如MLT项目的`melt`）
2. **次选**：构建翻译层，将项目格式的效果转换为渲染工具的语法（如MLT滤镜 → ffmpeg `-filter_complex`）
3. **最后**：生成渲染脚本，由用户手动运行

### 7.3 会话管理与状态持久化

CLI-Anything的每个CLI都实现了会话管理，包括：

**项目状态**：当前打开的项目、修改标记、项目路径。

**撤销/重做**：通过快照（snapshot）机制实现。每次变更操作前，保存当前状态的快照。撤销时恢复到上一个快照。

**自动保存**：一次性命令执行后，如果状态被修改，自动保存到项目文件。

**文件锁定**：使用排他文件锁防止并发写入损坏。具体的模式是：以`"r+"`模式打开文件，获取锁，然后在锁内截断并写入。

### 7.4 统一的REPL皮肤（ReplSkin）

ReplSkin是CLI-Anything的统一REPL界面组件。每个CLI都将`repl_skin.py`复制到自己的`utils/`目录中，确保一致的用户体验。

ReplSkin提供的功能：
- **品牌化启动横幅**：显示CLI名称、版本、SKILL.md路径
- **上下文感知的提示符**：显示当前项目名称和修改状态（`*`标记）
- **统一的消息格式**：✓成功、✗错误、⚠警告、●信息
- **格式化表格**：使用Unicode box-drawing字符
- **进度指示器**：`████░░░░ 50%`
- **命令历史**：持久化到`~/.cli-anything-<software>/history`
- **SKILL.md路径自动检测**：优先使用仓库根目录的规范路径，回退到打包的副本

### 7.5 输出验证方法论

CLI-Anything的测试文档中强调了输出验证的重要性：

**不要信任退出码**。"它没有报错"不等于"它工作正常"。

**验证方法**：
- 文件格式验证：检查magic bytes（PDF: `%PDF-`，ZIP: `PK`）
- 结构验证：验证ZIP/OOXML结构、XML格式正确性
- 内容验证：验证像素分析、音频RMS电平、持续时间检查
- 子进程验证：通过`subprocess.run`测试安装后的命令

**示例**：
```python
# 验证PDF输出
with open(result["output"], "rb") as f:
    assert f.read(5) == b"%PDF-"  # Validate format magic bytes

# 验证视频输出
probe = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_streams", output_path], capture_output=True, text=True)
stream_info = json.loads(probe.stdout)
assert stream_info["streams"][0]["codec_name"] == "h264"
```

---

## 第8章 命名空间包与分发架构

### 8.1 PEP 420命名空间包

CLI-Anything使用PEP 420命名空间包来组织所有CLI。这是一个关键的架构决策，确保多个CLI可以在同一个Python环境中共存而不冲突。

**核心规则**：
- `cli_anything/`目录**没有**`__init__.py`
- 每个子包（`gimp/`、`blender/`等）**有**自己的`__init__.py`

这意味着`cli_anything`是一个命名空间包，多个独立安装的PyPI包可以各自贡献一个子包到`cli_anything/`下。例如：
- `cli-anything-gimp`添加`cli_anything/gimp/`
- `cli-anything-blender`添加`cli_anything/blender/`
- 两者在同一个Python环境中和平共存

### 8.2 安装与发现

CLI的安装和发现经过精心设计：

```bash
# 安装
cd gimp/agent-harness
pip install -e .

# 验证
which cli-anything-gimp  # → /usr/local/bin/cli-anything-gimp

# 使用
cli-anything-gimp --help
cli-anything-gimp          # 进入REPL
cli-anything-gimp --json project new  # JSON输出
```

Agent通过标准的`which`命令发现CLI，通过`--help`了解CLI的功能。这种设计利用了Unix工具链的已有约定，无需额外的注册或配置步骤。

### 8.3 CLI-Hub：集中化的CLI注册表

CLI-Hub是CLI-Anything的集中化注册表和安装工具：

```bash
# 安装CLI-Hub
pip install cli-anything-hub

# 浏览可用的CLI
cli-hub list

# 安装一个CLI
cli-hub install blender
```

CLI-Hub的Web界面（https://hkuds.github.io/CLI-Anything/）提供了可视化的浏览和搜索功能。注册表数据存储在`registry.json`中，包含每个CLI的名称、描述、安装命令、分类和贡献者信息。

CLI-Hub还支持"元技能"（meta-skill）——一个SKILL.md文件，让AI Agent可以自主发现和安装CLI。Agent读取元技能后，可以浏览CLI目录，选择适合任务的CLI，然后自动安装和使用。

---

## 第二部分小结

本部分介绍了CLI-Anything的解决方案：

1. **六个设计原则**：使用真实软件、操作原生格式、利用现有工具、失败清晰、可幂等、可内省。
2. **七阶段流水线**：从代码分析到PyPI发布，全自动化。
3. **关键架构模式**：后端包装器、渲染鸿沟处理、会话管理、ReplSkin、输出验证。
4. **分发架构**：PEP 420命名空间包、CLI-Hub注册表、SKILL.md技能发现。

在下一部分，我们将深入解析七阶段流水线的每一个阶段，包括代码级的实现细节。
# 第三部分：七阶段流水线与核心架构深度解析

> "The pattern is always the same: build the data → call the real software → verify the output."
> — HARNESS.md

---

## 第9章 阶段1深度解析：代码分析

### 9.1 代码分析的目标与方法

代码分析是七阶段流水线的起点。它的目标不是理解软件的每一行代码，而是找到将软件"暴露"给Agent的关键切入点。

AI Agent（如Claude Code）在执行代码分析时，采用的是一种"自顶向下"的方法：

**第一步：识别软件的领域和功能**。通过阅读README、文档和顶层代码结构，理解软件是做什么的。例如，识别出Shotcut是一个视频编辑器，它的核心功能是时间线编辑、滤镜应用和视频导出。

**第二步：找到后端引擎**。大多数GUI应用将展示层（GUI代码）和逻辑层（核心引擎）分离。CLI-Anything关注的是逻辑层。例如：
- Shotcut → MLT框架（多媒体处理引擎）
- GIMP → GEGL（Generic Graphics Library）+ Script-Fu
- Blender → bpy（Blender Python API）
- LibreOffice → UNO API + headless模式
- Inkscape → 直接SVG/XML操作 + Inkscape CLI

**第三步：映射GUI动作到API调用**。每个GUI操作都对应一个或多个函数调用。例如：
- Shotcut的"添加轨道"→ MLT XML中的`<track>`元素
- GIMP的"新建图层"→ Script-Fu的`(gimp-layer-new ...)`调用
- Blender的"添加立方体"→ `bpy.ops.mesh.primitive_cube_add()`

**第四步：识别数据模型**。软件的项目文件格式是什么？状态如何持久化？
- Shotcut/Kdenlive → MLT XML（`.mlt`文件）
- GIMP → XCF格式（但可以通过Script-Fu操作）
- Blender → `.blend`格式（但可以通过bpy操作）
- LibreOffice → ODF（Open Document Format，ZIP压缩的XML）
- Inkscape → SVG（直接XML操作）
- Draw.io → mxGraph XML

**第五步：找到现有CLI工具**。许多后端自带命令行工具：
- `melt`（MLT框架的CLI）
- `ffmpeg`（多媒体处理）
- `blender --background --python`（Blender的无头模式）
- `libreoffice --headless`（LibreOffice的无头模式）
- `inkscape --actions`（Inkscape的CLI模式）
- `sox`（音频处理）

### 9.2 软件分类学

通过分析CLI-Anything支持的50+软件，我们可以建立一个软件分类学：

**按后端类型分类**：

| 后端类型 | 特征 | 代表软件 | CLI策略 |
|----------|------|----------|---------|
| 自带CLI工具 | 软件提供命令行接口 | LibreOffice, Inkscape, Blender | 包装现有CLI |
| 脚本API | 软件提供脚本语言绑定 | Blender (bpy), GIMP (Script-Fu) | 生成脚本并执行 |
| REST API | 软件提供HTTP API | ComfyUI, Ollama, n8n, Zoom | 包装HTTP调用 |
| 文件格式操作 | 通过修改项目文件控制软件 | Shotcut (MLT XML), Draw.io (mxGraph XML) | 直接操作XML/JSON |
| 混合模式 | 结合多种后端 | FreeCAD (Python API + 文件导出) | 多后端协调 |

**按数据模型分类**：

| 数据模型 | 特征 | 代表软件 |
|----------|------|----------|
| XML项目文件 | 层次化、可扩展 | Shotcut, Kdenlive, Draw.io, MuseScore |
| JSON项目文件 | 轻量级、易解析 | Blender CLI, s&box, OBS Studio |
| 二进制项目文件 | 高效、不可直接编辑 | GIMP (XCF), Blender (.blend) |
| ODF文档 | ZIP压缩的XML | LibreOffice |
| 纯SVG/XML | 直接可编辑 | Inkscape |
| 数据库 | 结构化存储 | Zotero (SQLite), Obsidian (vault) |
| 无状态 | 每次操作独立 | Mermaid, Eth2-Quickstart |

### 9.3 代码分析的实际案例：Blender

让我们以Blender为例，展示代码分析的实际过程。

**Blender的架构**：
- 核心引擎：bpy（Blender Python API）
- 项目文件：`.blend`（二进制）或`.blend-cli.json`（CLI-Anything使用的JSON格式）
- 渲染引擎：Cycles、EEVEE、Workbench
- 无头模式：`blender --background --python script.py`

**GUI动作到API的映射**：
| GUI操作 | bpy API调用 |
|---------|-------------|
| 添加立方体 | `bpy.ops.mesh.primitive_cube_add()` |
| 添加材质 | `bpy.data.materials.new(name)` |
| 设置渲染引擎 | `bpy.context.scene.render.engine = 'CYCLES'` |
| 渲染图像 | `bpy.ops.render.render(write_still=True)` |
| 保存文件 | `bpy.ops.wm.save_as_mainfile(filepath=path)` |

**CLI-Anything的Blender CLI策略**：
1. 使用JSON格式维护场景描述（而不是直接操作`.blend`文件）
2. 根据JSON描述生成bpy脚本
3. 通过`blender --background --python script.py`执行脚本
4. Blender执行真实的渲染，输出真实的图像

### 9.4 代码分析的实际案例：Shotcut/Kdenlive

Shotcut和Kdenlive使用MLT（Media Lovin' Toolkit）作为后端。这是一个重要的案例，因为它展示了"文件格式操作"策略。

**MLT的架构**：
- MLT是一个多媒体处理框架
- 项目文件是MLT XML格式（`.mlt`）
- `melt`是MLT的命令行工具
- Shotcut和Kdenlive都是MLT的GUI前端

**CLI-Anything的策略**：
1. 直接操作MLT XML——添加/删除轨道、片段、滤镜
2. 使用`melt`命令行工具进行渲染
3. 使用`ffmpeg`进行辅助处理（如帧提取、格式转换）

**关键挑战**：
- 滤镜翻译：MLT的滤镜参数需要翻译为ffmpeg的滤镜语法（当使用ffmpeg作为渲染后端时）
- 时间码精度：非整数帧率（如29.97fps）导致累积舍入误差
- 输出验证：需要验证渲染输出的格式、时长、分辨率

---

## 第10章 阶段2深度解析：CLI架构设计

### 10.1 命令组设计原则

CLI的命令组应该匹配软件的逻辑领域。通过分析CLI-Anything的50+个CLI，我们可以总结出以下通用命令组模式：

**通用命令组**（几乎所有CLI都有）：
| 命令组 | 功能 | 示例 |
|--------|------|------|
| `project` | 项目管理 | `new`, `open`, `save`, `info`, `close` |
| `session` | 会话管理 | `status`, `undo`, `redo`, `history` |
| `export` | 导出/渲染 | `render`, `convert`, `publish` |
| `preview` | 预览 | `capture`, `latest`, `live start/push/status/stop` |

**领域特定命令组**（取决于软件类型）：

| 软件类型 | 命令组示例 |
|----------|-----------|
| 图像编辑 | `layer`, `filter`, `selection`, `brush`, `canvas` |
| 3D建模 | `object`, `material`, `light`, `camera`, `modifier`, `animation` |
| 视频编辑 | `track`, `clip`, `transition`, `filter`, `timeline` |
| 音频处理 | `track`, `effect`, `selection`, `export` |
| 办公文档 | `writer`, `calc`, `impress`, `draw` |
| 图表绘制 | `shape`, `connector`, `page`, `style` |

### 10.2 状态模型设计

CLI-Anything的状态模型需要回答以下问题：

**什么需要持久化？**
- 当前打开的项目
- 项目修改标记
- 撤销/重做历史
- 会话配置（如渲染预设）

**状态存储在哪里？**
- 内存中：用于REPL会话
- 文件中：用于子命令模式（JSON项目文件）

**状态如何序列化？**
- 项目状态：JSON文件（`.blend-cli.json`、`.json`等）
- 会话状态：内存中的Python对象
- 撤销历史：状态快照列表

### 10.3 输出格式设计

CLI-Anything的输出格式遵循严格的设计规范：

**人类可读模式**（默认）：
- 成功消息：`✓ Created scene: ProductShot`
- 错误消息：`✗ Error: File not found`
- 警告消息：`⚠ Warning: Unsaved changes`
- 信息消息：`● Processing 24 clips...`
- 表格输出：使用Unicode box-drawing字符

**机器可读模式**（`--json`）：
```json
{
  "name": "ProductShot",
  "resolution_x": 1920,
  "resolution_y": 1080,
  "engine": "CYCLES",
  "samples": 128,
  "fps": 24,
  "objects": 0,
  "materials": 0,
  "modified": true
}
```

JSON输出的设计原则：
- 包含操作的完整结果（不仅仅是"成功"或"失败"）
- 包含足够的上下文（如文件路径、大小、格式）
- 使用一致的字段命名约定
- 支持嵌套结构（如对象列表、材质属性）

### 10.4 参数设计模式

CLI-Anything的参数设计遵循以下模式：

**全局参数**（适用于所有命令）：
| 参数 | 功能 |
|------|------|
| `--json` | JSON输出模式 |
| `--project <path>` | 指定项目文件路径 |
| `--dry-run` | 试运行模式（不保存更改） |

**命令特定参数**：
- 短选项和长选项：`-n` / `--name`
- 默认值：为常用参数提供合理的默认值
- 选择约束：使用`type=click.Choice([...])`限制选项
- 向量参数：用逗号分隔的字符串表示向量（如`--location "1.0,2.0,3.0"`）

---

## 第11章 阶段3深度解析：实现

### 11.1 目录结构模板

每个CLI的目录结构遵循统一的模板：

```
<software>/
└── agent-harness/
    ├── <SOFTWARE>.md          # 架构SOP文档
    ├── setup.py               # PyPI包配置
    ├── cli_anything/          # 命名空间包（无__init__.py）
    │   └── <software>/        # 子包
    │       ├── __init__.py
    │       ├── __main__.py    # python -m cli_anything.<software>
    │       ├── README.md
    │       ├── <software>_cli.py  # 主CLI入口（Click + REPL）
    │       ├── core/          # 核心模块
    │       │   ├── __init__.py
    │       │   ├── project.py     # 项目管理
    │       │   ├── session.py     # 会话管理
    │       │   ├── export.py      # 导出/渲染
    │       │   └── ...            # 领域特定模块
    │       ├── utils/         # 工具模块
    │       │   ├── __init__.py
    │       │   ├── <software>_backend.py  # 后端包装器
    │       │   └── repl_skin.py  # 统一REPL皮肤
    │       ├── skills/        # 技能定义
    │       │   └── SKILL.md
    │       └── tests/         # 测试套件
    │           ├── TEST.md
    │           ├── test_core.py
    │           └── test_full_e2e.py
    └── examples/              # 示例
```

### 11.2 Click框架的使用模式

CLI-Anything使用Python的Click框架构建CLI。以下是关键的使用模式：

**主CLI组**：
```python
@click.group(invoke_without_command=True)
@click.option("--json", "use_json", is_flag=True, help="Output as JSON")
@click.option("--project", "project_path", type=str, default=None)
@click.pass_context
def cli(ctx, use_json, project_path):
    """CLI description."""
    global _json_output
    _json_output = use_json
    if ctx.invoked_subcommand is None:
        ctx.invoke(repl, project_path=None)
```

**子命令组**：
```python
@cli.group()
def scene():
    """Scene management commands."""
    pass

@scene.command("new")
@click.option("--name", "-n", default="untitled")
@handle_error
def scene_new(name):
    """Create a new scene."""
    # ...
```

**错误处理装饰器**：
```python
def handle_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            if _json_output:
                click.echo(json.dumps({"error": str(e), "type": "file_not_found"}))
            else:
                click.echo(f"Error: {e}", err=True)
            if not _repl_mode:
                sys.exit(1)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
```

### 11.3 后端包装器的实现

后端包装器是CLI-Anything中最关键的组件之一。它负责与真实软件交互。

**典型的后端包装器结构**：
```python
import shutil
import subprocess
import os

def find_<software>():
    """Find the software executable."""
    name = "<software>"
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"{name} not found. Install with:\n"
            f"  apt install {name}\n"
            f"  # or visit https://<software>.org"
        )
    return path

def <operation>(input_path, output_path, **kwargs):
    """Perform operation using real software."""
    exe = find_<software>()
    cmd = [exe, ...]  # Build command
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Verify output
    if not os.path.exists(output_path):
        raise RuntimeError(f"Operation failed: {result.stderr}")
    return {"output": output_path, "method": "<software>-backend"}
```

### 11.4 会话管理的实现

会话管理通过`Session`类实现：

```python
class Session:
    def __init__(self):
        self._project = None
        self._project_path = None
        self._modified = False
        self._history = []  # Undo stack
        self._redo_stack = []  # Redo stack

    def set_project(self, project, path=None):
        self._project = project
        self._project_path = path
        self._modified = False

    def get_project(self):
        return self._project

    def has_project(self):
        return self._project is not None

    def snapshot(self, description):
        """Save current state for undo."""
        import copy
        self._history.append({
            "state": copy.deepcopy(self._project),
            "description": description,
        })
        self._redo_stack.clear()
        self._modified = True

    def undo(self):
        """Undo last operation."""
        if not self._history:
            raise ValueError("Nothing to undo")
        import copy
        current = copy.deepcopy(self._project)
        self._redo_stack.append({"state": current, "description": "current"})
        entry = self._history.pop()
        self._project = entry["state"]
        return entry["description"]

    def redo(self):
        """Redo last undone operation."""
        if not self._redo_stack:
            raise ValueError("Nothing to redo")
        import copy
        current = copy.deepcopy(self._project)
        self._history.append({"state": current, "description": "current"})
        entry = self._redo_stack.pop()
        self._project = entry["state"]
        return entry["description"]

    def save_session(self, path=None):
        """Save project to file with locking."""
        save_path = path or self._project_path
        if not save_path:
            raise ValueError("No path specified")
        # File locking for concurrent safety
        import fcntl
        with open(save_path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            f.truncate()
            json.dump(self._project, f, indent=2, default=str)
            fcntl.flock(f, fcntl.LOCK_UN)
        self._modified = False
        return save_path
```

---

## 第12章 测试架构深度解析

### 12.1 四层测试体系

CLI-Anything的测试体系包含四个层次：

**第一层：单元测试（test_core.py）**
- 使用合成数据，不依赖外部软件
- 测试每个核心函数的输入输出
- 覆盖正常路径、边界条件和错误处理
- 运行速度快，适合CI

**第二层：E2E测试-原生（test_full_e2e.py）**
- 测试项目文件生成管道
- 验证生成的文件格式正确（有效XML、正确ZIP结构等）
- 不调用真实软件

**第三层：E2E测试-真实后端（test_full_e2e.py）**
- 调用真实软件进行渲染和导出
- 验证输出文件存在、大小>0、格式正确
- 打印产物路径供人工检查
- **不跳过**：如果软件未安装，测试失败而非跳过

**第四层：CLI子进程测试（test_full_e2e.py）**
- 通过`subprocess.run`测试安装后的CLI命令
- 模拟真实用户/Agent的使用方式
- 使用`_resolve_cli`辅助函数解析CLI路径

### 12.2 `_resolve_cli`辅助函数

这是一个关键的测试辅助函数，确保测试使用的是安装后的命令而不是源码导入：

```python
def _resolve_cli(name):
    """Resolve installed CLI command; falls back to python -m for dev."""
    import shutil
    force = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "").strip() == "1"
    path = shutil.which(name)
    if path:
        print(f"[_resolve_cli] Using installed command: {path}")
        return [path]
    if force:
        raise RuntimeError(f"{name} not found in PATH. Install with: pip install -e .")
    module = name.replace("cli-anything-", "cli_anything.") + "." + name.split("-")[-1] + "_cli"
    print(f"[_resolve_cli] Falling back to: {sys.executable} -m {module}")
    return [sys.executable, "-m", module]
```

### 12.3 测试结果统计

CLI-Anything的测试数据令人印象深刻：

```
总计: 2,280个测试全部通过
- 1,682个单元测试
- 579个E2E测试
- 19个Node.js测试
通过率: 100%
```

各软件的测试数量：
| 软件 | 测试数 | 单元测试 | E2E测试 |
|------|--------|----------|---------|
| Blender | 208 | 150 | 58 |
| Inkscape | 202 | 148 | 54 |
| Audacity | 161 | 107 | 54 |
| LibreOffice | 158 | 89 | 69 |
| Kdenlive | 155 | 111 | 44 |
| Shotcut | 154 | 110 | 44 |
| OBS Studio | 153 | 116 | 37 |
| Draw.io | 138 | 116 | 22 |
| GIMP | 107 | 64 | 43 |
| Ollama | 98 | 87 | 11 |
| s&box | 244 | 157+17 | 50+20 |

---

## 第13章 Preview协议架构

### 13.1 Preview协议的诞生背景

CLI-Anything的早期版本专注于最终渲染——CLI生成项目文件，调用真实软件渲染，输出最终结果。但在实际使用中，团队发现了一个问题：**Agent在构建过程中需要中间反馈**。

想象一个场景：Agent正在用Blender构建一个复杂的3D场景。它添加了对象、材质、灯光，然后执行渲染——等待30秒后发现灯光位置不对。如果CLI能在每一步操作后提供一个快速的预览，Agent就可以立即发现并修正问题，而不是等到最终渲染才暴露。

Preview协议就是为了解决这个问题而设计的。它定义了一个跨软件的中间预览标准。

### 13.2 三层持久化模型

Preview协议定义了三个层次的持久化对象：

**Bundle（包）**：不可变的预览快照。包含：
- `manifest.json`：机器可读的元数据
- `summary.json`：人类/Agent可读的摘要
- `artifacts/`：预览产物（图像、视频、JSON等）

**Session（会话）**：可变的当前头部。包含：
- `session.json`：当前bundle标识、查看器命令、轮询状态

**Trajectory（轨迹）**：只追加的历史记录。包含：
- `trajectory.json`：命令到预览的完整历史

### 13.3 Bundle目录结构

```
<bundle_dir>/
  manifest.json        # 机器合约
  summary.json         # 人类/Agent摘要
  artifacts/
    hero.png           # 主预览图像
    gallery_01.png     # 画廊图像1
    gallery_02.png     # 画廊图像2
    preview.mp4        # 预览视频
    pipeline_diff.json # 管道差异
```

### 13.4 标准命令表面

参与Preview协议的CLI应暴露以下命令：

| 命令 | 功能 | Agent用途 |
|------|------|-----------|
| `preview recipes` | 列出可用的预览配方 | 了解预览能力 |
| `preview capture` | 生成新的预览包 | 获取中间反馈 |
| `preview latest` | 显示最新的预览包 | 快速检查状态 |
| `preview diff` | 生成差异预览包 | 比较前后变化 |
| `preview live start` | 启动实时预览会话 | 持续监控变化 |
| `preview live push` | 推送新的预览包 | 手动刷新预览 |
| `preview live status` | 查询实时预览状态 | 低成本内省 |
| `preview live stop` | 停止实时预览 | 清理资源 |

### 13.5 Producer vs Consumer分离

Preview协议的一个关键设计决策是**生产者和消费者的分离**：

**Producer（生产者）**：`cli-anything-<software> preview ...`
- 与真实后端交互
- 创建bundle、session和trajectory
- 负责实际的渲染和导出

**Consumer（消费者）**：`cli-hub previews ...`
- 读取已发布的预览状态
- 提供inspect、html、watch、open命令
- 不渲染，不合成预览产物

这种分离确保了：每个CLI专注于自己的领域渲染，而查看/分析功能集中在CLI-Hub中实现。

---

## 第三部分小结

本部分深入解析了CLI-Anything的核心架构：

1. **代码分析**：通过识别后端引擎、映射GUI动作、理解数据模型来找到CLI切入点。
2. **CLI设计**：命令组设计、状态模型、输出格式、参数约定。
3. **实现模式**：Click框架的使用、后端包装器、会话管理。
4. **测试架构**：四层测试体系、`_resolve_cli`辅助函数、输出验证方法论。
5. **Preview协议**：三层持久化模型、标准命令表面、Producer/Consumer分离。

在下一部分，我们将深入代码细节，展示各个核心模块的具体实现。


---

## 第14章 命名空间包技术详解

### 14.1 PEP 420的原理

PEP 420（Implicit Namespace Packages）允许一个Python包跨越多个目录。这对于CLI-Anything至关重要——`cli-anything-gimp`和`cli-anything-blender`是两个独立的PyPI包，但它们都在`cli_anything`命名空间下。

传统的Python包要求每个包含`__init__.py`的目录都是一个独立的包。如果`cli_anything/`有`__init__.py`，那么安装两个CLI包时就会冲突——它们都想"拥有"`cli_anything`这个包。

命名空间包通过**不**在`cli_anything/`目录中放置`__init__.py`来解决这个问题。Python会自动将`cli_anything`识别为命名空间包，允许多个独立的包各自贡献子包。

### 14.2 setup.py的关键配置

每个CLI的setup.py必须正确配置命名空间包：

```python
from setuptools import setup, find_packages

setup(
    name="cli-anything-gimp",
    version="1.0.0",
    packages=find_packages(),  # 自动发现 cli_anything.gimp
    entry_points={
        "console_scripts": [
            "cli-anything-gimp=cli_anything.gimp.gimp_cli:main",
        ],
    },
    package_data={
        "cli_anything.gimp": ["skills/*.md"],  # 包含SKILL.md
    },
)
```

关键点：
- `find_packages()`会自动发现`cli_anything.gimp`但不会发现`cli_anything`（因为没有`__init__.py`）
- `entry_points`定义了安装后的命令名
- `package_data`包含了SKILL.md等非Python文件

### 14.3 验证命名空间包

安装后，可以通过以下方式验证命名空间包配置正确：

```python
# 验证可以同时导入多个CLI
import cli_anything.gimp
import cli_anything.blender
import cli_anything.libreoffice

# 验证它们是独立的包
print(cli_anything.gimp.__file__)      # /path/to/cli_anything/gimp/__init__.py
print(cli_anything.blender.__file__)   # /path/to/cli_anything/blender/__init__.py
```

### 14.4 常见的命名空间包陷阱

**陷阱一：意外创建了`cli_anything/__init__.py`**。这会破坏命名空间包，导致只能安装一个CLI。

**陷阱二：使用了错误的`find_packages()`参数**。如果指定了`include=["cli_anything"]`，可能会导致问题。

**陷阱三：在开发模式下安装多个CLI时**。使用`pip install -e .`安装多个CLI时，需要确保每个CLI的`cli_anything/`目录没有`__init__.py`。

# 第四部分：代码实现与工程模式

> "We build structured interfaces TO software, not replacements."
> — CLI-Anything Architecture Principles

---

## 第14章 Click框架深度应用

### 14.1 为什么选择Click

CLI-Anything选择Click作为CLI框架，而非argparse或typer，原因如下：

**Click的`invoke_without_command=True`**：这个特性允许主命令组在没有子命令时执行默认行为（进入REPL模式）。这是CLI-Anything的核心交互模式。

**Click的装饰器模式**：Click使用装饰器定义命令，代码结构清晰，文档字符串自动成为帮助文本。

**Click的类型系统**：Click提供了丰富的参数类型（Choice、Path、IntRange等），内置参数验证。

**Click的生态**：Click有丰富的扩展（click-plugins、click-contrib），社区活跃。

### 14.2 主CLI组的完整模式

以下是CLI-Anything中所有CLI的主命令组的通用模式：

```python
@click.group(invoke_without_command=True)
@click.option("--json", "use_json", is_flag=True, help="Output as JSON")
@click.option("--project", "project_path", type=str, default=None,
              help="Path to project file")
@click.option("--dry-run", "dry_run", is_flag=True, default=False,
              help="Run without saving changes")
@click.pass_context
def cli(ctx, use_json, project_path, dry_run):
    """<Software> CLI — Stateful editing from the command line.
    
    Run without a subcommand to enter interactive REPL mode.
    """
    global _json_output
    _json_output = use_json

    # Load project if specified
    if project_path:
        sess = get_session()
        if not sess.has_project():
            proj = scene_mod.open_scene(project_path)
            sess.set_project(proj, project_path)

    # Default to REPL when no subcommand
    if ctx.invoked_subcommand is None:
        ctx.invoke(repl, project_path=None)
```

这个模式的关键元素：
1. `invoke_without_command=True`：允许无子命令执行
2. 全局`--json`标志：控制输出格式
3. 全局`--project`标志：指定项目文件
4. `ctx.invoked_subcommand is None`检查：判断是否进入REPL
5. `@click.pass_context`：传递上下文

### 14.3 自动保存机制

CLI-Anything通过`@cli.result_callback()`实现自动保存：

```python
@cli.result_callback()
def auto_save_on_exit(result, use_json, project_path, dry_run, **kwargs):
    """Auto-save project after one-shot commands if state was modified."""
    if _repl_mode:
        return  # REPL handles its own save
    if dry_run:
        return  # Don't save in dry-run mode
    sess = get_session()
    if sess.has_project() and sess._modified and sess.project_path:
        try:
            sess.save_session()
        except Exception as e:
            click.echo(f"Warning: Auto-save failed: {e}", err=True)
```

### 14.4 向量参数处理

许多3D/图形操作需要向量参数（如位置、旋转、缩放）。CLI-Anything使用逗号分隔的字符串表示向量：

```python
@click.option("--location", "-l", default=None, help="Location x,y,z")
@click.option("--rotation", "-r", default=None, help="Rotation x,y,z (degrees)")
@click.option("--scale", "-s", default=None, help="Scale x,y,z")

def parse_vector(vector_str):
    """Parse comma-separated vector string."""
    if vector_str is None:
        return None
    return [float(x) for x in vector_str.split(",")]
```

### 14.5 动态参数处理

某些命令需要接受任意数量的键值对参数：

```python
@click.option("--param", "-p", multiple=True, help="Parameter: key=value")

def parse_params(params):
    """Parse multiple key=value parameters."""
    result = {}
    for p in params:
        if "=" not in p:
            raise ValueError(f"Invalid param format: '{p}'. Use key=value.")
        k, v = p.split("=", 1)
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        result[k] = v
    return result
```

---

## 第15章 后端包装器模式详解

### 15.1 五种后端模式

通过分析50+个CLI的后端实现，我们总结出五种主要的后端模式：

**模式一：子进程调用**
```python
# LibreOffice、Blender、Inkscape、melt等
def render(input_path, output_path):
    exe = find_software()
    subprocess.run([exe, "--headless", "--convert-to", "pdf", input_path],
                   check=True, capture_output=True)
    return {"output": output_path}
```

**模式二：脚本生成+执行**
```python
# Blender bpy脚本
def render_scene(scene_json, output_path):
    script = generate_bpy_script(scene_json, output_path)
    script_path = write_temp_script(script)
    subprocess.run(["blender", "--background", "--python", script_path],
                   check=True, capture_output=True)
    return {"output": output_path, "script_path": script_path}
```

**模式三：REST API调用**
```python
# ComfyUI、Ollama、n8n、Zoom等
def generate_image(prompt, workflow):
    response = requests.post(f"{BASE_URL}/prompt", json={"prompt": prompt})
    task_id = response.json()["prompt_id"]
    # Poll for completion
    while True:
        status = requests.get(f"{BASE_URL}/history/{task_id}")
        if status.json().get(task_id, {}).get("completed"):
            break
        time.sleep(1)
    return {"output": get_output_path(task_id)}
```

**模式四：直接文件操作**
```python
# Shotcut MLT XML、Draw.io mxGraph XML
def add_track(mlt_xml, track_type="video"):
    tree = ET.parse(mlt_xml)
    root = tree.getroot()
    tractor = root.find(".//tractor")
    track = ET.SubElement(tractor, "track", {"producer": f"track{track_id}"})
    tree.write(mlt_xml)
    return {"track_id": track_id}
```

**模式五：混合模式**
```python
# FreeCAD：Python API + 文件导出
def export_step(document_path, output_path):
    # Generate FreeCAD macro
    macro = generate_export_macro(document_path, output_path)
    macro_path = write_temp_macro(macro)
    subprocess.run(["freecad", "--console", macro_path],
                   check=True, capture_output=True)
    return {"output": output_path}
```

### 15.2 软件发现机制

每个后端包装器都需要找到目标软件的可执行文件。CLI-Anything使用统一的发现机制：

```python
def find_<software>():
    """Find the <software> executable."""
    # Common executable names
    candidates = ["<software>", "<software>4", "<software>-cli"]
    
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    
    # Platform-specific paths
    if sys.platform == "darwin":
        mac_paths = ["/Applications/<software>.app/Contents/MacOS/<software>"]
        for p in mac_paths:
            if os.path.exists(p):
                return p
    
    if sys.platform == "win32":
        win_paths = [
            os.path.expandvars(r"C:\Program Files\<software>\<software>.exe"),
            os.path.expandvars(r"C:\Program Files (x86)\<software>\<software>.exe"),
        ]
        for p in win_paths:
            if os.path.exists(p):
                return p
    
    raise RuntimeError(
        f"<software> not found. Install with:\n"
        f"  Linux: apt install <software>\n"
        f"  macOS: brew install <software>\n"
        f"  Windows: https://<software>.org/download\n"
        f"Then ensure '<software>' is on PATH."
    )
```

### 15.3 输出验证模式

后端包装器的一个关键职责是验证输出：

```python
def verify_output(output_path, expected_format):
    """Verify output file is valid."""
    # Check existence
    if not os.path.exists(output_path):
        raise RuntimeError(f"Output file not created: {output_path}")
    
    # Check size
    size = os.path.getsize(output_path)
    if size == 0:
        raise RuntimeError(f"Output file is empty: {output_path}")
    
    # Check format
    with open(output_path, "rb") as f:
        header = f.read(16)
    
    format_checks = {
        "pdf": lambda h: h[:5] == b"%PDF-",
        "zip": lambda h: h[:2] == b"PK",
        "png": lambda h: h[:8] == b"\x89PNG\r\n\x1a\n",
        "jpeg": lambda h: h[:2] == b"\xff\xd8",
        "mp4": lambda h: b"ftyp" in h[:12],
        "wav": lambda h: h[:4] == b"RIFF",
    }
    
    check = format_checks.get(expected_format)
    if check and not check(header):
        raise RuntimeError(
            f"Output format mismatch: expected {expected_format}, "
            f"got header {header[:8].hex()}"
        )
    
    return {"path": output_path, "size": size, "format": expected_format}
```

---

## 第16章 ReplSkin实现详解

### 16.1 ANSI颜色系统

ReplSkin使用ANSI 256色系统，为每个软件分配独特的强调色：

```python
_ACCENT_COLORS = {
    "gimp":        "\033[38;5;214m",   # 暖橙色
    "blender":     "\033[38;5;208m",   # 深橙色
    "inkscape":    "\033[38;5;39m",    # 亮蓝色
    "audacity":    "\033[38;5;33m",    # 海军蓝
    "libreoffice": "\033[38;5;40m",    # 绿色
    "obs_studio":  "\033[38;5;55m",    # 紫色
    "kdenlive":    "\033[38;5;69m",    # 石板蓝
    "shotcut":     "\033[38;5;35m",    # 青绿色
}
_DEFAULT_ACCENT = "\033[38;5;75m"      # 天蓝色
```

### 16.2 横幅设计

ReplSkin的启动横幅使用Unicode box-drawing字符：

```
╭────────────────────────────────────────────────────────────────────────╮
║ ◆  cli-anything · Blender                                            ║
║    v1.0.0                                                            ║
║ ◇ Install:  npx skills add HKUDS/CLI-Anything --skill cli-anything-blender -g -y ║
║ ◇ Global skill:  ~/.agents/skills/cli-anything-blender/SKILL.md      ║
║                                                                        ║
║    Type help for commands, quit to exit                               ║
╰────────────────────────────────────────────────────────────────────────╯
```

### 16.3 上下文感知的提示符

REPL提示符显示当前项目状态：

```
◆ blender [ProductShot*] ❯ 
```

- `◆`：品牌图标
- `blender`：软件名称（使用强调色）
- `[ProductShot*]`：当前项目名称（`*`表示有未保存的更改）
- `❯`：输入指示符

### 16.4 消息格式

统一的消息格式确保一致的用户体验：

```python
def success(self, message):
    """✓ 绿色成功消息"""
    print(f"  {self._c(_GREEN + _BOLD, '✓')} {self._c(_GREEN, message)}")

def error(self, message):
    """✗ 红色错误消息"""
    print(f"  {self._c(_RED + _BOLD, '✗')} {self._c(_RED, message)}", file=sys.stderr)

def warning(self, message):
    """⚠ 黄色警告消息"""
    print(f"  {self._c(_YELLOW + _BOLD, '⚠')} {self._c(_YELLOW, message)}")

def info(self, message):
    """● 蓝色信息消息"""
    print(f"  {self._c(_BLUE, '●')} {self._c(_LIGHT_GRAY, message)}")
```

---

## 第17章 会话管理与状态持久化

### 17.1 Session类的完整实现

Session类是CLI-Anything状态管理的核心：

```python
import json
import copy
import fcntl
import sys
from typing import Any, Optional

class Session:
    """Manages project state with undo/redo and persistence."""
    
    def __init__(self):
        self._project: Optional[dict] = None
        self._project_path: Optional[str] = None
        self._modified: bool = False
        self._history: list = []  # Undo stack
        self._redo_stack: list = []  # Redo stack
    
    # --- Project Management ---
    
    def set_project(self, project: dict, path: Optional[str] = None):
        """Set the current project."""
        self._project = project
        self._project_path = path
        self._modified = False
        self._history.clear()
        self._redo_stack.clear()
    
    def get_project(self) -> dict:
        """Get the current project."""
        if self._project is None:
            raise ValueError("No project loaded. Use 'project new' or 'project open'.")
        return self._project
    
    def has_project(self) -> bool:
        """Check if a project is loaded."""
        return self._project is not None
    
    # --- Undo/Redo ---
    
    def snapshot(self, description: str):
        """Save current state for undo."""
        if self._project is None:
            return
        self._history.append({
            "state": copy.deepcopy(self._project),
            "description": description,
        })
        self._redo_stack.clear()
        self._modified = True
    
    def undo(self) -> str:
        """Undo last operation. Returns description of undone operation."""
        if not self._history:
            raise ValueError("Nothing to undo")
        # Save current state to redo stack
        self._redo_stack.append({
            "state": copy.deepcopy(self._project),
            "description": "current",
        })
        # Restore previous state
        entry = self._history.pop()
        self._project = entry["state"]
        return entry["description"]
    
    def redo(self) -> str:
        """Redo last undone operation. Returns description of redone operation."""
        if not self._redo_stack:
            raise ValueError("Nothing to redo")
        # Save current state to undo stack
        self._history.append({
            "state": copy.deepcopy(self._project),
            "description": "current",
        })
        # Restore redo state
        entry = self._redo_stack.pop()
        self._project = entry["state"]
        return entry["description"]
    
    def list_history(self) -> list:
        """List undo history."""
        return [{"index": i, "description": e["description"]}
                for i, e in enumerate(self._history)]
    
    # --- Persistence ---
    
    def save_session(self, path: Optional[str] = None) -> str:
        """Save project to file with exclusive locking."""
        save_path = path or self._project_path
        if not save_path:
            raise ValueError("No path specified for save")
        
        # Exclusive file locking to prevent concurrent corruption
        with open(save_path, "w") as f:
            if sys.platform != "win32":
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(self._project, f, indent=2, default=str)
            finally:
                if sys.platform != "win32":
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        self._modified = False
        self._project_path = save_path
        return save_path
    
    def load_session(self, path: str) -> dict:
        """Load project from file."""
        with open(path, "r") as f:
            project = json.load(f)
        self.set_project(project, path)
        return project
    
    # --- Status ---
    
    def status(self) -> dict:
        """Get session status."""
        return {
            "has_project": self.has_project(),
            "project_path": self._project_path,
            "modified": self._modified,
            "undo_depth": len(self._history),
            "redo_depth": len(self._redo_stack),
        }
```

### 17.2 并发安全

文件锁定机制防止并发写入损坏：

```python
# Linux/macOS: fcntl.flock
with open(path, "r+") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他锁
    try:
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2)
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁

# Windows: msvcrt.locking
import msvcrt
with open(path, "r+b") as f:
    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2)
    finally:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
```

---

## 第18章 测试实现详解

### 18.1 单元测试模式

单元测试遵循"Arrange-Act-Assert"模式：

```python
class TestScene:
    def test_create_scene_default(self):
        """Test creating a scene with default parameters."""
        # Arrange & Act
        proj = create_scene(name="TestScene")
        
        # Assert
        assert proj["scene"]["name"] == "TestScene"
        assert proj["scene"]["resolution_x"] == 1920
        assert proj["scene"]["resolution_y"] == 1080
        assert proj["scene"]["engine"] == "CYCLES"
    
    def test_create_scene_custom(self):
        """Test creating a scene with custom parameters."""
        proj = create_scene(
            name="Custom",
            resolution_x=1280,
            resolution_y=720,
            engine="EEVEE",
            samples=64,
            fps=30,
        )
        assert proj["scene"]["resolution_x"] == 1280
        assert proj["scene"]["engine"] == "EEVEE"
        assert proj["scene"]["fps"] == 30
    
    def test_create_scene_invalid_engine(self):
        """Test that invalid engine raises error."""
        with pytest.raises(ValueError, match="Unknown engine"):
            create_scene(engine="INVALID")
```

### 18.2 E2E测试-真实后端模式

```python
class TestBlenderRender:
    def test_render_cube_scene(self, tmp_dir):
        """Test rendering a scene with a cube via real Blender."""
        # Create scene
        proj = create_scene(name="RenderTest")
        add_object(proj, mesh_type="cube", name="TestCube")
        
        # Render via real Blender
        output_path = os.path.join(tmp_dir, "render.png")
        result = render_scene(proj, output_path)
        
        # Verify output
        assert os.path.exists(result["output"])
        assert result["file_size"] > 1000  # Not suspiciously small
        
        # Verify image format
        with open(result["output"], "rb") as f:
            header = f.read(8)
        assert header[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
        
        # Print artifact path for manual inspection
        print(f"\n  Render: {result['output']} ({result['file_size']:,} bytes)")
```

### 18.3 CLI子进程测试模式

```python
class TestCLISubprocess:
    CLI_BASE = _resolve_cli("cli-anything-blender")
    
    def _run(self, args, check=True):
        return subprocess.run(
            self.CLI_BASE + args,
            capture_output=True, text=True,
            check=check,
        )
    
    def test_help(self):
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "Blender CLI" in result.stdout
    
    def test_json_output(self, tmp_dir):
        out = os.path.join(tmp_dir, "test.json")
        result = self._run(["--json", "scene", "new", "-o", out])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "scene" in data
    
    def test_full_workflow(self, tmp_dir):
        """Test complete workflow: create, add objects, render."""
        proj = os.path.join(tmp_dir, "project.json")
        render_out = os.path.join(tmp_dir, "render.png")
        
        # Create project
        self._run(["scene", "new", "-o", proj])
        
        # Add object
        self._run(["--project", proj, "object", "add", "cube"])
        
        # Render
        self._run(["--project", proj, "render", "execute", render_out])
        
        # Verify
        assert os.path.exists(render_out)
        assert os.path.getsize(render_out) > 1000
```

---

## 第19章 SKILL.md生成与Agent发现

### 19.1 SKILL.md的结构

SKILL.md是CLI-Anything的Agent发现机制。每个CLI都生成一个SKILL.md文件：

```markdown
---
name: "cli-anything-blender"
description: "3D modeling, animation, and rendering via Blender"
---

# cli-anything-blender

## Installation

```bash
pip install git+https://github.com/HKUDS/CLI-Anything.git#subdirectory=blender/agent-harness
```

## Prerequisites

- Blender >= 4.2 (blender.org)
- Python 3.10+

## Command Groups

| Group | Description |
|-------|-------------|
| scene | Scene management (new, open, save, info) |
| object | 3D object management (add, remove, transform) |
| material | Material management (create, assign) |
| render | Rendering (settings, execute) |
| preview | Preview bundles (capture, live) |
| session | Session management (undo, redo) |

## Usage Examples

```bash
# Create a new scene
cli-anything-blender scene new --name "ProductShot"

# Add a cube
cli-anything-blender object add cube --location 0,0,1

# Render
cli-anything-blender render execute output.png

# JSON mode for agents
cli-anything-blender --json scene info
```

## Agent Guidance

- Use `--json` flag for machine-readable output
- Use `--help` on any command for detailed usage
- Enter REPL mode by running without arguments
- Use `session undo` to revert unwanted changes
```

### 19.2 自动化SKILL.md生成

CLI-Anything提供了`skill_generator.py`来自动生成SKILL.md：

```python
def generate_skill_md(software, cli_module, setup_info):
    """Generate SKILL.md from CLI metadata."""
    # Extract command groups from Click decorators
    commands = extract_commands(cli_module)
    
    # Generate YAML frontmatter
    frontmatter = {
        "name": f"cli-anything-{software}",
        "description": setup_info["description"],
    }
    
    # Generate markdown body
    body = generate_markdown(
        software=software,
        commands=commands,
        install_cmd=setup_info["install_cmd"],
        prerequisites=setup_info["requires"],
    )
    
    return f"---\n{yaml.dump(frontmatter)}---\n\n{body}"
```

### 19.3 SKILL.md的双重位置

SKILL.md文件存在于两个位置：

1. **规范位置**：`skills/cli-anything-<software>/SKILL.md`（仓库根目录）
2. **兼容位置**：`cli_anything/<software>/skills/SKILL.md`（包内）

ReplSkin在启动时显示SKILL.md的路径，Agent可以读取这个文件来了解CLI的完整功能。

---

## 第四部分小结

本部分深入解析了CLI-Anything的代码实现：

1. **Click框架应用**：主命令组模式、自动保存、向量参数、动态参数。
2. **后端包装器**：五种后端模式、软件发现、输出验证。
3. **ReplSkin**：ANSI颜色、横幅设计、提示符、消息格式。
4. **会话管理**：Session类、撤销/重做、文件锁定。
5. **测试实现**：单元测试、E2E测试、子进程测试的具体模式。
6. **SKILL.md**：Agent发现机制、自动生成、双重位置。

在下一部分，我们将通过50+个软件的领域案例，展示CLI-Anything在不同领域的实际应用。
# 第五部分：50+软件的领域实践

> "From creative workflows to production tools — CLI-Anything has been tested across 18+ major applications spanning image editing, 3D modeling, vector graphics, audio production, office suites, live streaming, video editing, and more."

---

## 第20章 创意与媒体领域

### 20.1 GIMP：图像编辑的CLI化

**软件概述**：GIMP（GNU Image Manipulation Program）是最知名的开源图像编辑软件，功能对标Adobe Photoshop。

**后端策略**：GIMP支持Script-Fu（基于Scheme的脚本语言）和Python-Fu两种脚本接口。CLI-Anything选择Python-Fu + PIL（Pillow）的混合策略。

**核心命令组**：
| 命令组 | 功能 | 示例 |
|--------|------|------|
| `project` | 项目管理 | `new --width 1920 --height 1080` |
| `layer` | 图层管理 | `add -n "Background" --type solid` |
| `filter` | 滤镜应用 | `apply-gaussian-blur --radius 5.0` |
| `selection` | 选区操作 | `rect --x 100 --y 100 --w 500 --h 300` |
| `export` | 导出 | `render output.png --format png` |

**技术挑战与解决方案**：
- **Script-Fu注入**：GIMP的Script-Fu存在路径注入漏洞。CLI-Anything在2026-03-14修复了这个问题，通过转义特殊字符防止注入。
- **批量操作**：GIMP的批处理模式（`gimp -i -b`）每次调用都有启动开销。CLI-Anything通过将多个操作合并到一个Script-Fu脚本中来减少调用次数。

**测试覆盖**：107个测试（64个单元测试 + 43个E2E测试）。

### 20.2 Blender：3D建模与渲染的CLI化

**软件概述**：Blender是功能最全面的开源3D创作套件，涵盖建模、材质、灯光、动画、渲染、合成、视频编辑等功能。

**后端策略**：使用bpy（Blender Python API）+ `blender --background --python`。

**核心命令组**：
| 命令组 | 功能 | 命令数 |
|--------|------|--------|
| `scene` | 场景管理 | 6 |
| `object` | 3D对象管理 | 8 |
| `material` | 材质管理 | 5 |
| `modifier` | 修改器管理 | 6 |
| `camera` | 摄像机管理 | 4 |
| `light` | 灯光管理 | 3 |
| `animation` | 动画管理 | 5 |
| `render` | 渲染 | 6 |
| `preview` | 预览 | 8 |
| `session` | 会话 | 4 |

**关键创新**：
- **JSON场景描述**：CLI-Anything不直接操作`.blend`文件，而是使用JSON格式描述场景，然后生成bpy脚本由Blender执行。
- **Preview协议**：Blender是Preview协议的3D/CAD试点之一。支持`quick`（单帧低分辨率）和`turntable`（旋转预览）配方。
- **Live Preview**：支持实时预览会话，Agent可以在每步操作后看到渲染结果。

**Demo亮点**：Agent使用Blender CLI构建了一个完整的"轨道中继无人机"（Orbital Relay Drone）3D模型，通过Preview + Live Preview + Trajectory实现了逐步构建的可视化反馈。

**测试覆盖**：208个测试（150个单元测试 + 58个E2E测试）。

### 20.3 Inkscape：矢量图形的CLI化

**软件概述**：Inkscape是开源矢量图形编辑器，功能对标Adobe Illustrator。

**后端策略**：直接SVG/XML操作 + `inkscape --actions` CLI导出。

**核心能力**：
- SVG元素的创建、修改、删除
- 路径操作（布尔运算、简化、偏移）
- 渐变和样式管理
- 图层管理
- 导出为PNG、PDF等格式

**技术亮点**：CLI-Anything直接操作SVG的XML结构，无需启动Inkscape即可完成大部分操作。只有在导出时才调用`inkscape --export-filename`。

**测试覆盖**：202个测试（148个单元测试 + 54个E2E测试）。

### 20.4 Krita：数字绘画的CLI化

**软件概述**：Krita是专业的开源数字绘画软件，专注于概念艺术、插画和纹理创作。

**后端策略**：Krita的CLI导出管道。

**核心能力**：画布管理、图层操作、滤镜应用、导出。

**测试覆盖**：通过E2E测试验证Krita CLI导出管道。

### 20.5 Audacity：音频处理的CLI化

**软件概述**：Audacity是最流行的开源音频编辑软件。

**后端策略**：使用sox（Sound eXchange）作为音频处理后端。

**核心命令组**：
| 命令组 | 功能 |
|--------|------|
| `project` | 项目管理 |
| `track` | 音轨管理 |
| `effect` | 音频效果（均衡、压缩、混响等） |
| `selection` | 选区操作 |
| `export` | 导出（WAV、MP3、FLAC等） |

**技术亮点**：Audacity CLI不使用Audacity本身的API（因为它没有稳定的脚本接口），而是使用sox作为后端。sox是一个强大的命令行音频处理工具，支持100+种音频格式和数百种效果。

**测试覆盖**：161个测试（107个单元测试 + 54个E2E测试）。

### 20.6 MuseScore：音乐记谱的CLI化

**软件概述**：MuseScore是开源的音乐记谱软件。

**后端策略**：使用mscore CLI + MSCX/MusicXML格式操作。

**核心能力**：乐谱创建、移调、乐器管理、导出（PDF、音频、MIDI）。

**测试覆盖**：56个测试。

---

## 第21章 视频编辑领域

### 21.1 Shotcut：视频编辑的CLI化

**软件概述**：Shotcut是开源的视频编辑软件，使用MLT框架作为后端。

**后端策略**：直接MLT XML操作 + `melt`渲染。

**核心命令组**：
| 命令组 | 功能 |
|--------|------|
| `project` | 项目管理 |
| `track` | 轨道管理 |
| `clip` | 片段管理 |
| `filter` | 滤镜应用 |
| `transition` | 转场效果 |
| `export` | 渲染导出 |
| `preview` | 预览包 |

**关键挑战**：
- **渲染鸿沟**：MLT的滤镜在渲染时应用。CLI-Anything需要确保`melt`正确读取和应用所有滤镜。
- **滤镜翻译**：当使用ffmpeg作为渲染后端时，MLT滤镜需要翻译为ffmpeg的`-filter_complex`语法。
- **时间码精度**：29.97fps等非整数帧率导致累积舍入误差。使用`round()`而非`int()`，并在测试中接受±1帧的容差。

**测试覆盖**：154个测试（110个单元测试 + 44个E2E测试）。

### 21.2 Kdenlive：KDE视频编辑器的CLI化

**软件概述**：Kdenlive是KDE项目的视频编辑软件，同样使用MLT框架。

**后端策略**：与Shotcut类似——MLT XML操作 + `melt`渲染。

**与Shotcut的对比**：Kdenlive和Shotcut使用相同的后端（MLT），但项目文件格式略有不同。CLI-Anything的Kdenlive CLI处理Kdenlive特有的项目结构（如`kdenlive`标签、`playlist`结构等）。

**测试覆盖**：155个测试（111个单元测试 + 44个E2E测试）。

### 21.3 Openscreen：屏幕录制编辑的CLI化

**软件概述**：Openscreen是屏幕录制编辑工具。

**后端策略**：ffmpeg后端。

**核心能力**：缩放、速度坡道、裁剪、注释、背景、导出。

**测试覆盖**：101个测试（78个单元测试 + 23个E2E测试）。

### 21.4 VideoCaptioner：AI视频字幕的CLI化

**软件概述**：VideoCaptioner是AI驱动的视频字幕工具。

**后端策略**：videocaptioner CLI（PyPI包）。

**核心能力**：语音转录、字幕优化/翻译、字幕烧录。

**测试覆盖**：26个测试。

---

## 第22章 办公与生产力领域

### 22.1 LibreOffice：办公套件的CLI化

**软件概述**：LibreOffice是最完整的开源办公套件，包含Writer（文字处理）、Calc（电子表格）、Impress（演示文稿）、Draw（绘图）等组件。

**后端策略**：ODF文件生成 + `libreoffice --headless`转换。

**核心命令组**：
| 命令组 | 功能 |
|--------|------|
| `document` | 文档管理 |
| `writer` | Writer操作（标题、段落、表格、图片） |
| `calc` | Calc操作（工作表、单元格、公式） |
| `impress` | Impress操作（幻灯片、布局） |
| `export` | 导出（PDF、DOCX、XLSX、PPTX） |

**关键架构决策**：
- **生成ODF而非直接操作**：CLI-Anything不直接调用LibreOffice的UNO API，而是生成有效的ODF文件，然后通过`libreoffice --headless`转换。
- **ODF是ZIP**：ODF文件本质上是ZIP压缩的XML。CLI-Anything使用Python的`zipfile`模块创建ODF文件。

**典型工作流**：
```bash
# 创建Writer文档
cli-anything-libreoffice document new -o report.json --type writer

# 添加内容
cli-anything-libreoffice --project report.json writer add-heading -t "Q1 Report"
cli-anything-libreoffice --project report.json writer add-table --rows 4 --cols 3

# 导出为PDF（通过真实LibreOffice）
cli-anything-libreoffice --project report.json export render output.pdf -p pdf --overwrite

# 验证输出
# 输出的PDF是LibreOffice渲染的真实PDF，不是Python生成的近似
```

**测试覆盖**：158个测试（89个单元测试 + 69个E2E测试）。

### 22.2 Obsidian：知识管理的CLI化

**软件概述**：Obsidian是流行的知识管理和笔记应用。

**后端策略**：Obsidian Local REST API。

**核心能力**：笔记管理、搜索、命令执行。

**测试覆盖**：48个单元测试 + 7个E2E测试。

### 22.3 Zotero：文献管理的CLI化

**软件概述**：Zotero是学术文献管理工具。

**后端策略**：本地SQLite + connector + Local API。

**核心能力**：搜索、导入、PDF管理、BibTeX、笔记等70+命令。

**特殊之处**：Zotero CLI是独立仓库托管的（standalone repo），通过registry.json注册到CLI-Hub。

### 22.4 Mubu：大纲笔记的CLI化

**软件概述**：Mubu（幕布）是大纲笔记和知识管理工具。

**后端策略**：本地Mubu数据 + 同步日志。

**测试覆盖**：96个测试。

---

## 第23章 3D/CAD与科学计算领域

### 23.1 FreeCAD：参数化3D CAD的CLI化

**软件概述**：FreeCAD是开源的参数化3D CAD建模器，广泛用于机械工程、产品设计和建筑。

**后端策略**：FreeCAD Python API + `freecad --console`宏执行。

**核心命令组**（258个命令，17个组）：
| 命令组 | 功能 |
|--------|------|
| `document` | 文档管理 |
| `parts` | 零件建模 |
| `sketcher` | 2D草图 |
| `body` | PartDesign主体 |
| `assembly` | 装配 |
| `mesh` | 网格操作 |
| `techdraw` | 工程图 |
| `draft` | 草图工具 |
| `fem` | 有限元分析 |
| `cam` | CAM加工 |
| `materials` | 材料管理 |
| `export` | 导出（STEP、STL、IGES等） |
| `preview` | 预览包 |

**关键创新**：
- **258个命令**：CLI-Anything中命令数最多的CLI之一。
- **Preview协议试点**：FreeCAD是Preview协议的3D/CAD试点。支持`quick`（单视图等轴测）和`quad`（四视图：等轴测、前、顶、右）配方。
- **宏生成**：CLI-Anything生成FreeCAD宏（Python脚本），通过`freecad --console`执行。

**Demo亮点**：Agent使用FreeCAD CLI构建了一个"好奇号"火星车模型（Curiosity Rover），通过Preview + Live Preview + Trajectory实现了逐步构建的可视化反馈。

**测试覆盖**：FreeCAD CLI包含完整的测试套件。

### 23.2 QGIS：地理信息系统的CLI化

**软件概述**：QGIS是开源的地理信息系统（GIS）软件。

**后端策略**：PyQGIS + `qgis_process`。

**核心能力**：项目创建、图层管理、处理算法、布局导出。

**测试覆盖**：22个测试。

### 23.3 CloudCompare：点云处理的CLI化

**软件概述**：CloudCompare是3D点云和网格处理软件。

**后端策略**：CloudCompare CLI。

**核心能力**：加载/保存、颜色操作、法线估计、Delaunay网格化、噪声过滤、ICP配准。

**测试覆盖**：88个测试（49个单元测试 + 39个E2E测试）。

### 23.4 Uni-Mol Tools：分子性质预测的CLI化

**软件概述**：Uni-Mol Tools是分子性质预测工具，用于药物发现。

**后端策略**：Uni-Mol分子机器学习后端。

**核心能力**：训练和预测5种任务类型（分类、回归、多类、多标签）。

**测试覆盖**：67个测试。

---

## 第24章 游戏开发领域

### 24.1 Godot Engine：游戏引擎的CLI化

**软件概述**：Godot是流行的开源游戏引擎。

**后端策略**：Godot 4.x无头子进程。

**核心能力**：项目管理、场景编辑、GDScript执行、导出。

**测试覆盖**：24个测试。

### 24.2 s&box：Source 2游戏引擎的CLI化

**软件概述**：s&box是基于Source 2引擎的游戏开发平台。

**后端策略**：直接文件I/O操作.scene/.prefab/.vmat/.vsnd JSON文件。

**核心能力**：场景、预制体、材质、声音、代码生成、资产图、项目验证。

**测试覆盖**：244个测试（157个单元测试 + 17个编排器测试 + 50个E2E测试 + 20个退出码测试）——CLI-Anything中测试数最多的CLI之一。

### 24.3 Slay the Spire II：游戏自动化的CLI化

**软件概述**：Slay the Spire II是一款流行的卡牌roguelike游戏。

**后端策略**：通过STS2_Bridge模组暴露的HTTP API。

**核心能力**：读取游戏状态、选择卡牌、选择路径、实时策略决策。

**特殊之处**：这是CLI-Anything中唯一的"游戏自动化"CLI。它不是为游戏开发者设计的，而是为游戏玩家/研究者设计的——让AI Agent可以玩Slay the Spire II。

**Demo亮点**：Agent使用CLI完整地进行了一场Slay the Spire II游戏。

---

## 第25章 开发与运维领域

### 25.1 LLDB：原生调试的CLI化

**软件概述**：LLDB是LLVM项目的调试器。

**后端策略**：LLDB Python API + DAP（Debug Adapter Protocol）服务器。

**核心能力**：断点管理、线程检查、变量查看、表达式求值。

**特殊之处**：LLDB CLI不仅是一个CLI接口，还包含一个stdio Debug Adapter Protocol服务器，可以与VS Code等IDE集成。

**测试覆盖**：27个测试。

### 25.2 RenderDoc：GPU帧捕获分析的CLI化

**软件概述**：RenderDoc是GPU帧捕获和分析工具。

**后端策略**：RenderDoc Python bindings。

**核心能力**：帧捕获分析、管线状态检查、着色器导出、纹理检查、绘制调用浏览。

**测试覆盖**：59个测试（45个单元测试 + 14个E2E测试）。

### 25.3 Nsight Graphics：GPU调试的CLI化

**软件概述**：NVIDIA Nsight Graphics是GPU调试和性能分析工具。

**后端策略**：ngfx/ngfx-capture编排 + GPU Trace摘要 + replay分析。

**核心能力**：Graphics/OpenGL捕获、GPU Trace摘要、C++捕获生成、ngfx-replay分析。

**测试覆盖**：51个测试。

### 25.4 Unreal Insights：性能分析的CLI化

**软件概述**：Unreal Insights是Unreal Engine的性能分析工具。

**后端策略**：后台trace会话 + 引擎匹配的UnrealInsights构建 + 无头导出。

**核心能力**：捕获会话控制、Timing Insights无头导出。

**测试覆盖**：50个测试。

### 25.5 n8n：工作流自动化的CLI化

**软件概述**：n8n是开源的工作流自动化平台。

**后端策略**：n8n REST API v1.1.1。

**核心能力**：工作流管理、执行、凭证管理、节点操作等55+命令。

**测试覆盖**：包含55+命令的测试。

### 25.6 PM2：Node.js进程管理的CLI化

**软件概述**：PM2是Node.js进程管理器。

**后端策略**：PM2 CLI包装。

**核心能力**：进程列表、启动、停止、重启、日志、指标。

---

## 第26章 网络与通信领域

### 26.1 Zoom：视频会议的CLI化

**软件概述**：Zoom是视频会议平台。

**后端策略**：Zoom REST API（OAuth2）。

**核心能力**：会议管理、参与者管理、录制检索。

**测试覆盖**：22个测试。

### 26.2 AdGuard Home：网络广告拦截的CLI化

**软件概述**：AdGuard Home是网络级广告拦截解决方案。

**后端策略**：AdGuard Home REST API。

**核心能力**：DNS管理、客户端管理、过滤规则、DHCP、日志查询。

**测试覆盖**：36个测试（24个单元测试 + 12个E2E测试）。

### 26.3 Browser：浏览器自动化的CLI化

**软件概述**：CLI-Anything的Browser CLI不是传统浏览器，而是通过DOMShell MCP服务器实现的浏览器自动化。

**后端策略**：DOMShell MCP + 可访问性树。

**核心能力**：页面导航、DOM操作、表单填写、网络捕获。

**安全特性**：包含URL验证和DOM清理，防止XSS和注入攻击。

---

## 第27章 AI与数据领域

### 27.1 ComfyUI：AI图像生成的CLI化

**软件概述**：ComfyUI是节点式的AI图像生成工作流工具。

**后端策略**：ComfyUI REST API。

**核心能力**：工作流管理、图像生成、队列管理、模型管理。

**测试覆盖**：70个测试（60个单元测试 + 10个E2E测试）。

### 27.2 Ollama：本地LLM推理的CLI化

**软件概述**：Ollama是本地大语言模型推理工具。

**后端策略**：Ollama REST API。

**核心能力**：模型管理、文本生成、嵌入向量、服务器管理。

**测试覆盖**：98个测试（87个单元测试 + 11个E2E测试）。

### 27.3 Exa：AI原生搜索的CLI化

**软件概述**：Exa是AI驱动的搜索引擎。

**后端策略**：exa-py SDK。

**核心能力**：语义搜索、内容提取、答案生成。

**测试覆盖**：40个测试。

### 27.4 ChromaDB：向量数据库的CLI化

**软件概述**：ChromaDB是开源的向量数据库。

**后端策略**：ChromaDB HTTP API。

**核心能力**：集合管理、文档管理、语义搜索。

---

## 第28章 其他领域

### 28.1 Draw.io：图表绘制的CLI化

**软件概述**：Draw.io（diagrams.net）是最流行的在线/桌面图表工具。

**后端策略**：mxGraph XML操作 + draw.io CLI导出。

**核心能力**：形状创建、连接器管理、页面管理、导出（PNG、PDF、SVG）。

**Demo亮点**：Agent使用Draw.io CLI从零创建了一个完整的HTTPS连接生命周期图。

**测试覆盖**：138个测试（116个单元测试 + 22个E2E测试）。

### 28.2 Mermaid：图表标记语言的CLI化

**软件概述**：Mermaid是基于文本的图表生成工具。

**后端策略**：Mermaid状态文件 + mermaid.ink渲染器。

**测试覆盖**：10个测试。

### 28.3 Firefly III：个人财务管理的CLI化

**软件概述**：Firefly III是开源的个人财务管理工具。

**后端策略**：Firefly III REST API。

**核心能力**：账户管理、交易管理、预算、账单、报告等。

### 28.4 WireMock：HTTP Mock服务器的CLI化

**软件概述**：WireMock是HTTP mock服务器，用于API测试。

**后端策略**：WireMock REST API。

**核心能力**：桩管理、请求日志、场景管理、录制/回放。

### 28.5 MacroCLI：GUI工作流宏的CLI化

**软件概述**：MacroCLI是一个创新性的工具，将GUI工作流转换为参数化的、Agent可调用的宏。

**后端策略**：多后端路由（原生API、文件转换、GUI Agent、GUI宏、可访问性控制、视觉模板匹配）。

**核心能力**：录制GUI操作、参数化、后端路由、宏执行。

**特殊之处**：MacroCLI是CLI-Anything生态中最具创新性的项目之一。它不为特定软件创建CLI，而是提供了一个通用的"GUI宏转CLI"框架。

---

## 第五部分小结

本部分展示了CLI-Anything在50+个软件上的领域实践：

1. **创意与媒体**：GIMP、Blender、Inkscape、Krita、Audacity、MuseScore
2. **视频编辑**：Shotcut、Kdenlive、Openscreen、VideoCaptioner
3. **办公与生产力**：LibreOffice、Obsidian、Zotero、Mubu
4. **3D/CAD与科学**：FreeCAD、QGIS、CloudCompare、Uni-Mol Tools
5. **游戏开发**：Godot、s&box、Slay the Spire II
6. **开发与运维**：LLDB、RenderDoc、Nsight Graphics、Unreal Insights、n8n、PM2
7. **网络与通信**：Zoom、AdGuard Home、Browser
8. **AI与数据**：ComfyUI、Ollama、Exa、ChromaDB
9. **其他领域**：Draw.io、Mermaid、Firefly III、WireMock、MacroCLI

每个CLI都遵循相同的架构模式，但后端策略各不相同——从直接文件操作到REST API调用，从脚本生成到MCP服务器。这种多样性展示了CLI-Anything方法论的通用性。


---

## 第29章 领域实践的关键洞察

### 29.1 后端策略的选择规律

通过分析50+个软件的后端策略，我们总结出以下选择规律：

**如果软件提供CLI工具**（如LibreOffice、Inkscape、Blender）→ 直接包装CLI工具。这是最可靠的策略，因为CLI工具是软件开发者维护的。

**如果软件提供脚本API**（如Blender的bpy、GIMP的Script-Fu）→ 生成脚本并执行。这提供了最大的灵活性，但需要处理脚本语言的特性。

**如果软件提供REST API**（如ComfyUI、Ollama、n8n）→ 包装HTTP调用。这适用于Web服务和有HTTP接口的本地服务。

**如果软件使用XML/JSON项目文件**（如Shotcut、Draw.io）→ 直接操作文件格式。这避免了启动软件的开销，但需要深入理解文件格式。

**如果没有以上任何接口**→ 使用MCP服务器或混合策略。

### 29.2 测试数量与软件复杂度的关系

| 软件复杂度 | 典型测试数 | 示例 |
|-----------|-----------|------|
| 简单（REST API包装） | 10-40 | Zoom, AdGuard Home, WireMock |
| 中等（文件格式操作） | 50-100 | Draw.io, MuseScore, Ollama |
| 复杂（多子系统） | 100-200 | GIMP, Audacity, LibreOffice |
| 极复杂（3D/视频） | 200+ | Blender, Inkscape, s&box |

### 29.3 跨软件的共性模式

尽管50+个软件涵盖的领域极其广泛，但CLI-Anything的实现中存在大量共性模式：

**会话管理**：所有CLI都使用相同的Session类实现撤销/重做。

**输出格式**：所有CLI都支持`--json`标志，使用相同的JSON输出约定。

**错误处理**：所有CLI都使用相同的`handle_error`装饰器。

**REPL界面**：所有CLI都使用相同的ReplSkin。

**测试结构**：所有CLI都遵循四层测试体系。

**目录结构**：所有CLI都遵循相同的目录模板。

这些共性模式使得CLI-Anything的生态系统具有一致性和可互换性——Agent学会了一个CLI的使用方式，就学会了所有CLI的使用方式。

### 29.4 领域特定的创新

每个领域也有自己的创新：

**3D领域**（Blender、FreeCAD）：Preview协议、Live Preview、Trajectory历史。

**视频领域**（Shotcut、Kdenlive）：滤镜翻译、时间码精度、渲染鸿沟处理。

**图像领域**（GIMP、Inkscape）：批量操作优化、格式转换管道。

**办公领域**（LibreOffice）：ODF生成、多组件支持（Writer/Calc/Impress）。

**游戏领域**（s&box、Slay the Spire II）：实时状态同步、游戏逻辑交互。

# 第六部分：CLI-Hub、Preview协议与生态

> "CLI-Hub lets agents autonomously discover and install the CLIs they need — zero human intervention required."

---

## 第29章 CLI-Hub生态系统

### 29.1 CLI-Hub的架构

CLI-Hub是CLI-Anything的集中化注册表和安装工具。它的架构包含三个层次：

**Web Hub层**：https://hkuds.github.io/CLI-Anything/ 提供可视化的浏览和搜索界面。

**CLI Hub层**：`pip install cli-anything-hub`安装的命令行工具，提供`install`、`update`、`uninstall`、`list`、`search`等命令。

**注册表层**：`registry.json`文件，包含所有CLI的元数据。

### 29.2 registry.json的结构

每个CLI在registry.json中注册以下信息：

```json
{
  "name": "blender",
  "display_name": "Blender",
  "version": "1.0.0",
  "description": "3D modeling, animation, and rendering via blender --background --python",
  "requires": "blender >= 4.2",
  "homepage": "https://www.blender.org",
  "source_url": null,
  "install_cmd": "pip install git+https://github.com/HKUDS/CLI-Anything.git#subdirectory=blender/agent-harness",
  "entry_point": "cli-anything-blender",
  "skill_md": "skills/cli-anything-blender/SKILL.md",
  "category": "3d",
  "contributors": [
    {"name": "CLI-Anything-Team", "url": "https://github.com/HKUDS/CLI-Anything"}
  ]
}
```

**关键字段说明**：
- `source_url`：null表示in-repo harness，非null表示standalone repo
- `install_cmd`：完整的pip install命令
- `skill_md`：SKILL.md的路径（相对路径或完整URL）
- `category`：分类标签，用于CLI-Hub的分类浏览

### 29.3 CLI-Hub的安装来源

CLI-Hub支持多种安装来源：

| 来源 | 示例 |
|------|------|
| PyPI | `pip install cli-anything-gimp` |
| Git subdirectory | `pip install git+https://github.com/HKUDS/CLI-Anything.git#subdirectory=gimp/agent-harness` |
| npm | `npm install -g sketch-cli` |
| Bundled/system | 使用已安装的系统工具 |
| Cargo | `cargo install clibrowser` |

### 29.4 元技能（Meta-Skill）

CLI-Hub提供了一个"元技能"——一个SKILL.md文件，让AI Agent可以自主发现和安装CLI：

```
Agent读取元技能 → 浏览CLI目录 → 选择适合任务的CLI → 自动安装 → 读取CLI的SKILL.md → 使用CLI
```

元技能的内容指向一个实时更新的CDN地址：
```
https://reeceyang.sgp1.cdn.digitaloceanspaces.com/SKILL.md
```

Agent读取这个文件后，可以看到20+个CLI的分类列表，每个CLI都有简短描述和pip install命令。

### 29.5 In-repo vs Standalone

CLI-Anything支持两种贡献模式：

**In-repo harness**：代码放在CLI-Anything仓库的`<software>/agent-harness/`目录下。适合：
- 核心团队维护的CLI
- 需要与主仓库同步更新的CLI
- 使用共享基础设施（如repl_skin.py）的CLI

**Standalone repo**：代码放在贡献者自己的仓库中，只在registry.json中注册。适合：
- 第三方贡献者维护的CLI
- 需要独立版本控制和发布周期的CLI
- 使用非Python技术栈的CLI（如Node.js、Rust）

---

## 第30章 Preview协议详解

### 30.1 协议的设计目标

Preview协议的设计遵循五个目标：

1. **保持渲染路径诚实**：预览必须来自真实软件。
2. **跨领域通用**：同一个协议处理视频时间线、CAD视图、Blender渲染、RenderDoc输出。
3. **足够便宜以频繁使用**：预览应该是快速、低分辨率、可缓存、小体积的。
4. **简单到可以渐进式采用**：一个CLI可以用一个命令组和一个小工具添加预览支持。
5. **在无头环境中工作**：bundle生成和验证不需要GUI。

### 30.2 Bundle的生命周期

```
创建 → 写入manifest.json → 写入summary.json → 写入artifacts → 不可变
```

Bundle一旦写入就不可变。如果项目发生变化，需要创建新的Bundle。

### 30.3 缓存规则

`preview capture`可以在以下条件满足时返回缓存的bundle：
- 项目/捕获指纹未变
- 配方和归一化预览参数未变
- 协议版本未变
- bundle仍然存在于磁盘上

`--force`必须绕过缓存。

### 30.4 Bundle ID的生成

推荐的bundle id格式：
```
<UTC时间戳>_<短指纹>_<配方>
```

示例：
```
20260419T104530Z_9f0a2c4b_quick
```

指纹应该从以下内容派生：
- 项目指纹或捕获指纹
- 配方名称
- 归一化的预览参数
- harness版本
- 协议版本

### 30.5 Trajectory的设计

Trajectory是只追加的命令到预览的历史记录。它回答的问题是："Agent执行了哪些命令，每个命令对应的预览状态是什么？"

Trajectory的存在是因为：
- Bundle是单个不可变快照，不是稳定的历史对象
- 同一个工作项目可能产生多个bundle
- Agent需要能够回溯到之前的构建检查点

### 30.6 性能预算

预览应该足够便宜，可以在Agent循环中生成：

| 产物类型 | 推荐限制 |
|----------|----------|
| 主图像 | 最长边 ≤ 1280 px |
| 画廊图像 | 3到8张 |
| 预览视频 | ≤ 8秒，≤ 720p |
| Bundle大小 | ≤ 25 MB |

---

## 第31章 多平台集成

### 31.1 Claude Code集成

CLI-Anything最初是为Claude Code设计的。Claude Code的集成通过插件系统实现：

```bash
# 添加市场
/plugin marketplace add HKUDS/CLI-Anything

# 安装插件
/plugin install cli-anything

# 使用
/cli-anything ./gimp
/cli-anything:refine ./gimp "batch processing"
/cli-anything:test ./gimp
/cli-anything:validate ./gimp
```

**插件的命令**：
| 命令 | 功能 |
|------|------|
| `/cli-anything <path>` | 构建完整的CLI（7阶段） |
| `/cli-anything:refine <path> [focus]` | 改进现有CLI |
| `/cli-anything:test <path>` | 运行测试 |
| `/cli-anything:validate <path>` | 验证标准合规性 |

### 31.2 Pi Coding Agent集成

Pi通过扩展系统集成：

```bash
# 安装扩展
bash .pi-extension/cli-anything/install.sh

# 使用
/cli-anything ./gimp
```

### 31.3 OpenCode集成

OpenCode通过命令文件集成：

```bash
# 复制命令文件
cp CLI-Anything/opencode-commands/*.md ~/.config/opencode/commands/
cp CLI-Anything/cli-anything-plugin/HARNESS.md ~/.config/opencode/commands/

# 使用
/cli-anything ./gimp
```

### 31.4 Codex集成

Codex通过技能系统集成：

```bash
# 安装技能
bash CLI-Anything/codex-skill/scripts/install.sh

# 使用
Use CLI-Anything to build a harness for ./gimp
```

### 31.5 OpenClaw集成

OpenClaw通过SKILL.md集成：

```bash
# 复制技能文件
cp CLI-Anything/openclaw-skill/SKILL.md ~/.openclaw/skills/cli-anything/SKILL.md

# 使用
@cli-anything build a CLI for ./gimp
```

### 31.6 Qodercli集成

```bash
# 注册插件
bash CLI-Anything/qoder-plugin/setup-qodercli.sh

# 使用
/cli-anything ./gimp
```

### 31.7 GitHub Copilot CLI集成

```bash
# 安装插件
copilot plugin install ./cli-anything-plugin

# 使用
/cli-anything ./gimp
```

### 31.8 Goose集成

Goose通过CLI provider（如Claude Code）间接使用CLI-Anything。

---

## 第32章 贡献与社区

### 32.1 贡献类型

CLI-Anything欢迎三种贡献：

**A) 新软件的CLI**：最有影响力的贡献。可以通过in-repo或standalone两种方式。

**B) 新功能**：改进现有CLI或插件框架。

**C) Bug修复**：修复现有CLI或插件中的问题。

### 32.2 贡献流程

1. Fork仓库，创建feature分支
2. 遵循HARNESS.md的方法论
3. 确保所有测试通过
4. 提交PR，填写PR模板
5. 等待review和合并

### 32.3 代码规范

- 遵循PEPE 8规范
- 使用类型注解
- 所有CLI命令必须支持`--json`标志
- 使用conventional commit messages

### 32.4 社区治理

CLI-Anything由香港大学数据科学实验室（HKUDS）维护。社区通过GitHub Issues、Discussions和PR进行协作。

---

## 第六部分小结

本部分介绍了CLI-Anything的生态系统：

1. **CLI-Hub**：集中化注册表、安装工具、元技能。
2. **Preview协议**：跨软件的中间预览标准，三层持久化模型。
3. **多平台集成**：Claude Code、Pi、OpenCode、Codex、OpenClaw、Qodercli、GitHub Copilot CLI、Goose。
4. **社区贡献**：in-repo和standalone两种模式，完善的贡献流程。

在下一部分，我们将展望CLI-Anything的未来方向和Agent-Native软件的趋势。


---

## 第33章 SKILL.md规范详解

### 33.1 SKILL.md的YAML Frontmatter

SKILL.md的头部是YAML格式的元数据：

```yaml
---
name: "cli-anything-blender"
description: "3D modeling, animation, and rendering via blender --background --python"
---
```

这些元数据被AI Agent框架用于技能发现。当Agent扫描可用技能时，它读取这些元数据来判断技能是否与当前任务相关。

### 33.2 SKILL.md的Markdown Body

SKILL.md的正文包含以下标准部分：

**安装指导**：如何安装CLI和它的依赖。

**命令组文档**：每个命令组的简要描述和可用子命令。

**使用示例**：常见的使用场景和命令序列。

**Agent特定指导**：
- 如何使用`--json`标志获取机器可读输出
- 如何处理错误
- 如何进行程序化使用
- 如何组合多个命令完成复杂任务

### 33.3 SKILL.md的自动生成

CLI-Anything提供了`skill_generator.py`来自动生成SKILL.md。这个脚本从Click装饰器中提取命令结构，从setup.py中提取包元数据，从README.md中提取描述，然后使用Jinja2模板生成标准化的SKILL.md。

生成的SKILL.md包含两个位置：
- **规范位置**：`skills/cli-anything-<software>/SKILL.md`
- **兼容位置**：`cli_anything/<software>/skills/SKILL.md`

### 33.4 SKILL.md与Agent框架的集成

不同Agent框架对SKILL.md的使用方式不同：

**Claude Code**：通过插件系统自动加载SKILL.md。插件的commands/目录下的.md文件引用SKILL.md中的内容。

**OpenClaw**：将SKILL.md复制到`~/.openclaw/skills/`目录，通过`@cli-anything`前缀调用。

**Pi**：通过扩展系统加载SKILL.md。扩展的install.sh脚本将SKILL.md复制到Pi的技能目录。

**Codex**：通过技能系统加载SKILL.md。安装脚本将SKILL.md复制到`$CODEX_HOME/skills/`。

---

## 第34章 HARNESS.md方法论深度解读

### 34.1 方法论的起源

HARNESS.md不是一开始就设计好的。它是CLI-Anything团队在构建18+个软件的CLI接口过程中，逐步总结和提炼出来的。

最初，团队在为Shotcut构建CLI时发现了"渲染鸿沟"问题——CLI操作了项目文件，但导出时使用了简单的ffmpeg拼接，导致滤镜效果丢失。这个问题促使团队制定了"使用真实软件渲染"的原则。

后来，在为Blender构建CLI时，团队发现了"Preview"的需求——Agent在构建3D场景时需要中间反馈。这催生了Preview协议。

每一次实践中的发现都被编码到HARNESS.md中，成为下一次实践的指导。这种"实践-总结-指导"的循环，使得HARNESS.md成为一个不断进化的活文档。

### 34.2 渐进式披露设计

HARNESS.md采用了"渐进式披露"（Progressive Disclosure）的设计。文档被组织为连续的阶段，每个阶段的结果为下一个阶段提供上下文。这种设计的目的是不要让Agent一开始就面对所有信息——通过分阶段输出，Agent可以在每个阶段专注于当前的任务。

### 34.3 关键教训总结

HARNESS.md中总结的关键教训：

| 教训 | 描述 |
|------|------|
| 使用真实软件 | CLI必须调用实际应用，不允许用Python替代 |
| 渲染鸿沟 | GUI应用在渲染时应用效果，CLI必须处理这个问题 |
| 滤镜翻译 | 不同格式的效果需要翻译，注意重复合并、参数空间差异 |
| 时间码精度 | 非整数帧率导致累积舍入，使用round()而非int() |
| 输出验证 | 不要信任退出码，验证magic bytes、ZIP结构、像素分析 |

### 34.4 目录结构规范

HARNESS.md定义了严格的目录结构，关键规则包括：
- `cli_anything/`目录**没有**`__init__.py`（PEP 420命名空间包）
- 每个子包**有**自己的`__init__.py`
- `tests/TEST.md`是必填的
- `README.md`是必填的
- `repl_skin.py`是从插件复制的

# 第七部分：Agent-Native软件的未来

> "Today's Software Serves Humans. Tomorrow's Users will be Agents."

---

## 第33章 Agent-Native软件的概念

### 33.1 什么是Agent-Native软件

"Agent-Native"（Agent原生）是一个新的软件设计理念。一个Agent-Native的软件不仅为人类用户提供GUI，还为AI Agent提供结构化的命令行接口。

这个概念的灵感来自"Cloud-Native"（云原生）的演进。在云计算的早期，软件是为单机设计的，然后被"搬到"云上。但真正的云原生软件从一开始就为云环境设计——微服务架构、容器化、声明式API。

类似地，当前的桌面软件是为人类设计的，然后通过CLI-Anything被"翻译"为Agent可用的接口。但未来的软件可能从一开始就为两种用户（人类和Agent）设计。

### 33.2 Agent-Native软件的特征

一个Agent-Native的软件应该具备以下特征：

**双模态接口**：同时提供GUI（给人类）和CLI/API（给Agent）。两者共享同一个后端引擎。

**结构化输出**：所有操作都可以输出JSON格式的结构化数据。

**自文档化**：通过`--help`、SKILL.md或OpenAPI规范，Agent可以自动发现和理解软件的功能。

**可内省**：Agent可以在操作前查询当前状态，在操作后验证结果。

**可幂等**：尽可能使操作幂等，支持安全的重试。

**可组合**：操作可以被组合成复杂的工作流。

**可回滚**：支持撤销/重做，Agent可以从错误中恢复。

### 33.3 从CLI-Anything到Agent-Native

CLI-Anything是通向Agent-Native软件的桥梁。它通过"后处理"的方式，为现有的GUI软件添加了Agent-Native的特征。

但更理想的方式是：软件在设计阶段就考虑Agent-Native的需求。例如：
- 一个3D建模软件在设计GUI的同时，也设计CLI命令树
- 一个视频编辑器在实现GUI操作的同时，也实现对应的CLI操作
- 一个办公套件在发布GUI版本的同时，也发布CLI版本

CLI-Anything的HARNESS.md文档就是这种"设计指南"——它告诉软件开发者，如何将你的软件设计成Agent-Native的。

---

## 第34章 AI Agent与工具使用的未来趋势

### 34.1 Agent框架的演进

当前的AI Agent框架（如LangChain、AutoGPT、CrewAI）主要关注Agent的推理和规划能力。但随着Agent需要操作的工具越来越多，工具发现和工具使用将成为关键瓶颈。

CLI-Anything提供了一种解决思路：为每个工具创建一个标准化的CLI接口，Agent通过`which`和`--help`发现和理解工具。

未来可能出现的趋势：
- **Agent工具浏览器**：类似于应用商店，Agent可以在其中浏览和安装工具
- **工具能力协商**：Agent在执行任务前，先与可用工具进行能力匹配
- **工具组合语言**：一种声明式语言，描述如何将多个工具组合成复杂的工作流

### 34.2 MCP与CLI-Anything的融合

Model Context Protocol（MCP）和CLI-Anything解决的是不同层次的问题：
- MCP：Agent如何发现和调用工具
- CLI-Anything：如何为没有API的软件创建工具

未来，这两个层次可能会融合：
- CLI-Anything生成的CLI可以被MCP服务器包装
- MCP服务器可以自动发现CLI-Anything生成的CLI
- Agent通过MCP协议透明地调用CLI-Anything的工具

### 34.3 多Agent协作与工具共享

当多个Agent需要使用同一个工具时，工具的并发访问和状态共享成为关键问题。

CLI-Anything的会话管理机制（文件锁定、撤销/重做）为多Agent协作提供了基础。但更复杂的场景（如多个Agent同时编辑同一个3D场景）需要更高级的协作机制。

### 34.4 Agent体验（AX）的设计

随着Agent越来越多地使用工具，"Agent体验"（Agent Experience, AX）将成为一个新的设计领域。

AX的设计原则包括：
- **可发现性**：Agent能快速找到需要的工具
- **可理解性**：Agent能快速理解工具的功能和用法
- **可操作性**：Agent能正确地调用工具
- **可恢复性**：Agent能从错误中恢复
- **可观测性**：Agent能理解操作的结果
- **可组合性**：Agent能将多个工具组合成复杂的工作流

CLI-Anything的`--json`、`--help`、`session undo`、`info/list/status`等设计，都是AX的具体实现。

---

## 第35章 技术演进方向

### 35.1 更多的软件支持

CLI-Anything当前支持50+个软件，但这只是冰山一角。潜在的扩展方向包括：

**CAD/EDA工具**：KiCad、LibreCAD、OpenSCAD
**数字音频工作站（DAW）**：Ardour、LMMS、Hydrogen
**科学计算**：ParaView、Gephi、Stellarium
**企业软件**：NextCloud、GitLab、Grafana、Mattermost
**更多AI工具**：Stable Diffusion WebUI、InvokeAI、Fooocus

### 35.2 自动化生成的改进

当前的CLI生成依赖AI Agent（如Claude Code）执行七阶段流水线。未来的改进方向：

**更智能的代码分析**：使用静态分析工具（如AST解析、调用图分析）辅助AI Agent理解代码结构。

**模板化生成**：为常见的软件类型（REST API包装、XML文件操作、脚本API包装）提供预定义的CLI模板。

**增量更新**：当软件更新时，自动检测变化并更新CLI，而不是从头重新生成。

### 35.3 Preview协议的扩展

Preview协议当前支持图像、视频和JSON产物。未来的扩展方向：

**3D预览**：支持glTF/OBJ等3D格式的预览
**音频预览**：支持波形图和频谱图的预览
**交互式预览**：支持在浏览器中旋转/缩放3D模型
**实时协作**：多个Agent同时查看同一个预览

### 35.4 性能优化

CLI-Anything的性能优化方向：

**缓存机制**：缓存频繁调用的后端操作结果
**批量操作**：将多个CLI调用合并为一个批量操作
**并行执行**：独立的操作可以并行执行
**增量渲染**：只重新渲染变化的部分

### 35.5 安全增强

CLI-Anything的安全增强方向：

**权限控制**：限制Agent可以执行的操作
**审计日志**：记录所有CLI操作
**沙箱执行**：在沙箱中执行不受信任的CLI操作
**输入验证**：更严格的参数验证和清理

---

## 第36章 行业影响与展望

### 36.1 对软件行业的影响

CLI-Anything代表的趋势——软件的"用户"从人类变成Agent——将对软件行业产生深远影响：

**软件设计的变化**：未来的软件可能需要同时为人类和Agent设计界面。GUI设计和CLI设计将成为同等重要的设计工作。

**软件分发的变化**：软件的分发渠道可能从"应用商店"扩展到"Agent工具注册表"。CLI-Hub就是这种注册表的早期原型。

**软件维护的变化**：软件更新不仅要考虑GUI的向后兼容，还要考虑CLI的向后兼容。CLI接口可能比GUI接口更稳定。

**商业模式的变化**：软件的商业模式可能从"按用户收费"扩展到"按Agent调用收费"。

### 36.2 对AI行业的影响

CLI-Anything对AI行业也有重要影响：

**Agent能力的提升**：CLI-Anything大幅扩展了Agent可以操作的工具范围。Agent不再局限于API和简单的文件操作，而是可以操作专业级的桌面软件。

**Agent可靠性的提升**：结构化的CLI接口比GUI自动化更可靠，减少了Agent操作失败的概率。

**Agent效率的提升**：CLI调用比GUI操作快几个数量级，Agent可以在更短的时间内完成更多的工作。

**新的研究方向**：CLI-Anything启发了新的研究方向，如Agent体验（AX）设计、Agent工具发现、Agent工具组合等。

### 36.3 对开源社区的影响

CLI-Anything对开源社区的影响：

**提高开源软件的可访问性**：CLI-Anything让更多的AI Agent可以使用开源软件，扩大了开源软件的用户群。

**促进社区协作**：CLI-Hub的in-repo和standalone贡献模式促进了社区协作。

**新的贡献形式**：为开源软件创建CLI接口成为一种新的贡献形式。

### 36.4 从"人机交互"到"Agent-机交互"

CLI-Anything代表了一个更大的趋势：**计算的用户正在从人类变成Agent**。

这个趋势的影响是深远的：
- 软件的"用户界面"概念需要重新定义
- 软件的"易用性"需要重新衡量——对Agent来说，"易用"意味着结构化、可预测、可验证
- 软件的"文档"需要重新设计——Agent需要的是`--help`和SKILL.md，而不是用户手册

CLI-Anything是这个趋势的早期实践者。它的方法论和工具为未来的Agent-Native软件提供了参考。

---

## 第七部分小结

本部分展望了CLI-Anything的未来：

1. **Agent-Native软件**：一种新的软件设计理念，同时为人类和Agent设计界面。
2. **技术趋势**：MCP融合、多Agent协作、Agent体验设计、性能优化、安全增强。
3. **行业影响**：对软件行业、AI行业和开源社区的深远影响。
4. **范式转换**：从"人机交互"到"Agent-机交互"的转变。

在最后一部分，我们将提供附录、术语表和资源链接。


---

## 第37章 深度案例：从零构建一个CLI

### 37.1 选择目标软件

让我们以一个假想的案例来展示CLI-Anything的完整流程。假设我们要为一个名为"PixelForge"的图像编辑软件构建CLI。

**软件信息**：
- 名称：PixelForge
- 类型：图像编辑
- 后端：Python API (pixelforge)
- 项目文件格式：JSON
- 导出：通过`pixelforge --export`命令

### 37.2 阶段1：代码分析

分析PixelForge的代码库后，我们发现：
- 核心引擎是一个Python库（pixelforge）
- GUI是用Tkinter构建的，与核心逻辑分离
- 项目文件是JSON格式，包含图层、滤镜、选区等信息
- 导出通过`pixelforge --export`命令行工具实现

### 37.3 阶段2：CLI架构设计

设计CLI的命令组：
| 命令组 | 功能 |
|--------|------|
| `project` | 项目管理（new, open, save, info） |
| `layer` | 图层管理（add, remove, list, get, set） |
| `filter` | 滤镜应用（blur, sharpen, contrast, brightness） |
| `selection` | 选区操作（rect, ellipse, free, invert） |
| `export` | 导出（render, convert） |
| `session` | 会话管理（status, undo, redo） |

### 37.4 阶段3：实现

按照目录结构模板创建文件：

```python
# cli_anything/pixel_forge/pixel_forge_cli.py
@click.group(invoke_without_command=True)
@click.option("--json", "use_json", is_flag=True)
@click.option("--project", "project_path", type=str, default=None)
@click.pass_context
def cli(ctx, use_json, project_path):
    """PixelForge CLI — Image editing from the command line."""
    global _json_output
    _json_output = use_json
    if ctx.invoked_subcommand is None:
        ctx.invoke(repl, project_path=None)
```

### 37.5 阶段4-6：测试

编写测试计划、实现测试、运行测试、记录结果。

### 37.6 阶段6.5-7：SKILL.md和发布

生成SKILL.md，创建setup.py，发布到PyPI。

---

## 第38章 CLI-Anything与其他自动化方案的对比

### 38.1 与Selenium/Playwright的对比

Selenium和Playwright是Web GUI自动化的标准工具。它们与CLI-Anything的对比：

| 维度 | Selenium/Playwright | CLI-Anything |
|------|---------------------|--------------|
| 目标 | Web浏览器 | 桌面软件 |
| 方法 | DOM操作 | CLI接口 |
| 可靠性 | 中（依赖DOM结构） | 高（结构化接口） |
| 速度 | 中（浏览器渲染） | 快（无GUI开销） |
| 通用性 | 仅Web | 任意软件 |

### 38.2 与RPA工具的对比

RPA工具（UiPath、Automation Anywhere）与CLI-Anything的对比：

| 维度 | RPA工具 | CLI-Anything |
|------|---------|--------------|
| 目标 | 企业流程自动化 | AI Agent工具使用 |
| 方法 | GUI录制/回放 | CLI接口 |
| 可靠性 | 低（GUI脆弱） | 高（结构化接口） |
| 维护成本 | 高（40-60%） | 低 |
| 适用场景 | 重复性流程 | Agent任务 |

### 38.3 与Computer Use的对比

Anthropic的Computer Use与CLI-Anything的对比：

| 维度 | Computer Use | CLI-Anything |
|------|--------------|--------------|
| 方法 | 截屏+点击 | CLI调用 |
| 速度 | 慢（0.5-2秒/操作） | 快（<0.1秒/操作） |
| 准确率 | 中 | 高 |
| 通用性 | 很高 | 高 |
| 可验证性 | 低 | 高 |

### 38.4 与原生API的对比

直接使用软件的原生API（如果存在）与CLI-Anything的对比：

| 维度 | 原生API | CLI-Anything |
|------|---------|--------------|
| 可用性 | 低（不是所有软件都有） | 高（任何有代码的软件） |
| 一致性 | 低（每个API不同） | 高（统一的CLI模式） |
| Agent体验 | 中 | 高 |
| 性能 | 高 | 高 |
| 学习成本 | 高（每个API不同） | 低（统一模式） |

---

## 第39章 哲学思考：软件的本质

### 39.1 软件是什么？

CLI-Anything促使我们重新思考"软件是什么"这个根本性问题。

传统上，我们认为软件是一个应用程序，用户通过GUI与它交互。但CLI-Anything展示了一个不同的视角：**软件是一个功能引擎，GUI只是它的一个界面**。

Blender不仅仅是一个3D建模GUI——它是一个3D渲染引擎，GUI只是它的一个界面。CLI-Anything为它创建了另一个界面——CLI。未来可能还会有更多的界面：语音界面、VR界面、Agent界面。

### 39.2 界面的多元化

CLI-Anything暗示了一个未来：**每个软件可能有多个界面**。

- GUI：给人类（鼠标、键盘、触摸）
- CLI：给Agent（命令、参数、JSON）
- API：给程序（HTTP、SDK）
- 语音：给对话（语音识别、TTS）
- VR/AR：给沉浸式体验

这些界面共享同一个后端引擎，但为不同的"用户"提供不同的交互方式。

### 39.3 Agent作为"用户"

CLI-Anything提出了一个有趣的问题：**Agent是软件的"用户"吗？**

如果是，那么软件设计就需要考虑Agent的需求。Agent需要什么？
- 结构化的接口
- 可预测的行为
- 可验证的结果
- 可恢复的操作

这些需求与人类的需求（直觉的界面、美观的设计、愉悦的体验）完全不同。

CLI-Anything的贡献在于：它第一次系统性地定义了"Agent需要什么样的软件界面"，并提供了一种通用的方法来创建这样的界面。

---

## 第七部分小结（补充）

本部分进一步展望了CLI-Anything的未来：

1. **深度案例**：从零构建一个CLI的完整流程。
2. **方案对比**：CLI-Anything与Selenium、RPA、Computer Use、原生API的对比。
3. **哲学思考**：软件的本质、界面的多元化、Agent作为"用户"。



---

## 第40章 构建Agent-Native软件的实践指南

### 40.1 从现有软件开始

如果你有一个现有的开源软件，想让它成为Agent-Native的，以下是推荐的步骤：

**第一步：评估CLI可行性**
- 软件是否有CLI工具或脚本API？
- 项目文件是否使用可解析的格式（XML、JSON、YAML）？
- 核心逻辑是否与GUI代码分离？

如果以上任何一个是"是"，CLI-Anything就可以应用。

**第二步：运行CLI-Anything**
```bash
/cli-anything <your-software>
```

**第三步：验证和迭代**
```bash
/cli-anything:validate <your-software>
/cli-anything:test <your-software>
/cli-anything:refine <your-software>
```

**第四步：发布**
```bash
cd <your-software>/agent-harness
pip install -e .
# 测试安装后的命令
cli-anything-<your-software> --help
```

### 40.2 从零开始设计Agent-Native软件

如果你正在从零开始设计一个软件，并希望它从一开始就是Agent-Native的，以下是推荐的设计原则：

**原则一：分离核心逻辑和GUI**。将软件的核心引擎与GUI代码完全分离。核心引擎应该可以通过命令行或API调用。

**原则二：使用可解析的项目文件格式**。选择XML、JSON或YAML作为项目文件格式，而不是二进制格式。

**原则三：提供命令行接口**。即使软件主要面向GUI用户，也应该提供基本的命令行接口。

**原则四：支持结构化输出**。所有命令都应该支持JSON格式的输出。

**原则五：实现撤销/重做系统**。使用命令模式实现撤销/重做，这不仅对人类有用，对Agent更重要。

**原则六：提供自文档化能力**。通过`--help`和元数据文件，让Agent可以自动发现和理解软件的功能。

### 40.3 Agent-Native软件的设计清单

在设计Agent-Native软件时，可以使用以下清单：

- [ ] 核心逻辑与GUI分离
- [ ] 项目文件使用可解析格式
- [ ] 提供命令行接口
- [ ] 所有命令支持`--json`输出
- [ ] 实现撤销/重做
- [ ] 提供`--help`文档
- [ ] 生成SKILL.md或等效的技能定义
- [ ] E2E测试覆盖真实后端
- [ ] CLI子进程测试验证安装后的命令
- [ ] 发布到PyPI或等效的包管理器

### 40.4 Agent-Native软件的度量

如何衡量一个软件的"Agent-Native程度"？以下是建议的度量指标：

**CLI覆盖率**：CLI支持的功能占软件总功能的百分比。

**Agent成功率**：Agent使用CLI完成任务的成功率。

**Agent效率**：Agent使用CLI完成任务的时间 vs 使用GUI完成相同任务的时间。

**SKILL.md完整性**：SKILL.md文档覆盖的命令组和参数的百分比。

**测试覆盖率**：E2E测试覆盖的功能占CLI总功能的百分比。

---

## 第41章 结语：通往Agent-Native世界的路

### 41.1 我们站在哪里

2026年，我们站在一个关键的技术转折点上。AI Agent正在从"聊天机器人"进化为"数字工人"。但这些数字工人面临的最大障碍不是推理能力的不足，而是**工具使用的瓶颈**。

CLI-Anything代表了一种解决这个瓶颈的方法：不是让Agent去适应为人类设计的界面，而是为Agent创造适合Agent的界面。

### 41.2 我们要去哪里

未来是Agent-Native的。我们预见：

**短期（1-2年）**：CLI-Anything支持100+个软件，CLI-Hub成为Agent工具发现的标准平台。

**中期（3-5年）**：主流开源软件开始内置CLI接口，Agent-Native成为软件设计的标准实践。

**长期（5-10年）**：软件的"用户"从人类扩展到Agent，软件行业进入"人机共用"时代。

### 41.3 每个人都可以参与

CLI-Anything是一个开源项目，欢迎所有人的参与：

- **软件开发者**：为你的软件创建CLI接口
- **AI研究者**：研究Agent工具使用的理论和方法
- **技术写作者**：改进文档和教程
- **测试工程师**：扩展测试覆盖
- **用户**：使用CLI-Anything，提供反馈

### 41.4 最后的话

CLI-Anything的愿景是：**让所有软件成为Agent-Native的软件**。

这个愿景的实现不需要推翻现有的软件——它只需要在现有软件之上添加一个CLI层。这个CLI层不替代GUI，而是与GUI共存。它不改变软件的核心逻辑，只是为软件提供了一个新的界面。

就像GUI在1984年（Macintosh发布）改变了人类与计算机的交互方式一样，CLI-Anything正在改变Agent与软件的交互方式。

**欢迎来到Agent-Native时代。**

# 第八部分：附录

---

## 附录A：术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| Agent | Agent | 能够感知环境、制定计划、采取行动的AI实体 |
| Agent体验 | Agent Experience (AX) | Agent使用工具时的体验设计 |
| Agent-Native | Agent-Native | 从设计之初就考虑Agent使用的软件设计理念 |
| Bundle | Bundle | Preview协议中的不可变预览快照 |
| CLI | Command Line Interface | 命令行接口 |
| Click | Click | Python的CLI框架 |
| DCC | Digital Content Creation | 数字内容创作（如3D建模、视频编辑） |
| E2E | End-to-End | 端到端测试 |
| GUI | Graphical User Interface | 图形用户界面 |
| HARNESS.md | HARNESS.md | CLI-Anything的方法论文档 |
| Harness | Harness | CLI-Anything生成的CLI接口包 |
| In-repo | In-repo | 代码放在CLI-Anything主仓库中的贡献模式 |
| LLM | Large Language Model | 大语言模型 |
| MCP | Model Context Protocol | 模型上下文协议 |
| MLT | Media Lovin' Toolkit | 多媒体处理框架 |
| ODF | Open Document Format | 开放文档格式 |
| Preview | Preview | 中间状态的可视化反馈 |
| REPL | Read-Eval-Print Loop | 交互式命令行环境 |
| RPA | Robotic Process Automation | 机器人流程自动化 |
| ReplSkin | ReplSkin | CLI-Anything的统一REPL界面组件 |
| SKILL.md | SKILL.md | AI Agent可发现的技能定义文件 |
| Session | Session | CLI-Anything的会话管理对象 |
| Standalone | Standalone | 代码放在贡献者自己仓库中的贡献模式 |
| Trajectory | Trajectory | Preview协议中的命令到预览历史记录 |

---

## 附录B：CLI-Anything支持的软件完整列表

| # | 软件名称 | 分类 | 后端策略 | 测试数 |
|---|----------|------|----------|--------|
| 1 | GIMP | 图像编辑 | Script-Fu + Pillow | 107 |
| 2 | Blender | 3D建模 | bpy + blender --background | 208 |
| 3 | Inkscape | 矢量图形 | SVG/XML + inkscape CLI | 202 |
| 4 | Audacity | 音频 | sox | 161 |
| 5 | LibreOffice | 办公 | ODF + libreoffice --headless | 158 |
| 6 | OBS Studio | 直播 | obs-websocket | 153 |
| 7 | Kdenlive | 视频编辑 | MLT XML + melt | 155 |
| 8 | Shotcut | 视频编辑 | MLT XML + melt | 154 |
| 9 | NSLogger | 日志 | NSLogger协议 | 139 |
| 10 | Draw.io | 图表 | mxGraph XML | 138 |
| 11 | Openscreen | 屏幕录制 | ffmpeg | 101 |
| 12 | Ollama | AI推理 | REST API | 98 |
| 13 | Mubu | 大纲笔记 | 本地数据 | 96 |
| 14 | CloudCompare | 点云 | CLI | 88 |
| 15 | ComfyUI | AI图像 | REST API | 70 |
| 16 | Uni-Mol Tools | 分子预测 | ML后端 | 67 |
| 17 | RenderDoc | GPU调试 | Python API | 59 |
| 18 | MuseScore | 音乐记谱 | mscore CLI | 56 |
| 19 | n8n | 工作流 | REST API | 55+ |
| 20 | Nsight Graphics | GPU调试 | ngfx CLI | 51 |
| 21 | AnyGen | AI生成 | REST API | 50 |
| 22 | Unreal Insights | 性能分析 | trace会话 | 50 |
| 23 | Obsidian | 知识管理 | REST API | 55 |
| 24 | AdGuard Home | 广告拦截 | REST API | 36 |
| 25 | Zoom | 视频会议 | REST API | 22 |
| 26 | Godot | 游戏引擎 | headless | 24 |
| 27 | s&box | 游戏引擎 | 文件I/O | 244 |
| 28 | Slay the Spire II | 游戏 | HTTP API | - |
| 29 | LLDB | 调试 | Python API | 27 |
| 30 | VideoCaptioner | 字幕 | CLI | 26 |
| 31 | Mermaid | 图表 | renderer | 10 |
| 32 | NotebookLM | AI研究 | CLI | 21 |
| 33 | Dify Workflow | AI工作流 | CLI | 11 |
| 34 | Zotero | 文献管理 | SQLite + API | - |
| 35 | Firefly III | 财务 | REST API | - |
| 36 | WireMock | 测试 | REST API | - |
| 37 | Exa | 搜索 | SDK | 40 |
| 38 | ChromaDB | 向量DB | HTTP API | - |
| 39 | MacroCLI | 宏 | 多后端 | - |
| 40 | Eth2-Quickstart | DevOps | shell | 18 |
| 41 | Browser | 浏览器 | MCP | - |
| 42 | Krita | 绘画 | CLI | - |
| 43 | Mailchimp | 邮件营销 | REST API | 36 |
| 44 | QGIS | GIS | PyQGIS | 22 |
| 45 | SeaClip | 看板 | REST API | - |
| 46 | PM2 | 进程管理 | CLI | - |
| 47 | Safari | 浏览器 | safari-mcp | - |
| 48 | Intelwatch | OSINT | CLI | - |
| 49 | QuietShrink | 视频压缩 | ffmpeg | - |
| 50 | CloudAnalyzer | 点云QA | Python API | - |
| 51 | Stata | 统计 | CLI | - |
| 52 | Ink/Stitch | 刺绣 | CLI | - |

**总计**: 2,280+ 测试，100% 通过率

---

## 附录C：关键文件路径

| 文件 | 路径 | 用途 |
|------|------|------|
| HARNESS.md | `cli-anything-plugin/HARNESS.md` | 方法论SOP（唯一真相来源） |
| registry.json | `registry.json` | CLI注册表 |
| repl_skin.py | `cli-anything-plugin/repl_skin.py` | 统一REPL皮肤 |
| preview_bundle.py | `cli-anything-plugin/preview_bundle.py` | Preview bundle工具 |
| skill_generator.py | `cli-anything-plugin/skill_generator.py` | SKILL.md生成器 |
| SKILL.md | `skills/cli-anything-<software>/SKILL.md` | 规范技能文件位置 |
| PREVIEW_PROTOCOL.md | `docs/PREVIEW_PROTOCOL.md` | Preview协议规范 |

---

## 附录D：常用命令参考

### CLI-Anything插件命令

```bash
# 构建CLI
/cli-anything <path-or-repo>

# 改进CLI
/cli-anything:refine <path> [focus]

# 运行测试
/cli-anything:test <path>

# 验证标准
/cli-anything:validate <path>
```

### 生成的CLI通用命令

```bash
# 帮助
cli-anything-<software> --help

# JSON模式
cli-anything-<software> --json <command>

# REPL模式
cli-anything-<software>

# 项目管理
cli-anything-<software> project new -o <path>
cli-anything-<software> project open <path>
cli-anything-<software> project save [path]

# 会话管理
cli-anything-<software> session status
cli-anything-<software> session undo
cli-anything-<software> session redo

# Preview
cli-anything-<software> preview recipes
cli-anything-<software> preview capture --recipe <name>
cli-anything-<software> preview latest
cli-anything-<software> preview live start
cli-anything-<software> preview live status
cli-anything-<software> preview live stop
```

### CLI-Hub命令

```bash
# 安装CLI-Hub
pip install cli-anything-hub

# 浏览CLI
cli-hub list

# 安装CLI
cli-hub install <name>

# 更新CLI
cli-hub update <name>

# 卸载CLI
cli-hub uninstall <name>

# 搜索CLI
cli-hub search <query>
```

### 测试命令

```bash
# 运行单元测试
python3 -m pytest cli_anything/<software>/tests/test_core.py -v

# 运行E2E测试
python3 -m pytest cli_anything/<software>/tests/test_full_e2e.py -v

# 运行所有测试
python3 -m pytest cli_anything/<software>/tests/ -v

# 强制使用安装后的命令
CLI_ANYTHING_FORCE_INSTALLED=1 python3 -m pytest cli_anything/<software>/tests/ -v -s
```

---

## 附录E：推荐阅读

### AI Agent相关

1. **API Agents vs. GUI Agents: Divergence and Convergence** (arXiv:2503.11069)
   - 深入分析了API Agent和GUI Agent的异同

2. **Model Context Protocol (MCP) Specification**
   - Anthropic提出的工具连接标准

3. **Toolformer: Language Models Can Teach Themselves to Use Tools**
   - LLM工具使用的开创性研究

### CLI设计相关

1. **The Art of Unix Programming** (Eric Raymond)
   - Unix CLI设计哲学的经典著作

2. **12 Factor CLI Apps**
   - 现代CLI应用的设计指南

3. **Click Documentation**
   - Python Click框架的官方文档

### 软件自动化相关

1. **RPA行业报告**
   - GUI自动化的商业化实践

2. **Computer Use Documentation**
   - Anthropic的Computer Use功能文档

3. **Selenium/Playwright Documentation**
   - Web GUI自动化的标准工具

---

## 附录F：资源链接

| 资源 | 链接 |
|------|------|
| GitHub仓库 | https://github.com/HKUDS/CLI-Anything |
| CLI-Hub网站 | https://hkuds.github.io/CLI-Anything/ |
| 中文文档 | https://github.com/HKUDS/CLI-Anything/blob/main/README_CN.md |
| 日文文档 | https://github.com/HKUDS/CLI-Anything/blob/main/README_JA.md |
| PyPI (CLI-Hub) | https://pypi.org/project/cli-anything-hub/ |
| 许可证 | Apache License 2.0 |

---

## 附录G：HARNESS.md核心要点速查

### 六个不可违反的原则

1. **使用真实软件**：CLI必须调用真实软件后端
2. **操作原生格式**：解析和修改软件的原生项目文件
3. **利用现有CLI工具**：使用软件自带的命令行工具
4. **失败大声且清晰**：错误消息必须明确、具体、可操作
5. **可幂等**：尽可能使操作幂等
6. **可内省**：提供探查命令

### 测试四层体系

1. 单元测试：合成数据，无外部依赖
2. E2E测试-原生：验证项目文件生成
3. E2E测试-真实后端：调用真实软件
4. CLI子进程测试：测试安装后的命令

### 渲染优先级

1. 最佳：使用应用的原生渲染器
2. 次选：构建翻译层
3. 最后：生成渲染脚本

---

## 后记

CLI-Anything是一个正在快速发展的项目。从2026年3月首次发布到现在，它已经从一个实验性的概念演变为一个包含50+软件、2280+测试、多种平台集成的成熟生态系统。

本书试图全面地记录CLI-Anything的设计理念、技术实现和工程实践。但技术在不断进步，CLI-Anything也在不断演进。我们鼓励读者：

1. **阅读源代码**：最好的文档是代码本身。CLI-Anything的代码结构清晰，注释充分，是学习的好材料。
2. **参与社区**：CLI-Anything欢迎各种形式的贡献。无论你是想为一个软件创建CLI，还是想改进插件框架，都欢迎提交PR。
3. **实践**：选择一个你熟悉的软件，按照HARNESS.md的方法论，尝试为它创建CLI接口。实践是最好的学习方式。

CLI-Anything的愿景是：**让所有软件成为Agent-Native的软件**。这个愿景的实现需要软件开发者、AI研究者和社区贡献者的共同努力。

让我们一起构建Agent-Native的未来。

---

*本书完*

*版本：v1.0 (2026年5月)*
*基于CLI-Anything项目的公开代码和文档撰写*
*许可证：Apache License 2.0*


---

## 附录H：CLI-Anything开发FAQ

### Q1: 为什么不用typer而用Click？

Click提供了`invoke_without_command=True`特性，允许无子命令时执行默认行为（进入REPL）。typer在当时不支持这个特性。此外，Click的社区更成熟，文档更完善。

### Q2: 为什么不用REST API而用CLI？

REST API需要为每个软件编写专门的HTTP服务器代码。CLI利用了软件已有的命令行接口，无需额外开发。此外，CLI是人类已经使用了50年的范式，有成熟的工具链和最佳实践。

### Q3: 为什么选择JSON作为状态格式？

JSON是人类可读的、机器可解析的、语言无关的。Python的json模块是标准库的一部分，无需额外依赖。JSON也便于调试——开发者可以直接用文本编辑器查看和修改项目文件。

### Q4: 如何处理软件版本差异？

CLI-Anything的后端包装器应该检测软件版本，并在版本不兼容时提供清晰的错误消息。对于不同版本之间的API差异，可以使用条件分支处理。

### Q5: 如何处理并发访问？

CLI-Anything使用文件锁定（fcntl.flock）防止并发写入损坏。对于更复杂的并发场景（如多Agent同时编辑），需要在应用层实现锁机制。

### Q6: 如何测试依赖外部软件的E2E测试？

E2E测试应该在安装了目标软件的环境中运行。如果软件未安装，测试应该失败（不是跳过）。这确保了CLI的实际可用性。

### Q7: 如何为闭源软件创建CLI？

CLI-Anything主要针对开源软件。对于闭源软件，如果它提供了CLI工具、脚本API或REST API，CLI-Anything仍然可以工作。如果没有任何程序化接口，CLI-Anything无法直接应用。

### Q8: 如何处理软件的GUI特有功能？

某些功能只在GUI中可用（如拖拽、视觉预览）。CLI-Anything不试图替代这些功能，而是提供等效的命令行操作。对于无法通过CLI实现的功能，应该在文档中明确说明。

---

## 附录I：CLI-Anything的局限性

### 局限性一：依赖强基础模型

CLI-Anything依赖前沿的大模型（如Claude Opus、GPT-5）来生成高质量的CLI。较弱的模型可能生成不完整或不正确的CLI，需要大量手动修正。

### 局限性二：依赖源代码

七阶段流水线分析和生成都基于源代码。对于只提供编译二进制文件的软件，CLI-Anything无法直接应用。

### 局限性三：可能需要迭代

一次`/cli-anything`运行可能无法覆盖所有功能。通常需要运行`/refine`多次来扩展CLI的覆盖范围。

### 局限性四：某些GUI功能无法替代

某些软件的核心价值在于GUI交互（如视觉设计、实时预览）。CLI-Anything为这些软件创建的CLI可能只覆盖了部分功能。

### 局限性五：性能开销

CLI-Anything的CLI层引入了额外的开销（JSON序列化、文件I/O、子进程调用）。对于性能敏感的场景，这可能是一个问题。

---

## 附录J：致谢与贡献者

### 核心团队

CLI-Anything由香港大学数据科学实验室（HKUDS）创建和维护。

### 社区贡献者

CLI-Anything的50+个CLI中，许多是由社区贡献者创建的：

- zhangxilong-43：Draw.io、Zoom、Sketch CLI
- AlexGabbia：Krita、FreeCAD CLI
- TianyuFan0504：Slay the Spire II CLI
- WEIFENG2333：VideoCaptioner CLI
- AiMiDi：LLDB、Nsight Graphics、Unreal Insights CLI
- furkankoykiran：Browser CLI
- omerarslan0：Godot CLI
- dorukozgen：Obsidian CLI
- 以及更多...

### 致谢

感谢所有为CLI-Anything贡献代码、文档和测试的社区开发者。感谢Apache 2.0许可证让这个项目能够被自由地研究和使用。

---

*附录完*


---

## 附录K：版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026年5月 | 初始版本，基于CLI-Anything 2026年4月代码库 |

---

*本书共计约10万字，涵盖41章正文和10个附录，系统性地解析了CLI-Anything项目的设计理念、技术实现和工程实践。*

*—— 完 ——*

（全书完）
