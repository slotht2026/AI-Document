# 前言

生成式人工智能（Generative AI）正在从根本上改变我们创造内容的方式。从GAN生成逼真的人脸，到扩散模型创造出令人惊叹的艺术作品，再到大语言模型写出流畅的文章和代码——生成式AI已经从学术研究走向了千家万户。

本书系统地介绍了生成式AI的核心技术，从基础理论到前沿应用。

全书分为四个部分：

**第一部分（第1-3章）** 建立基础：生成模型概论、变分自编码器、生成对抗网络。

**第二部分（第4-7章）** 深入扩散模型：基础、改进、理论分析、视频与3D生成。

**第三部分（第8-10章）** 探讨文本生成：自回归生成、文本到图像、大模型生成能力。

**第四部分（第11-13章）** 聚焦前沿：训练优化、安全伦理、未来展望。

---

*本书所有技术内容均基于公开来源的学术论文和技术报告。*
-e 

---


# 生成式AI：从GAN到扩散模型

## 书籍定位
- 体量：约100,000字
- 风格：数学推导+算法详解+应用前沿
- 读者：生成模型研究者、算法工程师、AI从业者

---

## 第一部分：基础（约15,000字）

### 第1章 生成模型概论（~5,000字）
1.1 什么是生成模型 / 1.2 生成模型的分类 / 1.3 评估指标 / 1.4 历史演进

### 第2章 变分自编码器（~5,000字）
2.1 隐变量模型 / 2.2 变分推断与ELBO / 2.3 重参数化技巧 / 2.4 VAE变体

### 第3章 生成对抗网络（~5,000字）
3.1 GAN基本原理 / 3.2 训练技巧 / 3.3 条件GAN / 3.4 GAN理论进展

## 第二部分：扩散模型（约20,000字）

### 第4章 扩散模型基础（~5,000字）
4.1 前向扩散过程 / 4.2 反向去噪过程 / 4.3 DDPM / 4.4 训练与采样

### 第5章 改进的扩散模型（~5,000字）
5.1 DDIM与确定性采样 / 5.2 分数匹配 / 5.3 Classifier-Free Guidance / 5.4 潜空间扩散

### 第6章 扩散模型的理论分析（~5,000字）
6.1 统一视角 / 6.2 采样加速 / 6.3 可控生成 / 6.4 与VAE/GAN的联系

### 第7章 视频与3D生成（~5,000字）
7.1 视频生成（SVD/Sora概念） / 7.2 3D生成 / 7.3 音频生成 / 7.4 多模态生成

## 第三部分：文本生成（约15,000字）

### 第8章 自回归文本生成（~5,000字）
8.1 语言模型与文本生成 / 8.2 解码策略 / 8.3 可控文本生成 / 8.4 评估方法

### 第9章 文本到图像生成（~5,000字）
9.1 DALL-E系列 / 9.2 Imagen / 9.3 Stable Diffusion / 9.4 条件生成与编辑

### 第10章 大模型生成能力（~5,000字）
10.1 上下文学习生成 / 10.2 代码生成 / 10.3 数学推理 / 10.4 创意写作

## 第四部分：前沿（约15,000字）

### 第11章 生成模型的训练与优化（~5,000字）
11.1 损失函数设计 / 11.2 训练稳定性 / 11.3 规模化训练 / 11.4 高效推理

### 第12章 生成模型的安全与伦理（~5,000字）
12.1 Deepfake检测 / 12.2 版权问题 / 12.3 偏见与公平 / 12.4 安全生成

### 第13章 生成式AI的未来（~5,000字）
13.1 统一生成架构 / 13.2 世界模拟器 / 13.3 个性化生成 / 13.4 结语

## 附录
- 附录A：生成模型数学基础速查
- 附录B：主流生成模型对比表
- 附录C：生成模型实验环境搭建

---

**预计总字数：~100,000字（13章 + 前言 + 附录）**
-e 

---


# 第8章 自回归文本生成

## 8.1 引言

自然语言是人类最复杂的信息载体。与图像的连续像素空间不同，文本由离散符号序列构成，这一本质差异使得文本生成面临独特的建模挑战。自回归（Autoregressive）方法是当前文本生成领域最核心的范式——它将联合概率分解为条件概率的连乘，逐词元（token）地预测下一个符号，从而完成从概率分布到自然语言的映射。

从早期的 N-gram 模型到循环神经网络（RNN），再到 Transformer 架构，自回归语言模型的演进史几乎是现代自然语言处理发展的缩影。GPT 系列、PaLM、LLaMA 等大规模预训练语言模型的成功，将自回归文本生成推向了一个新的高度：模型不仅能续写段落、翻译文本，还能进行推理、编程和多轮对话。

本章将系统介绍自回归文本生成的理论基础、解码策略、可控生成方法以及评估体系。

## 8.2 语言模型与自回归文本生成

### 8.2.1 语言模型的数学定义

语言模型的核心任务是对自然语言序列的概率分布进行建模。给定一个由 $n$ 个词元组成的序列 $\mathbf{x} = (x_1, x_2, \ldots, x_n)$，语言模型需要估计其联合概率 $P(\mathbf{x})$。根据概率论中的链式法则（Chain Rule），联合概率可以精确分解为：

$$P(\mathbf{x}) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdots = \prod_{t=1}^{n} P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

这一分解构成了自回归模型的理论基础。每个时间步 $t$ 的条件概率 $P(x_t \mid x_{<t})$ 由模型参数化为：

$$P(x_t = w \mid x_{<t}) = \frac{\exp(h_t^{\top} e_w)}{\sum_{w' \in \mathcal{V}} \exp(h_t^{\top} e_{w'})}$$

其中 $h_t \in \mathbb{R}^d$ 是模型在时间步 $t$ 的隐状态向量，$e_w \in \mathbb{R}^d$ 是词元 $w$ 的输出嵌入向量，$\mathcal{V}$ 是词表。分母对整个词表求和，本质上是一个 softmax 归一化。在大规模词表中（如 $\vert \mathcal{V} \vert = 50{,}000$ 甚至更大），这一计算的开销是显著的，因此实践中常采用分层 softmax 或自适应 softmax 等近似技术来降低计算复杂度。

### 8.2.2 从 N-gram 到 Transformer

语言模型的发展经历了几个关键阶段。

**N-gram 模型**通过马尔可夫假设将历史截断为前 $n-1$ 个词元：

$$P(x_t \mid x_{<t}) \approx P(x_t \mid x_{t-n+1}, \ldots, x_{t-1})$$

N-gram 模型简单高效，但存在严重的数据稀疏问题，且无法捕捉长距离依赖。

**循环神经网络（RNN）** 及其变体 LSTM、GRU 用隐状态 $h_t = f(h_{t-1}, x_{t-1})$ 递归地编码历史信息。理论上，RNN 可以记住任意长度的历史，但实践中梯度消失和爆炸问题限制了其建模远距离依赖的能力。此外，RNN 的顺序计算特性使其难以充分利用现代硬件的并行能力。

**Transformer**（Vaswani et al., 2017）通过自注意力机制彻底改变了这一格局。在自回归 Transformer（即 Decoder-only 架构）中，通过因果注意力掩码确保每个位置只能关注其左侧的上下文：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V$$

其中掩码矩阵 $M$ 的元素为：

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

这种架构使得每个位置的表示都能同时访问所有左侧上下文，突破了 RNN 的顺序瓶颈。GPT 系列模型正是基于这一架构，通过在大规模语料上进行自回归预训练，获得了强大的文本生成能力。

### 8.2.3 条件生成与续写

自回归文本生成不仅限于无条件采样，还可以通过条件化实现多种生成任务。条件语言模型将概率分解修改为：

$$P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{m} P(y_t \mid \mathbf{x}, y_1, y_2, \ldots, y_{t-1})$$

其中 $\mathbf{x}$ 是条件输入（如源文本、指令、前缀等），$\mathbf{y}$ 是待生成的目标序列。根据条件类型的不同，条件生成涵盖了广泛的任务：

- **文本续写**：给定前缀 $x_1, \ldots, x_k$，生成后续文本 $x_{k+1}, \ldots, x_n$
- **机器翻译**：$\mathbf{x}$ 为源语言句子，$\mathbf{y}$ 为目标语言翻译
- **摘要生成**：$\mathbf{x}$ 为长文档，$\mathbf{y}$ 为摘要
- **对话生成**：$\mathbf{x}$ 为对话历史和当前用户输入，$\mathbf{y}$ 为回复
- **指令遵循**：$\mathbf{x}$ 为自然语言指令，$\mathbf{y}$ 为执行结果

在大规模预训练语言模型（LLM）的语境下，所有这些任务都可以统一为"根据前文预测下文"的自回归范式，这就是所谓的"文本即通用接口"的思想。

## 8.3 解码策略

语言模型输出的是下一个词元的概率分布 $P(x_t \mid x_{<t})$，而最终生成的文本是一个确定性序列。从概率分布到具体序列的映射过程称为**解码（Decoding）**，不同的解码策略会显著影响生成文本的质量、多样性和可控性。

### 8.3.1 贪心解码

贪心解码（Greedy Decoding）在每个时间步选择概率最高的词元：

$$x_t = \arg\max_{w \in \mathcal{V}} P(w \mid x_{<t})$$

贪心解码是最简单直接的方法，计算开销最小。然而，它存在一个根本性缺陷：**局部最优不等于全局最优**。每一步都选择概率最高的词元，并不保证生成的完整序列具有最高的联合概率。贪心解码还容易产生退化现象——模型倾向于重复生成常见的短语模式，导致文本单调乏味。

### 8.3.2 Beam Search

束搜索（Beam Search）通过维护 $k$ 个候选序列（称为"束"或 beam）来缓解贪心解码的短视问题。在每个时间步，算法对所有 $k$ 个候选分别扩展词表中所有可能的下一个词元，得到 $k \times \vert \mathcal{V} \vert$ 个候选，然后保留得分最高的 $k$ 个继续搜索。

形式化地，束搜索的目标是最大化序列的对数概率和：

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} \sum_{t=1}^{n} \log P(x_t \mid x_{<t})$$

为了补偿不同序列长度带来的偏差，实践中常使用长度归一化：

$$\text{score}(\mathbf{x}) = \frac{1}{n^\alpha} \sum_{t=1}^{n} \log P(x_t \mid x_{<t})$$

其中 $\alpha \in [0, 1]$ 是长度惩罚因子。$\alpha = 0$ 退化为标准束搜索，$\alpha = 1$ 为完全长度归一化。

束搜索在机器翻译等需要忠实性、确定性输出的任务中表现出色。然而，在开放式生成任务中，束搜索生成的文本往往缺乏多样性和创造性，存在"退化"（degeneration）问题——生成的文本中高频短语过度重复（Holtzman et al., 2020）。

### 8.3.3 随机采样

与确定性解码不同，随机采样（Random Sampling）直接从模型输出的概率分布中随机抽取下一个词元：

$$x_t \sim P(\cdot \mid x_{<t})$$

这种方法引入了随机性，使得每次生成的结果不同，能够产生更多样化的文本。然而，原始概率分布中通常包含大量低概率的"噪声"词元，直接采样容易选中这些不合适的词元，导致生成文本质量下降、语义不连贯。

### 8.3.4 Top-k 采样

Top-k 采样（Fan et al., 2018）通过截断概率分布来减少低质量词元被选中的概率。具体做法是仅保留概率最高的 $k$ 个词元，将其概率重新归一化后进行采样：

$$P'(w \mid x_{<t}) = \begin{cases} \frac{P(w \mid x_{<t})}{\sum_{w' \in \mathcal{V}_k} P(w' \mid x_{<t})} & \text{if } w \in \mathcal{V}_k \\ 0 & \text{otherwise} \end{cases}$$

其中 $\mathcal{V}_k$ 是概率最高的 $k$ 个词元构成的集合。

Top-k 采样简单有效，但存在一个关键问题：$k$ 是固定的，无法自适应于不同的上下文。当模型非常确信下一个词元时（如"天空是___"），$k$ 可能过大，引入不必要的噪声；当模型不确定时（如"我喜欢___"），$k$ 可能过小，限制了合理的多样性。

### 8.3.5 Top-p 采样（核采样）

Top-p 采样（Holtzman et al., 2020），也称核采样（Nucleus Sampling），解决了 Top-k 采样缺乏自适应性的问题。其核心思想是动态确定截断阈值：将词元按概率从高到低排序，选择概率之和刚好超过阈值 $p$ 的最小词元集合作为采样池：

$$\mathcal{V}_p = \arg\min_{\mathcal{V}' \subseteq \mathcal{V}} \left| \mathcal{V}' \right| \quad \text{s.t.} \quad \sum_{w \in \mathcal{V}'} P(w \mid x_{<t}) \geq p$$

其中 $p \in (0, 1]$ 是预设的累积概率阈值。当概率分布比较集中时，采样池自然较小；当分布比较平坦时，采样池相应增大。Top-p 采样在实践中被广泛采用，GPT-2、GPT-3 的默认解码策略即为 Top-p 采样。

### 8.3.6 温度控制

温度（Temperature）是一种对模型输出 logits 进行缩放的技术，通过温度参数 $\tau$ 调节概率分布的"锐度"：

$$P(x_t = w \mid x_{<t}) = \frac{\exp(z_w / \tau)}{\sum_{w'} \exp(z_{w'} / \tau)}$$

其中 $z_w$ 是模型输出的原始 logit 值。温度参数的效果直观：

- $\tau \to 0$：概率分布趋于 one-hot（退化为贪心解码）
- $\tau = 1$：使用模型原始输出
- $\tau > 1$：概率分布趋于均匀（增加随机性和多样性）
- $\tau \to \infty$：完全均匀采样

温度控制可以与 Top-k 或 Top-p 采样结合使用。例如，先通过温度调整 logits，再进行 Top-p 截断和采样。在实际应用中，$\tau$ 的典型取值范围是 $[0.1, 2.0]$。

### 8.3.7 对比搜索

对比搜索（Contrastive Search，Su et al., 2022）是近年来提出的一种兼顾生成质量和多样性的解码策略。其核心思想是在选择下一个词元时，同时考虑模型置信度和与上下文的差异性：

$$x_t = \arg\max_{v \in \mathcal{V}_k} \left\{ (1 - \alpha) \cdot P(v \mid x_{<t}) - \alpha \cdot \max_{j < t} \text{sim}(h_v, h_{x_j}) \right\}$$

其中 $\mathcal{V}_k$ 是 Top-k 候选集合，$h_v$ 是候选词元 $v$ 的语义表示向量，$\text{sim}(\cdot, \cdot)$ 是余弦相似度，$\alpha \in [0, 1]$ 是平衡因子。

第一项鼓励选择模型认为高概率的词元（生成质量），第二项惩罚与已生成词元语义过于相似的候选（避免重复），两者通过 $\alpha$ 平衡。实验表明，对比搜索在困惑度（Perplexity）和生成多样性之间取得了优异的权衡，有效缓解了重复退化问题。

### 8.3.8 解码策略的比较与选择

不同解码策略各有适用场景：

| 策略 | 质量 | 多样性 | 适用场景 |
|------|------|--------|----------|
| 贪心解码 | 中 | 低 | 快速推理、确定性任务 |
| Beam Search | 高 | 低 | 翻译、摘要等忠实性任务 |
| 随机采样 | 低 | 高 | 创意写作（需配合温度） |
| Top-k 采样 | 中高 | 中高 | 对话生成 |
| Top-p 采样 | 高 | 中高 | 通用开放式生成 |
| 对比搜索 | 高 | 高 | 长文本生成、续写 |

实践中，Top-p 采样（$p = 0.9 \sim 0.95$）配合适中的温度（$\tau = 0.7 \sim 1.0$）是最常用的组合策略。

## 8.4 可控文本生成

无条件或简单条件化的自回归生成虽然强大，但在许多实际应用中，我们需要对生成文本的特定属性（如风格、情感、主题、长度等）进行精细控制。可控文本生成（Controllable Text Generation）正是在这一需求下发展起来的研究方向。

### 8.4.1 属性控制生成

属性控制生成的目标是在给定属性 $a$（如"正式的"、"积极的"、"科技类"）的条件下生成文本：

$$P(\mathbf{x} \mid a) = \prod_{t=1}^{n} P(x_t \mid x_{<t}, a)$$

**基于微调的方法**在训练阶段将属性标签作为条件输入。例如，CTRL（Keskar et al., 2019）在训练时为每个样本添加控制码（control code），生成时通过指定控制码来控制领域和风格。Conditional Transformer Language Model 将控制码拼接到输入序列的开头，使得模型在自回归生成时自然地受到属性条件的约束。

**基于提示（Prompt）的方法**利用大规模语言模型的上下文学习（In-Context Learning）能力，通过精心设计的提示来隐式指定属性。例如，在提示中加入"请用正式的语气写一封邮件"，模型便倾向于生成符合要求的文本。这种方法无需额外训练，但控制精度有限。

### 8.4.2 风格迁移

风格迁移（Style Transfer）的目标是将一段文本从源风格 $s$ 转换为目标风格 $s'$，同时保留原始内容的语义信息。形式化地：

$$\mathbf{x}' = \text{Transfer}(\mathbf{x}, s \to s')$$

**基于解码的方法**（Dathathri et al., 2020）通过在解码阶段引入额外的引导信号来实现风格控制。PPLM（Plug and Play Language Models）使用一个小型属性分类器的梯度来调节语言模型的隐状态：

$$\tilde{h}_t = h_t + \lambda \cdot \nabla_{h_t} \log P(a \mid h_t)$$

其中 $\lambda$ 控制引导强度，$P(a \mid h_t)$ 是属性分类器给出的属性概率。这种方法的优势在于无需修改语言模型本身，只需插入一个轻量级的引导模块。

**基于指令的方法**借助大规模语言模型的指令遵循能力，直接用自然语言描述迁移需求，如"将以下文本转换为幽默风格"。随着模型能力的提升，这种方法的效果已经非常接近甚至超过专门训练的风格迁移模型。

### 8.4.3 情感控制

情感控制生成是可控生成中研究最为充分的子方向之一。给定目标情感标签 $e \in \{$积极, 消极, 中性$\}$，模型需要生成具有指定情感倾向的文本。

**基于嵌入的方法**将情感嵌入向量注入到语言模型的条件生成过程中。例如，可以通过在每个时间步将情感嵌入向量与词元嵌入相加或拼接来实现：

$$\tilde{x}_t = [e_{x_t}; \mathbf{v}_e]$$

其中 $e_{x_t}$ 是词元嵌入，$\mathbf{v}_e$ 是情感嵌入向量。

**基于强化学习的方法**（Ziegler et al., 2019; Ouyang et al., 2022）使用情感分类器作为奖励模型，通过近端策略优化（PPO）等强化学习算法微调语言模型，使其生成的文本获得更高的情感奖励：

$$\mathcal{R}(\mathbf{x}) = P(e \mid \mathbf{x}) - \beta \cdot \text{KL}\left[P_\theta(\mathbf{x}) \| P_{\text{ref}}(\mathbf{x})\right]$$

其中 $P(e \mid \mathbf{x})$ 是情感分类器给出的奖励，第二项 KL 散度惩罚防止模型偏离原始预训练分布过远，$\beta$ 控制惩罚强度。这一框架（RLHF）也是 ChatGPT 等系统的核心训练范式之一。

### 8.4.4 主题控制

主题控制生成旨在引导模型生成围绕特定主题展开的文本。常用方法包括：

**关键词引导**：在生成过程中注入主题关键词作为硬约束或软引导。FUDGE（Yang and Klein, 2021）训练一个未来判别器来预测当前生成的前缀是否会导致包含目标主题的完整文本，并将判别器的输出作为解码时的附加得分：

$$\text{score}(x_t) = \log P(x_t \mid x_{<t}) + \lambda \cdot \log P(\text{topic} \mid x_{\leq t})$$

**基于检索的增强**：Retrieval-Augmented Generation（RAG）方法在生成时检索与目标主题相关的文档，将其作为上下文注入到输入中，从而引导模型生成主题相关的文本。这种方法在知识密集型任务中尤为有效。

### 8.4.5 可控生成的统一视角

从方法论的角度，可控文本生成技术可以归纳为三个层次：

1. **训练层控制**：在模型训练阶段引入控制信号（如 CTRL、RLHF）
2. **解码层控制**：在解码阶段修改或引导采样过程（如 PPLM、FUDGE、对比搜索）
3. **输入层控制**：通过设计输入提示或检索外部知识来间接控制输出（如 Prompt Engineering、RAG）

这三个层次并非互斥，实际系统中常常组合使用。例如，一个经过 RLHF 训练的模型（训练层），配合 Top-p 采样（解码层）和精心设计的系统提示（输入层），可以实现多层次的精细控制。

## 8.5 评估方法

如何客观、全面地评估生成文本的质量，是文本生成领域的核心挑战之一。由于自然语言的灵活性——同一语义可以有无数种合理表述——评估指标的设计始终存在争议。本节介绍主要的评估方法。

### 8.5.1 基于参考的自动指标

**BLEU**（Bilingual Evaluation Understudy，Papineni et al., 2002）是最经典的自动评估指标，最初为机器翻译设计。它基于 n-gram 精度的几何平均：

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

其中 $p_n$ 是 n-gram 精度，$w_n = 1/N$ 是权重（通常 $N = 4$），$BP$ 是简短惩罚（Brevity Penalty）：

$$BP = \begin{cases} 1 & \text{if } c > r \\ \exp(1 - r/c) & \text{if } c \leq r \end{cases}$$

其中 $c$ 是生成文本长度，$r$ 是参考文本长度。BLEU 衡量的是生成文本与参考文本之间的 n-gram 重叠程度。其优点是计算快速、可复现，但缺点是无法捕捉语义等价性——一个与参考文本语义相同但用词不同的句子可能获得很低的 BLEU 分数。

**ROUGE**（Recall-Oriented Understudy for Gisting Evaluation，Lin, 2004）是一组面向摘要评估的指标，主要有三个变体：

- **ROUGE-N**：基于 n-gram 召回率

$$\text{ROUGE-N} = \frac{\sum_{\text{ref}} \sum_{\text{n-gram} \in \text{ref}} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{\text{ref}} \sum_{\text{n-gram} \in \text{ref}} \text{Count}(\text{n-gram})}$$

- **ROUGE-L**：基于最长公共子序列（LCS）

$$\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot R_l \cdot P_l}{R_l + \beta^2 \cdot P_l}$$

其中 $R_l = \text{LCS}(X, Y) / |X|$，$P_l = \text{LCS}(X, Y) / |Y|$，$\beta$ 通常取 1.2。

- **ROUGE-W**：基于加权最长公共子序列，对连续匹配给予更高权重

ROUGE 的召回导向设计使其更关注参考信息的覆盖率，在摘要任务中被广泛采用。

**METEOR**（Banerjee and Lavie, 2005）通过引入同义词匹配、词干匹配和释义匹配来弥补 BLEU 的不足，与人类判断的相关性更高。**CIDEr**（Vedantam et al., 2015）使用 TF-IDF 加权的 n-gram 相似度，最初为图像描述评估设计，也适用于文本生成。**BERTScore**（Zhang et al., 2020）利用预训练语言模型的上下文嵌入计算生成文本与参考文本之间的语义相似度：

$$\text{BERTScore} = F_1 \cdot \frac{2 \cdot P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

其中精度和召回率通过 BERT 嵌入的余弦相似度计算。BERTScore 能更好地捕捉语义层面的相似性，是当前最先进的基于参考的自动指标之一。

### 8.5.2 困惑度

困惑度（Perplexity，PPL）是语言模型最常用的内在评估指标，衡量模型对测试数据的预测能力：

$$\text{PPL}(\mathbf{x}) = \exp\left(-\frac{1}{n} \sum_{t=1}^{n} \log P(x_t \mid x_{<t})\right)$$

直觉上，困惑度表示模型在每个位置平均"犹豫"在多少个词元之间。困惑度越低，说明模型对测试数据的预测越自信、越准确。

困惑度作为语言模型质量的代理指标具有重要价值，但也有明显的局限性。低困惑度并不一定意味着生成质量高——模型可能只是学会了复制训练数据中的高频模式。此外，不同模型、不同词表之间直接比较困惑度是不合理的，因为分词方式的差异会直接影响困惑度的数值。

### 8.5.3 多样性指标

生成文本的多样性是评估解码策略和模型质量的重要维度。常用的多样性指标包括：

**Distinct-N**（Li et al., 2016）计算生成文本中不同的 n-gram 占总 n-gram 数量的比例：

$$\text{Distinct-N} = \frac{|\{\text{unique n-grams}\}|}{|\{\text{all n-grams}\}|}$$

**Self-BLEU**（Zhu et al., 2018）通过计算生成样本之间的 BLEU 分数来衡量多样性。Self-BLEU 越低，说明生成的文本越多样：

$$\text{Self-BLEU} = \frac{1}{K} \sum_{i=1}^{K} \text{BLEU}(\mathbf{x}_i, \{\mathbf{x}_j\}_{j \neq i})$$

**熵（Entropy）** 衡量生成文本中 n-gram 分布的均匀程度：

$$H = -\sum_{g \in \mathcal{G}} P(g) \log P(g)$$

其中 $\mathcal{G}$ 是所有 n-gram 的集合，$P(g)$ 是 n-gram $g$ 在生成文本中出现的相对频率。较高的熵意味着更均匀的 n-gram 分布，即更高的多样性。

### 8.5.4 人工评估

尽管自动指标提供了便捷的量化评估手段，人工评估仍然是文本生成质量的黄金标准。人工评估通常从以下几个维度展开：

- **流畅性（Fluency）**：文本是否通顺、自然，语法是否正确
- **相关性（Relevance）**：文本是否与输入条件或上下文相关
- **信息量（Informativeness）**：文本是否包含有价值的、非冗余的信息
- **一致性（Coherence）**：文本内部的逻辑是否连贯
- **忠实性（Faithfulness）**：文本是否忠实于输入信息（对摘要、翻译等任务）

人工评估的方法包括绝对评分（如 Likert 量表）、相对排序（如 Elo 评分）和 A/B 测试。Chatbot Arena（Zheng et al., 2023）通过众包 A/B 测试建立了 LLM 的 Elo 排名系统，成为当前最有影响力的人工评估基准之一。

人工评估的主要挑战在于成本高、周期长、主观性强且难以复现。评估者之间的不一致性（inter-annotator disagreement）也是固有问题。因此，研究社区一直在探索自动指标与人工评估之间更好的相关性。

### 8.5.5 基于模型的评估

近年来，利用大语言模型本身作为评估器（LLM-as-a-Judge）的方法获得了广泛关注。Zheng et al.（2023）提出使用 GPT-4 等强模型对生成文本进行评分或排序，实验表明这种方法与人工评估具有较高的一致性。然而，这种评估方法也存在位置偏差（position bias）、冗长偏差（verbosity bias）和自我偏好（self-preference）等已知问题。

## 8.6 本章小结

自回归文本生成是当前生成式人工智能最活跃、影响最深远的研究方向之一。本章从语言模型的数学定义出发，介绍了从 N-gram 到 Transformer 的技术演进，系统梳理了贪心解码、束搜索、随机采样、Top-k/Top-p 采样、温度控制和对比搜索等主要解码策略的设计思想与适用场景。在可控生成方面，我们从训练层、解码层和输入层三个层次分析了属性控制、风格迁移、情感控制和主题控制等方法。最后，我们讨论了 BLEU、ROUGE、困惑度等自动指标以及人工评估的方法论。

自回归文本生成的研究仍在快速演进中。更大规模的模型、更精细的对齐技术（如 RLHF、DPO）、更高效的推理方法（如推测解码、稀疏注意力）以及更可靠的评估体系，都是当前和未来的重要研究方向。理解本章介绍的基础概念和核心方法，将为读者进入这些前沿领域奠定坚实的基础。

---

**参考文献**

- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*.
- Brown, T., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.
- Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration. *ICLR*.
- Fan, A., et al. (2018). Hierarchical Neural Story Generation. *ACL*.
- Su, Y., et al. (2022). A Contrastive Framework for Neural Text Generation. *NeurIPS*.
- Keskar, N., et al. (2019). CTRL: A Conditional Transformer Language Model for Controllable Generation. *arXiv*.
- Dathathri, S., et al. (2020). Plug and Play Language Models: A Simple Approach to Controlled Text Generation. *ICLR*.
- Yang, K. and Klein, D. (2021). FUDGE: Controlled Text Generation With Future Discriminators. *NAACL*.
- Ziegler, D., et al. (2019). Fine-Tuning Language Models from Human Preferences. *arXiv*.
- Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*.
- Papineni, K., et al. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. *ACL*.
- Lin, C. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop*.
- Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR*.
- Li, J., et al. (2016). A Diversity-Promoting Objective Function for Neural Conversation Models. *NAACL*.
- Zhu, Y., et al. (2018). Texygen: A Benchmarking Platform for Text Generation Models. *SIGIR*.
- Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS*.
- Su, Y., et al. (2022). Contrastive Search Is What You Need For Neural Text Generation. *TMLR*.
-e 

---


# 第9章 文本到图像生成

## 9.1 引言

文本到图像生成（Text-to-Image Generation）是生成式人工智能中最具视觉冲击力的任务之一：给定一段自然语言描述，模型自动合成与之语义一致的高保真图像。这一任务横跨计算机视觉与自然语言处理两大领域，要求模型同时具备对语言的深层理解和对视觉世界的精细建模能力。

从技术演进的角度看，文本到图像生成经历了三个关键阶段。早期方法主要依赖于生成对抗网络（GAN）的条件变体，通过将文本嵌入注入生成器来实现文本条件控制，但受限于 GAN 的训练不稳定性与模式坍缩问题，生成图像的分辨率和多样性均受到较大限制。2021 年，OpenAI 发布的 DALL-E 将自回归 Transformer 与 CLIP 表征相结合，首次展示了大规模语言-视觉预训练在文本到图像生成中的巨大潜力。随后，扩散模型的崛起带来了范式性变革——Google 的 Imagen 和 Stability AI 的 Stable Diffusion 分别从级联扩散与潜空间扩散两条技术路线，将文本到图像生成的质量推向了前所未有的高度。

本章将系统介绍这一领域的核心技术。§9.2 讲述 DALL-E 系列的技术路线，涵盖 CLIP 引导机制与自回归+扩散两阶段架构；§9.3 分析 Imagen 的级联扩散框架及其在大语言模型引导下的创新；§9.4 深入讨论 Stable Diffusion 的潜空间扩散、Classifier-Free Guidance 与开源生态；§9.5 则聚焦于条件生成与编辑的前沿方法，包括 InstructPix2Pix、ControlNet、IP-Adapter 以及图像修复与编辑技术。

---

## 9.2 DALL-E 系列：从自回归到扩散的范式跃迁

### 9.2.1 DALL-E：自回归 Transformer 与 CLIP 引导

DALL-E（Ramesh et al., 2021）是首个将大规模自回归模型应用于文本到图像生成的里程碑式工作。其核心思想是将图像和文本统一编码为离散 token 序列，然后用一个自回归 Transformer 对联合分布进行建模。

**离散视觉词表的构建。** DALL-E 使用离散变分自编码器（dVAE）将 $256 \times 256$ 的图像映射为 $32 \times 32$ 的离散 token 网格。dVAE 的编码器 $E$ 将图像 patch 映射到一个大小为 $K=8192$ 的码本（codebook）中的离散索引：

$$z_{i,j} = \arg\max_{k \in \{1,\ldots,K\}} \log q_\phi(k \mid x_{i,j})$$

其中 $q_\phi$ 是编码器输出的类别分布。解码器 $D$ 则从离散 token 重建图像，训练目标为重构损失与 KL 散度的加权和：

$$\mathcal{L}_{\text{dVAE}} = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x \mid z)\right] - \beta \cdot D_{\text{KL}}\left(q_\phi(z \mid x) \| p(z)\right)$$

**自回归 Transformer 的训练。** 给定文本描述 $y$，DALL-E 首先使用 BPE 编码器将其转化为最多 256 个文本 token，再与 1024 个图像 token 拼接为一个长度为 1280 的序列。一个包含 64 层 Transformer 块（隐藏维度 6144，注意力头数 62，总参数量约 120 亿）的自回归模型以自左向右、自上而下的顺序对联合分布进行建模：

$$p_\theta(x, y) = p_\theta(y) \prod_{t=1}^{T_{\text{img}}} p_\theta(x_t \mid y, x_{<t})$$

训练时，文本 token 的预测损失被屏蔽，仅对图像 token 计算交叉熵损失。

**CLIP 重排序（Reranking）。** 自回归模型在推理时通过 nucleus sampling 生成多个候选图像。DALL-E 引入了一个预训练的 CLIP 模型来对候选图像进行评分。CLIP 的文本编码器和图像编码器分别将文本 $y$ 和生成图像 $\hat{x}$ 映射到一个共享的嵌入空间，余弦相似度作为匹配分数：

$$s(y, \hat{x}) = \frac{f_{\text{CLIP}}^{\text{img}}(\hat{x})^\top f_{\text{CLIP}}^{\text{txt}}(y)}{\|f_{\text{CLIP}}^{\text{img}}(\hat{x})\| \cdot \|f_{\text{CLIP}}^{\text{txt}}(y)\|}$$

最终选择相似度最高的图像作为输出。这一机制有效地将语义对齐信息注入生成过程，显著提升了文本-图像一致性。

### 9.2.2 DALL-E 2：CLIP 引导的扩散生成

DALL-E 2（Ramesh et al., 2022）在架构上做出了根本性转变：放弃了纯自回归范式，转向以扩散模型为核心的生成流程。其整体框架包含两个关键模块：先验网络（Prior）和图像解码器（Decoder）。

**CLIP 嵌入空间中的先验学习。** 给定文本 $y$，CLIP 文本编码器生成文本嵌入 $T = f_{\text{CLIP}}^{\text{txt}}(y)$。先验网络的目标是从 $T$ 出发，生成对应的 CLIP 图像嵌入 $I = f_{\text{CLIP}}^{\text{img}}(x)$。DALL-E 2 探索了两种先验实现：自回归先验和扩散先验。扩散先验以加噪后的图像嵌入 $\tilde{I}_t$ 为输入，以文本嵌入 $T$ 为条件，学习去噪过程：

$$p_\psi(I \mid T) = \int p_\psi(I_{0:T} \mid T) \, dI_{1:T}, \quad \text{其中 } p_\psi(I_{t-1} \mid I_t, T) = \mathcal{N}(I_{t-1}; \mu_\psi(I_t, T, t), \sigma_t^2 I)$$

扩散先验的训练目标简化为：

$$\mathcal{L}_{\text{prior}} = \mathbb{E}_{t, I_0, \epsilon}\left[\|\epsilon - \epsilon_\psi(\tilde{I}_t, T, t)\|^2\right]$$

其中 $\epsilon \sim \mathcal{N}(0, I)$ 为添加的噪声，$\epsilon_\psi$ 为噪声预测网络。

**级联图像解码器。** 获得 CLIP 图像嵌入 $I$ 后，DALL-E 2 使用一个级联扩散模型将其解码为图像。该级联由三阶段组成：$64 \times 64$ 的基础扩散模型、$64 \to 256$ 的上采样扩散模型、$256 \to 1024$ 的上采样扩散模型。每一阶段都是一个条件扩散模型，以图像嵌入 $I$ 和（可选的）文本描述作为条件。

这种设计的精妙之处在于：CLIP 嵌入空间充当了一个高度语义化的"中间表示"，先验网络负责语义层面的"构思"，而级联解码器则负责将这一语义构思逐步"绘制"为高分辨率图像。与 DALL-E 相比，DALL-E 2 在图像质量、多样性和文本对齐度上均取得了质的飞跃。

### 9.2.3 DALL-E 3：合成描述与对齐增强

DALL-E 3（Betker et al., 2023）将研究重点从模型架构转向了数据质量与文本-图像对齐。其核心创新在于"合成描述"（synthetic captions）策略：使用一个专门训练的图像描述模型为训练集中的每张图像生成详细、准确的文本描述，替代原始的噪声标注（如 alt-text）。

实验表明，仅通过改进训练数据的文本描述质量——而不改变模型架构或训练流程——即可大幅提升生成图像对文本的遵循程度。DALL-E 3 还引入了"prompt 重写"机制：在推理时，使用一个语言模型将用户的原始输入重写为更详细、更结构化的描述，从而更好地激发模型的生成能力。

这一工作揭示了一个深刻洞见：在大规模模型时代，**数据质量可能比模型架构更关键**。

---

## 9.3 Imagen：大语言模型引导的级联扩散

### 9.3.1 核心架构

Imagen（Saharia et al., 2022）由 Google Research 提出，其核心主张是：**冻结的大语言模型在文本理解能力上远超专门训练的视觉-语言对比模型**，因此应将其作为文本编码器用于图像生成。

Imagen 的文本编码器采用 T5-XXL（约 46 亿参数），这是一个纯文本的编码器-解码器 Transformer，预训练于大规模文本语料之上。给定文本 $y$，T5-XXL 的编码器输出一系列上下文化文本嵌入 $c = \text{T5}(y) \in \mathbb{R}^{L \times d}$，其中 $L$ 为序列长度，$d$ 为隐藏维度。这些嵌入通过交叉注意力机制注入扩散 U-Net 的各层。

**扩散模型架构。** Imagen 的基础模型在 $64 \times 64$ 分辨率上进行训练，采用 U-Net 架构，通过交叉注意力层融合文本条件：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中 $Q$ 来自图像特征的线性投影，$K$ 和 $V$ 来自文本嵌入 $c$ 的线性投影。

**级联超分辨率。** 基础模型生成 $64 \times 64$ 图像后，通过两级超分辨率模型逐步上采样至 $256 \times 256$ 和 $1024 \times 1024$。超分辨率模型以低分辨率图像的条件增强（conditioning augmentation）版本为输入，以文本嵌入为条件：

$$x_{\text{low-res}}^{\text{aug}} = x_{\text{low-res}} + \eta, \quad \eta \sim \mathcal{N}(0, \sigma_{\text{aug}}^2 I)$$

条件增强通过向低分辨率输入添加高斯噪声来缓解训练-推理分布偏移。

### 9.3.2 动态阈值与高引导权重

Imagen 的一个关键技术贡献是**动态阈值**（Dynamic Thresholding）。在 Classifier Guidance 中，引导采样的去噪均值修正为：

$$\hat{\mu}_\theta(x_t, t, c) = \mu_\theta(x_t, t, c) + s \cdot \sigma_t^2 \nabla_{x_t} \log p_\phi(y \mid x_t)$$

其中 $s$ 为引导强度。当 $s$ 较大时（Imagen 使用 $s=30$），生成的均值会超出训练时见过的范围，导致图像出现过饱和等伪影。动态阈值的解决方案是：在每一步采样中，将像素值裁剪到 $[-s, s]$ 范围后，再缩放回 $[-1, 1]$：

$$\hat{x}_0^{(\text{clipped})} = \text{clip}(\hat{x}_0, -s, s), \quad \hat{x}_0^{(\text{rescaled})} = \frac{\hat{x}_0^{(\text{clipped})}}{s}$$

其中 $s$ 为当前 $|\hat{x}_0|$ 的某个百分位数（如 0.995）。这一技巧使得模型能够在高引导权重下稳定生成，显著提升图像的保真度和文本对齐度。

### 9.3.3 评估基准：DrawBench

Imagen 团队同时提出了 DrawBench——一个包含 200 个文本提示的结构化评估基准，涵盖组合性（compositionality）、计数（counting）、空间关系（spatial relations）、属性绑定（attribute binding）等多个维度。人类评估实验表明，Imagen 在 DrawBench 上的文本对齐度和图像质量均显著优于 DALL-E 2。

---

## 9.4 Stable Diffusion：潜空间扩散与开源生态

### 9.4.1 潜空间扩散模型（LDM）

Stable Diffusion（Rombach et al., 2022）的核心思想源自潜在扩散模型（Latent Diffusion Model, LDM）。与直接在像素空间进行扩散不同，LDM 首先通过一个预训练的自编码器（Autoencoder）将图像压缩到低维潜空间，再在潜空间中执行扩散过程。

**自编码器。** 给定图像 $x \in \mathbb{R}^{H \times W \times 3}$，编码器 $\mathcal{E}$ 将其压缩为潜变量 $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$，其中空间压缩比为 $f = H/h = W/w$（通常 $f=8$）。解码器 $\mathcal{D}$ 从潜变量重建图像 $\tilde{x} = \mathcal{D}(z)$。自编码器的训练目标包含重构损失、感知损失和对抗损失：

$$\mathcal{L}_{\text{AE}} = \underbrace{\|x - \tilde{x}\|_1}_{\text{重构}} + \underbrace{\mathcal{L}_{\text{LPIPS}}(x, \tilde{x})}_{\text{感知}} + \underbrace{\mathcal{L}_{\text{GAN}}(\tilde{x})}_{\text{对抗}}$$

**潜空间中的扩散。** 在潜空间 $z$ 上，前向扩散过程为：

$$z_t = \sqrt{\bar{\alpha}_t} \, z_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

去噪网络 $\epsilon_\theta(z_t, t, c)$ 预测噪声，训练目标为简化的去噪目标：

$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{z_0, t, \epsilon, c}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right]$$

其中 $c$ 为文本条件。通过在低维潜空间（如 $64 \times 64 \times 4$）而非像素空间（如 $512 \times 512 \times 3$）进行扩散，计算量降低了约 $f^2 = 64$ 倍，使得在消费级硬件上训练和推理成为可能。

### 9.4.2 文本条件注入：CLIP 文本编码器与交叉注意力

Stable Diffusion 使用 CLIP 的文本编码器（具体为 OpenCLIP ViT-H/14 或 CLIP ViT-L/14）将文本提示编码为嵌入序列 $c = f_{\text{CLIP}}^{\text{txt}}(y) \in \mathbb{R}^{L \times d}$。这些嵌入通过交叉注意力机制注入 U-Net 的中间层。U-Net 中的每个 Transformer 块包含三个注意力操作：自注意力（处理空间关系）、交叉注意力（融合文本条件）和前馈网络。

交叉注意力的计算为：

$$z_{\text{out}} = \text{softmax}\left(\frac{(z W_Q)(c W_K)^\top}{\sqrt{d}}\right)(c W_V)$$

其中 $z$ 为图像潜特征，$W_Q, W_K, W_V$ 为可学习的投影矩阵。这种机制允许模型在空间维度上自适应地关注文本描述的不同部分。

### 9.4.3 Classifier-Free Guidance

Classifier-Free Guidance（CFG）（Ho & Salimans, 2022）是提升条件生成质量的核心技术。其思想是同时训练一个有条件模型和一个无条件模型，然后在采样时通过外插来增强条件效果。

在训练时，以概率 $p_{\text{uncond}}$（通常为 10%）随机丢弃条件 $c$，将其替换为空条件 $\varnothing$（如空文本的嵌入）。去噪网络 $\epsilon_\theta(z_t, t, c)$ 在有条件和无条件两种模式下共享所有参数。在采样时，CFG 的去噪预测为：

$$\hat{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \varnothing) + w \cdot \left[\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \varnothing)\right]$$

其中 $w > 0$ 为引导强度（guidance scale）。当 $w = 1$ 时，等价于标准的条件生成；当 $w > 1$ 时，模型会"过度"遵循条件，生成更锐利、更符合文本描述的图像，但多样性会降低。

从贝叶斯视角，CFG 可以理解为对后验分布的温度缩放：

$$\log \hat{p}(z_t \mid c) \propto \log p(z_t) + w \cdot \log \frac{p(c \mid z_t)}{p(c)}$$

即放大了条件似然项的贡献。

### 9.4.4 采样器与加速

Stable Diffusion 的推理效率高度依赖于采样器的选择。DDPM 原始采样需要数百至数千步去噪，速度极慢。后续研究提出了多种加速采样策略：

- **DDIM**（Song et al., 2021）：将扩散过程推广为非马尔可夫过程，允许跳步采样，将步数从 1000 降至 50-200 步。其确定性采样公式为：

$$z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \frac{z_t - \sqrt{\bar{\alpha}_t}\hat{x}_0}{\sqrt{1 - \bar{\alpha}_t}} + \sigma_t \epsilon$$

- **DPM-Solver**（Lu et al., 2022）：利用扩散 ODE 的高阶求解器，在 10-20 步内即可获得高质量结果。
- **一致性模型**（Song et al., 2023）：通过蒸馏将多步扩散映射为单步生成，实现极低延迟推理。

### 9.4.5 开源生态与社区驱动

Stable Diffusion 最深远的影响在于其**开源性**。Stability AI 将模型权重、训练代码和推理代码完全公开，催生了一个庞大的开源生态：

- **SD 1.x / SD 2.x**：基于 CLIP ViT-L/14 和 OpenCLIP ViT-H/14 的基础模型，支持 $512 \times 512$ 和 $768 \times 768$ 生成。
- **SDXL**（Podell et al., 2023）：采用双 U-Net 架构（基础模型 + 精炼模型）、更大的 CLIP 文本编码器和 OpenCLIP，支持 $1024 \times 1024$ 原生分辨率。
- **LoRA 微调**：低秩适应（Low-Rank Adaptation）允许用户在消费级 GPU 上对模型进行轻量级微调，社区产生了数以万计的风格化和角色微调模型。
- **ComfyUI / AUTOMATIC1111**：可视化工作流和 Web 界面降低了使用门槛。

这种开源策略使得文本到图像生成从少数机构的实验室研究，迅速演变为全球数十亿用户可触达的创作工具。

---

## 9.5 条件生成与编辑

### 9.5.1 InstructPix2Pix：指令驱动的图像编辑

InstructPix2Pix（Brooks et al., 2023）提出了一种以自然语言指令驱动图像编辑的方法。给定输入图像 $x$ 和编辑指令（如"将天空变成夕阳"），模型直接输出编辑后的图像 $y$。

**训练数据构建。** 该方法的创新在于训练数据的自动生成流程。首先使用 GPT-3 生成大量（输入描述、编辑指令、输出描述）三元组，然后利用 Stable Diffusion 和 Prompt-to-Prompt 生成对应的（输入图像、输出图像）对。通过这种"AI 生成训练数据"的范式，构建了超过 100 万条训练样本。

**模型架构。** InstructPix2Pix 基于 Stable Diffusion 进行微调。输入图像通过 VAE 编码器编码为潜变量 $z_x = \mathcal{E}(x)$，然后与扩散噪声拼接为 $[z_t, z_x]$（在通道维度拼接）作为 U-Net 的输入。文本指令通过 CLIP 文本编码器编码后注入交叉注意力层。训练目标为：

$$\mathcal{L} = \mathbb{E}_{z_0, z_x, t, \epsilon, c}\left[\|\epsilon - \epsilon_\theta([z_t, z_x], t, c)\|^2\right]$$

推理时，引入两个关键的引导强度参数：图像条件强度 $s_i$ 和文本条件强度 $s_t$，以平衡编辑幅度与保真度：

$$\hat{\epsilon} = \epsilon_\theta(z_t, t, \varnothing) + s_i \cdot [\epsilon_\theta([z_t, z_x], t, \varnothing) - \epsilon_\theta(z_t, t, \varnothing)] + s_t \cdot [\epsilon_\theta([z_t, z_x], t, c) - \epsilon_\theta([z_t, z_x], t, \varnothing)]$$

### 9.5.2 ControlNet：空间条件的精细控制

ControlNet（Zhang & Agrawala, 2023）解决了文本提示在空间控制上的局限性。文本描述虽然可以指定语义内容，但难以精确控制物体的姿势、边缘结构或深度布局。ControlNet 通过引入额外的空间条件信号（如边缘图、深度图、人体姿势骨架等）来弥补这一不足。

**架构设计。** ControlNet 在冻结的 Stable Diffusion U-Net 基础上，复制了一份编码器部分的权重作为可训练的"控制分支"。给定空间条件 $c_s$（如 Canny 边缘图），控制分支处理 $c_s$ 并通过零卷积（zero convolution）将特征注入主 U-Net 的对应层：

$$y_c = \text{ControlNet}(z_t, t, c_s)$$
$$\hat{\epsilon}_\theta = \epsilon_\theta(z_t, t, c_{\text{text}}) + y_c$$

零卷积的权重和偏置初始化为零：

$$y = W \cdot x + b, \quad W = 0, \quad b = 0$$

这保证了训练初始阶段控制分支不产生任何输出，不会破坏预训练模型的能力，训练过程稳定且收敛快。

**多种空间条件。** ControlNet 的通用框架支持多种空间条件输入：

| 条件类型 | 提取方法 | 控制能力 |
|---------|---------|---------|
| Canny 边缘 | Canny 边缘检测器 | 物体轮廓和结构 |
| 深度图 | MiDaS / Depth Anything | 三维空间布局 |
| 人体姿势 | OpenPose 关键点检测 | 人体动作和姿态 |
| 法线图 | 粗糙法线估计 | 表面朝向和细节 |
| 语义分割 | 语义分割模型 | 场景布局和区域 |

### 9.5.3 IP-Adapter：图像提示适配器

IP-Adapter（Ye et al., 2023）探索了"图像提示"（image prompt）这一新的交互范式：用户除了提供文本提示外，还可以提供一张参考图像，模型将同时遵循文本和图像的语义来生成新图像。

**解耦的交叉注意力机制。** IP-Adapter 的核心创新在于将文本条件和图像条件的注入解耦。在标准的交叉注意力中，文本嵌入同时提供 $K$ 和 $V$。IP-Adapter 新增了一组独立的图像交叉注意力层：

$$z_{\text{text}} = \text{Attn}(Q_z, K_t, V_t), \quad z_{\text{image}} = \text{Attn}(Q_z, K_i, V_i)$$

其中 $K_t, V_t$ 来自文本嵌入，$K_i, V_i$ 来自 CLIP 图像编码器提取的图像嵌入。两者的输出相加：

$$z_{\text{out}} = z_{\text{text}} + z_{\text{image}}$$

新增的图像注意力层从零初始化，不破坏原始文本条件能力。推理时，可通过权重系数 $\lambda$ 调节图像条件的影响强度。

**应用场景。** IP-Adapter 在风格迁移、角色一致性保持、产品设计等领域展现出强大能力，成为 ComfyUI 等工具中使用最广泛的组件之一。

### 9.5.4 图像修复与编辑

图像修复（inpainting）是文本到图像生成的重要下游应用，目标是根据文本描述和掩码区域，在指定区域内生成新的内容。

**Stable Diffusion Inpainting。** 最直接的方法是将修复问题转化为条件生成问题：将原始图像的未掩码区域和掩码信息（二值掩码）与噪声潜变量拼接，作为 U-Net 的输入：

$$z_{\text{input}} = [z_t, \mathcal{E}(x \odot (1-m)), m]$$

其中 $m$ 为二值掩码（1 表示需要编辑的区域），$\odot$ 为逐元素乘法。模型学习在掩码区域内生成与周围区域协调的内容。Stable Diffusion Inpainting 通过在大量合成修复数据上微调，实现了高质量的区域编辑。

**SDEdit。** SDEdit（Meng et al., 2022）提供了一种更通用的编辑框架：将用户编辑（如涂鸦、粗略修改）视为"不完美"的中间结果，通过在适当的时间步添加噪声再进行去噪，实现编辑结果的"自然化"。编辑幅度可通过添加噪声的时间步 $t_0$ 控制：$t_0$ 越大，编辑越自由但保真度越低；$t_0$ 越小，保真度越高但编辑灵活性越差。

**Null-text Inversion。** 对于真实图像的编辑，需要将输入图像反演（inversion）到扩散过程的噪声空间。DDIM Inversion 是常用的方法，通过反向运行 DDIM 采样来获取初始噪声。然而，反演误差会累积导致重建质量下降。Null-text Inversion（Mokady et al., 2023）通过优化无条件嵌入 $\varnothing$（而非模型参数）来最小化每一步的反演误差：

$$\varnothing_t^* = \arg\min_{\varnothing} \|\hat{x}_0(z_t, c, \varnothing) - x_0^*\|^2$$

其中 $x_0^*$ 为参考图像的估计。优化后的 null-text 向量在不同时间步取不同值，精确地将输入图像映射到扩散模型的生成流形上，实现了高保真的真实图像编辑。

---

## 9.6 本章小结

本章系统介绍了文本到图像生成领域的核心方法与技术演进。从 DALL-E 的自回归范式，到 DALL-E 2 的 CLIP 引导扩散，再到 Imagen 的大语言模型引导级联扩散和 Stable Diffusion 的潜空间扩散，每一步进展都伴随着关键的技术突破：CLIP 嵌入空间的语义对齐、大语言模型的深层文本理解、潜空间压缩的计算效率以及 Classifier-Free Guidance 的条件增强。

条件生成与编辑技术（InstructPix2Pix、ControlNet、IP-Adapter）则将文本到图像生成从"一次性生成"推进到"可控编辑"的阶段，使用户能够以前所未有的精细度控制生成过程。这些技术共同构成了当前文本到图像生成的技术栈，广泛应用于创意设计、游戏开发、广告制作和数字艺术等领域。

展望未来，文本到图像生成面临的核心挑战包括：（1）**组合性推理**——精确理解"红色的猫坐在蓝色的椅子下面"这类涉及物体、属性和空间关系的复合描述；（2）**世界知识**——融入更深层的物理和常识推理能力；（3）**可控性与安全性**——在赋予用户创造力的同时防止滥用；（4）**视频与三维扩展**——将文本到图像的能力自然延伸到视频和三维场景生成。这些方向的研究正在快速推进，有望在不远的将来取得重要突破。

---

## 参考文献

- Betker, J., et al. (2023). *Improving Image Generation with Better Captions*. OpenAI Technical Report.
- Brooks, T., Holynski, A., & Efros, A. A. (2023). InstructPix2Pix: Learning to Follow Image Editing Instructions. *CVPR*.
- Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *NeurIPS Workshop*.
- Lu, C., et al. (2022). DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. *NeurIPS*.
- Meng, C., et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. *ICLR*.
- Mokady, R., et al. (2023). Null-text Inversion for Editing Real Images using Guided Diffusion Models. *CVPR*.
- Podell, D., et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. *arXiv:2307.01952*.
- Ramesh, A., et al. (2021). Zero-Shot Text-to-Image Generation. *ICML*.
- Ramesh, A., et al. (2022). Hierarchical Text-Conditional Image Generation with CLIP Latents. *arXiv:2204.06125*.
- Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR*.
- Saharia, C., et al. (2022). Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. *NeurIPS*.
- Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *ICLR*.
- Song, Y., et al. (2023). Consistency Models. *ICML*.
- Ye, H., et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. *arXiv:2308.06721*.
- Zhang, L., & Agrawala, M. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. *ICCV*.
-e 

---


# 第1章 生成模型概论

## 1.1 什么是生成模型

### 1.1.1 生成任务的定义

生成模型的核心目标是学习数据的分布 $p_{\text{data}}(x)$，并从中采样生成新的、逼真的数据样本。与判别模型（学习 $p(y|x)$，即给定输入预测标签）不同，生成模型试图理解数据本身是如何"产生"的。

形式化地说，给定一组独立同分布的训练样本 $\{x_1, x_2, \ldots, x_N\} \sim p_{\text{data}}(x)$，生成模型的目标是：
1. **密度估计**：学习一个参数化模型 $p_\theta(x)$ 来近似 $p_{\text{data}}(x)$
2. **采样生成**：从 $p_\theta(x)$ 中高效采样，生成新的逼真样本
3. **表示学习**：发现数据的潜在结构和低维表示

### 1.1.2 生成模型的应用

生成式AI的应用范围极其广泛：

**图像生成**：从随机噪声生成逼真的人脸、风景、艺术品。代表系统包括StyleGAN、DALL-E、Stable Diffusion、Midjourney等。

**文本生成**：自动写作、对话、翻译、摘要。大语言模型（GPT-4、Claude等）展示了惊人的文本生成能力。

**代码生成**：根据自然语言描述自动生成程序代码。GitHub Copilot、Cursor等工具已经改变了软件开发的方式。

**音频与音乐生成**：语音合成（TTS）、音乐创作、音效生成。

**视频生成**：从文本或图像生成视频片段。Sora等系统展示了视频生成的巨大潜力。

**3D生成**：从文本或图像生成3D模型和场景。

**科学应用**：蛋白质结构预测（AlphaFold）、药物分子设计、材料发现。

## 1.2 生成模型的分类

### 1.2.1 按建模方式分类

**显式密度模型**：显式地定义和计算 $p_\theta(x)$。
- 自回归模型：$p_\theta(x) = \prod_i p_\theta(x_i | x_{<i})$
- 变分自编码器（VAE）：通过隐变量 $z$ 建模 $p_\theta(x) = \int p_\theta(x|z)p(z)dz$
- 流模型（Normalizing Flows）：通过可逆变换精确计算密度
- 扩散模型：通过逐步去噪过程建模

**隐式密度模型**：不显式计算密度，但能从模型中采样。
- 生成对抗网络（GAN）：通过对抗训练学习生成器

### 1.2.2 按生成方式分类

**自回归生成**：逐元素生成（如逐词生成文本、逐像素生成图像）。优点是建模灵活，缺点是采样速度慢。

**并行生成**：一次生成所有元素（如GAN、扩散模型的某些变体）。优点是速度快，缺点是可能缺乏全局一致性。

**迭代生成**：通过多步迭代逐步精化（如扩散模型）。在质量和速度之间取得平衡。

### 1.2.3 按数据类型分类

- **图像生成**：像素级或潜空间生成
- **文本生成**：离散token序列生成
- **音频生成**：波形或频谱生成
- **视频生成**：时空序列生成
- **3D生成**：点云、网格、NeRF等表示

## 1.3 评估指标

### 1.3.1 图像生成评估

**FID（Fréchet Inception Distance）**：
$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

其中 $(\mu_r, \Sigma_r)$ 和 $(\mu_g, \Sigma_g)$ 分别是真实图像和生成图像在Inception网络特征空间中的均值和协方差。FID越低，生成质量越好。

**IS（Inception Score）**：
$$\text{IS} = \exp\left(\mathbb{E}[D_{\text{KL}}(p(y|x) \| p(y))]\right)$$

IS衡量生成图像的质量（条件分布的低熵）和多样性（边缘分布的高熵）。

**LPIPS（Learned Perceptual Image Patch Similarity）**：基于深度特征的感知相似度，用于评估图像编辑中保真度。

### 1.3.2 文本生成评估

**BLEU**：基于N-gram精确率的机器翻译评估。
$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

**ROUGE**：基于N-gram召回率的摘要评估。

**困惑度（Perplexity）**：语言模型的预测能力。
$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})\right)$$

**BERTScore**：基于BERT嵌入的语义相似度评估。

**人工评估**：最可靠但成本最高，评估流畅性、相关性、忠实度、创造性等维度。

### 1.3.3 评估的挑战

生成模型的评估面临根本性挑战：
1. **质量-多样性权衡**：提高生成质量可能降低多样性（模式坍缩）
2. **评估指标与人类判断的不一致**：FID等自动指标不总与人类偏好一致
3. **缺乏统一标准**：不同任务需要不同的评估方法
4. **可重复性问题**：随机种子、模型变体等影响结果

## 1.4 历史演进

### 1.4.1 早期探索（2006-2013）
- 受限玻尔兹曼机（RBM）
- 深度信念网络（DBN）
- 自编码器

### 1.4.2 GAN时代（2014-2019）
- 2014年：GAN提出
- 2015年：DCGAN（深度卷积GAN）
- 2017年：WGAN（Wasserstein GAN）
- 2018年：StyleGAN（风格化人脸生成）
- 2019年：StyleGAN2

### 1.4.3 扩散模型时代（2020-至今）
- 2020年：DDPM
- 2021年：DALL-E（文本到图像）
- 2022年：Stable Diffusion（潜空间扩散）
- 2023年：Sora概念（视频生成）
- 2024年：更强大的多模态生成系统

## 1.5 本章小结

本章建立了生成模型的基本框架，介绍了生成任务的定义、模型分类、评估方法和历史演进。生成式AI正在经历前所未有的快速发展，理解其基础原理是跟上前沿的前提。

---

*参考文献：Goodfellow et al. (2014), Kingma & Welling (2013), Ho et al. (2020)*
-e 

---


# 第2章 变分自编码器

> "我们能否让神经网络不仅学会编码数据，还能学会数据背后的概率分布？"
> —— Diederik P. Kingma & Max Welling, 2013

## 2.1 引言

在生成式模型的谱系中，变分自编码器（Variational Autoencoder, VAE）占据着承上启下的关键位置。它上承经典的隐变量模型与概率图模型理论，下启深度生成模型的现代范式，是理解后续扩散模型、归一化流等方法的必要基石。

VAE 由 Kingma 和 Welling 于 2013 年提出，其核心思想优雅而深刻：**用神经网络同时参数化生成过程和推断过程**，将变分推断转化为可微分的优化问题，从而在深度学习框架内实现端到端的概率生成建模。本章将从隐变量模型的基本概念出发，逐步推导变分下界、重参数化技巧，并介绍 VAE 的重要变体。

---

## 2.2 隐变量模型

### 2.2.1 基本设定

隐变量模型（Latent Variable Model）假设观测数据 $\mathbf{x}$ 是由某种不可直接观测的隐变量 $\mathbf{z}$ 经过随机过程生成的。形式化地，我们假设存在一个联合分布：

$$p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} | \mathbf{z}) \, p(\mathbf{z})$$

其中：

- $p(\mathbf{z})$ 是隐变量的**先验分布**，通常取为简单的分布，如标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$；
- $p_\theta(\mathbf{x} | \mathbf{z})$ 是由参数 $\theta$ 控制的**条件生成分布**（似然函数），通常由深度神经网络参数化；
- $\mathbf{z} \in \mathbb{R}^d$ 是低维隐空间中的一个点，$d \ll \text{dim}(\mathbf{x})$。

**生成过程**（Generation Process）非常直观：首先从先验 $p(\mathbf{z})$ 中采样一个隐变量 $\mathbf{z}$，然后从条件分布 $p_\theta(\mathbf{x} | \mathbf{z})$ 中采样得到观测数据 $\mathbf{x}$。这个过程可以被视为一个"解码器"：$\mathbf{z} \xrightarrow{\text{decode}} \mathbf{x}$。

### 2.2.2 边际似然与学习目标

我们的最终目标是最大化数据的**边际似然**（Marginal Likelihood），也称为模型证据：

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x} | \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z}$$

这个积分在绝大多数情况下是**不可解析计算**的。当 $p_\theta(\mathbf{x} | \mathbf{z})$ 由深度神经网络参数化时，被积函数高度非线性，不存在闭合形式的积分解。对于高维隐空间，数值积分同样不可行——这就是所谓的**配分函数困境**。

### 2.2.3 后验推断的困难

如果我们能够计算后验分布 $p_\theta(\mathbf{z} | \mathbf{x})$，许多问题将迎刃而解。根据贝叶斯定理：

$$p_\theta(\mathbf{z} | \mathbf{x}) = \frac{p_\theta(\mathbf{x} | \mathbf{z}) \, p(\mathbf{z})}{p_\theta(\mathbf{x})} = \frac{p_\theta(\mathbf{x} | \mathbf{z}) \, p(\mathbf{z})}{\int p_\theta(\mathbf{x} | \mathbf{z}') \, p(\mathbf{z}') \, d\mathbf{z}'}$$

然而，分母正是我们无法计算的边际似然。因此，**精确后验推断**在一般情况下是不可行的。

传统方法如马尔可夫链蒙特卡洛（MCMC）可以近似后验，但采样过程计算代价高昂且难以扩展到大规模数据。这正是变分推断登场的契机。

---

## 2.3 变分推断与证据下界

### 2.3.1 引入近似后验

变分推断（Variational Inference）的核心思想是：**用一个参数化的分布 $q_\phi(\mathbf{z} | \mathbf{x})$ 来近似真实但不可计算的后验 $p_\theta(\mathbf{z} | \mathbf{x})$**。在 VAE 中，$q_\phi(\mathbf{z} | \mathbf{x})$ 被称为**编码器**（Encoder）或**识别模型**（Recognition Model），由参数 $\phi$ 控制的神经网络实现。

通常假设 $q_\phi(\mathbf{z} | \mathbf{x})$ 为对角协方差的高斯分布：

$$q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}\big(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x}))\big)$$

其中 $\boldsymbol{\mu}_\phi(\mathbf{x})$ 和 $\boldsymbol{\sigma}^2_\phi(\mathbf{x})$ 是由编码器网络输出的均值向量和方差向量。

### 2.3.2 证据下界的推导

现在我们推导 VAE 的核心目标函数——**证据下界**（Evidence Lower Bound, ELBO）。从对数边际似然出发：

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$$

引入近似后验 $q_\phi(\mathbf{z} | \mathbf{x})$，利用 Jensen 不等式或直接代数推导：

$$\log p_\theta(\mathbf{x}) = \log \int q_\phi(\mathbf{z} | \mathbf{x}) \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} \, d\mathbf{z} \geq \int q_\phi(\mathbf{z} | \mathbf{x}) \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} \, d\mathbf{z}$$

更严格地，我们可以直接分解对数边际似然：

$$\log p_\theta(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}) \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \cdot \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]$$

$$= \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]}_{\mathcal{L}(\theta, \phi; \mathbf{x}) \; \text{(ELBO)}} + \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]}_{D_{\text{KL}}(q_\phi \| p_\theta) \geq 0}$$

由于 KL 散度非负，我们得到：

$$\log p_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi; \mathbf{x}) + D_{\text{KL}}\big(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p_\theta(\mathbf{z}|\mathbf{x})\big) \geq \mathcal{L}(\theta, \phi; \mathbf{x})$$

即 ELBO 是对数边际似然的下界。

### 2.3.3 ELBO 的两种等价形式

ELBO 可以进一步分解为两种等价且富有物理意义的形式。

**形式一：重建项 + KL 正则项**

$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x} | \mathbf{z}) \right]}_{\text{重建项（Reconstruction Term）}} - \underbrace{D_{\text{KL}}\big(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z})\big)}_{\text{KL 正则项（Regularization Term）}}$$

推导如下：

$$\mathcal{L} = \mathbb{E}_{q_\phi} \left[ \log \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right] = \mathbb{E}_{q_\phi} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right] - \mathbb{E}_{q_\phi} \left[ \log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p(\mathbf{z})} \right]$$

其中：

- **重建项** $\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]$ 度量从隐变量 $\mathbf{z}$ 重建原始数据的能力。对于连续数据，若 $p_\theta(\mathbf{x}|\mathbf{z})$ 为高斯分布，此项等价于最小化均方误差（MSE）；对于二值数据，等价于最小化交叉熵。
- **KL 正则项** $D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ 确保编码器输出的隐变量分布不偏离先验太远，起到正则化作用，防止过拟合并保证隐空间的连续性。

**形式二：负向 KL + 似然期望**

$$\mathcal{L} = \log p(\mathbf{z}) - \log q_\phi(\mathbf{z}|\mathbf{x}) + \log p_\theta(\mathbf{x}|\mathbf{z})$$

取期望后即得形式一。这一形式更直观地揭示了 ELBO 的逐样本计算方式。

### 2.3.4 KL 散度的解析计算

当先验 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ 且近似后验 $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ 时，KL 散度存在解析解：

$$D_{\text{KL}}(q_\phi \| p) = -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

其中 $d$ 为隐空间维度，$\mu_j$ 和 $\sigma_j^2$ 分别为第 $j$ 个分量的均值和方差。这一闭合形式的计算使得 KL 正则项无需蒙特卡洛估计，极大地提高了训练效率和梯度估计的稳定性。

### 2.3.5 变分间隙

ELBO 与真实对数似然之间的差距称为**变分间隙**（Variational Gap）：

$$\log p_\theta(\mathbf{x}) - \mathcal{L}(\theta, \phi; \mathbf{x}) = D_{\text{KL}}\big(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p_\theta(\mathbf{z}|\mathbf{x})\big) \geq 0$$

优化 ELBO 同时涉及两个对立的目标：最大化 $\mathcal{L}$ 关于 $\theta$ 提升生成质量；最大化关于 $\phi$ 使 $q_\phi$ 更接近真实后验，从而缩小变分间隙。这种对偶优化正是 VAE 训练的本质张力。

---

## 2.4 重参数化技巧

### 2.4.1 梯度估计的挑战

在 VAE 的训练中，我们需要对 ELBO 关于参数 $\theta$ 和 $\phi$ 同时求梯度。关于 $\theta$ 的梯度没有问题——期望中的被积函数可以直接对 $\theta$ 求导，再取蒙特卡洛估计：

$$\nabla_\theta \mathcal{L} \approx \frac{1}{L} \sum_{l=1}^{L} \nabla_\theta \log p_\theta(\mathbf{x} | \mathbf{z}^{(l)}), \quad \mathbf{z}^{(l)} \sim q_\phi(\mathbf{z}|\mathbf{x})$$

然而，关于 $\phi$ 的梯度则面临困难。直接对 $\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[f(\mathbf{z})]$ 求 $\phi$ 的梯度，需要处理"分布参数的梯度穿过随机采样操作"的问题。经典的 REINFORCE（得分函数）估计器虽然可以求解，但方差极高，在深度学习中几乎不可用。

### 2.4.2 重参数化技巧的核心思想

Kingma 和 Welling 提出的**重参数化技巧**（Reparameterization Trick）优雅地解决了这一问题。其核心思想是：**将随机变量的采样过程从计算图中分离出来，转化为确定性的可微变换**。

对于 $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$，我们不再直接从 $q_\phi$ 中采样 $\mathbf{z}$，而是：

1. 从标准正态分布采样一个**无参数的噪声变量**：$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. 通过确定性变换得到 $\mathbf{z}$：$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}$

其中 $\odot$ 表示逐元素乘法。这一变换的正确性由正态分布的线性变换性质保证：若 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，则 $\boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$。

### 2.4.3 梯度计算

经过重参数化后，ELBO 变为：

$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \log p_\theta(\mathbf{x} | \mathbf{z}(\boldsymbol{\epsilon}, \phi, \mathbf{x})) \right] - D_{\text{KL}}\big(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z})\big)$$

现在，$\boldsymbol{\epsilon}$ 的采样过程与参数 $\phi$ 无关，梯度可以直接穿过 $\mathbf{z}(\boldsymbol{\epsilon}, \phi, \mathbf{x})$ 的确定性变换进行反向传播：

$$\nabla_\phi \mathcal{L} \approx \frac{1}{L} \sum_{l=1}^{L} \nabla_\phi \left[ \log p_\theta(\mathbf{x} | \mathbf{z}^{(l)}) - \log q_\phi(\mathbf{z}^{(l)}|\mathbf{x}) + \log p(\mathbf{z}^{(l)}) \right]$$

其中 $\mathbf{z}^{(l)} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}^{(l)}$，$\boldsymbol{\epsilon}^{(l)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

### 2.4.4 蒙特卡洛近似与实际训练

在实践中，通常取 $L=1$（即每个样本仅采样一次），这已被证明在大批量训练中效果良好。完整的 VAE 训练算法（Kingma & Welling, 2013 算法 1）如下：

**输入**：数据集 $\mathcal{D}$，编码器参数 $\phi$，解码器参数 $\theta$

**重复**：
1. 从 $\mathcal{D}$ 中采样小批量 $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(M)}\}$
2. 采样噪声 $\boldsymbol{\epsilon}^{(i)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
3. 计算隐变量：$\mathbf{z}^{(i)} = \boldsymbol{\mu}_\phi(\mathbf{x}^{(i)}) + \boldsymbol{\sigma}_\phi(\mathbf{x}^{(i)}) \odot \boldsymbol{\epsilon}^{(i)}$
4. 计算 ELBO 的蒙特卡洛估计
5. 计算梯度 $\nabla_{\theta, \phi} \mathcal{L}$，更新参数（如使用 Adam 优化器）

由于整个流程是可微分的，可以无缝集成到标准的深度学习框架（PyTorch、TensorFlow 等）中，仅需实现一个自定义的采样层。

### 2.4.5 与其他梯度估计方法的比较

重参数化技巧与 REINFORCE/得分函数估计器的关键区别在于：

| 特性 | 重参数化 | REINFORCE |
|------|---------|-----------|
| 方差 | 低 | 高 |
| 是否需要可微 | 是（采样分布需重参数化） | 否 |
| 适用范围 | 连续可重参数化分布 | 任意分布 |
| 基线（Baseline） | 不需要 | 通常需要 |

重参数化技巧的低方差特性使其成为 VAE 训练的首选方法。对于离散隐变量（不可直接重参数化），后续研究提出了 Gumbel-Softmax 等连续松弛方法（Jang et al., 2017; Maddison et al., 2017），将其扩展到了更广泛的场景。

---

## 2.5 VAE 的变体

原始 VAE 框架提出后，研究者们从多个维度对其进行了扩展和改进。本节介绍几种最具影响力的变体。

### 2.5.1 β-VAE：解耦表示学习

**β-VAE**（Higgins et al., 2017）在 ELBO 中引入一个超参数 $\beta > 0$ 来调节 KL 正则项的权重：

$$\mathcal{L}_{\beta}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi} \left[ \log p_\theta(\mathbf{x} | \mathbf{z}) \right] - \beta \cdot D_{\text{KL}}\big(q_\phi(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z})\big)$$

当 $\beta = 1$ 时，退化为标准 VAE；当 $\beta > 1$ 时，更强的 KL 惩罚迫使隐空间更紧密地匹配先验，从而鼓励学习**解耦表示**（Disentangled Representation）。直观上，$\beta > 1$ 使得模型在信息瓶颈下运作——每个隐变量维度被鼓励捕获数据中独立变化的因子，而非冗余地编码信息。

β-VAE 在无监督解耦表示学习方面展现了令人瞩目的效果，例如在 dSprites 等数据集上，不同的隐变量维度能够自动对应物体的形状、大小、旋转、位置等独立属性。

### 2.5.2 VQ-VAE：离散隐空间

**VQ-VAE**（Vector Quantised VAE, van den Oord et al., 2017）从根本上改变了 VAE 的隐空间结构——从连续的高斯分布转向**离散的码本**（Codebook）。

其核心机制如下：

1. 维护一个大小为 $K$ 的码本 $\{e_1, e_2, \ldots, e_K\}$，$e_i \in \mathbb{R}^d$；
2. 编码器输出连续向量 $\mathbf{z}_e(\mathbf{x})$；
3. 通过最近邻查找将其量化为码本中的离散码：$\mathbf{z}_q(\mathbf{x}) = e_k$，其中 $k = \arg\min_j \|\mathbf{z}_e(\mathbf{x}) - e_j\|_2$；
4. 解码器以 $\mathbf{z}_q(\mathbf{x})$ 作为输入进行重建。

由于量化操作（argmin）不可微，VQ-VAE 使用**直通估计器**（Straight-Through Estimator）：前向传播使用量化后的向量，反向传播时将梯度直接从解码器传回编码器，绕过量化步骤。

VQ-VAE 的损失函数包含三项：

$$\mathcal{L}_{\text{VQ}} = \underbrace{\|\mathbf{x} - p_\theta(\mathbf{z}_q)\|^2}_{\text{重建损失}} + \underbrace{\|\text{sg}[\mathbf{z}_e(\mathbf{x})] - e\|^2}_{\text{码本损失}} + \underbrace{\beta \|\mathbf{z}_e(\mathbf{x}) - \text{sg}[e]\|^2}_{\text{承诺损失}}$$

其中 $\text{sg}[\cdot]$ 表示 stop-gradient 操作。码本损失更新码本嵌入以接近编码器输出，承诺损失则防止编码器输出偏离码本过远。

VQ-VAE 的优势在于：离散表示天然适合文本、语音等离散模态，且避免了连续 VAE 中常见的后验坍缩问题。后续的 VQ-VAE-2（Razavi et al., 2019）引入了层级结构，在图像生成质量上取得了与 GAN 可比的效果。

### 2.5.3 条件 VAE：可控生成

**条件 VAE**（Conditional VAE, CVAE, Sohn et al., 2015）在生成过程和推断过程中引入条件变量 $\mathbf{c}$（如类别标签、文本描述等）：

$$\text{编码器：} \quad q_\phi(\mathbf{z} | \mathbf{x}, \mathbf{c})$$
$$\text{解码器：} \quad p_\theta(\mathbf{x} | \mathbf{z}, \mathbf{c})$$
$$\text{先验：} \quad p(\mathbf{z} | \mathbf{c})$$

ELBO 相应地修改为：

$$\mathcal{L}_{\text{CVAE}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x},\mathbf{c})} \left[ \log p_\theta(\mathbf{x} | \mathbf{z}, \mathbf{c}) \right] - D_{\text{KL}}\big(q_\phi(\mathbf{z}|\mathbf{x},\mathbf{c}) \,\|\, p(\mathbf{z}|\mathbf{c})\big)$$

CVAE 使得生成过程可控：给定不同的条件 $\mathbf{c}$，模型可以生成符合特定属性的样本。例如，给定数字类别标签，MNIST 上的 CVAE 可以生成指定数字的手写体图像。

### 2.5.4 层级 VAE：更深的隐空间

标准 VAE 使用单层隐变量，其表达能力受限。**层级 VAE**（Hierarchical VAE）引入多层隐变量 $\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_L$，形成层级化的生成过程：

$$p(\mathbf{z}_L) \rightarrow p(\mathbf{z}_{L-1} | \mathbf{z}_L) \rightarrow \cdots \rightarrow p(\mathbf{z}_1 | \mathbf{z}_2) \rightarrow p_\theta(\mathbf{x} | \mathbf{z}_1)$$

对应的近似后验也为层级结构：

$$q_\phi(\mathbf{z}_1, \ldots, \mathbf{z}_L | \mathbf{x}) = q_\phi(\mathbf{z}_L | \mathbf{x}) \prod_{l=1}^{L-1} q_\phi(\mathbf{z}_l | \mathbf{z}_{l+1}, \mathbf{x})$$

代表性的层级 VAE 包括：

- **NVAE**（Nouveau VAE, Vahdat & Kolesnikov, 2020）：引入深度编码器-解码器架构和残差连接，在图像生成上首次使 VAE 达到与 GAN 可比的 FID 分数；
- **DRAW**（Gregor et al., 2015）：结合注意力机制的序列化编码-解码过程；
- **BIVA**（Maaløe et al., 2019）：使用多层随机变量和双向推断网络。

层级 VAE 的 ELBO 可以递归分解为：

$$\mathcal{L} = \mathbb{E}_{q_\phi} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}_1) \right] - \sum_{l=1}^{L} D_{\text{KL}}\big(q_\phi(\mathbf{z}_l | \cdot) \,\|\, p(\mathbf{z}_l | \cdot)\big)$$

其中 $\mathbf{z}_L$ 的先验为标准正态分布。

### 2.5.5 后验坍缩问题

**后验坍缩**（Posterior Collapse）是 VAE 训练中最常见也最棘手的问题之一。其表现为：编码器学会将 $q_\phi(\mathbf{z}|\mathbf{x})$ 退化为与输入无关的先验 $p(\mathbf{z})$，即 $q_\phi(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z})$ 对所有 $\mathbf{x}$ 成立。此时 KL 项为零，隐变量完全被忽略，解码器退化为与 $\mathbf{z}$ 无关的边际分布模型 $p_\theta(\mathbf{x})$。

**发生原因**：当解码器 $p_\theta(\mathbf{x}|\mathbf{z})$ 足够强大（如使用自回归解码器）时，它可以仅凭自身能力建模数据分布，无需利用隐变量的信息。此时优化器发现忽略 $\mathbf{z}$（使 KL 项为零）是最优策略，因为这不会损失重建质量，同时最小化了 KL 惩罚。

**缓解策略**包括：

1. **KL 退火**（KL Annealing）：训练初期使用 KL 权重 $\beta$ 从 0 逐渐增加到 1，让编码器先学习有效的表示，再逐步施加先验约束。其变体 **Cyclical Annealing**（Fu et al., 2019）周期性地重复退火过程，效果更稳定。

2. **自由比特**（Free Bits, Kingma et al., 2016）：为每个隐变量维度设定 KL 下界 $\lambda$，当某个维度的 KL 低于 $\lambda$ 时不对该维度施加惩罚：

   $$\mathcal{L}_{\text{free}} = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \sum_{j=1}^{d} \max\big(\lambda, \, D_{\text{KL}}^{(j)}\big)$$

3. **Delta-VAE**（Razavi et al., 2019）：引入一个信息瓶颈约束，确保编码器传递足够的信息。

4. **使用更弱的解码器**：例如使用参数共享或降低解码器容量，迫使模型利用隐变量。

5. **批量编码约束**（Batch Encoding Constraint）：在 batch 维度上约束编码器输出的多样性，防止单个样本的后验退化。

后验坍缩的本质是**优化路径**问题：在训练早期，解码器尚未学好，编码器有动力传递信息；但随着解码器变强，编码器逐渐"偷懒"。理解这一动态过程对 VAE 的实际应用至关重要。

---

## 2.6 VAE 的理论视角

### 2.6.1 作为编码-解码框架的 VAE

VAE 可以被视为概率版本的自编码器（Autoencoder）。传统自编码器学习确定性映射 $\mathbf{x} \rightarrow \mathbf{z} \rightarrow \hat{\mathbf{x}}$，最小化重建误差 $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$；VAE 则将编码和解码过程概率化，并加入 KL 正则化。关键区别在于：

- 传统自编码器的隐空间缺乏结构，可能产生不连续的"空洞"；
- VAE 的 KL 正则化确保隐空间平滑连续，支持有意义的插值和采样。

### 2.6.2 与 EM 算法的联系

VAE 可以被视为随机变分 EM（Expectation-Maximization）算法的在线版本。经典 EM 算法在 E 步精确计算后验，在 M 步更新模型参数；VAE 则用参数化的编码器替代 E 步的精确推断，用梯度下降替代 M 步的解析更新，使其能够扩展到大规模数据和复杂模型。

### 2.6.3 与主成分分析的关系

当解码器为线性模型且隐变量先验为标准正态时，VAE 退化为**概率主成分分析**（Probabilistic PCA, Tipping & Bishop, 1999）。这一联系表明 VAE 可以被视为 PCA 的非线性深度推广。

---

## 2.7 小结

本章系统介绍了变分自编码器的理论基础与技术细节。我们从隐变量模型出发，认识到边际似然计算和后验推断的根本困难；通过变分推断，我们将不可解的推断问题转化为可优化的 ELBO 目标函数；重参数化技巧使得梯度估计低方差且高效，将 VAE 纳入了标准的反向传播训练框架。

在 VAE 的诸多变体中，β-VAE 追求解耦表示，VQ-VAE 开辟了离散隐空间的新方向，条件 VAE 实现了可控生成，层级 VAE 则通过更深的隐结构提升了表达能力。后验坍缩问题是 VAE 训练中的核心挑战，多种缓解策略从不同角度提供了应对方案。

VAE 的意义不仅在于其本身作为一种生成模型的价值，更在于它为后续的生成模型研究奠定了概念基础：**隐空间的结构化、变分目标的设计、重参数化梯度估计**等思想在扩散模型（第5章）等后续工作中持续发挥重要作用。可以说，理解 VAE 是理解现代深度生成模型的必经之路。

---

## 参考文献

1. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint arXiv:1312.6114*.
2. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. *ICML*.
3. Higgins, I., et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR*.
4. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. *NeurIPS*.
5. Sohn, K., Lee, H., & Yan, X. (2015). Learning Structured Output Representation Using Deep Conditional Generative Models. *NeurIPS*.
6. Kingma, D. P., et al. (2016). Improved Variational Inference with Inverse Autoregressive Flow. *NeurIPS*.
7. Vahdat, A., & Kolesnikov, A. (2020). NVAE: A Deep Hierarchical Variational Autoencoder. *NeurIPS*.
8. Fu, H., et al. (2019). Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing. *NAACL*.
9. Gregor, K., et al. (2015). DRAW: A Recurrent Neural Network for Image Generation. *ICML*.
10. Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.
11. Tipping, M. E., & Bishop, C. M. (1999). Probabilistic Principal Component Analysis. *Journal of the Royal Statistical Society: Series B*.
-e 

---


# 第3章 生成对抗网络

## 3.1 引言

2014年，Ian Goodfellow等人提出了生成对抗网络（Generative Adversarial Networks, GAN），这一工作被Yann LeCun誉为"近十年来机器学习领域最有趣的想法"。GAN的核心思想源自博弈论：通过让两个神经网络——生成器（Generator）与判别器（Discriminator）——进行对抗博弈，最终使生成器学会产生以假乱真的样本。这一优雅的框架打破了传统生成模型对显式概率密度函数的依赖，开辟了一条通过隐式建模来学习数据分布的全新路径。

在GAN问世之前，生成模型的研究主要沿着两条路线展开：一是基于显式密度估计的方法，如变分自编码器（VAE）和自回归模型；二是基于模拟的方法，如马尔可夫链蒙特卡洛（MCMC）。GAN的出现提供了第三种范式——通过对抗训练间接学习数据分布，无需定义显式的似然函数。这一思想不仅在理论上具有深刻意义，更在图像生成、视频合成、文本生成等实际应用中展现出惊人的能力。

本章将从GAN的数学原理出发，逐步深入到训练技巧、条件扩展以及理论进展，系统地呈现这一领域的发展脉络。

## 3.2 GAN的基本原理

### 3.2.1 博弈论框架

GAN的训练过程可以被形式化为一个**极小极大博弈**（minimax game）。考虑两个玩家：

- **生成器 $G$**：接收一个来自先验分布 $p_z(z)$ 的随机噪声向量 $z$（通常为标准正态分布或均匀分布），将其映射到数据空间，目标是生成与真实数据分布 $p_{\text{data}}$ 无法区分的样本。
- **判别器 $D$**：接收一个样本 $x$（来自真实数据或生成器），输出一个标量值 $D(x) \in [0,1]$，表示该样本来自真实数据的概率。

二者的博弈目标由以下价值函数（value function）定义：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]$$

这个目标函数的直觉是：判别器 $D$ 试图最大化 $V(D,G)$，即正确区分真假样本；而生成器 $G$ 试图最小化 $V(D,G)$，即让判别器犯错。在理想情况下，当训练达到纳什均衡（Nash equilibrium）时，生成器产生的样本与真实数据在统计上不可区分，此时 $D(x) = \frac{1}{2}$ 对所有 $x$ 成立。

### 3.2.2 最优判别器的推导

为了理解GAN的训练动态，我们首先推导在给定生成器 $G$ 的情况下，最优判别器 $D^*$ 的形式。

对于固定的 $G$，我们需要最大化：

$$V(D, G) = \int_x \left[ p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx$$

其中 $p_g(x)$ 是生成器隐式定义的数据分布。对于积分内的每一项，考虑函数：

$$f(y) = a \log y + b \log(1 - y), \quad a = p_{\text{data}}(x), \quad b = p_g(x)$$

对 $y$ 求导并令其为零：

$$\frac{a}{y} - \frac{b}{1-y} = 0 \implies y^* = \frac{a}{a+b}$$

因此，最优判别器为：

$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

这个结果具有清晰的贝叶斯解释：$D^*(x)$ 实际上是样本 $x$ 来自真实数据的后验概率，其中先验 $p(\text{real}) = p(\text{fake}) = \frac{1}{2}$。

### 3.2.3 全局最优与JS散度

将最优判别器 $D^*$ 代入价值函数，我们得到：

$$C(G) = \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \right] + \mathbb{E}_{x \sim p_g} \left[ \log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)} \right]$$

经过代数变换，可以将其写为：

$$C(G) = -\log 4 + 2 \cdot JSD(p_{\text{data}} \| p_g)$$

其中 $JSD$ 表示 **Jensen-Shannon 散度**（Jensen-Shannon divergence）：

$$JSD(P \| Q) = \frac{1}{2} KL\left(P \left\| \frac{P+Q}{2} \right.\right) + \frac{1}{2} KL\left(Q \left\| \frac{P+Q}{2} \right.\right)$$

由于JSD始终非负，且当且仅当 $p_{\text{data}} = p_g$ 时为零，因此 $C(G)$ 的全局最小值为 $-\log 4$，在 $p_g = p_{\text{data}}$ 时达到。这从理论上证明了：**在有足够的模型容量和训练时间的条件下，GAN的训练过程收敛到真实数据分布**。

### 3.2.4 训练算法

GAN的原始训练算法采用交替优化的方式：

**算法1：GAN训练（Goodfellow et al., 2014）**

```
for 训练迭代次数 do
    for k步 do
        采样噪声 {z^(1), ..., z^(m)} ~ p_z(z)
        采样真实样本 {x^(1), ..., x^(m)} ~ p_data(x)
        通过梯度上升更新判别器：
            θ_D ← θ_D + ∇_{θ_D} (1/m) Σ [log D(x^(i)) + log(1 - D(G(z^(i))))]
    end for
    采样噪声 {z^(1), ..., z^(m)} ~ p_z(z)
    通过梯度下降更新生成器：
        θ_G ← θ_G - ∇_{θ_G} (1/m) Σ log(1 - D(G(z^(i))))
end for
```

在实践中，原始的 $\log(1 - D(G(z)))$ 损失在训练初期会导致梯度饱和问题（当 $D(G(z))$ 接近0时梯度极小）。因此，Goodfellow等人建议将生成器的目标替换为**最大化 $\log D(G(z))$**，即所谓的"非饱和"（non-saturating）损失：

$$\max_G \mathbb{E}_{z \sim p_z} [\log D(G(z))]$$

这一替换在梯度动力学上等价于原目标（具有相同的固定点），但在训练初期提供了更强的梯度信号。

## 3.3 训练技巧

原始GAN虽然在理论上优雅，但在实践中面临严重的训练不稳定和模式坍缩（mode collapse）问题。本节介绍几种关键的训练技巧。

### 3.3.1 谱归一化

**谱归一化**（Spectral Normalization, Miyato et al., 2018）是稳定判别器训练的一种有效技术。其核心思想是通过控制判别器每一层权重矩阵的谱范数（spectral norm）来约束Lipschitz常数。

对于权重矩阵 $W$，其谱范数定义为：

$$\sigma(W) = \max_{\|h\|_2 = 1} \|Wh\|_2$$

谱归一化将权重矩阵归一化为：

$$\bar{W} = \frac{W}{\sigma(W)}$$

在实践中，$\sigma(W)$ 通过幂迭代法（power iteration）高效近似计算，每次前向传播仅需额外的一次矩阵-向量乘法。谱归一化的优势在于：（1）计算开销极小；（2）不需要额外的超参数；（3）直接控制判别器的Lipschitz常数，从而避免梯度爆炸。

### 3.3.2 渐进训练

**渐进式增长**（Progressive Growing, Karras et al., 2018）是ProGAN中提出的一项突破性训练策略。其核心思想是：从低分辨率（如 $4 \times 4$）开始训练生成器和判别器，然后逐步添加新的层来增加分辨率（$4 \to 8 \to 16 \to \cdots \to 1024$）。

新层通过一个平滑的过渡（fade-in）机制引入：设新层的输出为 $x_{\text{new}}$，旧层的上采样输出为 $x_{\text{old}}$，则过渡期的输出为：

$$x = (1 - \alpha) \cdot x_{\text{old}} + \alpha \cdot x_{\text{new}}$$

其中 $\alpha$ 从0线性增长到1。这种渐进策略使得训练过程更加稳定，因为低分辨率阶段建立的全局结构可以在高分辨率阶段被精细化，而非从零开始学习所有尺度的特征。

### 3.3.3 StyleGAN的映射网络与自适应实例归一化

**StyleGAN**（Karras et al., 2019）在ProGAN的基础上引入了多项创新，其中最具影响力的是**映射网络**（mapping network）和**自适应实例归一化**（Adaptive Instance Normalization, AdaIN）。

传统GAN将噪声向量 $z$ 直接输入生成器。StyleGAN首先通过一个由8个全连接层组成的映射网络 $f$ 将 $z$ 映射到一个中间潜在空间 $w$：

$$w = f(z), \quad z \sim \mathcal{N}(0, I)$$

映射网络的作用是将纠缠的（entangled）高斯潜在空间"解纠缠"为更可分离的中间空间。在中间空间 $w$ 中，不同的方向对应不同的语义属性（如年龄、发色、姿态等），从而实现更精细的控制。

生成器的每一层通过AdaIN将风格信息注入。对于特征图 $x_i$（第 $i$ 个通道），AdaIN的计算为：

$$\text{AdaIN}(x_i, y) = y_{s,i} \cdot \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

其中 $\mu(x_i)$ 和 $\sigma(x_i)$ 分别是特征图的均值和标准差，$y_s$ 和 $y_b$ 是由 $w$ 经仿射变换得到的缩放和偏移参数。通过在不同分辨率的层注入不同的 $w$，StyleGAN实现了对不同尺度属性的精细控制——粗层（低分辨率）控制姿态、脸型等全局属性，细层（高分辨率）控制肤色、纹理等局部细节。

此外，StyleGAN引入了**噪声注入**（stochastic variation）机制，在每一层向特征图添加逐像素的随机噪声，用于生成如发丝、毛孔等随机细节，使生成结果更加逼真。

### 3.3.4 其他重要技巧

除了上述核心技术，GAN训练中还有若干被广泛采用的技巧：

- **两时间尺度更新规则**（TTUR, Heusel et al., 2017）：判别器和生成器使用不同的学习率，通常判别器学习率更大。这保证了在一定的正则化条件下，训练收敛到局部纳什均衡。
- **截断技巧**（Truncation trick）：在推理时，将潜在变量 $z$ 截断到较小的范围内（如限制在 $[-1, 1]$），以牺牲多样性换取生成质量的提升。
- **R1正则化**（Mescheder et al., 2018）：对判别器关于真实数据的梯度施加惩罚：

$$R_1 = \frac{\gamma}{2} \mathbb{E}_{x \sim p_{\text{data}}} [\|\nabla_x D(x)\|^2]$$

## 3.4 条件GAN

原始GAN生成的样本来自隐式学习到的数据分布，但缺乏对生成内容的可控性。**条件GAN**（Conditional GAN, Mirza & Osindero, 2014）通过引入条件信息 $y$（如类别标签、文本描述、参考图像等），将无条件的生成过程转变为可控的条件生成。

条件GAN的价值函数为：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x|y)] + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z|y)|y))]$$

条件信息 $y$ 可以同时输入生成器和判别器，使得生成过程和判别过程都在条件约束下进行。

### 3.4.1 文本到图像生成

**文本到图像**（Text-to-Image）生成是条件GAN最具代表性的应用之一。Reed等人（2016）提出的StackGAN首次实现了从文本描述生成高分辨率图像的突破。

文本条件的编码通常采用预训练的文本编码器（如LSTM或CLIP文本编码器）将文本描述 $s$ 映射为嵌入向量 $\varphi(s)$。StackGAN采用两阶段架构：

1. **Stage-I**：以文本嵌入为条件，生成低分辨率（$64 \times 64$）的粗略图像。
2. **Stage-II**：以Stage-I的输出和文本嵌入为条件，生成高分辨率（$256 \times 256$）的精细图像。

为了处理文本嵌入中的冗余信息，StackGAN引入了条件增强（Conditioning Augmentation）技术，对文本嵌入施加KL散度正则化，将其约束到一个平滑的高斯分布中，从而增强条件空间的多样性和连续性。

### 3.4.2 Pix2Pix：配对图像翻译

**Pix2Pix**（Isola et al., 2017）是条件GAN在**配对图像翻译**（paired image-to-image translation）任务中的经典应用。给定成对的训练数据 $\{(x_i, y_i)\}$（如语义分割图与对应的照片），Pix2Pix学习一个映射 $G: x \to y$。

Pix2Pix的目标函数结合了条件GAN损失和L1重建损失：

$$\mathcal{L} = \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

其中：

$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x,y} [\log D(x, y)] + \mathbb{E}_{x,z} [\log(1 - D(x, G(x, z)))]$$

$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y,z} [\|y - G(x, z)\|_1]$$

L1损失鼓励生成图像在像素级别上接近目标，而GAN损失则负责生成感知上逼真的细节。L1损失倾向于产生模糊的结果（因为它对所有可能输出取平均），而GAN损失则通过判别器的压力来"锐化"输出。

Pix2Pix采用U-Net架构作为生成器，该架构通过跳跃连接（skip connections）将编码器的特征直接传递到对应的解码器层，从而保留空间细节信息。判别器则采用PatchGAN架构，仅对图像的局部 $70 \times 70$ 区域进行真伪判别，这使得判别器的参数更少且对高频细节更加敏感。

### 3.4.3 CycleGAN：无配对风格迁移

在许多实际场景中，获取成对的训练数据是困难甚至不可能的（如照片与莫奈画作之间的翻译）。**CycleGAN**（Zhu et al., 2017）通过引入**循环一致性**（cycle consistency）约束，解决了无配对图像翻译的问题。

CycleGAN学习两个映射 $G: X \to Y$ 和 $F: Y \to X$，并施加循环一致性约束：

$$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim p_X} [\|F(G(x)) - x\|_1] + \mathbb{E}_{y \sim p_Y} [\|G(F(y)) - y\|_1]$$

完整的目标函数为：

$$\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{GAN}(G, D_Y) + \mathcal{L}_{GAN}(F, D_X) + \lambda \mathcal{L}_{\text{cyc}}(G, F)$$

循环一致性的直觉是：将一匹马翻译为斑马，再翻译回来，应该得到原来的马。这一约束有效地防止了映射的退化（例如将所有输入映射到同一个输出），并在不需要配对数据的情况下保证了翻译的语义一致性。

CycleGAN在风格迁移、季节转换、物体变形等任务上取得了令人印象深刻的效果，开创了无监督图像翻译的研究方向。后续工作如MUNIT（Huang et al., 2018）和DRIT（Lee et al., 2018）进一步将内容与风格解耦，实现了多模态的无配对翻译。

### 3.4.4 风格迁移中的应用

GAN在风格迁移领域的应用远不止CycleGAN。**Neural Style Transfer** 的GAN变体将风格迁移视为一个条件生成问题。AdaIN（Huang & Belongie, 2017）提出了一种简单而有效的方法：在特征空间中对齐内容图像和风格图像的统计量（均值和方差），实现了实时的任意风格迁移。

更近期的工作如**Gatys et al.**的方法利用预训练的VGG网络提取特征，通过最小化内容特征的Gram矩阵差异和风格特征的Gram矩阵差异来实现风格迁移。GAN框架下的风格迁移则通过对抗训练直接学习迁移映射，避免了逐优化的计算开销。

## 3.5 GAN的理论进展

### 3.5.1 Wasserstein GAN

原始GAN的JS散度目标存在一个根本性的理论缺陷：当真实分布 $p_{\text{data}}$ 和生成分布 $p_g$ 的支撑集不重叠时（这在高维空间中几乎总是成立），JS散度恒为 $\log 2$，梯度为零，导致训练信号消失。

**Wasserstein GAN**（WGAN, Arjovsky et al., 2017）提出用 **Wasserstein距离**（也称Earth Mover's Distance, EMD）替代JS散度。Wasserstein距离定义为：

$$W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x,y) \sim \gamma} [\|x - y\|]$$

其中 $\Pi(p_{\text{data}}, p_g)$ 是所有边际分布为 $p_{\text{data}}$ 和 $p_g$ 的联合分布的集合。直观上，Wasserstein距离衡量的是将一个分布"搬运"到另一个分布所需的最小"工作量"。

通过Kantorovich-Rubinstein对偶性，Wasserstein距离可以等价地写为：

$$W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1} \left\{ \mathbb{E}_{x \sim p_{\text{data}}} [f(x)] - \mathbb{E}_{x \sim p_g} [f(x)] \right\}$$

其中上确界取遍所有1-Lipschitz函数。WGAN将判别器（此时称为"critic"）约束为Lipschitz函数，目标函数变为：

$$\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim p_{\text{data}}} [D(x)] - \mathbb{E}_{z \sim p_z} [D(G(z))]$$

Wasserstein距离的关键优势在于其**连续性**：即使两个分布没有重叠，Wasserstein距离仍然能提供有意义的、平滑的梯度信号。这从根本上解决了原始GAN训练不稳定的问题。

### 3.5.2 梯度惩罚

原始WGAN通过**权重裁剪**（weight clipping）来强制Lipschitz约束——将判别器的权重限制在 $[-c, c]$ 范围内。然而，这种做法会导致优化困难和容量利用不足的问题。

**WGAN-GP**（Gulrajani et al., 2017）提出了**梯度惩罚**（gradient penalty）来替代权重裁剪。其核心思想是对判别器在真实数据和生成数据之间的插值样本处的梯度范数施加惩罚：

$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}} [(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

其中 $\hat{x} = \epsilon x + (1 - \epsilon) G(z)$，$\epsilon \sim U[0,1]$，$x \sim p_{\text{data}}$，$z \sim p_z$。梯度惩罚使得判别器在插值路径上的梯度范数趋向于1，从而近似满足1-Lipschitz约束。

实验表明，WGAN-GP显著提升了训练稳定性和生成质量，并且移除了WGAN中对学习率调整的敏感性。然而，梯度惩罚引入了额外的计算开销（需要计算二阶梯度），这在某些场景下可能成为瓶颈。

### 3.5.3 模式坍缩分析

**模式坍缩**（mode collapse）是GAN训练中最常见且最棘手的问题之一。它指的是生成器仅学会产生有限种类的样本，而忽略了数据分布中的大量模式。例如，在MNIST数据集上，生成器可能只生成数字"1"和"7"，而完全无法生成其他数字。

模式坍缩的根本原因可以从以下角度理解：

1. **极小极大优化的不对称性**：生成器可以"发现"判别器的弱点，并持续利用该弱点生成相似的样本，而非学习完整的分布。
2. **梯度信号的不充分性**：判别器仅提供"真/假"的二元信号，无法明确指示生成器应该覆盖哪些缺失的模式。
3. **优化动力学**：在非凸-非凹的极小极大博弈中，标准的梯度下降-上升方法可能收敛到不良的均衡点。

从理论角度看，模式坍缩与JS散度的特性密切相关。当生成分布坍缩到数据分布的一个子集时，JS散度仍然提供正的梯度（因为部分模式缺失），但梯度的方向可能指向已覆盖的模式而非缺失的模式。

为缓解模式坍缩，研究者提出了多种策略：

- **小批量判别**（Minibatch discrimination, Salimans et al., 2016）：让判别器同时观察整个小批量的样本，从而检测生成样本的多样性不足。
- **展开GAN**（UnrolledGAN, Metz et al., 2017）：在更新生成器时，将判别器的未来多步更新也纳入考虑，使生成器能够预见并避免模式坍缩。
- **多样性正则化**：在损失函数中加入鼓励多样性的项，如feature matching损失（匹配判别器中间层特征的统计量）。

### 3.5.4 收敛性分析

GAN的收敛性理论是一个活跃的研究领域。在最简单的设定下，考虑参数化的生成器和判别器，GAN的训练可以被建模为一个非凸-非凹的极小极大优化问题：

$$\min_\theta \max_\phi f(\theta, \phi)$$

**Mescheder等人（2017）** 的工作从连续时间动力学的角度分析了GAN的收敛性。他们证明，在判别器足够强（达到最优）的条件下，原始GAN和NS-GAN的梯度下降-上升动力学的局部渐近稳定性（local asymptotic stability）取决于学习率的选择。具体而言，如果判别器的学习率 $\eta_D$ 和生成器的学习率 $\eta_G$ 满足：

$$\eta_D \cdot \eta_G < \frac{2}{L^2}$$

其中 $L$ 是价值函数关于参数的Lipschitz常数，则训练过程在局部纳什均衡附近是稳定的。

**Nagarajan和Kolter（2017）** 进一步证明了原始GAN的梯度下降动力学在某些情况下是**不稳定的**——即使在全局最优判别器附近，生成器的梯度也可能将其推离均衡点。这一分析为WGAN等改进方法提供了理论动机。

**Heusel等人（2017）** 在TTUR框架下证明了：在满足一定正则化条件时，交替梯度下降方法在非凸-非凹极小极大问题上的收敛性。他们将GAN的训练建模为一个**双时间尺度随机近似**（two-timescale stochastic approximation）过程，并证明了在适当的条件下，训练轨迹收敛到一个**局部稳定纳什均衡**。

**Grnarova等人（2018）** 利用在线学习和博弈论的工具，将GAN的训练与**镜像下降**（mirror descent）算法联系起来，证明了在某些正则化条件下GAN训练的收敛性。

尽管取得了这些进展，GAN的收敛性理论仍然是一个开放问题。主要挑战在于：（1）实际的GAN架构涉及深度神经网络，使得理论分析极为困难；（2）极小极大优化的非凸-非凹性质排除了标准凸优化工具的直接应用；（3）模式坍缩等现象在理论分析中难以精确刻画。

## 3.6 本章小结

生成对抗网络自2014年提出以来，在短短数年间经历了爆发式的发展。从Goodfellow等人提出的极小极大博弈框架出发，GAN在理论和实践两个维度上都取得了长足的进步。

在基本原理层面，我们看到了GAN如何通过对抗训练隐式地学习数据分布，以及最优判别器与JS散度之间的深刻联系。在训练技巧层面，谱归一化、渐进训练和StyleGAN的映射网络等技术逐步解决了训练不稳定和生成质量的瓶颈。在条件扩展层面，从Pix2Pix到CycleGAN，条件GAN将生成模型从被动的分布学习转变为主动的内容创作工具。在理论进展层面，Wasserstein距离的引入从根本上改善了训练的稳定性，而对模式坍缩和收敛性的分析则加深了我们对GAN训练动力学的理解。

GAN的影响远远超出了图像生成本身。它开创的对抗训练范式已被广泛应用于自然语言处理、强化学习、药物发现等众多领域。尽管近年来扩散模型在图像生成质量上已经超越了GAN，但GAN的对抗思想仍然在实时生成、图像编辑和风格迁移等应用中保持着不可替代的地位。理解GAN的原理与技术，不仅是掌握生成模型发展史的关键，更是理解后续章节中扩散模型等新技术的重要基础。

## 参考文献

- Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014). Generative adversarial nets. *Advances in Neural Information Processing Systems*, 27.
- Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. *arXiv preprint arXiv:1411.1784*.
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein generative adversarial networks. *International Conference on Machine Learning*, 214-223.
- Gulrajani, I., Ahmed, F., Arjovsky, M., et al. (2017). Improved training of Wasserstein GANs. *Advances in Neural Information Processing Systems*, 30.
- Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral normalization for generative adversarial networks. *International Conference on Learning Representations*.
- Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). Progressive growing of GANs for improved quality, stability, and variation. *International Conference on Learning Representations*.
- Karras, T., Laine, S., & Aila, T. (2019). A style-based generator architecture for generative adversarial networks. *IEEE Conference on Computer Vision and Pattern Recognition*, 4401-4410.
- Isola, P., Zhu, J., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. *IEEE Conference on Computer Vision and Pattern Recognition*, 1125-1134.
- Zhu, J., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. *IEEE International Conference on Computer Vision*, 2223-2232.
- Reed, S., Akata, Z., Yan, X., et al. (2016). Generative adversarial text to image synthesis. *International Conference on Machine Learning*, 1060-1069.
- Mescheder, L., Geiger, A., & Nowozin, S. (2018). Which training methods for GANs do actually converge? *International Conference on Machine Learning*, 3481-3490.
- Heusel, M., Ramsauer, H., Unterthiner, T., et al. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *Advances in Neural Information Processing Systems*, 30.
- Salimans, T., Goodfellow, I., Zaremba, W., et al. (2016). Improved techniques for training GANs. *Advances in Neural Information Processing Systems*, 29.
- Metz, L., Poole, B., Pfau, D., & Sohl-Dickstein, J. (2017). Unrolled generative adversarial networks. *International Conference on Learning Representations*.
- Huang, X., & Belongie, S. (2017). Arbitrary style transfer in real-time with adaptive instance normalization. *IEEE International Conference on Computer Vision*, 1501-1510.
- Mescheder, L., Nowozin, S., & Geiger, A. (2017). The numerics of GANs. *Advances in Neural Information Processing Systems*, 30.
- Nagarajan, V., & Kolter, J. Z. (2017). Gradient descent GAN optimization is locally stable. *Advances in Neural Information Processing Systems*, 30.
-e 

---


# 第4章 扩散模型基础

## 4.1 引言

扩散模型（Diffusion Models）是近年来生成式人工智能领域最具突破性的范式之一。与生成对抗网络（GAN）和变分自编码器（VAE）不同，扩散模型并不依赖对抗训练或推断网络的近似，而是通过一个极其简洁的数学框架——逐步向数据添加噪声再学习逐步去噪——实现了令人惊叹的生成质量。2020年，Ho等人提出的去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）标志着扩散模型从理论走向实用的关键转折点，此后Stable Diffusion、DALL·E 2等里程碑式的应用均建立在这一基础之上。

本章将系统阐述扩散模型的数学基础与核心算法。我们从直观图景出发，逐步深入前向扩散过程、反向去噪过程的严格推导，并完整呈现DDPM的训练目标与采样算法，最后讨论实践中的关键技术细节。

---

## 4.2 前向扩散过程

### 4.2.1 基本思想

扩散模型的核心思想可以用一个日常类比来理解：想象一滴墨水滴入清水，墨滴会逐渐扩散，最终水变得均匀浑浊。前向扩散过程（forward diffusion process）所做的正是类似的"加噪"操作——从一张清晰图像出发，逐步添加高斯噪声，经过足够多步之后，原始数据的结构信息被完全抹除，变成一个各向同性的高斯噪声分布。

关键的洞察在于：虽然前向过程是不可逆的（信息丢失了），但如果我们知道每一步添加了多少噪声，理论上就可以训练一个神经网络来**逆转**每一步的加噪操作。这就是反向去噪过程的本质。

### 4.2.2 马尔可夫链形式化

设原始数据分布为 $q(\mathbf{x}_0)$，其中 $\mathbf{x}_0 \in \mathbb{R}^d$ 表示一个数据样本（例如一张图像展开后的向量）。前向扩散过程定义为一个马尔可夫链，它在 $T$ 个时间步内逐步向数据添加高斯噪声：

$$q(\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T \mid \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$

其中每一步的转移核为：

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I})$$

这里 $\beta_t \in (0, 1)$ 是第 $t$ 步的**噪声调度**（noise schedule）参数，控制该步添加噪声的强度。$\beta_t$ 越大，该步加入的噪声越多。序列 $\{\beta_t\}_{t=1}^T$ 构成一个预设的超参数序列，通常满足 $\beta_1 < \beta_2 < \cdots < \beta_T$，即越到后期添加的噪声越多。

### 4.2.3 加噪公式的重参数化

上述定义意味着每一步仅对前一步的输出施加微小扰动。然而在实际训练中，我们经常需要直接从 $\mathbf{x}_0$ 跳到任意时刻 $t$ 的加噪版本 $\mathbf{x}_t$，而不需要逐步模拟。为此，我们利用重参数化技巧（reparameterization trick）推导出闭合形式。

定义 $\alpha_t = 1 - \beta_t$，以及累积乘积：

$$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$

可以证明，从 $\mathbf{x}_0$ 直接采样 $\mathbf{x}_t$ 的公式为：

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1 - \bar{\alpha}_t)\mathbf{I})$$

等价地：

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**推导过程**如下。从单步递推开始：

$$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\,\boldsymbol{\epsilon}_{t-1}$$

将 $\mathbf{x}_{t-1}$ 展开：

$$\mathbf{x}_t = \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}\,\mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}}\,\boldsymbol{\epsilon}_{t-2}\right) + \sqrt{1 - \alpha_t}\,\boldsymbol{\epsilon}_{t-1}$$

$$= \sqrt{\alpha_t \alpha_{t-1}}\,\mathbf{x}_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1})}\,\boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t}\,\boldsymbol{\epsilon}_{t-1}$$

利用两个独立高斯分布之和的性质：$\mathcal{N}(0, \sigma_1^2\mathbf{I}) + \mathcal{N}(0, \sigma_2^2\mathbf{I}) = \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$，合并后两项噪声：

$$\mathbf{x}_t = \sqrt{\alpha_t \alpha_{t-1}}\,\mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}}\,\boldsymbol{\epsilon}$$

递推至 $\mathbf{x}_0$ 即可得到上述闭合形式。这一性质极为重要：它意味着训练时可以随机采样时间步 $t$，直接从 $\mathbf{x}_0$ 计算 $\mathbf{x}_t$，无需模拟整个马尔可夫链，大幅提高了训练效率。

### 4.2.4 噪声调度设计

噪声调度 $\{\beta_t\}$ 的设计直接影响模型的训练效果与生成质量。当 $T$ 足够大（通常 $T = 1000$），且 $\beta_t$ 从接近零递增到接近一时，可以保证：

$$\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$$

即前向过程的终态近似于标准高斯分布。这意味着我们从标准高斯噪声出发进行采样，理论上可以生成任意数据。

DDPM原始论文采用**线性噪声调度**（linear schedule）：$\beta_1 = 10^{-4}$ 线性增长到 $\beta_T = 0.02$。这种设计简单有效，但后续研究（Nichol & Dhariwal, 2021）表明，**余弦调度**（cosine schedule）能更好地利用整个时间轴，尤其在图像的低频结构保持方面表现更优：

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

其中 $s$ 是一个小偏移量（通常取 $s = 0.008$），防止 $\beta_t$ 在 $t$ 接近零时过小。

---

## 4.3 反向去噪过程

### 4.3.1 学习逆转马尔可夫链

前向过程 $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ 是已知的、预设的。然而反向过程 $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 是未知的——给定一个加噪图像，我们无法解析地求出去噪后的图像，因为这需要知道数据分布的全局结构。

扩散模型的核心思想是：**用一个参数化的神经网络来近似反向过程**。

定义反向过程为：

$$p_\theta(\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$

其中 $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是先验分布，每一步的反向转移核为：

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\, \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\, \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

神经网络的任务是预测均值 $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$，而方差 $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$ 可以选择固定或学习。

### 4.3.2 条件反向过程的解析形式

一个深刻的数学事实是：**当条件于 $\mathbf{x}_0$ 时，反向过程有解析解**。

利用贝叶斯定理，可以证明：

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1};\, \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\, \tilde{\beta}_t \mathbf{I})$$

其中：

$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\,\mathbf{x}_t$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\,\beta_t$$

**推导**：利用 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \propto q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)$，两个高斯分布的乘积仍为高斯分布，通过配方法即可得到上述结果。这个解析形式的含义是：如果我们**知道原始干净图像** $\mathbf{x}_0$，就能精确计算出从 $\mathbf{x}_t$ 到 $\mathbf{x}_{t-1}$ 的最优去噪步骤。

### 4.3.3 神经网络预测噪声

问题在于实际生成时我们并不知道 $\mathbf{x}_0$。DDPM的关键创新在于引入**噪声预测网络** $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$。

回顾前向过程的闭合形式 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}$，我们可以将 $\mathbf{x}_0$ 表示为：

$$\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}\right)$$

将这个关系代入条件反向过程的均值公式，经过代数运算可得：

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$

这就是DDPM的核心公式：**神经网络只需预测每一步加入的噪声** $\boldsymbol{\epsilon}$，就能计算出去噪均值。这一参数化方式相比直接预测 $\mathbf{x}_0$ 或预测均值 $\boldsymbol{\mu}$，具有更好的训练稳定性和采样质量。

---

## 4.4 DDPM完整推导

### 4.4.1 训练目标：从变分下界出发

DDPM的训练目标来源于最大化数据对数似然的变分下界（Variational Lower Bound, VLB），也称为证据下界（Evidence Lower Bound, ELBO）。

$$\log p(\mathbf{x}_0) \geq \mathbb{E}_q\left[\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}\right]$$

展开后，VLB可以分解为三项之和（经过推导，利用KL散度）：

$$L_{\text{VLB}} = L_0 + L_1 + L_2 + \cdots + L_{T-1} + L_T$$

其中：

$$L_0 = -\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)$$

这是重建损失，在离散化像素空间中用离散化的logistic分布建模。

$$L_{t-1} = D_{\text{KL}}\left(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\right), \quad t = 2, \ldots, T$$

这是每一步反向转移与真实反向后验之间的KL散度。

$$L_T = D_{\text{KL}}\left(q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, p(\mathbf{x}_T)\right)$$

这是前向过程终态与先验之间的KL散度，由于不含可训练参数，在训练时可以忽略。

### 4.4.2 损失函数的简化

当反向过程的方差固定为 $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \sigma_t^2 \mathbf{I}$（其中 $\sigma_t^2$ 取 $\tilde{\beta}_t$ 或 $\beta_t$），$L_{t-1}$ 中的KL散度退化为两个高斯分布均值之间的欧氏距离：

$$L_{t-1} = \frac{1}{2\sigma_t^2}\left\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\right\|^2 + C$$

其中 $C$ 是与 $\theta$ 无关的常数。将前面推导的 $\tilde{\boldsymbol{\mu}}_t$ 和 $\boldsymbol{\mu}_\theta$ 的表达式代入，经过化简可得：

$$L_{t-1} = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right\|^2 + C$$

Ho等人发现，去掉前面的系数权重，直接使用**简化的均方误差损失**，在实践中能获得更好的生成质量：

$$L_{\text{simple}} = \mathbb{E}_{t \sim \mathcal{U}(1,T),\, \mathbf{x}_0 \sim q(\mathbf{x}_0),\, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right\|^2\right]$$

这就是DDPM最终使用的训练损失。其含义极其直观：**给定时刻 $t$ 的加噪图像 $\mathbf{x}_t$，让神经网络预测添加的噪声 $\boldsymbol{\epsilon}$，最小化预测噪声与真实噪声之间的均方误差。**

### 4.4.3 训练算法

将上述推导整合，DDPM的训练算法如下：

---

**算法1：DDPM训练**

1. **重复**
2. &emsp;从数据集中采样 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$
3. &emsp;均匀采样时间步 $t \sim \mathcal{U}(\{1, 2, \ldots, T\})$
4. &emsp;采样噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
5. &emsp;计算加噪样本 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}$
6. &emsp;计算损失 $\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right\|^2$
7. &emsp;对 $\theta$ 做梯度下降步骤
8. **直到收敛**

---

训练流程的简洁性令人赞叹：每一步只需要一次前向扩散（单步，非链式）、一次网络前向传播和一次反向传播。网络的输入是加噪图像 $\mathbf{x}_t$ 和时间步 $t$，输出是同维度的噪声预测。

### 4.4.4 采样算法

训练完成后，生成新样本只需从纯噪声出发，逐步执行反向去噪：

---

**算法2：DDPM采样**

1. 采样 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. **对** $t = T, T-1, \ldots, 1$：
3. &emsp;采样 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（若 $t > 1$，否则 $\mathbf{z} = \mathbf{0}$）
4. &emsp;计算 $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}$
5. **返回** $\mathbf{x}_0$

---

注意第3步中，最后一步（$t=1$）不添加随机噪声（$\mathbf{z} = \mathbf{0}$），这是因为最终输出是确定性的。$\sigma_t$ 的选择通常取 $\sigma_t = \sqrt{\tilde{\beta}_t}$（后验方差的根号）或 $\sigma_t = \sqrt{\beta_t}$。

由于 $T$ 通常取1000，采样时需要调用神经网络1000次，这比GAN的单次前向传播慢了三个数量级，成为扩散模型最大的实际瓶颈。

### 4.4.5 隐变量视角

从概率图模型的角度看，扩散模型可以被理解为一个层级化的隐变量模型。$\mathbf{x}_1, \ldots, \mathbf{x}_T$ 是隐变量，$\mathbf{x}_0$ 是观测变量。与VAE仅有一层隐变量不同，扩散模型拥有 $T$ 层隐变量，每一层只负责去除一小部分噪声。这种"化整为零"的策略使得每一层的优化任务都相对简单，从而降低了整体训练难度。

---

## 4.5 网络架构与时间步嵌入

### 4.5.1 U-Net骨干网络

DDPM采用U-Net作为噪声预测网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 的骨干架构。U-Net最初为医学图像分割设计，其对称的编码器-解码器结构和跳跃连接（skip connections）使其特别适合像素级预测任务。

在DDPM的U-Net中：
- **编码器**逐层下采样，提取多尺度特征
- **解码器**逐层上采样，恢复空间分辨率
- **跳跃连接**将编码器的特征直接拼接到解码器的对应层
- **残差块**（ResNet blocks）作为基本构建单元
- **自注意力层**（self-attention）嵌入在较低分辨率的特征图中，捕获长距离依赖

### 4.5.2 时间步嵌入

时间步 $t$ 作为条件信息输入网络。DDPM采用正弦位置编码（sinusoidal positional encoding），与Transformer中的位置编码类似：

$$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$

时间步编码通过一个小型MLP后，以加法方式注入U-Net的每个残差块中。这种设计使网络能够感知当前所处的去噪阶段——在早期步骤（$t$较大）需要去除大量噪声、保留粗粒度结构；在后期步骤（$t$较小）则需要精细地恢复高频细节。

---

## 4.6 训练与采样细节

### 4.6.1 噪声预测 vs $\mathbf{x}_0$ 预测

虽然DDPM选择让网络预测噪声 $\boldsymbol{\epsilon}$，但这并非唯一选择。网络也可以直接预测干净图像 $\mathbf{x}_0$，或预测均值 $\boldsymbol{\mu}$。三种参数化之间的关系为：

| 参数化 | 网络输出 | 去噪均值 |
|--------|---------|---------|
| $\boldsymbol{\epsilon}$-prediction | $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ | $\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\boldsymbol{\epsilon}}\right)$ |
| $\mathbf{x}_0$-prediction | $\hat{\mathbf{x}}_0 = f_\theta(\mathbf{x}_t, t)$ | $\frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\hat{\mathbf{x}}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t$ |
| $\boldsymbol{\mu}$-prediction | $\hat{\boldsymbol{\mu}} = \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ | $\hat{\boldsymbol{\mu}}$ |

实践中，$\boldsymbol{\epsilon}$-prediction在 $\beta_t$ 较大（即噪声较强）时表现最佳，因为预测噪声的任务在高噪声区域更稳定。而 $\mathbf{x}_0$-prediction 在低噪声区域（$t$较小时）更准确。Salimans & Ho (2022) 提出了**v-prediction**，定义 $\mathbf{v} = \sqrt{\bar{\alpha}_t}\,\boldsymbol{\epsilon} - \sqrt{1-\bar{\alpha}_t}\,\mathbf{x}_0$，作为两者之间的折中，在整个时间轴上表现出更均匀的预测难度。

### 4.6.2 方差调度

DDPM原始工作将反向过程的方差 $\sigma_t^2$ 固定为超参数。具体有两种选择：

$$\sigma_t^2 = \beta_t \quad \text{或} \quad \sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t$$

前者是前向过程的步进方差，后者是条件反向过程的后验方差。实验证明两者差异不大，但 $\tilde{\beta}_t$ 在理论上更合理——它给出了在已知 $\mathbf{x}_0$ 条件下反向步骤的最优方差。

Nichol & Dhariwal (2021) 进一步提出让网络学习对数方差 $\log \sigma_t^2$，通过在VLB损失中加入对应的KL散度项进行训练。虽然这在理论上更完备，但实践中对最终生成质量的提升有限。

### 4.6.3 采样加速

DDPM需要 $T = 1000$ 步采样，每步调用一次神经网络，这在实际应用中效率极低。采样加速是扩散模型研究的重要方向。

**DDIM（去噪扩散隐式模型）**（Song et al., 2021）是最重要的加速方法之一。DDIM发现DDPM的采样过程可以被推广为一类非马尔可夫过程，其中存在一个**确定性**的特殊情况——给定相同的初始噪声 $\mathbf{x}_T$，DDIM总是生成相同的图像。更重要的是，DDIM允许跳步采样：从 $T$ 步中选取一个子序列 $\{\tau_1, \tau_2, \ldots, \tau_S\}$（$S \ll T$），仅在这些时间步上执行去噪，将采样步数从1000减少到50甚至10步，而质量下降有限。

DDIM的更新公式为：

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \sigma_t \boldsymbol{\epsilon}_t$$

当 $\sigma_t = 0$ 时，采样过程完全确定性，称为DDIM；当 $\sigma_t = \sqrt{\tilde{\beta}_t}$ 时，退化为标准DDPM。

**高阶求解器**是另一条加速路线。扩散SDE视角（Song et al., 2021）将离散的扩散过程理解为随机微分方程，从而可以借用ODE/SDE求解器（如Euler-Maruyama法、Heun法、DPM-Solver等）进行更高效的数值求解。DPM-Solver（Lu et al., 2022）专门针对扩散ODE设计了高阶求解器，在10-20步内即可达到高质量生成。

### 4.6.4 数据预处理与归一化

在处理图像数据时，DDPM将像素值从 $[0, 255]$ 线性缩放到 $[-1, 1]$，使数据分布与高斯噪声的尺度匹配。对于离散像素值，重建损失 $L_0$ 使用离散化的logistic似然而非简单的MSE，以更好地处理像素的离散性质。

---

## 4.7 理论联系：得分匹配与朗之万动力学

扩散模型与得分匹配（Score Matching）之间存在深刻的理论联系。数据分布的得分函数定义为：

$$\nabla_{\mathbf{x}} \log q(\mathbf{x})$$

Song & Ermon (2019) 证明，噪声预测网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 与噪声条件得分网络 $\mathbf{s}_\theta(\mathbf{x}_t, t)$ 之间存在简单关系：

$$\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

即预测噪声等价于预测噪声扰动数据分布的得分函数。采样过程则对应于朗之万动力学（Langevin dynamics）的退火版本——从高噪声级别逐步退火到低噪声级别，在每个级别上执行几步朗之万采样。

这一联系不仅提供了优美的理论解释，还催生了得分匹配生成模型（Score-based Generative Models）这一并行研究路线，与DDPM殊途同归。

---

## 4.8 本章小结

本章系统介绍了扩散模型的数学基础。我们看到，前向扩散过程通过马尔可夫链逐步向数据添加噪声，其闭合形式使得训练可以高效进行；反向去噪过程通过神经网络学习逆转噪声添加，DDPM选择预测噪声作为网络目标；从变分下界出发推导出的简化的MSE损失既理论完备又实践有效。

扩散模型的美在于其简洁性：一个网络、一个损失函数、一个采样循环。然而这种简洁背后是强大的理论支撑——变分推断、得分匹配、随机微分方程等多种数学视角在此交汇。尽管采样速度仍是主要瓶颈，DDIM等加速方法已将其推进到实际可用的范围。

在下一章中，我们将讨论扩散模型的高级变体与扩展，包括条件生成、文本到图像生成以及潜在空间扩散等前沿进展。

---

**参考文献**

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*.
- Nichol, A. Q., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML 2021*.
- Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *ICLR 2021*.
- Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *NeurIPS 2019*.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., et al. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR 2021*.
- Lu, C., Zhou, Y., Bao, F., et al. (2022). DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. *NeurIPS 2022*.
- Salimans, T., & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. *ICLR 2022*.
-e 

---


# 第5章 改进的扩散模型

> "扩散模型的强大之处不仅在于其生成质量，更在于其框架的可塑性——每一项改进都打开了新的可能性。"

## 引言

第4章介绍了去噪扩散概率模型（DDPM）的基本原理，展示了其在图像生成领域的惊人潜力。然而，原始DDPM存在几个显著的局限性：采样速度慢（通常需要数百至数千步迭代）、缺乏灵活的生成控制机制、以及在高分辨率图像上计算成本高昂。本章将系统介绍针对这些问题的四项关键改进：DDIM采样加速、分数匹配与统一随机微分方程框架、无分类器引导（Classifier-Free Guidance），以及潜空间扩散模型。这些改进共同构成了现代扩散模型系统（如Stable Diffusion、DALL·E 2、Imagen等）的理论基础。

---

## 5.1 DDIM与确定性采样

### 5.1.1 从DDPM到DDIM：非马尔可夫过程的引入

回顾DDPM的前向过程，它定义了一个马尔可夫链：

$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$$

其中每一步转移为高斯分布 $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$。在反向生成时，模型需要从 $x_T \sim \mathcal{N}(0, I)$ 出发，逐步执行 $p_\theta(x_{t-1}|x_t)$，共 $T$ 步才能得到最终样本。

Song等人（2020）在论文《Denoising Diffusion Implicit Models》中提出了一个关键洞察：**前向过程不必是马尔可夫的**。他们构造了一族非马尔可夫前向过程 $\{q_\sigma(x_{1:T}|x_0)\}$，这些过程共享相同的边缘分布 $q_\sigma(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$，但具有不同的联合分布。

具体而言，DDIM定义的反向过程为：

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2 I\right)$$

这里 $\sigma_t$ 是一个自由参数，控制着反向过程的随机性。当 $\sigma_t = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}} \cdot \sqrt{\frac{1-\bar{\alpha}_t/\bar{\alpha}_{t-1}}{1}}$ 时，退化为DDPM；当 $\sigma_t = 0$ 时，反向过程变为**完全确定性的**，这就是DDIM的核心。

### 5.1.2 加速采样机制

DDIM的一个革命性特性是它允许**跳步采样**。由于非马尔可夫过程不要求相邻时间步之间的依赖关系，我们可以从完整的时间步序列 $\{1, 2, \ldots, T\}$ 中选取一个子序列 $\{\tau_1, \tau_2, \ldots, \tau_S\}$（其中 $S \ll T$），仅在这些时间步上执行反向推理。

设原始DDPM训练使用 $T = 1000$ 步，DDIM可以选择 $S = 50$ 甚至 $S = 10$ 步的子序列进行采样。确定性DDIM的更新公式为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t, t) + \sigma_t \epsilon_t$$

当 $\sigma_t = 0$ 时，上式中的噪声项消失，整个采样过程变成一个确定性映射 $x_T \mapsto x_0$。这一加速效果极为显著：在CIFAR-10上，DDPM需要1000步才能达到FID 3.17，而DDIM仅需50步即可达到FID 4.67，10步时仍为6.84——以20倍的加速换取了相对较小的质量损失。

### 5.1.3 插值与语义操作能力

确定性映射 $f: x_T \mapsto x_0$ 的存在带来了一个意外收获：**潜空间插值**。由于 $x_T$ 是标准高斯分布中的向量，我们可以像在GAN的潜空间中一样进行线性插值。

给定两张图像 $x_0^{(1)}$ 和 $x_0^{(2)}$，首先通过确定性前向过程获取对应的噪声表示：

$$x_T^{(i)} = \sqrt{\bar{\alpha}_T}\,x_0^{(i)} + \sqrt{1-\bar{\alpha}_T}\,\epsilon^{(i)}, \quad i=1,2$$

然后在噪声空间中进行球面线性插值（spherical interpolation）：

$$\tilde{x}_T(\lambda) = \frac{(1-\lambda)x_T^{(1)} + \lambda x_T^{(2)}}{\|(1-\lambda)x_T^{(1)} + \lambda x_T^{(2)}\|} \cdot \sqrt{T}$$

最后通过DDIM的确定性反向过程将 $\tilde{x}_T(\lambda)$ 映射回图像空间。实验表明，这种插值在语义层面是平滑且有意义的——例如在两个人脸之间产生自然的过渡。这一能力此前被认为是生成对抗网络的独有优势，DDIM证明了扩散模型同样具备。

此外，确定性映射还使得**图像编辑**变得可行。通过DDIM反演（DDIM Inversion），我们可以将真实图像编码到噪声空间 $x_T$，然后在该空间中进行修改后再解码回图像。这一技术被广泛应用于图像修复（inpainting）、风格迁移、以及基于文本的图像编辑等任务中，成为后续诸多编辑方法（如Prompt-to-Prompt、Pix2Pix Zero等）的基础。

---

## 5.2 分数匹配与SDE统一框架

### 5.2.1 分数函数与Langevin动力学

分数匹配（Score Matching）是另一种理解扩散模型的视角。给定数据分布 $p(x)$，其**分数函数**（score function）定义为对数概率密度的梯度：

$$s(x) = \nabla_x \log p(x)$$

分数函数指向数据密度增长最快的方向，它是生成模型的关键——如果我们知道 $s(x)$，就可以通过**Langevin动力学**从噪声中生成样本：

$$x_{t+1} = x_t + \frac{\eta}{2}\nabla_x \log p(x_t) + \sqrt{\eta}\,z_t, \quad z_t \sim \mathcal{N}(0, I)$$

当步长 $\eta \to 0$ 且步数 $\to \infty$ 时，$x_t$ 的分布收敛到 $p(x)$。然而，直接估计高维数据的分数函数是困难的，因为 $\log p(x)$ 通常未知。

Song和Ermon（2019）提出了**噪声条件分数网络**（Noise Conditional Score Network, NCSN）：通过在多个噪声级别 $\{\sigma_1, \sigma_2, \ldots, \sigma_L\}$ 下对数据施加高斯扰动 $p_{\sigma_i}(x) = \int p(y)\mathcal{N}(x;y, \sigma_i^2 I)dy$，然后训练一个神经网络 $s_\theta(x, \sigma_i)$ 来估计各噪声级别的分数函数：

$$\mathcal{L}(\theta) = \sum_{i=1}^{L} \lambda(\sigma_i) \mathbb{E}_{p_{\sigma_i}(x)} \left[\|s_\theta(x, \sigma_i) - \nabla_x \log p_{\sigma_i}(x)\|^2\right]$$

其中 $\lambda(\sigma_i) = \sigma_i^2$ 是权重因子。由于扰动后的分数函数 $\nabla_x \log p_{\sigma_i}(x)$ 可以通过去噪分数匹配（denoising score matching）高效计算，这一框架变得实用可行。

### 5.2.2 前向-反向SDE统一框架

Song等人（2020）在论文《Score-Based Generative Modeling through Stochastic Differential Equations》中提出了一个优雅的统一框架。核心思想是：当噪声级别从有限集合变为连续序列时，加噪过程可以描述为一个**随机微分方程**（SDE）：

$$dx = f(x, t)\,dt + g(t)\,dw$$

其中 $f(x, t)$ 是漂移系数，$g(t)$ 是扩散系数，$w$ 是标准维纳过程（布朗运动）。这个SDE将数据分布 $p_0$ 逐步演化为先验分布 $p_T \approx \mathcal{N}(0, I)$。

对应的**反向SDE**由Anderson（1982）的结果给出：

$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar{w}$$

其中 $\bar{w}$ 是反向时间的维纳过程。关键在于：反向SDE只需要知道各时刻的分数函数 $\nabla_x \log p_t(x)$，而这正是分数网络 $s_\theta(x, t)$ 所估计的。

Song等人展示了不同的SDE选择对应不同的扩散模型：

| SDE类型 | 漂移系数 $f(x,t)$ | 扩散系数 $g(t)$ | 对应模型 |
|---------|-------------------|-----------------|---------|
| VP-SDE | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)}$ | DDPM |
| VE-SDE | $0$ | $\sqrt{\frac{d[\sigma^2(t)]}{dt}}$ | NCSN |
| sub-VP-SDE | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})}$ | 改进的NCSN |

此外，该框架还引入了**概率流ODE**（Probability Flow ODE）：

$$dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt$$

这是一个确定性ODE，其轨迹定义了与SDE相同的边缘分布 $\{p_t(x)\}_{t=0}^{T}$，但没有随机噪声项。概率流ODE与DDIM的确定性采样本质上是等价的，它们都可以实现精确的似然计算（通过瞬时变量替换公式）和确定性生成。

这一统一框架的价值在于：它将DDPM、NCSN、以及各种变体纳入同一个数学语言体系，使得模型设计可以系统化地在SDE参数空间中进行搜索和优化。

---

## 5.3 Classifier-Free Guidance

### 5.3.1 从Classifier Guidance到Classifier-Free Guidance

条件生成是生成模型的核心需求——我们不仅希望生成逼真的图像，更希望控制生成的内容。Dhariwal和Nichol（2021）提出了**Classifier Guidance**：利用一个在噪声图像上训练的分类器 $p_\phi(y|x_t)$ 来引导生成过程。具体地，条件分数函数被修正为：

$$\nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + s \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$

其中 $s$ 是引导强度（guidance scale）。然而，这种方法需要额外训练一个噪声鲁棒的分类器，增加了复杂性。

Ho和Salimans（2022）提出了**Classifier-Free Guidance**（CFG），巧妙地消除了对额外分类器的依赖。其核心思想是：同时训练一个条件去噪模型 $\epsilon_\theta(x_t, t, c)$ 和一个无条件去噪模型 $\epsilon_\theta(x_t, t, \varnothing)$（通常通过在训练时以一定概率随机丢弃条件 $c$ 来实现），然后将两者的预测进行线性组合：

$$\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w \cdot [\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)]$$

其中 $w$ 是引导强度参数。当 $w = 1$ 时，退化为标准的条件生成；当 $w > 1$ 时，模型被"推向"更强地遵循条件信号的方向。

从分数函数的角度理解，CFG等价于从以下隐式分布中采样：

$$\hat{p}_\theta(x_t|c) \propto p_\theta(x_t) \cdot \left(\frac{p_\theta(x_t|c)}{p_\theta(x_t)}\right)^w$$

当 $w > 1$ 时，这个分布比原始条件分布更加"尖锐"——它放大了条件信号与无条件信号之间的差异，使得生成结果更加忠于给定的条件。

### 5.3.2 训练策略与条件丢弃

CFG的训练极为简洁。在每个训练步骤中，以概率 $p_\text{uncond}$（通常取10%~20%）将条件 $c$ 替换为空标记 $\varnothing$：

$$\mathcal{L}_\text{CFG} = \mathbb{E}_{t, x_0, \epsilon, c} \left[\|\epsilon - \epsilon_\theta(x_t, t, c')\|^2\right]$$

其中 $c'$ 以概率 $1-p_\text{uncond}$ 等于真实条件 $c$，以概率 $p_\text{uncond}$ 等于 $\varnothing$。这意味着单个网络同时学会了条件生成和无条件生成，无需额外的分类器。

### 5.3.3 引导强度的调节

引导强度 $w$ 对生成质量有显著影响：

- **$w = 1$**：标准条件生成，多样性高但条件一致性可能不足。
- **$w > 1$**（如 $w = 7.5$，Stable Diffusion的默认值）：生成结果更加清晰、更加符合文本描述，但多样性降低。过大的 $w$ 会导致图像过度饱和、出现伪影。
- **$w < 1$**：弱于标准条件生成，更加多样但可能偏离条件。

在文本到图像生成中，CFG已经成为事实上的标准技术。Stable Diffusion使用 $w = 7.5$，Imagen使用 $w = 1.5$（在其归一化尺度下）。有趣的是，CFG不仅提升了条件一致性，还显著提高了生成图像的视觉质量——即使在无条件生成中，适当增大 $w$ 也能改善结果，这表明CFG本质上起到了一种"锐化"生成分布的作用。

---

## 5.4 潜空间扩散模型

### 5.4.1 像素空间扩散的计算瓶颈

前述所有改进虽然提升了扩散模型的采样速度和生成质量，但一个根本性的计算瓶颈仍然存在：**像素空间的维度太高**。一张 $512 \times 512 \times 3$ 的RGB图像有786,432个维度，在如此高维空间中进行扩散过程，每一步都需要对全分辨率图像进行去噪，计算代价极为高昂。

以DDPM在 $256 \times 256$ 图像上的训练为例，单个去噪U-Net需要处理约20万个像素，每个像素包含3个通道。对于 $512 \times 512$ 甚至更高分辨率的图像，计算量呈二次增长，使得大规模训练变得不切实际。

### 5.4.2 Stable Diffusion架构

Rombach等人（2022）在论文《High-Resolution Image Synthesis with Latent Diffusion Models》中提出了**潜空间扩散模型**（Latent Diffusion Model, LDM），巧妙地将扩散过程从像素空间转移到低维潜空间。Stable Diffusion正是基于这一架构构建的，它由三个核心组件组成：

**（1）变分自编码器（VAE）**

VAE的编码器 $\mathcal{E}$ 将图像 $x \in \mathbb{R}^{H \times W \times 3}$ 映射到低维潜空间表示 $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$，其中空间分辨率通常压缩 $f$ 倍（Stable Diffusion中 $f = 8$），即 $h = H/f$，$w = W/f$。通道数 $c$ 通常为4。因此，$512 \times 512 \times 3$ 的图像被压缩为 $64 \times 64 \times 4$ 的潜表示，维度降低了48倍。

VAE的解码器 $\mathcal{D}$ 负责将潜表示还原为图像：$\hat{x} = \mathcal{D}(z)$。VAE通过重建损失和KL散度正则化进行训练：

$$\mathcal{L}_\text{VAE} = \|x - \mathcal{D}(\mathcal{E}(x))\|^2 + \lambda_\text{KL} \cdot D_\text{KL}(q(z|x) \| \mathcal{N}(0, I))$$

为了提升重建质量，Stable Diffusion的VAE采用了感知损失（perceptual loss）和对抗损失（patch-based adversarial loss）：

$$\mathcal{L}_\text{VAE} = \mathcal{L}_\text{rec} + \lambda_\text{perc}\mathcal{L}_\text{perc} + \lambda_\text{adv}\mathcal{L}_\text{adv} + \lambda_\text{KL}\mathcal{L}_\text{KL}$$

**（2）潜空间U-Net**

在潜空间中，扩散模型的前向和反向过程完全在 $z$ 空间进行。去噪网络采用U-Net架构，但在 $64 \times 64$ 的分辨率上操作，而非 $512 \times 512$。

Stable Diffusion的U-Net包含：
- **下采样路径**：多个ResNet块和自注意力层，逐步降低空间分辨率。
- **中间块**：在最低分辨率上进行全局特征交互。
- **上采样路径**：与下采样路径对称，通过跳跃连接（skip connection）保留多尺度信息。
- **时间步嵌入**：通过正弦位置编码将时间步 $t$ 注入每一层。
- **交叉注意力层**：接收文本编码器的输出，实现文本条件的融合。

U-Net的输入是噪声潜表示 $z_t$、时间步 $t$ 和文本条件 $c_\text{text}$，输出是预测的噪声 $\hat{\epsilon}$（或预测的 $z_0$，取决于参数化方式）。训练目标为：

$$\mathcal{L}_\text{LDM} = \mathbb{E}_{z_0, t, \epsilon \sim \mathcal{N}(0,I), c_\text{text}} \left[\|\epsilon - \epsilon_\theta(z_t, t, c_\text{text})\|^2\right]$$

**（3）文本编码器（CLIP Text Encoder）**

文本条件通过预训练的CLIP文本编码器进行处理。Stable Diffusion v1.x使用CLIP ViT-L/14的文本编码器，将文本提示（prompt）编码为77个token的序列，每个token维度为768。这些文本嵌入通过U-Net中的交叉注意力机制与视觉特征交互：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

其中 $Q$ 来自视觉特征（通过线性投影），$K$ 和 $V$ 来自文本嵌入。这使得模型能够根据文本描述精确地控制生成内容。

Stable Diffusion v2.x和后续版本改用了OpenCLIP的文本编码器，具有更大的容量和更好的文本理解能力。

### 5.4.3 两阶段训练策略

LDM的训练分为两个独立的阶段：

**第一阶段：VAE训练**

在大规模图像数据集上训练VAE，学习高质量的图像编码-解码。这一步的目标是获得一个既能高压缩又能忠实重建的潜空间。VAE训练完成后即固定参数，不再更新。

**第二阶段：潜空间扩散模型训练**

在VAE的潜空间中训练扩散模型。具体流程为：
1. 从数据集中采样图像 $x$；
2. 通过冻结的VAE编码器获取潜表示 $z_0 = \mathcal{E}(x)$；
3. 随机采样时间步 $t \sim \text{Uniform}(\{1, \ldots, T\})$ 和噪声 $\epsilon \sim \mathcal{N}(0, I)$；
4. 构造噪声潜表示 $z_t = \sqrt{\bar{\alpha}_t}\,z_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$；
5. 训练U-Net预测噪声 $\epsilon_\theta(z_t, t, c_\text{text})$。

这种两阶段策略的优雅之处在于：VAE的训练和扩散模型的训练完全解耦，可以分别优化。VAE专注于压缩效率，扩散模型专注于生成质量。此外，同一个VAE可以搭配不同分辨率或不同条件的扩散模型，提供了极大的灵活性。

### 5.4.4 计算效率的质变

潜空间扩散带来的效率提升是革命性的。以 $512 \times 512$ 图像生成为例：

| 模型 | 操作空间 | 分辨率 | 参数量 | 单步耗时 |
|------|---------|--------|-------|---------|
| 像素空间DDPM | $\mathbb{R}^{512\times512\times3}$ | $512 \times 512$ | ~2B+ | 数秒 |
| Stable Diffusion | $\mathbb{R}^{64\times64\times4}$ | $64 \times 64$ | ~860M | ~50ms（GPU） |

维度从786,432降低到16,384，压缩比达48倍。值得注意的是，这种压缩是有损的——VAE无法完美重建所有细节，尤其是在高频纹理和微小文字方面。然而，扩散模型的生成过程可以在一定程度上弥补这些损失，因为模型学习到的先验知识能够"补全"潜空间中缺失的细节。实际应用中，Stable Diffusion的VAE在感知质量上表现出色，其重建结果在人眼主观评价中几乎与原图无异。这意味着：（1）训练成本大幅降低，使得在消费级GPU上微调成为可能；（2）采样速度显著提升，50步DDIM采样在现代GPU上可在几秒内完成；（3）内存占用大幅减少，使得批处理和高分辨率生成变得可行。

---

## 5.5 技术协同与现代系统

上述四项改进并非孤立存在，而是协同工作的。一个典型的现代文本到图像扩散系统（如Stable Diffusion）的工作流程如下：

1. **文本编码**：CLIP文本编码器将输入提示编码为条件嵌入。
2. **初始化**：从标准高斯分布采样 $z_T \sim \mathcal{N}(0, I)$。
3. **迭代去噪**（DDIM采样，50步）：
   - U-Net预测条件噪声 $\epsilon_\theta(z_t, t, c_\text{text})$ 和无条件噪声 $\epsilon_\theta(z_t, t, \varnothing)$；
   - 通过CFG组合：$\hat{\epsilon} = \epsilon_\theta(z_t, t, \varnothing) + w \cdot [\epsilon_\theta(z_t, t, c_\text{text}) - \epsilon_\theta(z_t, t, \varnothing)]$；
   - DDIM更新：$z_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{z}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\hat{\epsilon}$。
4. **解码**：VAE解码器将 $z_0$ 还原为像素空间图像。

这种架构的影响力是深远的。它使得扩散模型从学术研究走向了大规模的实际应用——Midjourney、DALL·E 3、Stable Diffusion XL等系统都建立在这些基础之上，影响了设计、娱乐、教育等众多领域。

从更宏观的视角来看，扩散模型的改进历程体现了一个重要的研究范式：首先建立坚实的理论基础（DDPM），然后从不同角度（采样效率、理论统一、条件控制、计算效率）进行系统性优化，最终将各组件有机整合为实用系统。这种从理论到实践、从单一技术到系统集成的发展路径，为整个生成模型领域提供了宝贵的经验。

---

## 5.6 本章小结

本章介绍了扩散模型领域的四项关键改进。DDIM通过引入非马尔可夫前向过程实现了确定性采样和显著的加速效果；分数匹配与SDE统一框架将扩散模型、分数匹配和Langevin动力学纳入同一理论体系；Classifier-Free Guidance以极简的训练策略实现了强大的条件生成控制；潜空间扩散模型通过将扩散过程转移到低维空间解决了计算效率的根本瓶颈。

这些改进共同推动了扩散模型从理论研究走向实际应用，为后续章节讨论的条件生成、图像编辑、3D生成等高级应用奠定了基础。

---

**参考文献**

1. Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. *International Conference on Learning Representations (ICLR)*.
2. Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *Advances in Neural Information Processing Systems (NeurIPS)*.
3. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-Based Generative Modeling through Stochastic Differential Equations. *International Conference on Learning Representations (ICLR)*.
4. Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *NeurIPS Workshop on Deep Generative Models and Downstream Applications*.
5. Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. *Advances in Neural Information Processing Systems (NeurIPS)*.
6. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
7. Anderson, B. D. (1982). Reverse-time diffusion equation models. *Stochastic Processes and their Applications*, 12(3), 313-326.
-e 

---


# 第6章 扩散模型的理论分析

> "理解扩散模型为什么有效，比知道它怎么工作更重要。"

## 引言

前两章介绍了扩散模型的基本框架和主要改进方法。但一个自然的问题是：扩散模型为什么能生成如此高质量的样本？其背后的数学结构是什么？与其他生成模型（如VAE、GAN）有什么深层联系？本章将从概率论、随机微分方程和信息论的角度，对扩散模型进行系统的理论分析。

---

## 6.1 变分下界与训练目标

### 6.1.1 扩散模型的变分下界

与其他隐变量模型类似，扩散模型可以通过最大化数据的对数似然的变分下界（ELBO）来训练。对于隐变量 $x_{1:T}$ 和观测数据 $x_0$，变分下界为：

$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)} \left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

通过一系列数学推导，可以将ELBO分解为三个可解释的项：

$$\mathcal{L} = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \underbrace{\sum_{t=2}^{T} D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} + \underbrace{\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]}_{L_0}$$

**$L_T$ 项**：前向过程末端与先验的KL散度。由于前向过程的设计，当 $T$ 足够大时，$q(x_T|x_0)$ 近似于标准正态分布，此项趋近于零。

**$L_{t-1}$ 项**：反向过程每一步与真实后验的KL散度。这是训练的核心——让模型学习正确的反向转移。

**$L_0$ 项**：重建损失，衡量从 $x_1$ 重建 $x_0$ 的质量。

### 6.1.2 简化训练目标的推导

DDPM的训练目标是预测噪声 $\epsilon$，但这并非唯一的参数化方式。通过数学推导，可以证明以下三种参数化方式是等价的：

**噪声预测**：$\epsilon_\theta(x_t, t) \approx \epsilon$（DDPM原始方式）

**均值预测**：$\mu_\theta(x_t, t) \approx \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)$

**$x_0$ 预测**：$x_0^\theta(x_t, t) \approx \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}$

Ho等人（2020）发现，噪声预测参数化在实践中效果最好。一个可能的解释是：噪声预测对所有时间步的权重更加均匀，而 $x_0$ 预测在小 $t$ 时（信号强、噪声小）会给予过大的权重。

### 6.1.3 简化目标与完整目标的关系

DDPM使用简化目标 $\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$，去掉了ELBO中每项的权重系数。这个简化有以下理论动机：

- 简化目标等价于对ELBO的均匀加权近似
- 实验表明简化目标生成质量更好（尽管ELBO可能更差）
- 这暗示似然与感知质量之间存在张力

---

## 6.2 分数匹配视角

### 6.2.1 分数函数与分数网络

数据分布 $p(x)$ 的分数函数定义为：

$$s(x) = \nabla_x \log p(x)$$

分数函数指向数据密度增长最快的方向，是分布的"梯度场"。训练一个分数网络 $s_\theta(x)$ 来估计这个函数，就是分数匹配的核心思想。

Song & Ermon（2019）证明了一个深刻的联系：**扩散模型的噪声预测目标本质上就是去噪分数匹配（Denoising Score Matching）**。具体地：

$$\epsilon_\theta(x_t, t) = -\sqrt{1-\bar{\alpha}_t} \cdot s_\theta(x_t, t)$$

其中 $s_\theta(x_t, t)$ 是在噪声水平 $t$ 下对 $\nabla_{x_t} \log q(x_t)$ 的估计。这意味着预测噪声和估计分数是同一件事的两种表述。

### 6.2.2 Langevin 动力学

有了分数函数，就可以通过朗之万动力学（Langevin Dynamics）从分布中采样：

$$x_{t+1} = x_t + \frac{\eta}{2} \nabla_x \log p(x_t) + \sqrt{\eta} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

当步长 $\eta \to 0$ 且步数 $\to \infty$ 时，$x_t$ 的分布收敛到 $p(x)$。

扩散模型可以看作是"学习了的"朗之万动力学：噪声预测网络提供了每一步的分数估计，而前向-反向框架提供了一个从粗到细的多尺度采样策略。

### 6.2.3 噪声条件分数网络（NCSN）

Song & Ermon（2019）提出了噪声条件分数网络，使用多个噪声水平训练分数网络。这与DDPM的时间步条件异曲同工：

$$\mathcal{L}_{\text{NCSN}} = \sum_{i=1}^{L} \lambda(\sigma_i) \mathbb{E}_{x \sim p_{\text{data}}} \mathbb{E}_{\tilde{x} \sim q(\tilde{x}|x)} \left[\|s_\theta(\tilde{x}, \sigma_i) - \nabla_{\tilde{x}} \log q(\tilde{x}|x)\|^2\right]$$

其中 $\{\sigma_1, \ldots, \sigma_L\}$ 是噪声水平序列，$\lambda(\sigma_i)$ 是权重函数。

---

## 6.3 随机微分方程框架

### 6.3.1 连续时间扩散过程

Song等人（2021）提出了用随机微分方程（SDE）统一所有扩散模型的框架。在连续时间极限下，前向扩散过程可以表示为：

$$dx = f(x, t)dt + g(t)dw$$

其中 $f(x, t)$ 是漂移系数，$g(t)$ 是扩散系数，$w$ 是标准维纳过程（布朗运动）。

对应的反向SDE为：

$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

其中 $\bar{w}$ 是反向维纳过程。关键在于：**只要知道分数函数 $\nabla_x \log p_t(x)$，就可以从噪声生成样本**。

### 6.3.2 三种SDE形式

Song等人（2021）展示了三种具体的SDE形式，它们分别对应不同类型的扩散模型：

**VP-SDE（方差保持）**：对应DDPM。$f(x, t) = -\frac{1}{2}\beta(t)x$，$g(t) = \sqrt{\beta(t)}$。方差始终保持在1附近。

**VE-SDE（方差爆炸）**：对应SMLD/NCSN。$f(x, t) = 0$，$g(t) = \sqrt{\frac{d[\sigma^2(t)]}{dt}}$。方差随时间增长而爆炸。

**sub-VP-SDE**：VP-SDE的变体，方差增长更慢，在某些场景下表现更好。

### 6.3.3 概率流ODE

除了随机的SDE，还存在一个确定性的常微分方程（ODE），称为概率流ODE，它与SDE共享相同的边缘分布：

$$dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt$$

概率流ODE的重要性在于：

- **精确似然计算**：可以通过ODE求解器精确计算对数似然
- **确定性生成**：给定初始噪声，生成结果完全确定
- **高效的ODE求解器**：可以使用自适应步长的ODE求解器加速采样

### 6.3.4 训练统一目标

在SDE框架下，所有扩散模型的训练目标统一为：

$$\mathcal{L} = \mathbb{E}_{t \sim U(0,T)} \mathbb{E}_{x_0 \sim p_{\text{data}}} \mathbb{E}_{x_t \sim p_{t|0}(x_t|x_0)} \left[\lambda(t) \|s_\theta(x_t, t) - \nabla_{x_t} \log p_{t|0}(x_t|x_0)\|^2\right]$$

其中 $\lambda(t)$ 是时间步权重函数。DDPM的简化目标对应 $\lambda(t) = 1$，而完整ELBO对应 $\lambda(t) = \beta(t)^2 / (2\sigma^2(t))$。

---

## 6.4 与VAE的深层联系

### 6.4.1 扩散模型作为层级VAE

扩散模型可以看作是一个特殊的层级VAE，其中：

- 隐变量数量等于时间步数 $T$（通常是1000）
- 编码器（前向过程）是固定的，不需要学习
- 解码器（反向过程）需要学习
- 所有隐变量共享同一个网络

与标准VAE相比，扩散模型的"编码器"是预定义的高斯噪声过程，这消除了编码器学习的不确定性，使得训练更加稳定。

### 6.4.2 潜空间VAE的联系

Stable Diffusion等潜空间扩散模型进一步加强了与VAE的联系：

```
输入图像 → VAE编码器 → 潜空间表示 → 扩散模型在潜空间操作 → VAE解码器 → 输出图像
```

这实际上是VAE和扩散模型的组合：VAE负责学习一个好的潜空间表示，扩散模型负责在该潜空间中进行高质量生成。

### 6.4.3 似然与感知质量的权衡

VAE和扩散模型都面临一个根本性的权衡：

- **最大化ELBO（似然）**：倾向于覆盖所有数据模式，但可能生成模糊样本
- **最大化感知质量**：倾向于生成清晰样本，但可能丢失多样性

扩散模型通过多步去噪过程部分缓解了这个权衡——每一步只需要做"微小"的修正，因此每一步的误差都很小，累积误差也可以控制。

---

## 6.5 信息论分析

### 6.5.1 逐层信息压缩

从信息论角度，扩散模型的反向过程可以理解为逐层增加信息量：

- $x_T$（纯噪声）：信息量 = 0（或噪声的熵）
- $x_{T-1}$：开始注入少量结构信息
- ...
- $x_1$：大部分结构信息已恢复
- $x_0$：完整的样本信息

每一步去噪都是一个"信息恢复"过程：模型从带噪输入中恢复出被噪声掩盖的信号。这与率失真理论（Rate-Distortion Theory）有深刻的联系。

### 6.5.2 去噪即压缩

一个有趣的视角是：去噪过程可以看作是一种"解压缩"。

前向过程（加噪）= 有损压缩：$x_0 \to x_t$，信息逐渐丢失
反向过程（去噪）= 解压缩：$x_t \to x_0$，信息逐渐恢复

噪声预测网络 $\epsilon_\theta(x_t, t)$ 实际上在做的是：给定压缩版本 $x_t$，预测被压缩掉的部分（噪声）。这与传统压缩-解压缩框架的思路一致。

### 6.5.3 率失真视角

Tibshirani（2023）等人建立了扩散模型与率失真理论的联系。在率失真框架下：

- 率（Rate）对应于去噪步数：步数越多，传输的信息越多
- 失真（Distortion）对应于生成质量：步数越多，质量越好

这为理解扩散模型的加速采样提供了理论基础：加速采样本质上是在率失真曲线上寻找更高效的点。

---

## 6.6 最优传输视角

### 6.6.1 扩散与最优传输

最优传输（Optimal Transport）研究如何以最小的"成本"将一个分布变换为另一个分布。扩散模型的前向过程可以看作一种从数据分布到高斯分布的传输，反向过程则是从高斯分布到数据分布的传输。

De Bortoli等人（2021）证明，在一定条件下，扩散模型的反向过程近似于最优传输映射。这为设计更高效的扩散过程提供了理论指导。

### 6.6.2 OT-ODE与OT-SDE

基于最优传输理论，可以设计更高效的扩散过程：

**OT-ODE**：使用最优传输映射的确定性ODE，采样路径最短。

**OT-SDE**：结合最优传输和随机性的SDE，在效率和多样性之间取得平衡。

实验表明，基于最优传输的扩散过程可以在更少的步数内达到更好的生成质量。

---

## 6.7 收敛性与逼近理论

### 6.7.1 生成误差分解

扩散模型的总生成误差可以分解为三个部分：

$$\text{总误差} = \text{近似误差} + \text{估计误差} + \text{离散化误差}$$

**近似误差**：由于网络容量有限，无法完美表示分数函数。随网络增大而减小。

**估计误差**：由于训练数据有限，分数估计存在统计误差。随数据量增大而减小。

**离散化误差**：由于使用有限步数进行采样。随步数增加而减小。

### 6.7.2 收敛保证

Chen等人（2023）在一定正则性假设下，证明了扩散模型的收敛性：当网络足够大、数据足够多、步数足够多时，生成分布可以在Wasserstein距离下任意接近真实数据分布。

这些理论结果虽然有较强的假设条件，但为扩散模型的有效性提供了数学基础。

---

## 6.8 本章小结

本章从多个理论角度分析了扩散模型：

- **变分下界**：提供了训练目标的概率论基础
- **分数匹配**：揭示了扩散模型与分数函数估计的等价性
- **SDE框架**：统一了不同类型的扩散模型，提供了连续时间视角
- **VAE联系**：将扩散模型置于生成模型的统一框架中
- **信息论分析**：解释了去噪即信息恢复的本质
- **最优传输**：为设计高效扩散过程提供了理论指导
- **收敛性**：为扩散模型的有效性提供了数学保证

这些理论分析不仅加深了我们对扩散模型的理解，也为设计更好的扩散模型提供了指导。下一章将从另一个角度——视频与3D生成——探讨扩散模型的应用前沿。

---

*参考文献：Song et al. (2021) Score-Based SDE, Ho et al. (2020) DDPM, Song & Ermon (2019) NCSN, De Bortoli et al. (2021) OT, Chen et al. (2023) Convergence*
-e 

---


# 第7章 视频与3D生成

> "从静态图像到动态世界，生成式AI正在学习理解时间、空间与三维结构。"

## 7.1 引言

前几章讨论的生成模型主要聚焦于二维图像的生成。然而，真实世界是三维的、动态的。如何让生成模型跨越静态图像的边界，进入视频、三维场景乃至音频的生成，是近年来生成式AI领域最具挑战性也最激动人心的前沿方向。

本章将系统介绍视频生成、三维内容生成、音频生成以及多模态统一生成的核心方法与最新进展。这些方向不仅在学术上具有深刻意义，更在影视制作、游戏开发、虚拟现实、自动驾驶仿真等领域展现出巨大的应用潜力。

---

## 7.2 视频生成

视频本质上是一组具有时间连续性的图像序列。与图像生成相比，视频生成需要同时建模空间结构与时间动态，其核心挑战在于：（1）如何在保持帧间一致性的同时生成丰富的时间变化；（2）如何在计算可承受的范围内处理高维数据。

### 7.2.1 时空扩散模型

扩散模型在图像生成中取得的成功自然被扩展到视频领域。时空扩散模型的核心思想是在扩散过程中同时建模空间与时间维度。

**基础框架。** 给定一段包含 $T$ 帧的视频 $\mathbf{x} = (x_1, x_2, \ldots, x_T)$，其中每帧 $x_t \in \mathbb{R}^{H \times W \times 3}$，前向扩散过程定义为：

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I})$$

其中 $\beta_t$ 为噪声调度参数。反向去噪过程学习条件分布 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$，训练目标为简化的去噪损失：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

其中 $\boldsymbol{\epsilon}_\theta$ 是去噪网络，通常采用3D U-Net或Transformer架构来同时捕获空间与时间信息。

**Stable Video Diffusion (SVD)。** SVD（Blattmann et al., 2023）是在Stable Diffusion图像模型基础上扩展的视频生成模型。其核心设计包括：（1）在预训练的2D U-Net中插入时间注意力层（temporal attention layers），使模型能够学习帧间的时间依赖关系；（2）采用图像-视频联合训练策略，首先在大规模图像数据上预训练空间层，然后在视频数据上微调时间层。SVD的网络架构可以形式化为：

$$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) = \text{TemporalBlock}\left(\text{SpatialBlock}(\mathbf{x}_t, t, c)\right)$$

其中 SpatialBlock 为空间自注意力与卷积层，TemporalBlock 为沿时间轴的自注意力层，$c$ 为条件信号（如参考图像）。SVD采用了级联式生成策略：先生成低分辨率视频，再通过时间超分辨率模型逐级提升分辨率与时长。

**Video Diffusion Models (VDM)。** Ho et al.（2022）提出的VDM是首个将扩散模型完整应用于视频生成的工作。VDM直接在视频的潜空间（latent space）中进行扩散，通过将视频编码为低维潜表示来降低计算成本：

$$\mathcal{L}_{\text{VDM}} = \mathbb{E}_{\mathbf{z}_0, \boldsymbol{\epsilon}, t}\left[\lambda(t)\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, c)\|^2\right]$$

其中 $\mathbf{z}_0 = \mathcal{E}(\mathbf{x})$ 为视频编码器输出的潜表示，$\lambda(t)$ 为时间依赖的权重函数。

### 7.2.2 Sora与大规模视频生成

OpenAI于2024年发布的Sora代表了视频生成领域的范式转变。Sora的关键创新并非单一的技术突破，而是将多种已知技术在前所未有的规模上整合。

**Patch化视频表示。** Sora将视频分割为时空patch（spacetime patches），类似于ViT将图像分割为patch的方式。给定视频 $\mathbf{x} \in \mathbb{R}^{T \times H \times W \times 3}$，通过3D patch embedding将其映射为token序列：

$$\mathbf{z} = \text{Patchify}(\mathbf{x}) \in \mathbb{R}^{N \times d}$$

其中 $N = \frac{T}{p_t} \times \frac{H}{p_h} \times \frac{W}{p_w}$ 为patch数量，$(p_t, p_h, p_w)$ 为时空patch大小，$d$ 为嵌入维度。这种表示使得模型可以处理不同分辨率、时长和宽高比的视频。

**Diffusion Transformer (DiT) 架构。** Sora采用了DiT作为骨干网络，将Transformer的强大序列建模能力与扩散过程相结合。DiT的核心组件包括：

$$\text{DiTBlock}(\mathbf{z}) = \mathbf{z} + \text{MLP}\left(\text{AdaLN}(\text{MSA}(\mathbf{z}), t, c)\right)$$

其中 AdaLN（Adaptive Layer Normalization）通过时间步 $t$ 和条件 $c$ 自适应调节特征，MSA为多头自注意力机制。DiT的优势在于其良好的scaling特性——更大的模型和更多的数据能够持续提升生成质量。

**Scaling规律。** Sora的一个重要发现是视频生成模型遵循与大语言模型类似的scaling规律。计算量（以FLOPs衡量）与生成质量之间存在幂律关系：

$$\text{FID}(C) \propto C^{-\alpha}$$

其中 $C$ 为训练计算量，$\alpha$ 为scaling指数。这一发现表明，通过持续扩大模型规模和数据量，视频生成质量还有巨大的提升空间。

### 7.2.3 自回归视频生成

与扩散模型的整体生成范式不同，自回归方法将视频生成分解为逐步预测下一帧（或下一帧的token）的过程。

**VideoGPT（Yan et al., 2021）。** VideoGPT采用两阶段方法：首先使用VQ-VAE将视频编码为离散token序列，然后用自回归Transformer建模token的联合分布：

$$p(\mathbf{z}) = \prod_{i=1}^{N} p(z_i | z_1, z_2, \ldots, z_{i-1})$$

这种方法的优势在于可以利用成熟的语言模型技术（如GPT架构），但缺点是自回归解码速度较慢，且长程依赖建模面临挑战。

**VideoPoet（Kondratyuk et al., 2024）。** Google的VideoPoet采用了多模态token化策略，将视频、音频、文本统一为离散token序列，并在一个自回归Transformer中进行联合建模。其训练目标为：

$$\mathcal{L}_{\text{AR}} = -\sum_{i=1}^{N} \log p_\theta(z_i | z_{<i}, c)$$

其中 $c$ 可以是文本描述、参考图像或音频信号。这种统一的token化方案为多模态生成提供了优雅的框架。

### 7.2.4 时空一致性与长视频生成

生成长视频（数十秒至数分钟）的核心难题是维持时间一致性与语义连贯性。当前主要的解决方案包括：

**滑动窗口策略。** 将长视频分解为重叠的短片段，每个片段在条件信号（如全局嵌入或前一片段的末帧）指导下生成。设窗口大小为 $L$，滑动步长为 $S$，则第 $k$ 个片段的生成可以表示为：

$$\mathbf{x}^{(k)} = \text{Gen}(c, \mathbf{x}_{kS+L-S:kS+L}), \quad k = 0, 1, 2, \ldots$$

**分层生成。** 先生成关键帧序列建立全局叙事结构，再在关键帧之间插值生成中间帧。这种方法能够更好地控制长程语义，但需要额外的关键帧规划模块。

**运动一致性约束。** 为了维持帧间运动的平滑性，一些方法在训练损失中引入光流一致性项：
$$\mathcal{L}_{\text{flow}} = \sum_{t=1}^{T-1} \|\mathbf{w}_t - \hat{\mathbf{w}}_t\|_1$$
其中 $\mathbf{w}_t$ 为第 $t$ 帧到第 $t+1$ 帧的真实光流，$\hat{\mathbf{w}}_t$ 为从生成帧估计的光流。此外，自适应帧率采样策略也被用于在训练中平衡快速运动和慢速运动场景的学习。

---

## 7.3 三维内容生成

三维生成的目标是从文本、图像或其他模态的输入中生成三维物体或场景。与视频相比，三维生成需要建模完整的空间结构，包括几何形状、外观材质和光照条件。

### 7.3.1 点云生成

点云是三维数据最直接的表示形式之一，由一组无序的三维点 $\mathcal{P} = \{p_1, p_2, \ldots, p_N\}$，$p_i \in \mathbb{R}^3$（可附加法线、颜色等属性）组成。

**Point-E（Nichol et al., 2022）。** OpenAI的Point-E采用两阶段流程：（1）文本到图像模型生成参考图像；（2）图像到点云模型将参考图像转换为3D点云。第二阶段使用扩散模型在点云空间中进行生成：

$$q(\mathcal{P}_t | \mathcal{P}_{t-1}) = \mathcal{N}(\mathcal{P}_t; \sqrt{\bar{\alpha}_t}\,\mathcal{P}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$。去噪网络采用基于Transformer的架构，对点云中的每个点预测噪声分量。

**点云生成中的置换不变性。** 点云的本质特点是点的顺序无关性。生成模型需要满足置换不变性约束。对于函数 $f$，置换不变性要求：

$$f(\pi(\mathcal{P})) = f(\mathcal{P}), \quad \forall \pi \in \mathfrak{S}_N$$

其中 $\mathfrak{S}_N$ 为 $N$ 个元素的对称群。扩散模型天然满足这一约束，因为每个点的去噪过程独立于点的排列顺序。

点云生成的优势在于其表示简洁且易于与深度学习模型集成，但其缺点是缺乏拓扑信息，无法直接用于渲染和物理仿真，通常需要后续的表面重建步骤。此外，点云的固定点数限制了生成结果的分辨率，自适应点密度分配是当前的研究热点。

### 7.3.2 网格生成

三角网格（Triangle Mesh）是计算机图形学中最广泛使用的三维表示，由顶点集合 $\mathbf{V} \in \mathbb{R}^{M \times 3}$ 和面集合 $\mathbf{F} \subset \{1,\ldots,M\}^3$ 组成。

**PolyGen（Nash et al., 2020）。** Google的PolyGen开创性地将Transformer应用于网格生成。它将网格生成分解为两步：（1）自回归地预测顶点序列；（2）自回归地预测面（三角形）序列。顶点预测的概率模型为：

$$p(\mathbf{V}) = \prod_{i=1}^{M} p(v_i | v_1, \ldots, v_{i-1})$$

面的预测则以顶点为条件：

$$p(\mathbf{F}|\mathbf{V}) = \prod_{j=1}^{|\mathbf{F}|} p(f_j | f_1, \ldots, f_{j-1}, \mathbf{V})$$

**MeshDiffusion（Siddiqui et al., 2023）。** 该方法将网格表示为四面体网格（tetrahedral mesh）上的信号，然后在该结构化表示上应用扩散模型。通过四面体化的体表示，每个顶点可以关联标量场值，扩散过程在这些标量值上进行，最终通过Marching Tetrahedra提取表面网格。

### 7.3.3 NeRF生成

神经辐射场（NeRF）通过一个连续函数 $F_\theta$ 将三维坐标 $(x, y, z)$ 和观察方向 $(\theta, \phi)$ 映射为颜色和密度：

$$F_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

其中 $\mathbf{c} \in \mathbb{R}^3$ 为RGB颜色，$\sigma \in \mathbb{R}^+$ 为体积密度。渲染方程通过体渲染积分得到像素颜色：

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

其中 $T(t) = \exp\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)$ 为累积透射率，$\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ 为射线。

**DreamFusion（Poole et al., 2022）。** DreamFusion提出了Score Distillation Sampling（SDS），利用预训练的2D扩散模型作为监督信号来优化NeRF。给定文本描述 $y$，SDS损失定义为：

$$\mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \boldsymbol{\epsilon}}\left[w(t) \left(\boldsymbol{\epsilon}_\phi(\mathbf{z}_t; t, y) - \boldsymbol{\epsilon}\right) \frac{\partial \mathbf{z}}{\partial \theta}\right]$$

其中 $\mathbf{z}$ 是从NeRF渲染并编码到潜空间的图像，$\boldsymbol{\epsilon}_\phi$ 是预训练扩散模型的去噪预测，$w(t)$ 为权重函数。直观上，SDS将扩散模型的去噪梯度反向传播到NeRF参数 $\theta$，引导NeRF生成与文本描述一致的三维内容。

**Magic3D（Lin et al., 2023）。** Magic3D采用粗到细（coarse-to-fine）策略：先在低分辨率下用NeRF进行粗略优化，再转换为DMTet（Deep Marching Tetrahedra）表示进行高分辨率精细化。这种两阶段方法在质量和效率之间取得了更好的平衡。

### 7.3.4 3D Gaussian Splatting生成

3D Gaussian Splatting（3DGS）（Kerbl et al., 2023）是近年来兴起的一种显式三维表示方法。场景由一组三维高斯原语（Gaussian primitives）表示：

$$G(\mathbf{x}) = \sum_{i=1}^{N} c_i \cdot \alpha_i \cdot \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i)\right)$$

每个高斯原语由以下参数描述：中心位置 $\boldsymbol{\mu}_i \in \mathbb{R}^3$，协方差矩阵 $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3\times 3}$（通过旋转四元数 $\mathbf{q}_i$ 和缩放向量 $\mathbf{s}_i$ 参数化），不透明度 $\alpha_i \in [0,1]$，以及球谐系数（用于视角相关颜色）。

渲染通过可微分的splatting实现。对于像素 $p$，其颜色通过alpha-compositing计算：

$$C(p) = \sum_{i=1}^{N'} c_i \cdot \alpha_i' \cdot \prod_{j=1}^{i-1}(1 - \alpha_j')$$

其中 $N'$ 为覆盖像素 $p$ 的高斯数量，$\alpha_i' = \alpha_i \cdot \exp\left(-\frac{1}{2}\Delta_p^T \boldsymbol{\Sigma}_{2D,i}^{-1} \Delta_p\right)$ 为2D投影后的权重。

**基于3DGS的生成。** 将3DGS与生成模型结合是当前的研究热点：

- **DreamGaussian（Tang et al., 2024）** 将DreamFusion的SDS框架迁移到3DGS表示，利用3DGS的高效渲染加速优化过程。
- **GaussianDreamer** 进一步引入了3D感知的扩散先验，通过在3DGS上直接应用2D扩散模型的梯度来实现文本到3D的生成。
- **Instant3D** 和 **LGM** 等方法探索了前馈式（feed-forward）生成策略：训练一个网络直接从单张图像预测3DGS参数，实现实时的3D生成。

3DGS的优势在于其渲染速度（实时渲染）和显式表示的可编辑性，使其成为3D生成的重要表示选择。然而，3DGS也面临一些挑战：大规模场景中高斯数量可能膨胀至数百万，导致存储和优化成本显著增加；此外，高斯原语的各向异性形状在表达尖锐边缘和薄结构时存在固有局限。研究者们正在探索自适应高斯分裂与合并策略、以及高斯与网格的混合表示来应对这些问题。

---

## 7.4 音频生成

虽然严格意义上音频不属于"视频与3D"的范畴，但音频生成（特别是语音合成和音乐生成）与视觉生成在技术框架上高度相通，且在多模态生成系统中不可或缺。

### 7.4.1 语音合成

现代语音合成（Text-to-Speech, TTS）系统通常采用两阶段架构：（1）声学模型将文本转换为梅尔频谱图（Mel-spectrogram）；（2）声码器将梅尔频谱图转换为波形。

**VITS（Kim et al., 2021）。** VITS将变分推断与端到端TTS相结合，通过条件VAE框架实现高质量语音合成。其生成过程可以表示为：

$$p_\theta(x|c) = \int p_\theta(x|z) p_\theta(z|c) \, dz$$

其中 $x$ 为波形，$c$ 为文本条件，$z$ 为潜变量。训练时使用ELBO目标，并引入对抗训练提升音质。

**扩散语音合成。** Diff-TTS和Grad-TTS将扩散模型引入语音合成。Grad-TTS（Popov et al., 2021）在梅尔频谱图空间中进行扩散：

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

去噪网络以文本编码为条件，逐步从纯噪声生成梅尔频谱图。WaveGrad（Chen et al., 2021）进一步将扩散模型应用于波形级别，直接在原始音频波形上进行扩散去噪。

### 7.4.2 音乐生成

音乐生成需要建模长期结构（如曲式、和声进行）和多轨协调关系。

**MusicLM（Agostinelli et al., 2023）。** Google的MusicLM采用层级式自回归生成：首先生成粗粒度的音乐token序列建立全局结构，再逐步细化为高保真音频token。其核心是对音频使用SoundStream神经编码器进行离散化：

$$\mathbf{z}_q = \text{SoundStream}(\mathbf{x}) \in \{0, 1, \ldots, K-1\}^{T' \times L}$$

其中 $K$ 为码本大小，$T'$ 为时间帧数，$L$ 为量化层数（残差向量量化）。

**MusicGen（Copet et al., 2023）。** Meta的MusicGen提出了码本交错（codebook interleaving）策略，将多层码本的预测从并行转化为顺序序列，有效降低了模型复杂度：

$$\hat{\mathbf{z}}_q^{(l)} = \text{Transformer}(\mathbf{z}_q^{(1:l-1)}, c)$$

其中 $l$ 为残差量化层索引，$c$ 为文本条件。

### 7.4.3 波形扩散

波形扩散模型直接在原始音频波形（通常采样率为16kHz-48kHz）上进行扩散，避免了频谱图转换带来的信息损失。

**AudioLDM（Liu et al., 2023）。** AudioLDM将潜扩散模型应用于音频生成，其创新在于：（1）使用CLAP（Contrastive Language-Audio Pretraining）模型提取文本-音频对齐的条件嵌入；（2）在梅尔频谱图的潜空间中进行扩散。训练目标为：

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, c, t, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \text{CLAP}(c))\|^2\right]$$

**Stable Audio（Evans et al., 2024）。** Stability AI的Stable Audio将潜扩散模型扩展到长音频生成，引入了时间条件机制，允许用户精确控制生成音频的时长。该模型在超过80万首音乐和音效上训练，支持文本到音频和文本到音乐的生成。

---

## 7.5 多模态生成

多模态生成是将上述各个方向统一起来的终极目标——构建能够理解和生成多种模态内容的统一模型。

### 7.5.1 文本到视频（Text-to-Video）

文本到视频生成需要将自然语言描述转化为连贯的视频序列。除了7.2节介绍的基础视频生成方法外，文本到视频还面临以下特殊挑战：

**语义对齐。** 生成的视频需要忠实反映文本描述的内容。条件机制通常通过交叉注意力实现：

$$\text{CrossAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}$$

其中 $\mathbf{Q}$ 来自视频特征，$\mathbf{K}, \mathbf{V}$ 来自文本编码器（如T5或CLIP）的输出。

**运动建模。** 文本中的动词语义需要被准确映射为视觉运动。当前方法通过在大规模视频-文本对上训练来隐式学习这种映射，但复杂运动的精确控制仍是开放问题。

**代表模型。** 包括Runway的Gen系列、Pika Labs、Kuaishou的Kling、以及字节跳动的PixelDance等。这些模型在时长（从4秒到超过1分钟）、分辨率（最高1080p）和运动质量上持续突破。

### 7.5.2 图像到视频（Image-to-Video）

图像到视频生成以一张参考图像为起点，生成合理的运动和变化。这在动画化静态照片、虚拟试衣、自动驾驶数据增强等场景中具有广泛应用。

**SVD的条件机制。** SVD将参考图像通过VAE编码后与噪声潜变量在通道维度拼接：

$$\mathbf{z}_t' = \text{Concat}(\mathbf{z}_t, \text{VAE}_{\text{enc}}(\mathbf{x}_{\text{ref}}))$$

然后通过修改后的U-Net处理拼接后的张量。这种简单而有效的方式使模型能够保持与参考图像的视觉一致性。

**相机控制。** 在图像到视频生成中，相机运动的精确控制是一个重要研究方向。方法包括：（1）在条件信号中注入相机参数（如外参矩阵 $\mathbf{R}, \mathbf{t}$）；（2）使用Plücker坐标表示的光线作为额外条件（Bian et al., 2023）：

$$\mathbf{l} = (\mathbf{d}, \mathbf{o} \times \mathbf{d}) \in \mathbb{R}^6$$

其中 $\mathbf{d}$ 为光线方向，$\mathbf{o}$ 为光线原点。

### 7.5.3 统一生成架构

追求一个模型同时处理多种模态的生成是该领域的圣杯。

**NVIDIA Omniverse与Cosmos。** NVIDIA的Cosmos平台提出了世界基础模型（World Foundation Model）的概念，旨在构建能够理解物理世界规律的生成模型。这类模型不仅要生成视觉上逼真的内容，还要遵循物理定律（如重力、碰撞、流体动力学）。

**统一token化方法。** 将不同模态统一为离散token序列是最有前景的统一架构方向之一。具体而言：

$$\mathcal{T}_{\text{image}}: \mathbf{x}_{\text{img}} \rightarrow \{z_1^{(v)}, \ldots, z_{N_v}^{(v)}\}, \quad z_i^{(v)} \in \{1, \ldots, K_v\}$$

$$\mathcal{T}_{\text{audio}}: \mathbf{x}_{\text{aud}} \rightarrow \{z_1^{(a)}, \ldots, z_{N_a}^{(a)}\}, \quad z_j^{(a)} \in \{1, \ldots, K_a\}$$

$$\mathcal{T}_{\text{video}}: \mathbf{x}_{\text{vid}} \rightarrow \{z_1^{(v)}, \ldots, z_{N_t}^{(v)}\}, \quad z_k^{(v)} \in \{1, \ldots, K_v\}$$

所有模态的token被组织为统一的序列，由一个大型Transformer进行自回归建模：

$$p(\mathbf{z}) = \prod_{i=1}^{N} p(z_i | z_{<i})$$

**Transfusion（Zhou et al., 2024）** 提出了混合训练目标——在同一个Transformer中同时进行离散token的交叉熵损失训练和连续数据的扩散损失训练：

$$\mathcal{L}_{\text{Transfusion}} = \lambda_{\text{CE}} \mathcal{L}_{\text{CE}} + \lambda_{\text{diff}} \mathcal{L}_{\text{diff}}$$

这种方法避免了离散化带来的信息损失，同时保持了自回归建模的灵活性。

**Emu3（Wang et al., 2024）** 展示了仅使用next-token prediction就能统一处理图像、文本和视频生成的可能性。通过精心设计的tokenizer和大规模训练，一个纯自回归模型可以在多种模态上达到与专用模型相当的性能。

---

## 7.6 挑战与展望

尽管视频与3D生成取得了显著进展，仍面临诸多挑战：

**计算效率。** 视频生成的计算成本极高。生成一段60秒的1080p视频可能需要数百GPU小时。3DGS和NeRF的生成虽然较快，但在复杂场景下仍需大量优化迭代。

**物理真实性。** 当前模型生成的内容在视觉上可能逼真，但往往违反物理规律——水不会正确流动、物体穿模、刚体运动不符合力学定律。将物理先验融入生成模型是重要方向。

**长程一致性。** 在长视频生成中，角色外观、场景布局和叙事逻辑的长程一致性仍是核心难题。3D生成中，复杂场景的整体一致性和细节质量也需进一步提升。

**评估标准。** 视频和3D生成缺乏像FID那样被广泛接受的评估指标。CLIP Score衡量文本-视觉语义对齐程度，FVD（Fréchet Video Distance）评估视频分布的真实性，LPIPS衡量感知相似度，但这些指标各有局限——它们无法充分捕捉时间一致性、物理合理性和长程叙事质量。构建全面的评估体系需要结合自动指标与人工评估，这是该领域亟需解决的基础问题。

**可控性与可编辑性。** 精确控制生成内容（如指定角色动作、相机轨迹、光照条件）的能力仍然有限。3D表示的可编辑性（如场景中的物体移动、材质替换）也是实际应用的关键需求。

展望未来，以下方向值得期待：

- **世界模型（World Models）**：不仅生成视觉内容，还学习物理世界的因果结构和动力学规律。Sora已经展示了视频生成模型在模拟物理世界方面的初步能力，未来的世界模型将能够支持机器人决策、自动驾驶仿真等高风险应用。
- **实时交互式生成**：用户可以实时编辑和引导生成过程，实现所见即所得的创意工具。3DGS的实时渲染特性为这一方向奠定了基础。
- **4D生成（3D空间+时间）**：生成可自由视角观看的动态三维场景，将视频生成与3D生成融为一体。这需要同时建模几何、外观和运动的联合分布。
- **统一多模态智能**：一个模型理解并生成所有模态，真正实现通感（synesthesia）。从语言到视觉、从视觉到声音、从声音到触觉——模态之间的边界将逐渐消融。

---

## 7.7 本章小结

本章系统介绍了从视频生成到三维内容生成、从音频合成到多模态统一生成的前沿方法。扩散模型在这些领域中展现出强大的生成能力，而Transformer架构的引入则为统一建模提供了新的范式。Sora的出现表明，当计算规模和数据规模足够大时，视频生成能够涌现出对物理世界的朴素理解。3DGS等新型表示则为三维生成提供了更高效的载体。音频领域的扩散模型和自回归方法正在缩小合成语音与自然语音之间的差距。多模态统一生成架构预示着一个模型通晓所有模态的未来。

从GAN到扩散模型，从2D到3D再到4D，生成式AI的边界正在以前所未有的速度扩展。这不仅是技术的进步，更是人类理解世界、创造世界方式的深刻变革。

---

## 参考文献

1. Blattmann, A., et al. (2023). Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. *arXiv:2311.15127*.
2. Ho, J., et al. (2022). Video Diffusion Models. *NeurIPS 2022*.
3. Brooks, T., et al. (2024). Video Generation Models as World Simulators. *OpenAI Technical Report*.
4. Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers. *ICCV 2023*.
5. Yan, W., et al. (2021). VideoGPT: Video Generation using VQ-VAE and Transformers. *arXiv:2104.10157*.
6. Kondratyuk, D., et al. (2024). VideoPoet: A Large Language Model for Zero-Shot Video Generation. *ICML 2024*.
7. Nichol, A., et al. (2022). Point-E: A System for Generating 3D Point Clouds from Complex Prompts. *arXiv:2212.08751*.
8. Nash, C., et al. (2020). PolyGen: An Autoregressive Generative Model of 3D Meshes. *ICML 2020*.
9. Poole, B., et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion. *ICLR 2023*.
10. Kerbl, B., et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *SIGGRAPH 2023*.
11. Tang, J., et al. (2024). DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation. *ICLR 2024*.
12. Kim, J., et al. (2021). Conditional Variational Autoencoder with Adarial Learning for End-to-End Text-to-Speech. *ICML 2021*.
13. Popov, V., et al. (2021). Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech. *ICML 2021*.
14. Agostinelli, A., et al. (2023). MusicLM: Generating Music From Text. *arXiv:2301.11325*.
15. Copet, J., et al. (2023). Simple and Controllable Music Generation. *NeurIPS 2023*.
16. Liu, H., et al. (2023). AudioLDM: Text-to-Audio Generation with Latent Diffusion Models. *ICML 2023*.
17. Evans, Z., et al. (2024). Fast Timing-Conditioned Latent Audio Diffusion. *arXiv:2402.04825*.
18. Zhou, C., et al. (2024). Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model. *arXiv:2408.11039*.
19. Wang, X., et al. (2024). Emu3: Next-Token Prediction is All You Need. *arXiv:2409.18869*.
20. Lin, C.-H., et al. (2023). Magic3D: High-Resolution Text-to-3D Content Creation. *CVPR 2023*.
21. Siddiqui, Y., et al. (2023). MeshDiffusion: Score-based Generative 3D Mesh Modeling. *ICLR 2023*.
22. Bian, W., et al. (2023). Ctrl-Camera: Controllable Camera Generation for Text-to-3D Scene. *arXiv:2311.17105*.
23. Chen, N., et al. (2021). WaveGrad: Estimating Gradients for Waveform Generation. *ICLR 2021*.
-e 

---


# 第10章 大模型生成能力

## 10.1 引言

自2017年Transformer架构问世以来，语言模型经历了从百万参数到千亿参数的跨越式发展。GPT-3（Brown et al., 2020）的发布标志着一个转折点：当模型规模突破某一临界阈值后，语言模型不再仅仅是文本补全工具，而是展现出令人惊叹的通用生成能力。这种能力的涌现（emergence）催生了一系列全新的应用场景——从上下文学习到代码生成，从数学推理到创意写作——深刻重塑了人们对人工智能创造力边界的认知。

本章将系统探讨大语言模型在四个核心生成维度上的能力：上下文学习生成、代码生成、数学推理生成与创意写作。我们将追溯每项能力的技术演进脉络，分析其背后的机制原理，并审视当前的局限与未来的发展方向。

## 10.2 上下文学习生成

### 10.2.1 Few-shot生成

传统的机器学习范式要求对模型进行显式的参数更新——通过梯度下降在标注数据上微调模型权重。GPT-3的发布打破了这一范式。Brown等人（2020）发现，当模型规模足够大时，仅通过在提示（prompt）中提供少量示例，模型就能"学会"完成新任务，而无需任何参数更新。这种能力被称为上下文学习（In-Context Learning, ICL）。

Few-shot生成的核心机制在于：模型在预训练阶段已经接触了海量的任务模式，当推理时接收到少量示例后，能够识别出隐含的任务结构，并将其泛化到新的输入上。具体而言，给定一个提示模板：

```
将以下英文翻译为法文：
sea → mer
cheese → fromage
house →
```

模型能够正确输出"maison"，尽管它在推理过程中并未发生任何权重变化。

这一发现引发了大量后续研究。Min等人（2022）在《Rethinking the Role of Demonstrations》中提出，Few-shot示例的核心作用并非"教会"模型新任务，而是帮助模型从预训练知识中定位正确的任务格式和分布。实验表明，即使将示例中的标签随机打乱，模型在某些任务上的表现依然相当稳定，这暗示示例的主要功能是激活模型已有的能力，而非注入新的知识。

后续研究进一步探索了影响Few-shot性能的关键因素。Liu等人（2022）发现，示例的选择顺序、与测试输入的语义相似度以及示例的格式都会显著影响生成质量。这一发现催生了一系列示例检索与优化方法，如基于嵌入相似度的动态示例选择策略。

Few-shot生成能力的出现具有深远意义。它使得大模型能够以极低的成本快速适应新任务——无需收集大规模训练数据，无需昂贵的微调过程，只需设计合适的提示即可。这为"提示工程"（prompt engineering）这一新兴领域奠定了基础。

### 10.2.2 指令遵循

如果说Few-shot生成展现的是模型"被动"适应任务的能力，那么指令遵循（instruction following）则体现了模型"主动"理解并执行人类意图的能力。指令遵循要求模型不仅能完成特定任务，还能理解自然语言描述的复杂指令，并按照指令要求生成符合预期的输出。

InstructGPT（Ouyang et al., 2022）是这一方向的里程碑工作。OpenAI的研究者发现，单纯通过大规模预训练获得的语言模型，其生成内容并不总是与人类意图一致——模型可能会生成有害内容、编造事实，或者给出冗长而无用的回答。为了解决这一问题，InstructGPT引入了基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）框架。

RLHF包含三个关键步骤：首先，收集人类标注者对模型输出的偏好排序数据；其次，训练一个奖励模型（reward model）来预测人类偏好；最后，使用近端策略优化（Proximal Policy Optimization, PPO）算法，以奖励模型的评分作为信号来微调语言模型。通过这一流程，模型学会了生成更有帮助、更诚实、更无害的输出。

指令遵循能力的核心挑战在于泛化。FLAN（Wei et al., 2022）和FLAN-T5（Chung et al., 2023）的工作表明，通过在大量多样化的任务上进行指令微调（instruction tuning），模型能够获得对未见任务的零样本泛化能力。FLAN在62个数据集、超过1800个任务上进行微调，使得模型在面对全新指令时也能展现出合理的理解与执行能力。

这一方向的后续发展催生了开源指令微调生态的繁荣。Alpaca（Stanford, 2023）利用GPT-3.5生成的52K条指令数据微调LLaMA模型，成本仅为数百美元；Vicuna（Chiang et al., 2023）则通过收集ShareGPT网站上的用户对话数据实现了接近GPT-4水平的对话能力。这些工作证明，高质量的指令数据是解锁大模型指令遵循能力的关键。

### 10.2.3 思维链生成

2022年，Wei等人（2022）在《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》中提出了一个简单却深刻的方法：在Few-shot示例中不仅给出最终答案，还展示完整的推理步骤。实验表明，这种"思维链"（Chain-of-Thought, CoT）提示方法能够显著提升模型在算术推理、常识推理和符号推理等任务上的表现。

思维链的核心思想是将复杂问题分解为一系列中间步骤，使模型能够"展示其工作过程"。以一个数学问题为例：

```
问：Roger有5个网球，他又买了2罐网球，每罐有3个。他现在有多少网球？
答：Roger最初有5个网球。他买了2罐，每罐3个，即2×3=6个。5+6=11。答案是11。
```

这种方法的效果随模型规模的增大而增强——在小模型上，思维链甚至可能导致性能下降，这与"涌现能力"的概念一致。Kojima等人（2022）进一步发现，仅需一句简单的提示"Let's think step by step"就能激发模型的零样本推理能力，表明大模型已经隐式地学习了推理的结构。

思维链的变体不断涌现。Wang等人（2023）提出自洽性（Self-Consistency），通过采样多条推理路径并取多数投票来提高答案的可靠性。Yao等人（2023）提出思维树（Tree of Thoughts, ToT），将线性的推理链扩展为树状的搜索空间，允许模型在推理过程中进行回溯和探索。这些方法将思维链从简单的线性推理提升为更加灵活的推理框架。

思维链生成的意义超越了技术本身。它为大模型的推理过程提供了可解释性——我们不仅能看到模型的最终答案，还能审视其推理过程是否合理。这种透明性对于建立对AI系统的信任至关重要。

## 10.3 代码生成

### 10.3.1 Codex与代码生成的突破

2021年，OpenAI发布的Codex（Chen et al., 2021）是代码生成领域的里程碑事件。Codex基于GPT-3架构，在来自GitHub的159GB Python代码数据上进行了微调。其最引人注目的成就是在HumanEval基准测试上的表现：HumanEval包含164个手写的Python编程问题，每个问题提供函数签名和文档字符串，要求模型生成正确的函数实现。Codex在单次采样中能够解决28.8%的问题，当生成100个候选解并选择最可能的一个时，解决率提升至72.31%。

Codex的成功揭示了一个关键洞察：代码是一种高度结构化的自然语言，其语法约束和逻辑结构为模型提供了强大的学习信号。与自然语言文本相比，代码具有明确的语法规则、可验证的执行结果和丰富的注释-实现对应关系，这些特性使得代码成为大模型训练的理想数据源。

GitHub Copilot的发布将Codex的能力直接带给了全球开发者。作为IDE中的智能编程助手，Copilot能够根据当前代码上下文实时生成代码补全建议，涵盖了从简单变量命名到复杂算法实现的广泛场景。这一产品标志着AI代码生成从学术研究走向了大规模商业化应用。

### 10.3.2 Code Llama与开源代码模型

2023年，Meta发布的Code Llama（Rozière et al., 2023）将开源代码生成模型推向了新的高度。Code Llama基于LLaMA 2架构，通过在500B token的代码数据上进行持续预训练，并经过指令微调和RLHF优化，实现了与闭源模型相竞争的性能。

Code Llama家族包含三个变体：基础版Code Llama专注于代码补全；Code Llama-Python针对Python进行了专门优化；Code Llama-Instruct则通过指令微调实现了更好的对话式代码生成能力。在HumanEval基准上，Code Llama 34B达到了48.8%的pass@1分数，而经过进一步优化的70B版本更是达到了57.3%，接近GPT-3.5的水平。

Code Llama的贡献不仅在于性能，更在于其对开源社区的赋能。它使得研究者和开发者能够在本地部署和定制代码生成模型，促进了代码生成技术的民主化。后续的StarCoder（Li et al., 2023）、DeepSeek-Coder（Guo et al., 2024）等开源模型进一步推动了这一趋势。

### 10.3.3 代码补全与生成的机制

大模型代码生成的技术基础在于对代码语义和结构的深度理解。在预训练阶段，模型通过海量代码数据的学习，建立了对以下维度的内在表征：

**语法层面**，模型学会了编程语言的语法规则和惯用模式。它能够生成语法正确的代码，并遵循语言特定的编码规范。**语义层面**，模型理解变量的作用域、类型系统、控制流和数据流关系。它能够追踪变量的定义和使用，生成语义一致的代码。**任务层面**，模型能够从自然语言描述或代码上下文中推断出编程意图，并将其转化为具体的代码实现。

代码补全与代码生成在技术实现上存在细微但重要的差异。代码补全是更具约束性的任务——模型需要在给定的代码框架内填充缺失的部分，这要求模型对现有代码的上下文有深入理解。代码生成则更为开放——从自然语言描述出发，模型需要从头构建完整的代码实现，这要求模型具备更强的语义理解和代码规划能力。

Aider、Continue等开源工具的出现，进一步拓展了代码生成的应用边界。这些工具不仅支持代码补全，还能够进行代码重构、测试生成、Bug修复和代码审查，展现了大模型在软件工程全生命周期中的潜力。

### 10.3.4 代码质量评估

评估AI生成代码的质量是一个多维度的挑战。最直观的评估指标是功能正确性——生成的代码是否能够通过预定义的测试用例。HumanEval基准采用的pass@k指标就是这一思路的体现：生成k个候选解，只要其中有一个通过所有测试用例即视为成功。

然而，功能正确性只是冰山一角。生产级代码还需要考虑以下维度：

**安全性**方面，模型可能生成存在安全漏洞的代码——如SQL注入、跨站脚本攻击或缓冲区溢出。Pearce等人（2022）的研究表明，GitHub Copilot在特定场景下会生成不安全的代码，这对代码生成的安全审计提出了迫切需求。

**可维护性**方面，AI生成的代码可能缺乏良好的结构设计、命名规范和文档注释，增加了后续维护的难度。**效率**方面，模型可能生成功能正确但时间或空间复杂度不佳的解决方案。**风格一致性**方面，生成的代码需要与项目现有的编码风格保持一致。

为应对这些挑战，研究社区提出了多维度的评估框架。CodeBLEU（Ren et al., 2020）通过结合语法匹配和数据流匹配来评估代码质量；一些研究则引入了代码异味检测、复杂度分析等静态分析工具作为评估维度。最新的评估趋势强调人机协作——将AI生成的代码交由人类开发者审查，评估其实际可用性和改进潜力。

## 10.4 数学推理生成

### 10.4.1 数学证明生成

数学证明生成是检验大模型推理能力的理想试金石。与自然语言生成不同，数学证明要求严格的逻辑一致性——每一步推导都必须建立在公理、定义或已证明的引理之上，任何逻辑跳跃都会导致证明失效。

GPT-f（Polu & Sutskever, 2020）是将大语言模型应用于形式化数学证明的先驱工作。研究者在Lean定理证明器的数学库上微调GPT-2模型，使其能够生成形式化的证明步骤。在Metamath基准测试中，GPT-f发现了一些此前未被记录的证明策略，甚至产生了新的数学定理。这一结果表明，大模型不仅能"复述"已有的证明，还能在某种程度上进行创造性的数学探索。

AlphaProof（DeepMind, 2024）将这一方向推向了新的高度。在2024年国际数学奥林匹克竞赛（IMO）中，AlphaProof结合了大语言模型的直觉推理能力和形式化验证器的严格检验能力，成功解决了四道题目中的两道（加上一道银牌级题目），达到了银牌水平的成绩。其核心思路是利用大模型生成证明的"草稿"，然后通过Lean的形式化验证器检验每一步的正确性，对于不正确的步骤进行修正和重试。

这一方向的开放挑战在于：当前的形式化数学库（如Mathlib）虽然庞大，但仍然只覆盖了数学知识的一小部分。如何让大模型在更广泛的数学领域中进行有效的证明搜索，是一个重要的研究问题。

### 10.4.2 数学题求解

与形式化证明不同，数学题求解关注的是从问题描述出发，通过一系列数学操作得出最终答案。这一任务要求模型同时具备问题理解、数学知识检索和多步推理能力。

GSM8K（Cobbe et al., 2021）是一个广泛使用的数学推理基准，包含8500道小学水平的数学应用题，每道题需要2到8步计算。该基准的核心设计理念是"可分解性"——每道题的解题过程由一系列独立的计算步骤组成，便于评估模型的中间推理质量。

思维链提示的引入显著提升了大模型在GSM8K上的表现。GPT-4在链式推理下能够解决约92%的GSM8K问题，而其前身GPT-3.5仅为57%。这一进步不仅归功于模型规模的增长，更反映了模型推理能力的根本性提升。

然而，GSM8K等小学水平的基准已经逐渐被"解决"。MATH基准（Hendrycks et al., 2021）将难度提升到了竞赛数学水平，包含代数、几何、数论、概率和微积分等领域的12500道题目。在这一更具挑战性的基准上，模型的表现仍有较大提升空间。2024年的进展表明，Gemini Ultra和GPT-4 Turbo在MATH基准上的准确率分别达到了53.2%和52.9%，而经过专门优化的模型（如MetaMath）能够进一步提升这一数字。

### 10.4.3 过程奖励模型

传统的奖励模型（outcome reward model, ORM）只关注最终答案的正确性，而忽略了解题过程的质量。然而，在数学推理中，正确的推理过程比正确的最终答案更为重要——一个通过错误推理偶然得出正确答案的解法，其价值远不如一个每一步都正确推导的解法。

过程奖励模型（Process Reward Model, PRM）的概念正是为了弥补这一不足而提出的。Lightman等人（2023）在《Let's Verify Step by Step》中系统比较了结果奖励模型和过程奖励模型的差异。他们收集了大规模的人类标注数据，对GSM8K和MATH问题的每个推理步骤进行正确性判断，然后训练了一个过程奖励模型。

实验结果令人瞩目：在MATH基准上，使用过程奖励模型进行最佳候选选择（best-of-N sampling）的性能，显著优于使用结果奖励模型的方案。具体而言，当生成1800个候选解时，过程奖励模型能够将准确率从34%提升至56%。这一发现表明，对推理过程的精细化评估比仅评估最终结果更加有效。

过程奖励模型的训练需要大量步骤级别的标注数据，这在实践中是一个显著的成本障碍。为缓解这一问题，王等人（2024）提出了自动过程奖励标注方法（Math-Shepherd），通过蒙特卡洛采样来自动生成步骤级别的正确性标签：对于每个推理步骤，从该步骤出发进行多次独立采样完成后续推理，如果大多数采样都得出正确答案，则该步骤被标记为正确。这种方法在降低标注成本的同时保持了较好的评估质量。

过程奖励模型与思维链生成的结合，开启了"可验证推理"的新范式。在这种范式下，模型不仅生成推理链，还对每一步进行自我评估和修正，从而在保持推理深度的同时提高推理的可靠性。

## 10.5 创意写作

### 10.5.1 故事生成

故事生成是大语言模型最具挑战性也最具吸引力的应用之一。与信息性文本不同，故事需要连贯的叙事结构、立体的人物塑造、合理的冲突设计和引人入胜的节奏把控。这些要素的组合使得高质量故事生成成为对模型综合能力的终极考验。

从技术演进来看，故事生成经历了从模板填充到自由创作的范式转变。早期的系统如Tale-Spin（Meehan, 1977）依赖于手工编写的叙事脚本，生成的故事虽然结构完整但缺乏变化。神经语言模型的出现使得故事生成进入了数据驱动的时代，但RNN和LSTM时代的模型受限于长距离依赖问题，生成的故事往往在几段之后就失去连贯性。

大语言模型的出现从根本上改变了这一局面。GPT-3和后续模型展现出令人印象深刻的故事生成能力——它们能够维持多段落的叙事连贯性，塑造具有内在一致性的角色，并在一定程度上控制故事的风格和基调。Fan等人（2018）在《Hierarchical Neural Story Generation》中提出的分层生成框架——先生成故事大纲，再逐段扩展为完整故事——至今仍是故事生成的主流范式之一。

然而，当前的大模型在故事生成方面仍面临显著挑战。**长期一致性**是首要难题：在长故事中，模型可能会遗忘早期设定的角色特征、场景描述或情节线索，导致叙事出现矛盾。**情节创新**是另一挑战：模型倾向于生成符合训练数据中常见叙事模式的故事，难以产生真正新颖的情节结构。此外，模型在**情感深度**和**主题表达**方面仍有不足，生成的故事往往停留在表面的情节推进，缺乏对人性、社会等深层主题的探索。

### 10.5.2 诗歌创作

诗歌创作要求模型不仅要理解语言的语义，还要掌握其音韵、节奏和形式美感。这一任务天然地考验着大模型对语言艺术层面的理解。

中国古代诗歌的生成是一个具有代表性的研究场景。清华大学的九歌（Jiuge）系统（Yi et al., 2018）是这一方向的先驱工作，通过多模型协作的方式生成符合格律要求的古典诗歌。系统包含多个专门模型，分别负责主题建模、意象选择和韵律控制，最终通过集成策略生成完整的诗歌。

大语言模型的出现使得诗歌创作变得更加流畅。GPT-4等模型能够生成形式多样的诗歌——从严格遵循格律的古典诗词到自由奔放的现代诗，从莎士比亚风格的十四行诗到俳句。模型通过预训练阶段对海量诗歌文本的学习，隐式地掌握了各种诗歌形式的结构特征。

然而，评估AI生成诗歌的质量是一个开放问题。诗歌的美学价值涉及意象的原创性、情感的真挚性、语言的凝练性等多个难以量化的维度。现有的自动评估指标（如困惑度、BLEU分数）难以捕捉这些层面的质量。目前，诗歌质量的评估仍然高度依赖人类评审，这限制了该领域的系统性研究进展。

一个值得讨论的哲学问题是：AI生成的诗歌是否具有"创造力"？如果创造力定义为产生新颖且有价值的作品的能力，那么当前的大模型确实在某种程度上满足这一定义——它们能够将已有的语言元素以新颖的方式组合，产生出人意料的意象和比喻。但如果将创造力理解为源于主体性体验的表达，那么AI的"创作"更接近于精妙的模仿与重组。这一争论尚无定论，但无疑是值得深入思考的问题。

### 10.5.3 风格模仿

风格模仿（style imitation）要求模型在保持内容生成能力的同时，精确模仿特定作者或特定文体的写作风格。这一任务的难点在于"风格"是一个多维度的概念，涵盖了词汇选择、句式结构、修辞手法、叙事视角、语气基调等多个层面。

大模型在风格模仿方面展现出令人印象深刻的能力。通过在提示中指定目标风格或提供风格样本，模型能够调整其输出的语言特征以匹配目标风格。例如，给定"以鲁迅的风格改写以下段落"的指令，模型能够生成具有鲁迅特色的短句、反讽和冷峻笔调的文本。

Ippolito等人（2022）的研究系统探讨了大模型在风格迁移方面的能力与局限。他们发现，模型在模仿显著的表层风格特征（如句长、词汇正式程度）方面表现较好，但在模仿深层风格特征（如叙事节奏、隐喻系统、世界观表达）方面仍有不足。此外，当目标风格与模型训练数据中的主导风格差异较大时（如模仿实验性文学的先锋风格），模型的表现会显著下降。

风格模仿的一个重要应用场景是**创作辅助**。作者可以利用大模型来探索不同的写作视角和风格，生成多个风格化版本作为创作的起点。这种"人机协作"的模式正在改变文学创作的流程。

### 10.5.4 长文本生成的挑战

长文本生成是大模型面临的核心技术挑战之一。当前主流大模型的上下文窗口虽然从GPT-3时代的2048 token扩展到了GPT-4 Turbo的128K token甚至更多，但上下文窗口的扩展并不能完全解决长文本生成的挑战。

**连贯性维持**是长文本生成的首要难题。人类作家在写作长篇作品时，会通过提纲、角色设定表和情节规划来维持整体一致性。大模型则缺乏这种显式的规划机制——它依赖于注意力机制来关注上下文中的相关信息，但随着文本长度的增加，注意力的分散效应会导致模型逐渐"遗忘"早期设定的内容。

**结构规划**是另一关键挑战。高质量的长文本（如小说、论文）需要清晰的层次结构——章节划分、主题发展、线索交织和节奏控制。当前的大模型在生成单一段落时表现出色，但在宏观结构规划方面仍显薄弱。Yang等人（2023）提出的"递归摘要"方法尝试通过分层生成策略来解决这一问题：先生成高层大纲，再逐层细化为具体段落，最后整合为完整的长文本。

**重复与退化**也是长文本生成中的常见问题。随着生成长度的增加，模型倾向于重复使用相似的句式、词汇甚至段落结构，导致文本的多样性下降。这一问题的根源在于自回归生成的累积效应——早期的生成选择会对后续生成产生持续的影响，使得模型陷入局部最优的生成模式。

为应对这些挑战，研究社区提出了多种策略。**检索增强生成**（Retrieval-Augmented Generation, RAG）通过在生成过程中检索相关信息来维持长距离一致性。**分段生成与人工审核**的交互式范式允许人类在关键节点介入，提供方向性指导。**规划-执行分离**的架构将宏观结构规划与微观文本生成解耦，分别由不同的模块或不同的推理过程来处理。

## 10.6 本章小结

本章系统探讨了大语言模型在四个核心维度上的生成能力。上下文学习生成展现了大模型无需参数更新即可适应新任务的惊人能力，从Few-shot学习到指令遵循再到思维链推理，大模型正在成为通用的任务求解器。代码生成从Codex到Code Llama的演进，不仅改变了软件开发的工作流程，更证明了大模型理解结构化逻辑的深度能力。数学推理生成通过思维链、过程奖励等技术的结合，正在逼近形式化推理的严格性要求。创意写作则展现了大模型在语言艺术层面的潜力与局限——它们能够生成流畅、多样的文本，但在长期一致性、深层风格模仿和宏观结构规划方面仍有显著提升空间。

展望未来，大模型的生成能力将在以下方向继续演进：多模态生成（文本、图像、音频、视频的统一生成）、可控生成（更精细的生成过程控制）、可验证生成（对生成内容的事实性、逻辑性和安全性进行形式化验证）以及个性化生成（根据用户偏好和需求定制生成内容）。这些发展方向共同指向一个愿景：大模型将成为人类创造力的延伸——不是替代人类的创造者，而是增强人类表达能力的强大工具。

## 参考文献

- Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.
- Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code. *arXiv:2107.03374*.
- Chung, H. W., et al. (2023). Scaling Instruction-Finetuned Language Models. *JMLR*.
- Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*.
- Fan, A., et al. (2018). Hierarchical Neural Story Generation. *ACL*.
- Guo, D., et al. (2024). DeepSeek-Coder: When the Large Language Model Meets Programming. *arXiv:2401.14196*.
- Hendrycks, D., et al. (2021). Measuring Mathematical Problem Solving with the MATH Dataset. *NeurIPS*.
- Ippolito, D., et al. (2022). Unsupervised Text Style Transfer with Padded Masked Language Models. *EMNLP*.
- Kojima, T., et al. (2022). Large Language Models are Zero-Shot Reasoners. *NeurIPS*.
- Li, R., et al. (2023). StarCoder: May the Source Be with You! *arXiv:2305.06161*.
- Lightman, H., et al. (2023). Let's Verify Step by Step. *arXiv:2305.20050*.
- Liu, J., et al. (2022). What Makes Good In-Context Examples for GPT-3? *DeeLIO Workshop, ACL*.
- Meehan, J. R. (1977). TALE-SPIN, An Interactive Program that Writes Stories. *IJCAI*.
- Min, S., et al. (2022). Rethinking the Role of Demonstrations. *EMNLP*.
- Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*.
- Pearce, H., et al. (2022). Asleep at the Keyboard? Assessing the Security of GitHub Copilot's Code Contributions. *IEEE S&P*.
- Polu, S., & Sutskever, I. (2020). Generative Language Modeling for Automated Theorem Proving. *arXiv:2009.03393*.
- Ren, S., et al. (2020). CodeBLEU: A Method for Automatic Evaluation of Code Synthesis. *arXiv:2009.10297*.
- Rozière, B., et al. (2023). Code Llama: Open Foundation Models for Code. *arXiv:2308.12950*.
- Wang, P., et al. (2024). Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations. *ACL*.
- Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR*.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.
- Wei, J., et al. (2022). Finetuned Language Models are Zero-Shot Learners. *ICLR*.
- Yang, K., et al. (2023). DOC: Improving Long Story Coherence with Detailed Outline Control. *ACL*.
- Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS*.
- Yi, X., et al. (2018). Jiuge: A Human-Machine Collaborative Chinese Classical Poetry Generation System. *ACL Demo*.
-e 

---


# 第11章 生成模型的训练与优化

> *"训练一个生成模型，远比设计它的架构更加困难。"*
> — Ian Goodfellow, 2016

生成模型的理论框架固然重要，但真正决定模型成败的往往是训练与优化的工程实践。从GAN时代的模式坍缩到扩散模型时代的计算成本，训练与优化始终是生成式AI领域最具挑战性的课题。本章系统梳理生成模型训练中的核心问题：损失函数设计、训练稳定性、规模化训练与高效推理，为读者提供从理论到实践的完整指引。

---

## 11.1 损失函数设计

损失函数是连接模型输出与期望生成质量之间的桥梁。不同于判别模型仅需优化单一分类或回归目标，生成模型需要在多个维度上平衡生成质量、多样性和训练可行性。本节讨论四种核心损失函数及其组合策略。

### 11.1.1 重建损失（Reconstruction Loss）

重建损失衡量生成样本与目标样本之间的像素级或特征级差异，是自编码器（VAE）、流模型（Flow）和扩散模型的基础损失。

**像素级重建损失**。最直接的形式是均方误差（MSE）：

$$\mathcal{L}_{\text{rec}} = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \| x - G(x) \|_2^2 \right]$$

其中 $G$ 为生成器（或解码器），$x$ 为真实数据。MSE 倾向于生成模糊结果，因为它对所有像素赋予等权重，无法捕捉感知上的重要差异。

**L1 损失**在图像生成中常优于 MSE，因其对异常值更鲁棒，且在超分辨率等任务中能产生更锐利的边缘：

$$\mathcal{L}_{1} = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \| x - G(x) \|_1 \right]$$

Wang 等人（2004）的研究表明，L1 损失在结构保持方面优于 L2 损失，这一结论在 Pix2Pix（Isola et al., 2017）等工作中得到广泛验证。

**KL 散度重建**。在变分自编码器中，重建损失与 KL 正则项共同构成证据下界（ELBO）：

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(z|x)} \left[ \log p(x|z) \right] - \beta \cdot D_{\text{KL}}(q(z|x) \| p(z))$$

其中 $\beta$ 控制正则化强度。$\beta$-VAE（Higgins et al., 2017）通过调节 $\beta > 1$ 来学习更解耦的表示，但代价是重建质量下降。

### 11.1.2 对抗损失（Adversarial Loss）

对抗损失是 GAN 框架的核心，通过判别器 $D$ 与生成器 $G$ 的博弈来匹配生成分布与真实分布。

**原始 GAN 损失**（Goodfellow et al., 2014）：

$$\min_G \max_D \mathcal{L}_{\text{GAN}}(G, D) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p(z)} [\log(1 - D(G(z)))]$$

当最优判别器 $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$ 时，全局最小值对应 $p_g = p_{\text{data}}$，此时 $\min_G \max_D \mathcal{L}_{\text{GAN}} = -\log 4$。

然而，原始 GAN 在实际训练中面临严重问题。当判别器过于强大时，$\log(1 - D(G(z)))$ 饱和，梯度消失。因此实践中常使用**非饱和启发**：

$$\mathcal{L}_G^{\text{ns}} = -\mathbb{E}_{z \sim p(z)} [\log D(G(z))]$$

**Wasserstein 距离**（Arjovsky et al., 2017）从根本上解决了梯度消失问题：

$$\mathcal{L}_{\text{WGAN}}(G, D) = \mathbb{E}_{x \sim p_{\text{data}}} [D(x)] - \mathbb{E}_{z \sim p(z)} [D(G(z))]$$

其中 $D$ 需满足 1-Lipschitz 约束。WGAN-GP（Gulrajani et al., 2017）通过梯度惩罚替代权重裁剪来实现此约束：

$$\mathcal{L}_{\text{gp}} = \lambda_{\text{gp}} \cdot \mathbb{E}_{\hat{x}} \left[ (\| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1)^2 \right]$$

其中 $\hat{x}$ 为真实样本与生成样本之间的随机插值点，$\lambda_{\text{gp}}$ 通常取 10。

### 11.1.3 感知损失（Perceptual Loss）

感知损失在预训练网络的特征空间中度量相似性，比像素级损失更符合人类感知。

**特征匹配损失**（Johnson et al., 2016）：

$$\mathcal{L}_{\text{perceptual}} = \sum_{l} \frac{1}{C_l H_l W_l} \| \phi_l(x) - \phi_l(G(x)) \|_2^2$$

其中 $\phi_l$ 为预训练 VGG-19 网络第 $l$ 层的特征图，$C_l, H_l, W_l$ 为该层的维度。不同层捕捉不同层次的语义信息：浅层关注纹理细节，深层关注结构语义。

**风格损失**（Gatys et al., 2016）通过 Gram 矩阵捕捉纹理统计特性：

$$\mathcal{L}_{\text{style}} = \sum_{l} \| G_l^{\phi}(x) - G_l^{\phi}(G(x)) \|_F^2$$

其中 $G_l^{\phi} \in \mathbb{R}^{C_l \times C_l}$ 为第 $l$ 层特征图的 Gram 矩阵，定义为 $(G_l^{\phi})_{ij} = \frac{1}{C_l H_l W_l} \sum_k \phi_l^{(i)}(k) \phi_l^{(j)}(k)$。

**LPIPS 指标**（Zhang et al., 2018）将感知相似性学习为加权特征距离：

$$d(x, x') = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\hat{y}_{hw}^l - \hat{y}_{hw}'^l) \|_2^2$$

LPIPS 不仅作为评估指标，也可直接用作训练损失，尤其在图像编辑和风格迁移中表现出色。

### 11.1.4 扩散损失（Diffusion Loss）

扩散模型通过逐步去噪过程学习数据分布，其损失函数设计经历了从简单到精炼的演进。

**简化目标**（Ho et al., 2020）。DDPM 的训练目标简化为预测噪声：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|_2^2 \right]$$

其中 $t \sim \text{Uniform}(1, T)$，$\epsilon \sim \mathcal{N}(0, I)$，$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$。这一简化形式虽然忽略了不同时间步的信噪比差异，但在实践中表现优异。

**加权扩散损失**。Nichol & Dhariwal（2021）提出根据信噪比加权：

$$\mathcal{L}_{\text{weighted}} = \mathbb{E}_{t, x_0, \epsilon} \left[ w(t) \| \epsilon - \epsilon_\theta(x_t, t) \|_2^2 \right]$$

其中 $w(t)$ 可取 SNR$(t)$ 或 SNR$(t) / (\text{SNR}(t) + 1)$ 等形式。加权策略对生成质量有显著影响，尤其在低噪声（大 $t$）阶段。

**v-预测目标**（Salimans & Ho, 2022）。预测速度 $v = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1 - \bar{\alpha}_t} x_0$ 而非噪声，在数值上更稳定：

$$\mathcal{L}_v = \mathbb{E}_{t, x_0, \epsilon} \left[ \| v - v_\theta(x_t, t) \|_2^2 \right]$$

此目标在渐进式蒸馏（progressive distillation）中尤为重要。

### 11.1.5 损失函数组合策略

现代生成模型通常组合多种损失函数。以图像超分辨率为例，总损失可表示为：

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{rec}} + \lambda_2 \mathcal{L}_{\text{adv}} + \lambda_3 \mathcal{L}_{\text{perceptual}} + \lambda_4 \mathcal{L}_{\text{style}}$$

各权重 $\lambda_i$ 的选取需要精心调节。实践中的经验法则包括：

1. **渐进式引入**：先用重建损失预训练，再逐步引入对抗损失和感知损失
2. **数值尺度匹配**：确保各损失项处于相近数量级，避免某一项主导优化方向
3. **自适应权重**：如 GradNorm（Chen et al., 2018）自动平衡多任务梯度

Stable Diffusion 的成功很大程度上归功于其损失设计：仅使用简化的扩散损失训练潜在空间模型，避免了 GAN 式多损失组合的复杂性，同时通过分类器引导（classifier guidance）或无分类器引导（classifier-free guidance）在推理时控制生成质量。

---

## 11.2 训练稳定性

生成模型的训练稳定性是一个持续存在的挑战。本节分析四种常见不稳定现象及其应对策略。

### 11.2.1 模式坍缩（Mode Collapse）

模式坍缩是 GAN 训练中最棘手的问题之一，表现为生成器仅产生有限种类的输出，无法覆盖完整的训练分布。

**现象与诊断**。模式坍缩可分为完全模式坍缩（所有输入映射到同一输出）和部分模式坍缩（仅覆盖部分模式）。可通过以下指标监测：
- FID（Fréchet Inception Distance）突然升高
- 生成样本的多样性视觉下降
- 判别器损失接近零但生成器损失不收敛

**成因分析**。从优化视角看，模式坍缩源于生成器找到了"欺骗"判别器的捷径——生成少量判别器无法区分的样本，而非学习完整分布。这与 GAN 的非凸-非凹博弈结构直接相关。

**应对策略**：

1. **Unrolled GAN**（Metz et al., 2017）：生成器优化时考虑判别器未来 $k$ 步的更新：
$$\mathcal{L}_G^{\text{unrolled}} = \mathcal{L}_G(D^*(G))$$
其中 $D^*$ 是对判别器执行 $k$ 步梯度下降后的结果。

2. **多样性正则化**（Mao et al., 2019）：在损失中显式鼓励多样性：
$$\mathcal{L}_{\text{div}} = -\mathbb{E}_{z_1, z_2} [\| G(z_1) - G(z_2) \| / \| z_1 - z_2 \|]$$

3. **Minibatch Discrimination**（Salimans et al., 2016）：让判别器同时看到一批样本，识别生成样本之间的相似性。

4. **渐进式训练**（Karras et al., 2018）：从低分辨率逐步增加到高分辨率，降低模式坍缩风险。

### 11.2.2 梯度消失与梯度爆炸

深度生成网络中梯度问题尤为突出，因为生成路径通常比判别路径更深。

**梯度消失**在原始 GAN 中表现为当 $D(G(z))$ 接近 0 时，$\nabla_G \log(1 - D(G(z))) \to 0$。WGAN 通过将优化目标从 JS 散度切换到 Wasserstein 距离，保证了处处非零的梯度信号。

**梯度爆炸**常出现在 RNN 型生成器或深层卷积网络中。应对方法包括：

1. **梯度裁剪**（Gradient Clipping）：
$$g \leftarrow \min\left(1, \frac{c}{\|g\|_2}\right) \cdot g$$
其中 $c$ 为裁剪阈值。

2. **谱归一化**（Spectral Normalization, Miyato et al., 2018）：将判别器每一层的权重矩阵除以其谱范数：
$$W \leftarrow W / \sigma(W)$$
其中 $\sigma(W)$ 为 $W$ 的最大奇异值。这确保了判别器的 Lipschitz 常数为 1，稳定训练过程。

3. **权重初始化**：正交初始化（orthogonal initialization）或 He 初始化在生成模型中表现稳定。BigGAN（Brock et al., 2019）发现初始化尺度对训练稳定性有决定性影响。

### 11.2.3 训练震荡

训练震荡表现为损失函数在训练过程中剧烈波动，无法收敛到稳定点。

**GAN 中的判别器-生成器振荡**是典型的训练震荡。当判别器与生成器的能力不匹配时，两者可能陷入"追逐"循环：判别器适应当前生成器后，生成器更新导致分布剧变，判别器再次适应，如此往复。

**缓解策略**：

1. **判别器-生成器平衡**：维持 $n_D / n_G$ 比率（通常 $n_D = 1 \sim 5$，$n_G = 1$），确保两者学习速度匹配。

2. **Exponential Moving Average（EMA）**：维护生成器参数的移动平均：
$$\theta_{\text{EMA}} \leftarrow \mu \cdot \theta_{\text{EMA}} + (1 - \mu) \cdot \theta$$
其中 $\mu = 0.999$ 或 $0.9999$。EMA 有效平滑训练波动，是 BigGAN 和 StyleGAN 系列的标准实践。

3. **学习率调度**：余弦退火（cosine annealing）或线性预热（linear warmup）可减少训练初期的震荡。扩散模型训练中，预热阶段尤为重要。

### 11.2.4 超参数敏感性

生成模型对超参数极为敏感，微小的变化可能导致训练崩溃或生成质量骤降。

**关键超参数**：

| 超参数 | 典型范围 | 敏感程度 | 说明 |
|--------|---------|---------|------|
| 学习率 $\eta$ | $10^{-5} \sim 10^{-4}$ | 极高 | GAN 通常使用不同学习率给 G 和 D |
| 批大小 $B$ | $8 \sim 256$ | 高 | 影响梯度估计方差和模式覆盖 |
| $\beta_1$（Adam） | $0 \sim 0.5$ | 高 | WGAN-GP 建议 $\beta_1 = 0$ 或 $0.5$ |
| 正则化系数 | $1 \sim 100$ | 中 | 如梯度惩罚 $\lambda_{\text{gp}}$ |
| EMA 衰减率 | $0.999 \sim 0.9999$ | 中 | 过低导致滞后，过高失去平滑效果 |

**自动超参数搜索**。Karras et al.（2020）在 StyleGAN2 中引入了自适应权重调整策略，根据判别器最后一层特征的尺度自动调节正则化强度。这种"自适应"方法减少了手动调参的需要。

对于扩散模型，噪声调度（noise schedule）是最重要的超参数。线性调度（Ho et al., 2020）、余弦调度（Nichol & Dhariwal, 2021）和学习型调度（Kingma et al., 2021）各有优劣。SNR-aware 调度策略在高分辨率生成中尤为重要。

---

## 11.3 规模化训练

随着生成模型规模从百万参数增长到数十亿参数，高效利用计算资源成为关键。本节讨论四种核心规模化技术。

### 11.3.1 分布式训练

单 GPU 无法容纳现代生成模型的训练需求。分布式训练通过多设备协同来扩展计算能力。

**数据并行（Data Parallelism）**。最基础的分布式策略，将数据批次分配到多个 GPU 上，每个 GPU 持有完整模型副本：

$$g = \frac{1}{N} \sum_{i=1}^{N} g_i$$

其中 $g_i$ 为第 $i$ 个 GPU 上的梯度，$N$ 为 GPU 数量。All-Reduce 操作同步梯度。

**PyTorch 的 DistributedDataParallel（DDP）**通过 Ring-AllReduce 算法高效同步梯度，通信开销与 GPU 数量呈亚线性关系。对于生成模型，DDP 需要特别注意判别器与生成器的同步问题。

**模型并行（Model Parallelism）**。当模型超过单 GPU 显存时，将模型层分布到多个设备上。分为：
- **张量并行（Tensor Parallelism）**：将单个算子（如矩阵乘法）分割到多个 GPU
- **流水线并行（Pipeline Parallelism）**：将模型按层分割，不同层在不同 GPU 上执行

**ZeRO 优化器**（Rajbhandari et al., 2020）通过分割优化器状态、梯度和参数来降低显存占用：
- ZeRO-1：分割优化器状态，显存降低 $4\times$
- ZeRO-2：分割梯度，显存降低 $8\times$
- ZeRO-3：分割参数，显存降低 $N\times$（$N$ 为 GPU 数量）

Stable Diffusion 的训练即采用了 DeepSpeed ZeRO-2 策略，在 256 个 A100 GPU 上进行训练。

### 11.3.2 混合精度训练

混合精度训练利用 FP16 或 BF16 的低精度计算来加速训练，同时保持 FP32 的精度来维护数值稳定性。

**基本机制**（Micikevicius et al., 2018）：

1. 维护 FP32 主权重副本
2. 前向传播使用 FP16/BF16
3. 损失缩放（Loss Scaling）防止下溢：$\mathcal{L}_{\text{scaled}} = s \cdot \mathcal{L}$
4. 梯度更新在 FP32 精度下执行

**BF16 vs FP16**。BF16（Brain Floating Point）具有与 FP32 相同的指数位宽（8 位），但尾数仅 7 位。这意味着 BF16 的动态范围与 FP32 相同，但精度较低。在实践中，BF16 无需损失缩放即可稳定训练，逐渐成为大规模生成模型训练的默认选择。

**显存节省**。混合精度训练将显存占用降低近一半，同时在 NVIDIA Tensor Core 上获得 2-3 倍的计算加速。对于扩散模型训练，混合精度几乎已成为标配。

### 11.3.3 梯度累积

当硬件限制无法使用大批次时，梯度累积通过多次前向-反向传播模拟大批次效果：

$$g_{\text{accum}} = \frac{1}{K} \sum_{k=1}^{K} g_k$$

其中 $K$ 为累积步数。等效批大小为 $B_{\text{eff}} = B \times K \times N$（$B$ 为单 GPU 批大小，$N$ 为 GPU 数量）。

**在生成模型中的应用**。扩散模型训练通常需要较大批次（如 256 或更高）以获得稳定的噪声估计。当显存不足以容纳大批次时，梯度累积成为必需。实践中的注意事项：

1. **BatchNorm 替代**：梯度累积改变了每个物理批次的统计特性，应使用 GroupNorm 或 LayerNorm 替代 BatchNorm
2. **学习率调整**：线性缩放规则 $\eta \propto B_{\text{eff}}$ 在大范围内有效，但需配合预热
3. **EMA 更新频率**：EMA 应在每次参数更新后（而非每次物理批次后）更新

### 11.3.4 数据并行与高效数据加载

大规模生成模型训练的瓶颈往往在数据加载而非计算。

**数据加载优化**：

1. **WebDataset 格式**：将数据打包为 tar 文件，减少小文件 I/O 开销
2. **预取与缓存**：使用内存映射（memory-mapping）和预取线程
3. **数据增强的分布式实现**：确保不同 GPU 上的增强操作使用不同随机种子

**LAION 等大规模数据集**的处理需要特别注意数据质量和去重。Schuhmann et al.（2022）在构建 LAION-5B 时采用了 CLIP 过滤、NSFW 过滤和近似去重策略，这些预处理步骤对最终模型质量至关重要。

**检查点策略**。大规模训练的检查点应包含：
- 模型参数（EMA 和非 EMA）
- 优化器状态
- 学习率调度器状态
- 全局步数和随机数生成器状态

分布式检查点（如 PyTorch DCP）可并行保存和加载，避免训练中断的长时间恢复。

---

## 11.4 高效推理

训练完成后，生成模型的推理效率直接决定了其在实际应用中的可行性。本节讨论五种核心加速技术。

### 11.4.1 模型压缩

模型压缩旨在减少模型参数量和计算量，同时尽量保持生成质量。

**结构化剪枝**。移除整个通道、注意力头或层。与非结构化剪枝相比，结构化剪枝可直接在现有硬件上获得加速。Fang et al.（2023）提出对扩散模型的 U-Net 进行通道剪枝，在移除 30% 参数后仍保持相当的 FID 分数。

**非结构化剪枝**。以稀疏矩阵形式存储权重，保留绝对值最大的参数。SparseGPT（Frantar & Alistarh, 2023）和 Wanda（Sun et al., 2024）展示了在生成模型上进行 50-60% 稀疏化而不显著损失质量的可能性。

### 11.4.2 知识蒸馏

知识蒸馏将大型教师模型的知识转移到小型学生模型。

**渐进式蒸馏**（Salimans & Ho, 2022）是扩散模型特有的蒸馏策略。将 $T$ 步的去噪过程压缩为 $T/2$ 步，学生模型学习执行两步合并的去噪：

$$\mathcal{L}_{\text{distill}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon_\theta^{\text{student}}(x_t, t) - \epsilon_\theta^{\text{teacher}}(\hat{x}_{t/2}, t/2) \|_2^2 \right]$$

通过反复应用此过程，可将采样步数从 1000 压缩到 4 步甚至 1 步。

**对抗蒸馏**（Adversarial Distillation）。SDXL-Turbo（Sauer et al., 2024）和 SDXL-Lightning（Wang et al., 2024）采用对抗损失训练蒸馏模型，进一步减少采样步数：

$$\mathcal{L}_{\text{adv-distill}} = \mathcal{L}_{\text{distill}} + \lambda \cdot \mathcal{L}_{\text{GAN}}$$

其中 $\mathcal{L}_{\text{GAN}}$ 以真实图像为真实样本，学生模型生成的图像为生成样本。

**一致性模型**（Consistency Models, Song et al., 2023）。通过学习将任意噪声水平直接映射到干净数据，实现单步生成：

$$f_\theta(x_t, t) = x_\epsilon, \quad \forall t \in [\epsilon, T]$$

其中 $f_\theta$ 需满足自一致性：$f_\theta(x_t, t) = f_\theta(x_{t'}, t')$，对任意 $t, t' \in [\epsilon, T]$。

### 11.4.3 量化

量化将模型权重和激活从高精度浮点数映射到低精度表示。

**权重量化**。将 FP16 权重量化为 INT8 或 INT4：

$$W_q = \text{round}\left(\frac{W - z}{s}\right), \quad s = \frac{\max(W) - \min(W)}{2^b - 1}$$

其中 $s$ 为缩放因子，$z$ 为零点，$b$ 为量化位数。

**DiffusionBits**（Shih et al., 2024）和 Q-Diffusion（Li et al., 2023）专门针对扩散模型提出了量化策略。关键发现是：

1. 扩散模型对权重量化比激活量化更鲁棒
2. 不同时间步的权重敏感性不同——早期去噪步需要更高精度
3. 混合精度量化（对敏感层保留高精度）优于统一量化

**INT8 量化**可在几乎不损失生成质量（FID 变化 < 1）的情况下将模型大小减半，推理速度提升 1.5-2 倍。**INT4 量化**进一步压缩模型，但通常需要校准数据和量化感知微调。

### 11.4.4 剪枝

在生成模型中，剪枝可分为训练后剪枝和训练中剪枝。

**训练后剪枝**。基于权重重要性指标移除冗余参数。常用的重要性指标包括：

1. **幅度剪枝**：移除绝对值最小的权重
2. **梯度敏感性**：$\text{score}(w) = |w \cdot \nabla_w \mathcal{L}|$
3. **Hessian 感知**：基于二阶信息估计参数重要性

**结构化剪枝在扩散模型中**。U-Net 的跳跃连接（skip connection）和注意力层对剪枝的敏感性差异显著。Diff-Pruning（Lee et al., 2024）通过泰勒展开估计每层的重要性分数，在 40% 稀疏率下保持了 95% 的生成质量。

**训练中剪枝（彩票假说）**。Frankle & Carlin（2019）的彩票假说表明，密集网络中存在稀疏子网络（"中奖彩票"），可独立训练达到相当性能。在生成模型中，这启发了动态稀疏训练方法，训练过程中逐步增加稀疏度。

### 11.4.5 缓存加速

缓存加速通过复用中间计算结果来减少推理开销。

**KV 缓存**。在自回归生成模型（如 Transformer 解码器）中，缓存注意力机制的 Key 和 Value 向量，避免对已生成 token 的重复计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

缓存 $K, V$ 后，每步仅需计算新 token 的 $q, k, v$ 并追加到缓存中。

**去噪步缓存**。在扩散模型中，相邻时间步的中间特征具有高度相似性。DeepCache（Ma et al., 2024）利用这一特性，缓存 U-Net 中间层的特征图，每隔若干步才重新计算：

$$h_l^{t} \approx h_l^{t-\Delta t}, \quad \text{当 } \Delta t \text{ 较小时}$$

实验表明，缓存间隔为 2-3 步时，推理速度可提升 1.5-2.3 倍，FID 退化在 5% 以内。

**注意力缓存与稀疏注意力**。对于高分辨率图像生成，注意力计算的 $O(n^2)$ 复杂度成为瓶颈。Flash Attention（Dao et al., 2022）通过 IO-aware 算法优化注意力计算，减少 HBM 访问次数。xFormers 等库提供了多种高效注意力实现。

**时间步调度优化**。并非所有去噪步都同等重要。DDIM（Song et al., 2021）通过确定性采样将 1000 步压缩到 50 步。进一步的非均匀时间步调度（如 DPM-Solver++, Lu et al., 2022）可在 20 步内获得与 50 步相当的质量。

---

## 11.5 实践建议与总结

### 11.5.1 训练清单

在启动生成模型训练前，建议按以下清单检查：

1. **损失函数**：选择合适的损失组合，从简单损失开始逐步引入复杂损失
2. **监控指标**：实时监控 FID/IS、判别器/生成器损失、梯度范数
3. **稳定性措施**：EMA、谱归一化、梯度裁剪
4. **分布式策略**：根据模型大小选择 DDP 或模型并行
5. **检查点**：频繁保存，包含完整训练状态
6. **数据质量**：去重、过滤、增强的完整流水线

### 11.5.2 推理优化优先级

在部署生成模型时，建议按以下优先级实施优化：

1. **采样步数减少**（DDIM/DPM-Solver）——收益最大，实现简单
2. **模型蒸馏**——显著加速，质量可控
3. **量化**——硬件友好，部署成本低
4. **缓存加速**——无需修改模型，即插即用
5. **剪枝**——需要专门硬件支持，收益因架构而异

### 11.5.3 技术趋势

生成模型的训练与优化正朝着以下方向发展：

- **训练效率**：DiT（Peebles & Xie, 2023）等纯 Transformer 架构简化了训练流水线
- **推理速度**：一致性模型和对抗蒸馏推动单步生成成为可能
- **规模化**：混合专家（MoE）架构在不增加推理成本的情况下扩大模型容量
- **自动化**：自动损失设计和超参数搜索减少人工调参需求

生成模型的训练与优化是一个不断演进的领域。掌握本章介绍的核心技术和设计原则，将为读者在实践中训练和部署高质量生成模型奠定坚实基础。

---

**参考文献**

1. Goodfellow, I., et al. (2014). Generative Adversarial Nets. *NeurIPS*.
2. Arjovsky, M., et al. (2017). Wasserstein GAN. *ICML*.
3. Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs. *NeurIPS*.
4. Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.
5. Nichol, A. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML*.
6. Salimans, T. & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. *ICLR*.
7. Song, Y., et al. (2023). Consistency Models. *ICML*.
8. Karras, T., et al. (2018). Progressive Growing of GANs. *ICLR*.
9. Miyato, T., et al. (2018). Spectral Normalization for GANs. *ICLR*.
10. Micikevicius, P., et al. (2018). Mixed Precision Training. *ICLR*.
11. Rajbhandari, S., et al. (2020). ZeRO: Memory Optimizations. *SC*.
12. Dao, T., et al. (2022). FlashAttention. *NeurIPS*.
13. Ma, X., et al. (2024). DeepCache: Accelerating Diffusion Models for Free. *CVPR*.
14. Lu, C., et al. (2022). DPM-Solver++. *NeurIPS*.
15. Peebles, W. & Xie, S. (2023). Scalable Diffusion Models with Transformers. *ICCV*.
16. Brock, A., et al. (2019). Large Scale GAN Training for High Fidelity Image Synthesis. *ICLR*.
17. Karras, T., et al. (2020). Analyzing and Improving the Image Quality of StyleGAN. *CVPR*.
18. Higgins, I., et al. (2017). β-VAE: Learning Basic Visual Concepts. *ICLR*.
19. Zhang, R., et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*.
20. Sauer, A., et al. (2024). Adversarial Diffusion Distillation. *ECCV*.
-e 

---


# 第12章 生成模型的安全与伦理

## 引言

生成式人工智能（Generative AI）在过去十年间经历了从生成对抗网络（GAN）到变分自编码器（VAE），再到扩散模型（Diffusion Models）和大型语言模型（LLM）的跨越式发展。这些技术在图像合成、文本生成、语音克隆和视频制作等领域展现出了前所未有的能力，其生成内容的质量已达到难以与真实内容区分的程度。然而，技术能力的急剧提升也带来了深刻的安全隐患和伦理挑战。

Deepfake技术的泛滥威胁着信息生态的信任基础；AI生成内容的版权归属问题在法律界引发了激烈争论；训练数据中隐含的社会偏见通过生成模型被放大和传播；有害内容的自动化生产对社会安全构成直接威胁。这些问题不再是学术讨论中的假设性议题，而是当下亟需解决的现实挑战。

本章将从四个核心维度系统阐述生成模型的安全与伦理问题：Deepfake检测技术、版权争议与法律边界、偏见与公平性、以及安全生成的实践框架。我们的目标不是对生成式AI进行价值否定，而是为负责任的技术发展提供理论基础和实践指引。

---

## 12.1 Deepfake检测

### 12.1.1 人脸伪造检测

Deepfake一词最早出现于2017年，指利用深度学习技术生成或篡改人脸图像和视频的技术。随着GAN和扩散模型的发展，人脸伪造技术已从早期粗糙的换脸演进为高度逼真的面部重演（Face Reenactment）、面部交换（Face Swap）和面部合成（Face Synthesis）。

**伪造类型与生成机制。** 当前主流的人脸伪造技术可分为四类：第一，面部交换，将源人脸的身份特征替换到目标人脸之上，代表方法包括FaceSwap、DeepFaceLab以及基于StyleGAN的FSGAN（Nirkin et al., 2019）；第二，面部重演，驱动目标人脸模仿源人脸的表情和姿态，典型代表为First Order Motion Model（Siarohin et al., 2019）；第三，属性编辑，修改面部的特定属性如年龄、性别或肤色，StarGAN（Choi et al., 2018）和AttGAN（He et al., 2019）是该方向的代表；第四，完全人脸合成，生成不存在的人脸图像，StyleGAN系列（Karras et al., 2019, 2020, 2021）在此方面达到了照片级逼真度。

**检测方法。** 人脸伪造检测的核心思路是捕捉真实人脸与伪造人脸之间的细微差异。早期方法主要依赖卷积神经网络（CNN）进行二分类。Rossler et al.（2019）提出的FaceForensics++基准数据集系统评估了多种检测方法，发现基于XceptionNet的检测器在压缩视频上仍能达到较高准确率。

随着伪造技术的迭代，检测方法也不断演进。频域分析成为重要的检测线索——Afchar et al.（2018）提出的MesoNet通过关注图像的中频特征来区分真伪；Li et al.（2020）发现伪造图像在频谱中表现出上采样产生的周期性伪影。注意力机制的引入使检测器能够聚焦于伪造痕迹最显著的区域，Zhao et al.（2021）提出的Multi-Attentional Deepfake Detection利用多个注意力头捕获不同尺度的伪造特征。

生物信号提供了另一条重要检测路径。Li et al.（2020）观察到Deepfake视频中人物的脉搏信号（remote photoplethysmography, rPPG）会因面部篡改而出现异常，因为当前的伪造技术难以准确保持血流引起的细微颜色变化。类似地，眨眼频率、头部运动一致性和唇音同步性（Lip Sync）等生理和行为特征也被用于检测。

值得注意的是，检测器面临着泛化能力的根本性挑战。在一个伪造方法上训练的检测器往往难以泛化到未见过的伪造方法。为此，Wang et al.（2020）提出了基于身份感知的检测框架，通过学习身份一致性特征来提高跨伪造方法的泛化能力。Celeb-DF（Li et al., 2020）和WildDeepfake（Zi et al., 2020）等更具挑战性的数据集也被构建出来，以推动检测技术向真实场景靠拢。

### 12.1.2 音视频伪造检测

音频伪造（Voice Deepfake）技术利用语音合成（TTS）和语音转换（Voice Conversion）技术生成或篡改语音内容。近年来，基于神经网络的语音合成系统如VITS（Kim et al., 2021）和大规模语音语言模型已能生成高度自然的语音，使得音频伪造的威胁日益严峻。

音频伪造检测主要依赖声学特征分析。Mel频谱图、恒Q变换（CQT）和群延迟（Group Delay）等特征被广泛用于训练检测模型。Tak et al.（2021）提出的RawNet2直接从原始波形中学习特征，避免了手工特征设计的局限性。在ASVspoof挑战赛（Yamagishi et al., 2021）中，检测系统需要区分真实语音与由多种合成和转换方法生成的伪造语音，该竞赛已成为音频伪造检测领域最重要的基准。

视频伪造检测面临更复杂的挑战，因为视频同时包含视觉和听觉信息。多模态检测方法通过分析唇音一致性、音频-视觉同步性和面部运动的物理合理性来判断视频真伪。Mittal et al.（2020）提出的Emotions Don't Lie方法利用音频-视觉情感不一致性来检测视频伪造。随着扩散模型在视频生成领域的应用（如Sora等），视频伪造检测将面临更加严峻的挑战。

### 12.1.3 检测方法与基准

Deepfake检测领域的研究高度依赖标准化的基准数据集和评测协议。以下数据集在推动该领域发展中发挥了关键作用：

- **FaceForensics++**（Rossler et al., 2019）：包含1000个视频片段，涵盖Deepfakes、Face2Face、FaceSwap和NeuralTextures四种伪造方法，提供原始、轻度压缩和重度压缩三个质量级别。
- **Celeb-DF**（Li et al., 2020）：包含5639个高质量Deepfake视频，更接近真实场景中的伪造质量。
- **DFDC**（Dolan et al., 2020）：Facebook发起的Deepfake Detection Challenge数据集，包含超过10万个视频片段。
- **ASVspoof**系列：音频伪造检测的标准评测基准，涵盖多种语音合成和转换攻击方法。

检测性能的评估通常采用准确率（Accuracy）、等错误率（Equal Error Rate, EER）和曲线下面积（AUC-ROC）等指标。然而，当前的评估体系存在若干不足：首先，大多数基准数据集基于实验室环境构建，与真实世界中的伪造存在分布差距；其次，图像压缩和社交媒体传输会显著降低检测性能；最后，对抗性攻击可以有效规避现有检测器。

对抗鲁棒性是检测领域面临的核心挑战之一。Neekhara et al.（2021）展示了针对Deepfake检测器的对抗攻击，通过添加人眼不可察觉的扰动即可使检测器失效。这促使研究者开发更加鲁棒的检测方法，包括对抗训练（Adversarial Training）和输入预处理等防御策略。

---

## 12.2 版权问题

### 12.2.1 AI生成内容的版权归属

AI生成内容（AI-Generated Content, AIGC）的版权归属是当前法律界最具争议的问题之一。传统版权法的核心原则之一是"作者身份"（Authorship），即作品必须由具有创作意图的人类作者创作才能获得版权保护。然而，生成式AI系统能够独立产生具有高度创造性的内容，这直接挑战了传统版权法的基本假设。

2023年，美国版权局（USCO）在Zarya of the Dawn案中作出重要裁定：由Midjourney生成的图像不受版权保护，但人类对这些图像的创意性选择和编排可以获得版权保护。这一裁定确立了一个重要原则——纯AI生成的内容不具有可版权性，但人类对AI生成内容的创造性贡献可以构成版权保护的基础。

在司法层面，Thaler v. Perlmutter案（2023）中，美国联邦法院明确裁定，由AI系统DABUS自主生成的艺术品不能获得版权保护，因为缺乏人类作者。法官Howell在判决中指出："版权法旨在保护和激励人类创造力，而非保护由机器生成的创造力。"

然而，中国在此问题上展现出更为灵活的态度。2023年北京互联网法院在李某诉刘某案中裁定，如果用户在使用AI生成图像过程中进行了充分的智力投入——包括选择提示词、调整参数和筛选结果——则生成的图像可以作为美术作品获得著作权保护。这一判决在全球范围内率先承认了AI辅助创作的可版权性。

欧盟在这一问题上采取了中间立场。根据现行的版权指令，作品需要体现作者的"自由创作选择"才能获得保护，这为AI生成内容的版权保护留下了空间，但具体标准仍有待进一步明确。

### 12.2.2 训练数据版权

生成模型的训练过程涉及大规模数据的收集和使用，这引发了训练数据的版权合规问题。2023年以来，多起针对AI公司的版权诉讼集中爆发，深刻影响着这一领域的发展方向。

Getty Images诉Stability AI案（2023年1月）是最早也最具影响力的案例之一。Getty Images指控Stability AI在训练Stable Diffusion时未经授权使用了超过1200万张Getty Images的版权图片，包括完整的水印。此案的核心争议在于：使用版权作品训练AI模型是否构成版权侵权，以及是否属于合理使用（Fair Use）。

Andersen et al.诉Stability AI、Midjourney和DeviantArt案（2023年1月）代表了艺术家群体的集体诉求。三位视觉艺术家指控这些公司的AI模型"是对艺术家风格的21世纪拼贴工具"，其训练过程构成了对版权作品的大规模复制。

在文本生成领域，《纽约时报》诉OpenAI和Microsoft案（2023年12月）将训练数据版权争议推向了新高度。《纽约时报》指控GPT模型能够近乎逐字地复制其新闻报道的内容，这不仅涉及训练数据的版权问题，还涉及AI模型是否可以作为版权作品的"替代品"。

合理使用原则在这些案件中成为核心法律争论点。在美国法律框架下，合理使用的四要素分析包括：使用目的和性质、版权作品的性质、使用的数量和实质性、以及对版权作品潜在市场的影响。支持者认为AI训练是一种"变革性使用"（Transformative Use），类似于搜索引擎对网页内容的索引；反对者则认为AI训练的大规模复制不具有变革性，且AI生成的内容可能直接替代原始版权作品的市场。

在监管层面，欧盟《人工智能法案》（EU AI Act, 2024）要求通用AI模型的提供者公布训练数据的"充分详细摘要"，并遵守欧盟版权法的规定。日本在2023年修订了版权法，明确AI训练可以使用版权作品而无需获得权利人许可，但排除了"以损害创作者利益为主要目的"的使用。这些不同的法律框架反映了各国在技术创新与版权保护之间的不同权衡。

### 12.2.3 风格模仿的法律边界

风格（Style）本身的可版权性是一个高度复杂的问题。传统版权法通常保护具体的表达（Expression），而不保护抽象的风格或思想。然而，当AI能够精准模仿特定艺术家的风格并批量生产作品时，这一传统边界受到了挑战。

在Andersen案中，原告特别指出AI模型能够"以Andersen的风格"生成图像，构成对其个人风格的不当利用。然而，"风格"在法律上是否构成可保护的知识产权仍然是一个悬而未决的问题。美国最高法院在Star Athletica v. Varsity Brands案（2017）中确立了"可分离性"标准，即艺术元素必须能够与实用功能分离才能获得版权保护，但这一标准是否适用于"风格"仍有争议。

在人格权层面，许多司法管辖区已经对未经授权的肖像使用进行了限制。美国多个州（如加利福尼亚州、纽约州、德克萨斯州）已经制定了针对Deepfake和数字仿真的法律。中国《民法典》明确规定了对个人肖像权和声音权的保护，未经授权使用AI克隆他人声音或面容可能构成侵权。

从伦理角度看，即使法律尚未明确禁止风格模仿，AI的大规模风格模仿也可能对原创艺术家的生计造成实质影响。一些行业组织和艺术家群体呼吁建立"选择退出"（Opt-out）机制，允许艺术家将其作品从AI训练数据集中移除。Glaze（Shan et al., 2023）和Nightshade（Shan et al., 2024）等工具的出现为艺术家提供了技术层面的自我保护手段，这些工具通过向图像添加人眼不可察觉的扰动来干扰AI模型的训练或模仿。

---

## 12.3 偏见与公平

### 12.3.1 生成模型中的社会偏见

生成模型中的偏见问题是一个横跨技术和社会科学的交叉议题。偏见在生成模型中的表现形式多样且影响深远。

在文本生成领域，大型语言模型已被证实会复制和放大训练数据中存在的性别、种族和文化偏见。Bordia和Bowman（2019）系统研究了语言模型中的性别偏见，发现模型在生成文本时会将特定职业与特定性别不均衡地关联——例如，"护士"更多与女性代词关联，"程序员"则更多与男性代词关联。Bolukbasi et al.（2016）的经典工作揭示了词向量空间中的性别偏见结构，这一发现被后续研究扩展到了大型语言模型中。

在图像生成领域，偏见的表现更加直观。Luccioni et al.（2023）对Stable Diffusion的系统评估发现，该模型在生成"CEO"、"医生"等职业相关图像时，绝大多数生成结果为白人男性形象；而在生成"护士"、"清洁工"等图像时，则倾向于生成女性或有色人种形象。Bianchi et al.（2023）进一步发现，Stable Diffusion在生成"犯罪者"形象时过度关联有色人种，而在生成"恐怖分子"形象时则过度关联中东面孔，这种偏见的自动化传播可能加剧社会歧视。

文本到图像模型中的偏见还体现在文化代表性上。当用户输入"婚礼"这一提示词时，模型几乎只会生成西式婚礼场景，而忽略全球多元的婚礼文化。类似地，"家庭"、"节日"等概念的生成结果也严重偏向西方文化视角，反映了训练数据中西方互联网内容的主导地位。

### 12.3.2 数据偏见的来源与传导

生成模型中的偏见根植于训练数据。互联网规模的训练数据不可避免地反映了现实社会中已有的不平等结构。ImageNet和LAION-5B等大规模数据集中存在的偏见已被多项研究所记录。

Schuhmann et al.（2022）对LAION-5B数据集的分析揭示了多个层面的问题：首先，数据集在地理和语言分布上严重不均衡，英语内容占据主导地位，发展中国家的代表性严重不足；其次，数据集中的文本-图像对存在噪声和错误关联，部分图片的文字描述包含有害的刻板印象；最后，数据集的过滤机制虽然移除了明显不安全的内容，但隐含的偏见难以通过简单的过滤来消除。

数据偏见通过模型训练过程被学习和固化，并在生成阶段被放大。放大效应（Amplification Effect）指的是模型在生成过程中产生的偏见程度可能超过训练数据中的偏见程度。这一现象已在多项研究中被观察到，其机制可能与模型对分布中模式的过度拟合有关。

此外，提示词工程（Prompt Engineering）的使用模式也可能引入额外的偏见。用户在使用生成模型时往往倾向于使用英语提示词，这进一步强化了模型对西方文化视角的偏好。在某些情况下，即使用户并未在提示词中指定种族或性别特征，模型也会基于其学到的偏见生成特定类型的结果。

### 12.3.3 偏见评估与缓解

偏见的量化评估是缓解偏见的第一步。研究者开发了多种评估框架和指标来测量生成模型中的偏见。

在文本生成领域，BOLD（Dhamala et al., 2021）基准测试评估了语言模型在不同人口群体上的生成公平性；BBQ（Parrish et al., 2022）通过问答形式测试模型中的社会偏见。在图像生成领域，FID（Fréchet Inception Distance）等生成质量指标已被扩展为公平性感知的版本，用于评估模型在不同人口群体上的生成质量是否一致。

偏见缓解策略可分为数据层面、模型层面和后处理层面三类。

**数据层面的干预**包括：数据再平衡（Rebalancing），即调整训练数据中不同群体的代表性比例；数据增强（Augmentation），即通过合成或人工标注补充缺失的群体代表性；以及数据去偏（Debiasing），即识别和移除训练数据中有偏见的样本。Sheng et al.（2020）的研究表明，简单的数据平衡策略可以有效减少语言模型中的部分偏见。

**模型层面的干预**包括：对抗去偏（Adversarial Debiasing），通过对抗训练使模型学习与敏感属性无关的表示；约束优化（Constrained Optimization），在训练目标中加入公平性约束；以及提示词引导的去偏（Prompt-based Debiasing），通过精心设计的提示词引导模型生成更公平的结果。Chuang et al.（2023）提出了用于文本到图像模型的去偏微调方法，通过在特定概念上进行微调来减轻模型中的刻板印象。

**后处理层面的干预**包括：输出过滤（Output Filtering），对模型输出进行偏见检测和过滤；以及结果再排序（Re-ranking），在多个生成结果中选择偏见最小的结果呈现给用户。然而，后处理方法往往难以全面覆盖所有偏见形式，且可能影响生成内容的质量和多样性。

需要强调的是，偏见缓解不存在银弹式的解决方案。任何单一的干预措施都可能在减少某类偏见的同时引入新的偏见，或在公平性和生成质量之间产生权衡。系统性的偏见治理需要技术手段、制度设计和持续评估的有机结合。

---

## 12.4 安全生成

### 12.4.1 有害内容过滤

生成模型被滥用来制造有害内容——包括但不限于虚假信息、仇恨内容、非自愿色情、暴力内容和欺诈材料——是当前最紧迫的安全问题之一。

有害内容过滤的第一道防线是训练数据的净化。LAION在构建LAION-5B数据集时采用了多阶段过滤流程：首先利用NSFW分类器移除色情和暴力内容；其次通过CLIP嵌入的相似度过滤低质量或不相关的文本-图像对；最后利用哈希算法移除已知的CSAM（儿童性虐待材料）内容。然而，这些过滤手段并非完美，有害内容仍可能以更隐蔽的形式存在于训练数据中。

在模型推理阶段，安全分类器被部署为内容生成的守门人。OpenAI在DALL·E系统中部署了多层安全过滤机制：在输入端，提示词分类器检测并拒绝可能生成有害内容的提示；在输出端，生成的图像经过安全分类器的二次检查，不安全的内容在呈现给用户之前被移除。Stable Diffusion的Safety Checker组件也采用了类似的架构。

然而，有害内容过滤面临根本性的技术挑战。对抗性攻击者可以通过提示词注入（Prompt Injection）、越狱（Jailbreaking）和后缀攻击（Suffix Attack）等手段绕过安全过滤。Zou et al.（2023）的研究表明，通过在提示词末尾添加优化的对抗性后缀，可以使大型语言模型生成被禁止的内容。这一发现表明，基于黑名单或关键词的过滤方法存在固有的局限性。

更有前景的防御策略包括：基于内容理解的深层语义过滤，而非简单的模式匹配；持续学习和适应的动态安全系统，能够识别新出现的规避策略；以及红队测试（Red Teaming），通过主动寻找系统漏洞来加强防御。Perez et al.（2022）提出了利用语言模型自身来生成对抗性测试用例的方法，为自动化红队测试开辟了新方向。

### 12.4.2 水印技术

数字水印（Digital Watermarking）被视为解决AI生成内容溯源和真实性验证问题的关键技术。水印技术的目标是在生成内容中嵌入可检测但不可见的信号，从而标识内容的AI生成属性。

**生成阶段水印。** 在内容生成过程中嵌入水印是目前最主流的方法。Kirchenbauer et al.（2023）提出了针对大型语言模型的文本水印方案：在文本生成过程中，通过一个基于密钥的伪随机函数将词汇表分为"绿色列表"和"红色列表"，然后在采样时偏向于选择绿色列表中的词汇。这种偏向在统计上是可检测的，但对生成文本的质量影响微乎其微。该方案的理论基础表明，随着生成文本长度的增加，水印检测的统计功效趋于完美。

对于图像生成模型，Fernandez et al.（2023）和Wen et al.（2023）分别提出了针对扩散模型的水印嵌入方法。这些方法利用扩散模型的采样过程，在去噪步骤中引入受控的扰动来嵌入水印信息。Stable Signature（Fernandez et al., 2023）通过微调模型的解码器，使其生成的每张图像都携带特定的水印信号，而无需在推理时进行额外处理。

Google DeepMind开发的SynthID系统代表了水印技术的工程化实践。SynthID能够在AI生成的图像和文本中嵌入不可见的水印，即使内容经过裁剪、压缩或截图处理后仍可被检测到。2024年，Meta也宣布在其AI生成内容中部署水印系统。

**检测与验证。** 水印的检测需要拥有嵌入时使用的密钥或算法。检测器需要在误报率（将人类创作内容误判为AI生成）和漏报率（未能检测到AI生成内容）之间取得平衡。鲁棒性是水印技术面临的主要挑战之一——水印需要抵抗各种内容变换（裁剪、旋转、压缩、噪声添加等）而不失效，同时又不能影响内容的正常使用。

CPC（Constrained Perturbation Coding）和频域水印等传统水印技术已被扩展应用于AI生成内容，但扩散模型生成的图像的内在随机性为水印嵌入和检测带来了独特的技术挑战。此外，水印技术还面临"去水印攻击"的威胁——攻击者可能通过再生成、微调或其他后处理手段移除或破坏水印。

### 12.4.3 负责任的生成AI实践

构建负责任的生成AI生态系统需要技术开发者、平台运营者、政策制定者和用户群体的共同努力。

**技术治理框架。** Anthropic、OpenAI和Google DeepMind等主要AI实验室已建立了内部安全评估流程。Anthropic的负责任扩展政策（Responsible Scaling Policy）定义了AI系统的风险等级，并为不同等级设定了相应的安全要求。OpenAI在其《Preparedness Framework》中建立了从风险评估到缓解措施的系统性流程。这些框架的共同特征是将安全评估嵌入模型开发的全生命周期，而非仅在部署后进行事后补救。

**开源与闭源的权衡。** 生成模型的开源发布策略是一个具有争议性的话题。支持开源的观点认为，开放模型权重可以促进学术研究、加速安全技术的发展，并防止AI能力被少数机构垄断。反对者则指出，开源模型一旦发布便无法撤回，可能被恶意行为者无限制地使用。Meta发布的LLaMA系列模型引发了这一讨论的高潮——尽管Meta在发布时设置了使用限制，但模型权重很快被泄露并在互联网上广泛传播。

在实践层面，分层开放（Tiered Access）模式正在成为一种折中方案：基础模型向学术研究者开放，应用级接口向经过审核的开发者开放，而最终用户通过受控的API访问。这种模式在保护技术安全的同时维持了研究社区的开放性。

**行业自律与标准制定。** 2023年，多家AI公司自愿签署了白宫AI安全承诺，承诺在发布前对AI系统进行安全测试。在标准层面，NIST发布的AI风险管理框架（AI RMF）为组织提供了系统性的AI风险管理指引。ISO/IEC 42001标准（2023年12月发布）建立了AI管理体系的国际标准，为组织实施负责任的AI实践提供了制度框架。

**检测生态的构建。** 面对Deepfake等安全威胁，建立开放的检测生态至关重要。Meta、Microsoft和Partnership on AI联合发起的Deepfake Detection Challenge推动了检测技术的发展。Reality Defender、Sensity AI等初创公司正在将检测技术商业化，为企业和媒体机构提供Deepfake检测服务。然而，当前的检测生态仍面临"军备竞赛"的困境——随着生成技术的快速迭代，检测方法需要不断更新以应对新的伪造手段。

**教育与素养。** 技术手段之外，提升公众的媒体素养和AI意识同样重要。芬兰等国家已将媒体素养教育纳入国民教育体系，帮助公民识别虚假信息和AI生成内容。多个国际组织和非营利机构也在积极推动AI素养的普及。

---

## 本章小结

本章从Deepfake检测、版权问题、偏见与公平、安全生成四个维度系统阐述了生成模型面临的安全与伦理挑战。这些问题相互交织、彼此影响：偏见驱动有害内容的产生，版权争议影响数据获取和模型训练，Deepfake威胁信息生态的信任基础，而安全生成实践则是应对上述挑战的综合性框架。

面对这些挑战，我们既不应因噎废食地放弃生成式AI技术的巨大潜力，也不应盲目乐观地忽视其现实风险。负责任的技术发展需要在创新与安全之间寻找动态平衡，这种平衡需要技术创新、法律制度、行业自律和公众参与的协同推进。

未来的方向可能包括：更加精细化的偏见评估和缓解工具；适应AI时代特点的新型版权框架；鲁棒且可扩展的水印和检测基础设施；以及全球协调的AI治理机制。正如生成式AI本身是一个快速演进的领域，其安全与伦理框架也需要持续的迭代和完善。

生成式AI的未来，取决于我们今天如何选择治理它。

---

## 参考文献

1. Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). MesoNet: a compact facial video forgery detection network. *IEEE WIFS*.
2. Bianchi, F., et al. (2023). Easily accessible text-to-image generation amplifies demographic stereotypes at large scale. *FAccT*.
3. Bolukbasi, T., et al. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. *NeurIPS*.
4. Bordia, S., & Bowman, S.R. (2019). Identifying and Reducing Gender Bias in Word-Level Language Models. *NAACL*.
5. Choi, Y., et al. (2018). StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation. *CVPR*.
6. Chuang, C.Y., et al. (2023). Debiasing text-to-image diffusion models. *NeurIPS Workshop*.
7. Dhamala, J., et al. (2021). BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation. *FAccT*.
8. Dolan, B., et al. (2020). The Deepfake Detection Challenge (DFDC) Preview Dataset. *arXiv:1910.08854*.
9. Fernandez, P., et al. (2023). Stable Signature: Rooting Watermarks in Latent Diffusion Models. *ICCV*.
10. He, Z., et al. (2019). AttGAN: Facial Attribute Editing by Only Changing What You Want. *IEEE TIP*.
11. Karras, T., et al. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks. *CVPR*.
12. Karras, T., et al. (2020). Analyzing and Improving the Image Quality of StyleGAN. *CVPR*.
13. Karras, T., et al. (2021). Alias-Free Generative Adversarial Networks. *NeurIPS*.
14. Kim, J., et al. (2021). Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS). *ICML*.
15. Kirchenbauer, J., et al. (2023). A Watermark for Large Language Models. *ICML*.
16. Li, L., et al. (2020). Celeb-DF: A Large-Scale Challenging Dataset for Deepfake Forensics. *CVPR*.
17. Li, Y., et al. (2020). Face X-Ray for More General Face Forgery Detection. *CVPR*.
18. Luccioni, A., et al. (2023). Stable Bias: Evaluating Societal Representations in Diffusion Models. *NeurIPS*.
19. Mittal, T., et al. (2020). Emotions Don't Lie: An Audio-Visual Deepfake Detection Method Using Affective Cues. *ACM MM*.
20. Neekhara, P., et al. (2021). Adversarial Deepfakes: Evaluating Vulnerability of Deepfake Detectors to Adversarial Examples. *WACV*.
21. Nirkin, Y., et al. (2019). FSGAN: Subject Agnostic Face Swapping and Reenactment. *ICCV*.
22. Parrish, A., et al. (2022). BBQ: A Hand-Built Bias Benchmark for Question Answering. *ACL Findings*.
23. Perez, E., et al. (2022). Red Teaming Language Models with Language Models. *EMNLP*.
24. Rossler, A., et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. *ICCV*.
25. Schuhmann, C., et al. (2022). LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. *NeurIPS*.
26. Shan, S., et al. (2023). Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models. *USENIX Security*.
27. Shan, S., et al. (2024). Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models. *IEEE S&P*.
28. Sheng, E., et al. (2020). The Woman Worked as a Babysitter: On Biases in Language Generation. *EMNLP*.
29. Siarohin, A., et al. (2019). First Order Motion Model for Image Animation. *NeurIPS*.
30. Tak, H., et al. (2021). Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2vec 2.0 and Data Augmentation. *Odyssey*.
31. Wang, R., et al. (2020). Multi-Attentional Deepfake Detection. *CVPR*.
32. Wen, Y., et al. (2023). A Private Watermark for Diffusion Models. *ICML*.
33. Yamagishi, J., et al. (2021). ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection. *IEEE SLT*.
34. Zhao, H., et al. (2021). Multi-Attentional Deepfake Detection. *CVPR*.
35. Zi, B., et al. (2020). WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection. *ACM MM*.
36. Zou, A., et al. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. *arXiv:2307.15043*.
-e 

---


# 第13章 生成式AI的未来

> "预测未来的最好方式，就是创造它。"——艾伦·凯

生成式人工智能（Generative AI）在过去十年间经历了从实验室概念到产业基础设施的跃迁。从2014年生成对抗网络（GAN）的横空出世，到变分自编码器（VAE）对隐变量建模的理论贡献，再到扩散模型以去噪为核心范式重新定义高质量生成的标准，每一轮范式转换都深刻重塑了机器对"创造"的理解。站在2024年的技术前沿回望，生成式AI正从单一模态的专项工具演化为跨模态、跨任务的通用生成平台。本章将从四个维度——统一生成架构、世界模拟器、个性化生成以及开放问题与社会责任——勾勒生成式AI的未来图景。

## 13.1 统一生成架构：从专用到通用

### 13.1.1 统一性的历史驱动

生成式AI的早期发展呈现出"一模态一架构"的特征。图像生成由GAN和扩散模型主导，文本生成由自回归语言模型（如GPT系列）主导，音频和视频生成则各自发展出专门的模型结构。这种分化不仅增加了工程复杂度，也阻碍了跨模态知识的迁移。一个自然的问题浮现：是否存在一种统一的架构，能够在一个模型中同时处理离散文本、连续图像、音频乃至视频的生成？

答案正在变得清晰。2023年至2024年间，一批突破性的统一生成架构相继提出，标志着生成式AI从"专才"向"通才"的关键转型。

### 13.1.2 Transfusion：自回归与扩散的融合

Transfusion（Zhou et al., 2024）提出了一个优雅的方案：将自回归语言建模与扩散过程统一在同一个Transformer架构中。其核心洞察在于，文本的离散token与图像的连续表示可以共享同一模型，但采用不同的训练目标——文本部分使用下一个token预测的交叉熵损失，图像部分使用扩散去噪损失。两个目标在同一次前向传播中计算，并通过加权求和共同优化。

这种方法的技术贡献体现在几个方面。首先，它避免了将连续信号强制量化为离散token所带来的信息损失——这是许多早期多模态模型（如DALL-E的第一代）的固有缺陷。其次，共享的Transformer骨干网络天然支持跨模态注意力，使得文本与图像之间能够进行深层语义对齐，而不仅仅是浅层的条件注入。最后，Transfusion在推理时可以灵活切换模式：在文本区域执行自回归采样，在图像区域执行迭代去噪，从而在单一模型中实现两种范式的优势互补。

实验表明，Transfusion在图像生成质量上可以匹敌同等规模的专用扩散模型，同时保持了语言模型的文本理解与生成能力，为真正统一的多模态生成模型奠定了架构基础。

### 13.1.3 Emu3：纯自回归的视觉生成

如果说Transfusion代表了"混合范式"路线，那么Emu3（BAAI, 2024）则坚定地走向了"纯自回归"阵营。Emu3的核心主张是：经过精心设计的视觉tokenizer可以将图像压缩为高质量的离散token序列，使得视觉生成可以完全复用下一token预测（next-token prediction）的范式，无需引入扩散过程。

Emu3的视觉tokenizer采用了改进的向量量化变分自编码器（VQ-VAE）架构，通过多尺度量化和语义对齐训练，显著提升了视觉token的重建质量。在此基础上，Emu3在一个统一的Transformer中同时处理文本token和视觉token，使用单一的自回归目标进行训练。这一设计的优雅之处在于：它将多模态生成问题简化为一个序列预测问题，所有的多模态理解和生成能力都涌现自同一套机制。

Emu3的成功揭示了一个深刻的见解：token化（tokenization）的质量往往是统一架构成败的关键。当视觉tokenizer能够充分保留语义信息和视觉细节时，自回归模型在视觉生成上的表现可以与扩散模型相媲美，同时天然继承了语言模型的上下文学习、指令跟随等能力。

### 13.1.4 多模态统一模型的格局与展望

在Transfusion和Emu3之外，一系列多模态统一模型也在探索不同的技术路径。Chameleon（Meta, 2024）采用了"早期融合"策略，在预训练阶段就将所有模态的token混合在一个序列中，实现了真正的模态无差别处理。Show-o（Microsoft, 2024）则尝试在单一流匹配（flow matching）框架中统一离散与连续生成。Janus（DeepSeek, 2024）提出了一种解耦的视觉编码策略，分别用于理解和生成的视觉编码器共享同一语言骨干，避免了两个任务之间的干扰。

这些工作的共同趋势是：统一生成架构正在从"技术可行性验证"阶段迈向"大规模实用化"阶段。未来的统一模型可能不再区分"文本模型"和"图像模型"，而是像人类大脑一样，在一个统一的认知系统中同时处理语言、视觉、听觉等多种模态的信息生成与理解。这将对AI基础设施的设计范式产生根本性影响——从部署多个专用模型转向部署单个通用生成模型。

然而，统一架构也面临挑战。不同模态的生成难度和计算需求差异巨大，如何在训练中平衡各模态的学习进度、在推理中高效调度计算资源，仍是活跃的研究方向。此外，统一模型的评估标准也需要发展——当一个模型同时进行文本对话和图像生成时，传统的单一指标评估体系不再适用。

## 13.2 世界模拟器：视频生成作为世界模型

### 13.2.1 从"生成好看"到"模拟真实"

如果说统一架构代表了生成式AI在"形式"上的统一，那么世界模拟器（World Simulator）则代表了其在"功能"上的升华。传统视频生成的目标是产生视觉上逼真的帧序列；而世界模拟器的目标更为宏大：构建一个能够理解和模拟物理世界运行规律的内部模型，使其生成的视频不仅"好看"，而且"合理"——符合物理定律、因果逻辑和空间几何约束。

这一概念并非全新。在强化学习领域，世界模型（World Model）一直是核心研究方向之一，旨在让智能体在内部模拟环境中进行规划和决策。然而，传统世界模型通常在低维状态空间中操作，与高维像素级的视频生成存在巨大鸿沟。生成式AI的最新进展正在弥合这一鸿沟。

### 13.2.2 Sora：通往物理世界的窗口

2024年2月，OpenAI发布的Sora技术报告（"Video Generation as World Simulators"）在学术界和工业界引发了巨大反响。Sora展示了长达一分钟的高质量视频生成能力，其分辨率可达1080p，画面质量和时间一致性远超此前所有公开模型。但更令研究者兴奋的是，Sora展现出了对三维空间、物体运动和物理交互的初步理解。

Sora的技术架构建立在时空patch（spacetime patch）的表示之上。视频被分解为一系列在时间和空间上局部化的patch，这些patch作为基本token被送入一个扩散Transformer（DiT）进行处理。这种表示方式天然地编码了时空局部性，使得模型能够学习到物体在时间和空间中的连续性。

OpenAI在技术报告中明确将Sora定位为"世界模拟器"，而非单纯的"视频生成器"。报告中展示了Sora在以下方面的能力：三维一致性——相机围绕物体旋转时，物体保持一致的三维结构；物体持久性——被遮挡的物体在重新出现时保持身份不变；与数字世界的交互——模拟Minecraft等游戏环境的画面和物理逻辑。尽管这些能力仍不完美（例如在复杂物理交互中仍会出现违反物理定律的现象），但它们暗示了一种可能性：足够强大的视频生成模型，可能通过大规模数据的统计学习，隐式地获得对物理世界的近似理解。

### 13.2.3 物理推理的挑战与路径

将视频生成模型转化为真正的世界模拟器，核心挑战在于物理推理（physical reasoning）。当前的视频生成模型本质上是统计学习系统，它们从大量视频数据中学习像素级的分布模式，但并未显式地建模物理定律。当遇到训练数据中罕见的物理场景（如复杂的流体动力学、多体碰撞），模型的生成结果往往会出现物理不一致。

研究者正在从多条路径探索解决方案。第一条路径是**物理注入**——在模型架构或训练过程中引入物理先验知识。例如，Physics-Informed Neural Networks（PINN）的思想可以被扩展到视频生成中，通过物理损失函数约束生成结果满足基本的守恒定律。第二条路径是**模拟器蒸馏**——使用物理引擎（如MuJoCo、Isaac Gym）生成大量物理正确的视频数据，将其与真实视频混合训练，使模型在统计学习的同时获得物理正确的归纳偏置。第三条路径是**神经符号混合**——将神经网络的感知能力与符号推理系统的逻辑能力结合，让模型在生成视频的同时进行显式的物理推理。

此外，一些工作正在探索世界模型的层级化构建。例如，UniSim（Google DeepMind, 2023）尝试构建一个通用模拟器，能够根据文本或动作指令生成对应的视频结果，从而为机器人学习和具身智能提供训练环境。Genie（Google DeepMind, 2024）则从无标注视频中学习可交互的生成式环境，用户可以通过动作控制生成内容的变化。

### 13.2.4 世界模拟器的应用前景

世界模拟器的应用前景远超传统视频生成的范畴。在**具身智能**领域，一个精确的世界模拟器可以为机器人提供无限的训练环境，大幅降低真实世界试错的成本和风险。在**自动驾驶**领域，世界模拟器可以生成各种极端天气和罕见交通场景的视频数据，用于安全测试和算法验证。在**科学发现**领域，世界模拟器可以帮助研究者可视化和预测复杂系统（如气候、分子动力学）的演化。在**创意产业**中，世界模拟器将使电影制作、游戏开发和虚拟现实内容创作进入一个新纪元——创作者可以用自然语言描述场景，由模拟器自动生成物理正确的视觉内容。

然而，世界模拟器也带来了深刻的伦理问题。如果一个系统能够生成与真实世界无法区分的视频，那么"眼见为实"这一人类认知的基本假设将被根本动摇。深度伪造（deepfake）的技术门槛将进一步降低，信息真伪的辨别将变得更加困难。这些挑战将在本章最后一节详细讨论。

## 13.3 个性化生成：千人千面的创造力

### 13.3.1 从通用到个人

大规模生成模型的一个根本性张力在于：它们学习了海量数据的总体分布，但每个用户的需求往往是高度个性化的。一个设计师可能希望AI生成特定风格的插画，一个品牌可能需要保持一致的视觉调性，一个普通用户可能想把自己的宠物狗放入各种创意场景。个性化生成（Personalized Generation）正是为了解决这一张力而发展起来的技术方向。

个性化生成的核心问题可以表述为：如何在不重新训练整个大模型的前提下，使其能够学习和生成特定主体（subject）、风格（style）或概念（concept）的内容？这一问题的解决方案在过去两年间取得了令人瞩目的进展。

### 13.3.2 DreamBooth：少样本主体学习

DreamBooth（Ruiz et al., 2022）是个性化生成领域的里程碑式工作。其核心思想惊人地简洁：给定某个特定主体（如一只狗、一辆车、一个人）的3-5张照片，通过微调整个扩散模型的参数，使其能够将一个稀有的文本标识符（如"a [V] dog"）与该主体的视觉特征绑定。此后，用户可以通过包含该标识符的文本提示生成该主体在任意场景中的图像。

DreamBooth的技术贡献在于几个关键设计。首先，它引入了"先验保留损失"（prior preservation loss），通过在微调过程中同时生成该类别的通用样本来防止模型"遗忘"——即过度拟合到少量训练样本而丧失生成其他同类物体的能力。其次，DreamBooth展示了扩散模型具有惊人的少样本学习能力：仅凭几张图片，模型就能学习到主体的身份特征（如狗的毛色、花纹、体型），并将其泛化到训练中未出现过的姿态、光照和场景。

DreamBooth的局限性在于其微调成本。完整微调一个大模型需要大量GPU内存和时间，这限制了其在消费级设备上的部署。这一局限直接催生了后续的参数高效微调方法。

### 13.3.3 LoRA：参数高效的微调革命

低秩适应（Low-Rank Adaptation, LoRA）（Hu et al., 2022）最初为语言模型提出，但其在扩散模型个性化中的应用产生了更为深远的影响。LoRA的核心思想是：预训练模型的参数更新矩阵通常是低秩的，因此可以将更新分解为两个低秩矩阵的乘积，从而将可训练参数量减少几个数量级。

在扩散模型的语境下，LoRA微调通常只需要训练原始模型参数量的0.1%至1%，即可实现高质量的个性化。一个典型的DreamBooth微调需要约4GB显存和数分钟时间，而使用LoRA后可以在消费级GPU上以更少的时间完成。更关键的是，LoRA的低秩适配器（adapter）文件通常只有几MB到几十MB大小，可以通过简单地加载和卸载不同的适配器来切换不同的个性化概念，实现了"一个基础模型+多个轻量适配器"的灵活架构。

LoRA的普及催生了一个繁荣的社区生态。用户在Civitai、Hugging Face等平台上分享自己训练的LoRA适配器，涵盖了各种风格（如赛博朋克、水彩画、浮世绘）、角色（来自动漫、游戏的虚构角色）和主题。这种"适配器市场"的模式深刻改变了生成式AI的使用方式：用户不再需要从零训练模型，而是可以组合现有的适配器来快速实现创意目标。

### 13.3.4 概念学习与文本反转

除了对模型参数进行微调，另一条个性化路径是学习新的文本嵌入（text embedding）。Textual Inversion（Gal et al., 2022）提出了一种"学习你说的话"的方法：不修改模型的任何参数，而是为新的概念学习一个新的文本embedding向量。给定3-5张包含特定概念的图像，Textual Invasion优化一个"伪token"的embedding，使其能够通过文本提示激活模型中与该概念对应的生成能力。

这种方法的优雅之处在于它完全不改变预训练模型，因此不会引入任何灾难性遗忘的风险，且不同概念的embedding可以自由组合。然而，Textual Inversion的表达能力受限于文本编码器的嵌入空间维度，对于复杂视觉概念（如特定人脸）的捕捉能力不如DreamBooth。

后续工作如Custom Diffusion（Kumari et al., 2023）进一步探索了概念学习的高效策略，通过只微调模型中的键值投影（key-value projection）来实现概念注入，在质量和效率之间取得了更好的平衡。UniTune（Voynov et al., 2023）则探索了在统一框架中同时处理多个概念学习任务的可能性。

### 13.3.5 用户定制的未来形态

展望未来，个性化生成正在向以下方向演进：

**实时个性化**：随着适配器加载和切换的效率提升，未来的生成系统可能支持实时的概念混合和风格调整。用户可以在生成过程中动态添加、移除或调整个性化元素，实现类似"实时绘画"的交互体验。

**多概念组合**：当前的个性化方法在同时处理多个自定义概念时仍存在属性泄露（attribute leakage）和身份混淆的问题。未来的方法需要更好地支持多主体场景，例如同时将"我的猫"和"我的狗"放入同一张图片中，同时保持各自的身份特征。

**3D与视频个性化**：从2D图像扩展到3D模型和视频的个性化生成是一个自然而富有挑战的方向。DreamFusion（Google, 2022）已经展示了从文本生成3D模型的可能性，将个性化概念注入3D生成是下一个前沿。在视频领域，保持主体在时间维度上的一致性比单张图像更加困难，但也更具应用价值。

**隐私与所有权**：个性化生成技术使得任何人都可以使用他人的面部图像训练模型并生成其虚拟形象，这引发了深刻的隐私和伦理问题。如何在技术层面防止非授权的面部克隆、如何建立数字肖像权的法律框架，是与技术发展并行的社会议题。

## 13.4 结语：技术脉络、开放问题与社会责任

### 13.4.1 技术脉络回顾

回顾生成式AI的发展历程，可以清晰地辨识出几条交织的技术脉络。

**从对抗到去噪**：GAN时代的核心创新是通过对抗训练绕过显式似然估计的困难，但训练不稳定和模式坍塌始终是其阿喀琉斯之踵。扩散模型通过将生成过程分解为渐进的去噪步骤，用简单的均方误差损失替代了不稳定的对抗博弈，不仅提升了训练稳定性，还在生成质量上实现了质的飞跃。这一范式转移的技术根源在于：将一个困难的生成问题分解为一系列简单的条件去噪问题，每个步骤都可以被可靠地学习。

**从单一到多模态**：早期的生成模型通常专注于单一模态。VAE和GAN主要处理图像，RNN和后来的Transformer处理文本。CLIP（Radford et al., 2021）的出现开启了视觉-语言对齐的新时代，为文本引导的图像生成提供了强大的语义桥接。此后，多模态生成能力迅速扩展到音频、视频、3D等领域，统一架构的探索正如本章11.1节所述。

**从生成到理解**：一个微妙但重要的趋势是，生成能力正在反哺理解能力。研究表明，训练良好的生成模型内部隐式地学到了丰富的语义表示，这些表示可以用于下游的理解任务。世界模拟器的概念更是将生成与理解推向了统一——一个能够生成物理合理视频的模型，必然对物理世界有一定程度的"理解"。

**从实验室到基础设施**：生成式AI正在从研究原型演变为产业基础设施。Stable Diffusion的开源生态、Midjourney的产品化、各云厂商的生成式AI API服务，都标志着这一技术栈的成熟。未来的AI基础设施可能将生成能力作为基础服务提供，如同今天的计算和存储一样。

### 13.4.2 开放问题

尽管取得了巨大进展，生成式AI仍面临多个根本性的开放问题。

**评估标准的困境**：如何评估生成模型的质量？FID（Fréchet Inception Distance）衡量分布层面的相似性，但无法评估单个样本的质量。人类评估成本高昂且存在主观偏差。CLIP Score衡量文本-图像对齐但忽略美学质量。目前尚无一个统一的、自动化的、与人类判断高度一致的评估指标。这一困境的根源在于"生成质量"本身就是一个多维度、上下文依赖的概念。

**可控性与忠实性的权衡**：用户对生成内容的控制粒度越细，模型需要遵守的约束就越多，但过度约束可能导致生成质量下降或多样性丧失。如何在可控性和生成自由度之间找到最优平衡，是一个需要持续探索的工程和科学问题。

**长尾分布与罕见概念**：当前模型在训练数据中频繁出现的概念上表现出色，但对于罕见概念、文化特定的视觉符号或专业领域的知识，生成质量显著下降。如何使生成模型更好地覆盖长尾分布，是一个涉及数据、架构和训练策略的综合问题。

**因果推理与反事实生成**：当前的生成模型本质上是统计相关性学习器，它们擅长捕捉"什么与什么一起出现"，但不擅长回答"如果X没有发生会怎样"这类反事实问题。将因果推理能力引入生成模型，使其能够进行真正意义上的反事实生成和因果干预模拟，是一个远未解决的前沿问题。

**能源与效率**：大规模生成模型的训练和推理消耗巨大的计算资源和能源。一次Stable Diffusion XL的推理需要数秒的GPU时间，而视频生成的计算成本更高。如何在保持生成质量的同时大幅降低计算需求——通过模型压缩、蒸馏、稀疏化或全新的高效架构——是推动生成式AI可持续发展的关键问题。

### 13.4.3 展望与社会责任

生成式AI的未来不仅是技术问题，更是社会问题。随着生成能力的快速提升，以下社会责任议题需要技术社区和整个社会共同面对。

**真实性危机**：当AI生成的文本、图像、音频和视频达到与人类创作无法区分的水平时，数字内容的真实性验证将成为一个系统性挑战。水印技术（如C2PA标准）、检测算法和媒体素养教育需要协同推进。技术社区有责任在模型设计中内建可追溯性机制，而非将责任完全推给下游应用。

**版权与知识产权**：生成模型的训练数据通常包含大量受版权保护的作品，这引发了关于"训练使用"是否构成合理使用的激烈法律辩论。Getty Images诉Stability AI案、《纽约时报》诉OpenAI案等诉讼正在塑造这一领域的法律边界。未来的解决方案可能包括：基于授权数据的训练、对训练数据贡献者的补偿机制、以及生成内容的版权归属框架。

**偏见与公平性**：生成模型会忠实反映甚至放大训练数据中的社会偏见。当被要求生成"CEO"的图像时，模型可能过度代表某些性别和种族群体。这些偏见不仅影响生成内容的质量，还可能在社会层面强化刻板印象。去偏技术（debiasing）和公平性评估需要成为模型开发流程的标准组成部分。

**就业与创意产业**：生成式AI对创意产业的影响是双面的。一方面，它降低了创意表达的门槛，使更多人能够将想象转化为视觉或文字作品；另一方面，它对插画师、摄影师、文案作者等传统创意工作者的生计构成了切实威胁。社会需要在拥抱技术红利的同时，建立相应的再培训体系和社会安全网。

**安全与滥用**：生成式AI的滥用风险包括但不限于：深度伪造用于诈骗和诽谤、生成虚假新闻和宣传材料、制造非自愿色情内容、辅助网络攻击等。负责任的AI开发需要包括安全护栏（safety guardrails）、使用政策和滥用检测机制。然而，过度限制也可能抑制正当的创新和表达自由，如何在安全与开放之间找到平衡是一个持续的治理挑战。

**环境影响**：训练一个大型生成模型的碳排放量可能相当于数辆汽车全生命周期的排放。随着模型规模的持续增长，生成式AI的环境足迹不容忽视。发展绿色AI——通过高效架构、可再生能源供电和碳补偿机制——应成为行业的自觉追求。

### 13.4.4 结语

生成式AI正处于一个激动人心的历史节点。统一生成架构正在消融模态之间的壁垒，世界模拟器正在赋予AI对物理世界的直觉，个性化生成正在将创造力民主化。这些趋势共同指向一个愿景：未来的AI系统将不仅仅是工具，而是能够理解、模拟和共同创造的智能伙伴。

然而，技术的力量越大，伴随的责任也越重。生成式AI的未来不仅取决于算法的突破和工程的优化，更取决于我们作为技术共同体和社会整体，能否建立起与之匹配的治理框架、伦理准则和价值导向。正如扩散模型通过逐步去噪从混沌中生成秩序，我们也需要通过持续的对话、反思和行动，在技术的喧嚣中找到通往负责任创新的路径。

生成式AI的故事远未结束。它的下一章，将由技术突破与社会责任的共同书写来完成。

---

**参考文献**

- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
- Ruiz, N., et al. (2022). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. *CVPR 2023*.
- Gal, R., et al. (2022). An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. *ICLR 2023*.
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.
- Zhou, C., et al. (2024). Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model. *arXiv:2408.11039*.
- BAAI. (2024). Emu3: Next-Token Prediction is All You Need. *arXiv:2409.18869*.
- OpenAI. (2024). Video Generation as World Simulators (Sora Technical Report).
- Kumari, K., et al. (2023). Multi-Concept Customization of Text-to-Image Diffusion. *CVPR 2023*.
- Poole, B., et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion. *ICLR 2023*.
- Yang, M., et al. (2024). Chameleon: Mixed-Modal Early-Fusion Foundation Models. *arXiv:2405.09818*.
- Xie, J., et al. (2024). Show-o: One Single Transformer to Unify Multimodal Understanding and Generation. *arXiv:2408.12528*.
- Wu, Y., et al. (2024). Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation. *arXiv:2410.13848*.
- Du, Y., et al. (2023). UniSim: Learning Interactive Real-World Simulators. *arXiv:2310.06114*.
- Bruce, J., et al. (2024). Genie: Generative Interactive Environments. *arXiv:2402.15391*.
- Voynov, K., et al. (2023). UniTune: Text-Driven Image Editing by Fine-Tuning an Image Generation Model on a Single Image. *arXiv:2210.09477*.
-e 

---


# 附录：生成模型数学基础与算法速查

## 核心数学工具

### KL散度
$$D_{\text{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

### ELBO（证据下界）
$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z))$$

### Wasserstein距离
$$W(p, q) = \inf_{\gamma \in \Pi(p,q)} \mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$$

### InfoNCE损失
$$\mathcal{L} = -\log \frac{\exp(sim(z_i, z_j^+)/\tau)}{\sum_{k=1}^{N} \exp(sim(z_i, z_k)/\tau)}$$

## 主流生成模型对比

| 模型 | 类型 | 生成质量 | 多样性 | 采样速度 | 训练稳定性 |
|------|------|---------|--------|---------|-----------|
| GAN | 隐式 | 高 | 中 | 快 | 不稳定 |
| VAE | 显式 | 中 | 高 | 快 | 稳定 |
| Flow | 显式 | 中 | 高 | 快 | 稳定 |
| Diffusion | 显式 | 高 | 高 | 慢→快 | 稳定 |
| AR | 显式 | 高 | 高 | 慢 | 稳定 |

## 扩散模型核心公式

前向过程：$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$

训练目标：$\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$

CFG生成：$\hat\epsilon = (1+w)\epsilon_\theta(x_t, t, c) - w\epsilon_\theta(x_t, t, \emptyset)$

---

*注：本附录仅收录核心公式，完整推导请参阅各章正文。*
-e 

---


