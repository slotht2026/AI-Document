# Kronos：金融K线基础模型深度解析

> 基于源码的全面技术解读与实践指南

---

# 第一部分：基础篇

---

## 第一章 前言与导读

### 1.1 写作背景

2025年8月，一篇题为"Kronos: A Foundation Model for the Language of Financial Markets"的论文在arXiv上发表（arXiv:2508.02739），随即引起了金融量化和人工智能领域的广泛关注。2025年11月，该论文被人工智能顶级会议AAAI 2026正式接收。Kronos成为全球首个专为金融K线数据设计的开源基础模型，其训练数据覆盖全球45个交易所、超过120亿条K线记录。

这一里程碑事件标志着金融时间序列分析正式进入了"基础模型时代"。正如GPT系列模型改变了自然语言处理的范式，Kronos正在尝试改变金融数据分析的范式——不再针对每个特定任务从头训练模型，而是先在海量金融数据上预训练一个通用的基础模型，然后通过微调适配各种下游任务。

本书的写作目的是基于Kronos的完整源码，为读者提供一份深入、全面、可操作的技术解读。我们不仅会解释模型的设计理念和数学原理，还会逐行分析关键代码，帮助读者真正理解这个系统的每一个细节。

### 1.2 读者对象

本书适合以下读者：

**量化研究人员和交易员**：如果你希望了解如何将最新的AI技术应用于金融数据分析，本书将为你展示一个完整的从模型训练到回测验证的流程。

**机器学习工程师**：如果你对时间序列基础模型的架构设计感兴趣，本书将详细讲解分层Tokenizer、Decoder-only Transformer、二进制球面量化等前沿技术。

**AI研究者**：如果你正在研究基础模型在特定领域的应用，Kronos提供了一个将通用AI架构成功适配到高噪声、高复杂度金融数据的优秀案例。

**学生和爱好者**：如果你对AI和金融的交叉领域充满好奇，本书将从基础概念讲起，逐步深入到前沿技术。

### 1.3 本书结构

本书分为五个部分：

**第一部分（基础篇）**：包括本章导读、金融K线数据概述、时间序列预测技术演进，为后续深入理解Kronos奠定基础。

**第二部分（架构篇）**：深入分析Kronos的核心技术架构，包括分层Tokenizer的设计、Decoder-only预测模型的实现、自回归推理机制。

**第三部分（工程篇）**：讲解数据预处理、训练策略、分布式训练等工程实践细节。

**第四部分（实践篇）**：提供完整的微调实践指南，包括Qlib集成、自定义CSV数据微调、回测系统设计、Web UI搭建。

**第五部分（展望篇）**：将Kronos与同类模型进行对比，讨论金融AI的未来发展方向。

### 1.4 如何阅读本书

对于希望快速上手的读者，建议先阅读第四章（项目全景）和第十一章（自定义CSV数据微调），了解整体架构后直接开始实践。

对于希望深入理解技术原理的读者，建议按顺序阅读第一至第三章，然后重点研读第五至第七章的核心架构分析。

对于关注量化交易应用的读者，建议重点阅读第十至第十二章，了解微调流程和回测系统设计。

### 1.5 源码版本说明

本书的分析基于Kronos的开源代码仓库。所有代码引用均指向实际的源文件路径。读者可以通过以下地址获取完整代码：

```
https://gitcode.com/GitHub_Trending/kronos14/Kronos
```

项目的HuggingFace模型地址为 `NeoQuasar/Kronos-*` 系列。

---

## 第二章 金融K线数据概述

### 2.1 K线的起源与历史

K线图（Candlestick Chart）起源于18世纪的日本米市交易。当时，一位名叫本间宗久（Munehisa Homma）的米商发明了这种记录价格变动的方法，用以分析米价的走势规律。这套方法后来被称为"酒田战法"或"日本蜡烛图技术"。

K线之所以被称为"蜡烛图"，是因为其形状酷似蜡烛。每根K线由一个实体（Real Body）和上下两条影线（Shadow/Wick）组成。实体表示开盘价和收盘价之间的区间，影线表示最高价和最低价的延伸范围。

在20世纪80年代末，美国技术分析师史蒂夫·尼森（Steve Nison）将日本蜡烛图技术引入西方金融分析界，出版了经典的《日本蜡烛图技术》一书，使得K线分析成为全球通用的金融技术分析工具。

### 2.2 OHLCV数据详解

每根K线包含五个基本数据点，通常称为OHLCV：

**开盘价（Open）**：该时间段内第一笔成交的价格。开盘价反映了市场参与者在该时段开始时的预期和情绪。

**最高价（High）**：该时间段内的最高成交价格。最高价代表了买方力量在该时段内所能达到的极限。

**最低价（Low）**：该时间段内的最低成交价格。最低价代表了卖方力量在该时段内所能达到的极限。

**收盘价（Close）**：该时间段内最后一笔成交的价格。收盘价被广泛认为是K线中最重要的价格，因为它反映了该时段结束时市场的综合判断。

**成交量（Volume）**：该时间段内成交的总数量。成交量是衡量市场活跃度和确认价格趋势的重要指标。

此外，某些数据源还包含**成交额（Amount）**，即该时间段内成交的总金额。成交额等于各笔成交量乘以对应价格的总和。

在Kronos的数据模型中，输入特征包含6个维度：`['open', 'high', 'low', 'close', 'volume', 'amount']`。这6个维度共同构成了一根K线的完整信息。

### 2.3 K线的时间粒度

K线可以按照不同的时间粒度进行构建：

**分钟级K线**：包括1分钟、5分钟、15分钟、30分钟、60分钟等。分钟级K线适合短线交易者分析价格的微观波动。Kronos的演示数据中就包含了5分钟级别的K线数据。

**日级K线**：每个交易日形成一根K线，是最常用的分析粒度。日级K线包含了开盘价、最高价、最低价、收盘价四个价格点和当日成交量。

**周级和月级K线**：将一周或一个月的交易数据压缩为一根K线，适合长期投资者分析大趋势。

** tick级数据**：每笔成交形成一个数据点，是最高频率的原始数据，通常需要聚合后才能形成有意义的K线。

不同粒度的K线反映了市场在不同时间尺度上的行为模式。Kronos的一个重要设计目标就是能够处理各种时间粒度的K线数据，这得益于其通用的Tokenizer和预测架构。

### 2.4 全球交易所概览

Kronos的训练数据覆盖了全球45个主要交易所，这些交易所分布在不同的时区、遵循不同的交易规则，共同构成了全球金融市场的24小时不间断交易网络。

**美洲地区**：纽约证券交易所（NYSE）、纳斯达克（NASDAQ）、芝加哥商品交易所（CME）、多伦多证券交易所（TSX）等。

**欧洲地区**：伦敦证券交易所（LSE）、法兰克福证券交易所（FWB）、巴黎泛欧交易所（Euronext）、瑞士证券交易所（SIX）等。

**亚太地区**：东京证券交易所（TSE）、上海证券交易所（SSE）、深圳证券交易所（SZSE）、香港交易所（HKEX）、韩国交易所（KRX）、新加坡交易所（SGX）、澳大利亚证券交易所（ASX）等。

**加密货币交易所**：币安（Binance）、Coinbase、OKX等。加密货币市场由于24/7交易的特点，提供了极为丰富的高频数据。

这种多样化的数据来源使得Kronos能够学习到不同市场、不同资产类别、不同时间粒度上的通用K线模式，这是其泛化能力的重要基础。

### 2.5 K线形态分析

传统的K线形态分析是技术分析的核心内容之一。常见的K线形态可以分为单根K线形态和多根K线组合形态。

**单根K线形态**：

- **大阳线/大阴线**：实体较长，表示强烈的买方或卖方力量。
- **十字星（Doji）**：开盘价和收盘价几乎相等，实体极小，表示市场犹豫不决。
- **锤子线（Hammer）**：下影线很长，实体在上方，通常出现在下跌趋势底部，暗示可能反转。
- **上吊线（Hanging Man）**：形状与锤子线相同，但出现在上涨趋势顶部，暗示可能反转。
- **射击之星（Shooting Star）**：上影线很长，实体在下方，出现在上涨趋势中。

**多根K线组合形态**：

- **吞没形态（Engulfing）**：后一根K线的实体完全包含前一根K线的实体，是强烈的反转信号。
- **晨星/暮星（Morning Star / Evening Star）**：三根K线组合，分别表示底部反转和顶部反转。
- **三只乌鸦/三白兵**：连续三根同方向K线，表示趋势的延续。

Kronos通过在海量K线数据上的预训练，能够隐式地学习到这些形态模式及其统计意义，而不需要显式地编程识别每一种形态。

### 2.6 传统K线分析方法的局限性

传统的K线分析方法存在以下局限性：

**主观性强**：形态的识别高度依赖分析师的主观判断，不同的分析师可能对同一段行情得出不同的结论。

**规则僵化**：传统的形态识别规则是固定的，无法适应市场的动态变化。市场环境、资产类别、时间粒度的不同都会影响形态的有效性。

**无法处理高维关系**：传统的技术分析通常只能同时考虑少数几个指标，难以捕捉多维数据之间的复杂关系。

**缺乏概率框架**：传统分析给出的通常是确定性的"买入/卖出"信号，而不是概率性的预测，这使得风险管理和仓位控制变得困难。

**无法迁移学习**：在一个市场或资产上总结的分析经验，很难直接迁移到其他市场或资产上。

这些局限性正是Kronos等AI模型试图解决的问题。通过数据驱动的深度学习方法，Kronos能够自动从海量数据中学习复杂的模式，提供概率性的预测，并且具有跨市场、跨资产的泛化能力。

### 2.7 K线数据的统计特性

金融K线数据具有以下独特的统计特性，这些特性对建模提出了特殊的挑战：

**非平稳性**：金融时间序列的统计特性（均值、方差、自相关等）会随时间变化。这意味着在历史数据上训练的模型可能无法很好地泛化到未来。

**厚尾分布**：金融收益率的分布通常具有"肥尾"特性，极端事件（如暴涨暴跌）发生的概率远高于正态分布的预测。这使得基于正态分布假设的模型容易低估风险。

**波动率聚集**：高波动时期和低波动时期会交替出现，形成"波动率聚集"现象。这是GARCH等波动率模型的理论基础。

**杠杆效应**：价格下跌通常伴随着波动率上升，而价格上涨时波动率往往较低。这种不对称性被称为"杠杆效应"。

**长程依赖**：金融时间序列可能存在长程依赖关系，即很久以前的事件可能对当前价格产生影响。这使得简单的滑动窗口方法可能遗漏重要信息。

**高噪声特性**：金融数据中的信噪比极低，这使得模型很容易过拟合到噪声上，而无法学到真正的信号。

Kronos的设计充分考虑了这些特性。例如，其分层Tokenizer通过离散化连续数据来降低噪声的影响；其Decoder-only Transformer架构能够捕捉长程依赖；其在45个交易所数据上的预训练有助于学习通用的、可迁移的模式。

---

## 第三章 时间序列预测的演进

### 3.1 统计方法时代

时间序列预测的理论基础可以追溯到20世纪初。在深度学习兴起之前，统计方法一直是时间序列预测的主流工具。

**自回归移动平均模型（ARIMA）**：由Box和Jenkins在1970年代系统化提出的ARIMA模型，是时间序列预测的经典方法。ARIMA模型将时间序列分解为自回归（AR）项、差分（I）项和移动平均（MA）项三个部分。ARIMA(p,d,q)模型的数学表达式为：

$$\phi(B)(1-B)^d X_t = \theta(B)\epsilon_t$$

其中 $\phi(B)$ 是AR多项式，$\theta(B)$ 是MA多项式，$B$ 是后移算子，$d$ 是差分阶数。

ARIMA模型假设时间序列是线性的、平稳的（或经过差分后平稳），这在金融数据中往往不成立。

**广义自回归条件异方差模型（GARCH）**：Engle在1982年提出的ARCH模型和Bollerslev在1986年提出的GARCH模型，专门用于建模金融时间序列的波动率。GARCH(p,q)模型的条件方差方程为：

$$\sigma_t^2 = \omega + \sum_{i=1}^{q}\alpha_i\epsilon_{t-i}^2 + \sum_{j=1}^{p}\beta_j\sigma_{t-j}^2$$

GARCH模型能够捕捉波动率聚集现象，但仍然基于线性假设。

**向量自回归模型（VAR）**：当需要同时预测多个相关时间序列时，VAR模型提供了一个自然的框架。VAR模型将每个变量表示为所有变量滞后值的线性组合。

**指数平滑方法**：包括简单指数平滑、Holt线性趋势方法、Holt-Winters季节性方法等。这些方法通过赋予近期数据更大的权重来适应时间序列的变化。

### 3.2 机器学习方法时代

随着机器学习技术的发展，研究者们开始尝试用非线性模型来捕捉金融时间序列中的复杂模式。

**支持向量回归（SVR）**：SVR通过核函数将数据映射到高维空间，在高维空间中进行线性回归。SVR的优势在于能够处理非线性关系，并且通过正则化控制过拟合。

**随机森林和梯度提升树**：集成学习方法在金融预测中表现出色。XGBoost、LightGBM等梯度提升框架因其优秀的预测性能和可解释性，在量化投资领域得到广泛应用。

**隐马尔可夫模型（HMM）**：HMM假设时间序列由一个隐藏的状态序列生成，每个状态对应不同的观测分布。在金融领域，HMM常用于市场状态识别（如牛市、熊市、震荡市）。

### 3.3 深度学习方法时代

深度学习的兴起为时间序列预测带来了革命性的变化。

**循环神经网络（RNN）**：RNN是处理序列数据的自然选择，其隐藏状态可以传递历史信息。然而，原始RNN存在梯度消失问题，难以学习长程依赖。

**长短期记忆网络（LSTM）**：Hochreiter和Schmidhuber在1997年提出的LSTM通过引入门控机制（遗忘门、输入门、输出门）解决了梯度消失问题。LSTM在金融时间序列预测中得到了广泛应用。

**门控循环单元（GRU）**：Cho等人在2014年提出的GRU是LSTM的简化版本，将遗忘门和输入门合并为更新门，减少了参数量，在某些任务上性能与LSTM相当。

**卷积神经网络（CNN）**：虽然CNN最初是为图像处理设计的，但一维CNN在时间序列分析中也表现出色。CNN能够通过卷积核自动提取局部模式特征。

**WaveNet**：DeepMind在2016年提出的WaveNet使用因果膨胀卷积（Causal Dilated Convolution）来处理序列数据，能够以对数复杂度捕捉长程依赖。WaveNet最初用于语音合成，但其架构思想对时间序列预测产生了深远影响。

### 3.4 Transformer时代

2017年Vaswani等人提出的Transformer架构彻底改变了序列建模的范式。Transformer通过自注意力机制（Self-Attention）实现了对序列中任意位置之间关系的直接建模，摆脱了RNN的顺序处理限制。

**自注意力机制**：Transformer的核心是缩放点积注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q$、$K$、$V$ 分别是查询、键、值矩阵，$d_k$ 是键向量的维度。

**位置编码**：由于Transformer本身不包含位置信息，需要通过位置编码来注入序列顺序信息。原始Transformer使用正弦/余弦函数生成固定位置编码。

**在时间序列中的应用**：

- **Informer**（2021）：通过ProbSparse注意力机制降低Transformer的计算复杂度，使其能够处理长序列时间序列。
- **Autoformer**（2021）：引入自相关机制替代传统的自注意力，在效率和性能之间取得平衡。
- **PatchTST**（2023）：将时间序列分割为多个Patch，每个Patch作为一个token输入Transformer，显著提高了效率。
- **iTransformer**（2024）：反转Transformer的处理维度，将变量维度作为token，时间维度作为序列长度。

### 3.5 基础模型时代

2023年以来，时间序列领域开始出现"基础模型"（Foundation Model）的概念，即在大规模数据上预训练的通用模型，可以通过微调或零样本学习适配各种下游任务。

**TimesFM**（Google, 2024）：TimesFM是一个基于Decoder-only Transformer的时间序列基础模型，在真实世界数据和合成数据上进行了大规模预训练。TimesFM使用Patch作为token，支持零样本预测。

**Chronos**（Amazon, 2024）：Chronos将时间序列值量化为离散token，然后使用语言模型的架构进行预训练。Chronos的独特之处在于它将时间序列预测完全转化为语言建模问题。

**MOMENT**（2024）：MOMENT是一个通用的时间序列基础模型，通过掩码自编码器（Masked Autoencoder）在大规模时间序列数据上进行预训练。

**Timer**（2024）：Timer是一个基于GPT架构的时间序列基础模型，使用下一token预测的方式进行预训练。

**TimeGPT**（Nixtla, 2024）：TimeGPT是最早公开的时间序列基础模型之一，基于Transformer架构，在超过1000亿个时间点上进行预训练。

### 3.6 Kronos的独特定位

在上述基础模型中，Kronos具有独特的定位：

**专注于金融K线数据**：与通用时间序列基础模型不同，Kronos专门针对金融K线数据设计。这使得它能够更好地处理金融数据的特殊性质，如OHLCV多维结构、高噪声特性、波动率聚集等。

**分层Tokenizer设计**：Kronos使用分层的二进制球面量化（BSQ）将连续的K线数据转换为离散token，这种设计既保留了数据的精细信息，又实现了高效的离散表示。

**Decoder-only自回归架构**：与使用编码器-解码器架构的模型不同，Kronos采用纯Decoder-only架构，与GPT系列语言模型的设计哲学一致。这种架构在生成式任务上表现出色。

**多维度联合建模**：Kronos不是单独预测每个价格维度，而是联合预测整个OHLCV向量，这保留了各维度之间的相关性信息。

**覆盖范围最广**：Kronos在45个全球交易所的120亿条K线上进行预训练，是目前覆盖范围最广的金融时间序列基础模型。

---

# 第二部分：架构篇

---

## 第四章 Kronos项目全景

### 4.1 项目概述

Kronos是一个Decoder-only的基础模型家族，专门为金融市场的"语言"——K线序列而设计。与通用时间序列基础模型（TSFMs）不同，Kronos针对金融数据独特的高噪声特性进行了优化。

Kronos采用了一个创新的两阶段框架：

1. **第一阶段**：一个专门的Tokenizer首先将连续的、多维的K线数据（OHLCV）量化为分层的离散token。
2. **第二阶段**：一个大型的自回归Transformer在这些token上进行预训练，使其能够作为各种量化任务的统一模型。

### 4.2 设计哲学

Kronos的设计哲学可以概括为以下几点：

**数据为本**：Kronos的核心竞争力来自于其庞大的训练数据集——覆盖45个全球交易所的120亿条K线记录。这种规模的数据使得模型能够学习到跨市场、跨资产的通用模式。

**K线即语言**：Kronos将K线序列视为一种特殊的"语言"，每根K线对应一个或多个"token"。这种类比使得我们可以借鉴自然语言处理领域的成功经验。

**两阶段解耦**：将连续数据的离散化（Tokenizer）与序列建模（Predictor）解耦，使得两个组件可以独立优化和复用。

**层次化表示**：分层Tokenizer将一根K线编码为两个层次的token——粗粒度的s1 token和细粒度的s2 token。这种设计既保留了关键信息，又控制了词汇表的大小。

### 4.3 模型家族

Kronos提供了多种规模的预训练模型，以适应不同的计算资源和应用场景：

| 模型名称 | Tokenizer | 上下文长度 | 参数量 | 开源状态 |
|----------|-----------|-----------|--------|----------|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M | ✅ |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M | ✅ |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M | ✅ |
| Kronos-large | Kronos-Tokenizer-base | 512 | 499.2M | ❌ |

**Kronos-mini**：最轻量的模型，仅有4.1M参数，上下文长度为2048。适合资源受限的环境或需要快速推理的场景。

**Kronos-small**：平衡性能和速度的选择，24.7M参数，上下文长度为512。适合大多数常规应用。

**Kronos-base**：基础模型，102.3M参数，提供更好的预测质量。适合对精度要求较高的场景。

**Kronos-large**：最大模型，499.2M参数，目前未开源。

### 4.4 模型配置分析

从源码中，我们可以看到Kronos模型的具体配置参数。以Kronos-small为例：

```python
# 源自 model/kronos.py 中的 Kronos 类
class Kronos(nn.Module, PyTorchModelHubMixin):
    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, 
                 ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p, 
                 token_dropout_p, learn_te):
```

Kronos-small的典型配置：
- `s1_bits=10`：s1 token的位数，词汇表大小为 $2^{10}=1024$
- `s2_bits=10`：s2 token的位数，词汇表大小为 $2^{10}=1024$
- `n_layers=12`：Transformer层数
- `d_model=832`：模型维度
- `n_heads=16`：注意力头数
- `ff_dim=2048`：前馈网络维度
- `ffn_dropout_p=0.2`：前馈网络dropout率
- `attn_dropout_p=0.0`：注意力dropout率
- `resid_dropout_p=0.2`：残差连接dropout率
- `token_dropout_p=0.0`：token dropout率
- `learn_te=True`：使用可学习的时间编码

Tokenizer的典型配置：
- `d_in=6`：输入维度（OHLCV 6维）
- `d_model=256`：模型维度
- `n_heads=4`：注意力头数
- `ff_dim=512`：前馈网络维度
- `n_enc_layers=4`：编码器层数
- `n_dec_layers=4`：解码器层数
- `s1_bits=10, s2_bits=10`：量化位数
- `beta=0.05`：commit loss权重
- `gamma0=1.0`：每样本熵惩罚权重
- `gamma=1.1`：codebook熵惩罚权重
- `zeta=0.05`：总熵惩罚权重
- `group_size=4`：量化分组大小

### 4.5 项目结构

Kronos的代码库结构清晰，各模块职责分明：

```
Kronos/
├── model/                      # 核心模型代码
│   ├── __init__.py            # 模型注册和导出
│   ├── kronos.py              # 主模型：Tokenizer, Predictor, Predictor类
│   └── module.py              # 基础模块：BSQ, Attention, FFN, Embedding等
├── finetune/                   # Qlib微调流程
│   ├── config.py              # 训练配置
│   ├── dataset.py             # 数据集类
│   ├── train_tokenizer.py     # Tokenizer训练脚本
│   ├── train_predictor.py     # Predictor训练脚本
│   ├── qlib_data_preprocess.py # Qlib数据预处理
│   ├── qlib_test.py           # 回测脚本
│   └── utils/
│       └── training_utils.py  # 训练工具函数
├── finetune_csv/               # 自定义CSV微调流程
│   ├── config_loader.py       # YAML配置加载器
│   ├── finetune_tokenizer.py  # Tokenizer微调
│   ├── finetune_base_model.py # Predictor微调
│   ├── train_sequential.py    # 顺序训练脚本
│   ├── configs/               # 配置文件模板
│   ├── data/                  # 示例数据
│   └── examples/              # 训练结果示例
├── examples/                   # 使用示例
│   ├── prediction_example.py  # 基本预测示例
│   ├── prediction_batch_example.py # 批量预测示例
│   ├── run_backtest_kronos.py # 回测示例
│   └── yuce/                  # 预测结果示例
├── webui/                      # Web用户界面
│   ├── app.py                 # Flask应用
│   ├── templates/             # HTML模板
│   └── prediction_results/    # 预测结果存储
├── tests/                      # 测试用例
│   └── test_kronos_regression.py # 回归测试
├── figures/                    # 文档图片
├── requirements.txt            # 依赖列表
└── README.md                   # 项目文档
```

### 4.6 开发团队与论文

Kronos由Yu Shi、Zongliang Fu、Shuo Chen、Bohan Zhao、Wei Xu、Changshui Zhang和Jian Li共同开发。论文"Kronos: A Foundation Model for the Language of Financial Markets"于2025年8月在arXiv发表（arXiv:2508.02739），并于2025年11月被AAAI 2026正式接收。

AAAI（Association for the Advancement of Artificial Intelligence）是人工智能领域最顶级的国际会议之一，Kronos被AAAI接收，证明了其在学术界的重要地位和创新价值。

### 4.7 Live Demo

Kronos团队提供了一个在线演示系统，展示模型对BTC/USDT交易对未来24小时的预测结果。演示系统使用Plotly.js进行交互式可视化，用户可以直观地看到模型预测的K线图与实际走势的对比。

---

## 第五章 核心架构：分层Tokenizer

### 5.1 Tokenizer的设计动机

在自然语言处理中，Tokenizer负责将连续的文本转换为离散的token序列。类似地，在Kronos中，Tokenizer负责将连续的K线数据（6维浮点数向量）转换为离散的token序列。

这一设计的核心挑战在于：如何在离散化的过程中既保留原始数据的关键信息，又控制词汇表的大小？

直接的量化方法（如标量量化）会丢失数据的连续性信息；而过于精细的量化（如使用很大的词汇表）则会增加后续自回归模型的建模难度。

Kronos的解决方案是**分层量化**：将每个K线数据点编码为两个层次的token——s1（粗粒度）和s2（细粒度）。s1捕捉数据的主要模式，s2捕捉细节差异。这种设计类似于将一个数字表示为高位和低位两个部分。

### 5.2 二进制球面量化（BSQ）

Kronos的Tokenizer使用了二进制球面量化（Binary Spherical Quantization, BSQ）技术，这一技术源自论文"Image and Video Tokenization with Binary Spherical Quantization"（arXiv:2406.07548）。

#### 5.2.1 基本原理

BSQ的核心思想是将高维连续向量映射到超球面上的离散点。具体来说，给定一个 $d$ 维的连续向量 $z$，BSQ首先将其L2归一化到单位球面上，然后对每个维度进行二值化（大于0为1，小于等于0为-1），得到一个 $d$ 维的二进制向量。

从源码中可以看到BSQ的实现：

```python
# 源自 model/module.py 中的 BinarySphericalQuantizer 类
def quantize(self, z):
    assert z.shape[-1] == self.embed_dim
    zhat = torch.where(z > 0,
                       torch.tensor(1, dtype=z.dtype, device=z.device),
                       torch.tensor(-1, dtype=z.dtype, device=z.device))
    return z + (zhat - z).detach()  # Straight-through estimator
```

这里使用了直通估计器（Straight-Through Estimator, STE）：前向传播时使用量化后的值，反向传播时将梯度直接传递给输入。这是因为二值化操作本身不可导。

#### 5.2.2 码本结构

在BSQ中，码本（Codebook）不是一个显式的查找表，而是由所有可能的二进制向量组成。对于 $d$ 维的二进制向量，码本的大小为 $2^d$。例如，当 `s1_bits=10, s2_bits=10` 时，总码本大小为 $2^{20} = 1,048,576$。

Kronos使用分层设计，将总维度分为s1_bits和s2_bits两部分。s1对应的码本大小为 $2^{10}=1024$，s2对应的码本大小也为 $2^{10}=1024$。这种分层设计有几个优势：

1. **降低建模复杂度**：自回归模型只需建模 $2^{10}$ 大小的词汇表，而不是 $2^{20}$
2. **层次化信息编码**：s1捕捉粗粒度信息，s2捕捉细粒度信息
3. **条件生成**：可以先生成s1，再根据s1条件生成s2

#### 5.2.3 指数与码字的转换

BSQ使用二进制编码来建立索引和码字之间的映射关系：

```python
# 源自 model/module.py
def codes_to_indexes(self, zhat):
    """将码字转换为索引"""
    return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

def indexes_to_codes(self, indices):
    """将索引转换为码字"""
    indices = indices.unsqueeze(-1)
    codes_non_centered = torch.remainder(
        torch.floor_divide(indices, self.basis), 2
    )
    return codes_non_centered * 2 - 1
```

其中 `self.basis` 是一个预计算的权重向量 `[2^(d-1), 2^(d-2), ..., 2^1, 2^0]`，用于将二进制向量转换为十进制索引。

#### 5.2.4 分组量化

为了进一步优化量化效果，BSQ将维度分成若干组（group_size），每组独立进行量化。这使得熵惩罚可以在组级别上计算，提高了训练的稳定性。

```python
# 源自 model/module.py
assert embed_dim % group_size == 0, "embed_dim must be divisible by group_size"
self.num_groups = embed_dim // group_size
self.group_size = group_size
```

默认配置中 `group_size=4`，因此 `embed_dim=20` 被分为5组，每组4个维度。

### 5.3 KronosTokenizer架构

KronosTokenizer是一个编码器-解码器架构的Transformer模型，负责将连续的K线数据编码为离散token，并能够从token解码回连续数据。

#### 5.3.1 编码器

编码器由多层Transformer Block组成，负责将输入的K线数据映射到量化空间：

```python
# 源自 model/kronos.py 中的 KronosTokenizer
self.embed = nn.Linear(self.d_in, self.d_model)  # 输入投影
self.encoder = nn.ModuleList([
    TransformerBlock(self.d_model, self.n_heads, self.ff_dim, ...)
    for _ in range(self.enc_layers - 1)
])
self.quant_embed = nn.Linear(self.d_model, self.codebook_dim)  # 量化前投影
```

编码过程：
1. 输入投影：`d_in` → `d_model`
2. Transformer编码器处理
3. 量化前投影：`d_model` → `codebook_dim`（即 `s1_bits + s2_bits`）

#### 5.3.2 量化层

量化层使用BSQ将连续的编码向量转换为离散的token：

```python
# 源自 model/kronos.py
bsq_loss, quantized, z_indices = self.tokenizer(z)
```

`self.tokenizer` 是一个 `BSQuantizer` 实例，它内部包含 `BinarySphericalQuantizer`。BSQuantizer在BSQ的基础上增加了L2归一化和分层索引计算：

```python
# 源自 model/module.py 中的 BSQuantizer
def forward(self, z, half=False, collect_metrics=True):
    z = F.normalize(z, dim=-1)  # L2归一化
    quantized, bsq_loss, metrics = self.bsq(z, collect_metrics=collect_metrics)
    if half:
        q_pre = quantized[:, :, :self.s1_bits]
        q_post = quantized[:, :, self.s1_bits:]
        z_indices = [self.bits_to_indices(q_pre), self.bits_to_indices(q_post)]
    else:
        z_indices = self.bits_to_indices(quantized)
    return bsq_loss, quantized, z_indices
```

当 `half=True` 时，量化结果被分为两部分：s1索引和s2索引。这是分层设计的关键。

#### 5.3.3 解码器

解码器负责从离散token重建连续的K线数据：

```python
# 源自 model/kronos.py
self.post_quant_embed_pre = nn.Linear(self.s1_bits, self.d_model)  # s1部分
self.post_quant_embed = nn.Linear(self.codebook_dim, self.d_model)  # 完整codebook
self.decoder = nn.ModuleList([
    TransformerBlock(self.d_model, self.n_heads, self.ff_dim, ...)
    for _ in range(self.dec_layers - 1)
])
self.head = nn.Linear(self.d_model, self.d_in)  # 输出投影
```

解码过程有两个路径：
1. **s1路径**：仅使用s1_bits进行解码，用于预训练时的辅助损失
2. **完整路径**：使用完整的codebook进行解码，用于最终的重建

```python
# 源自 model/kronos.py 中的 forward 方法
quantized_pre = quantized[:, :, :self.s1_bits]
z_pre = self.post_quant_embed_pre(quantized_pre)
z = self.post_quant_embed(quantized)

# 解码器处理
for layer in self.decoder:
    z_pre = layer(z_pre)
z_pre = self.head(z_pre)

for layer in self.decoder:
    z = layer(z)
z = self.head(z)
```

### 5.4 损失函数设计

KronosTokenizer的训练损失由三个部分组成：

#### 5.4.1 重建损失

重建损失衡量解码后的数据与原始数据之间的差异：

```python
# 源自 finetune/train_tokenizer.py
recon_loss_pre = F.mse_loss(z_pre, batch_x)  # s1路径的重建损失
recon_loss_all = F.mse_loss(z, batch_x)       # 完整路径的重建损失
recon_loss = recon_loss_pre + recon_loss_all
```

使用均方误差（MSE）作为重建损失的度量。s1路径的重建损失作为一个辅助损失，鼓励s1 token单独也能提供合理的重建质量。

#### 5.4.2 Commit损失

Commit损失惩罚量化后的向量与原始连续向量之间的差异，促使编码器的输出接近量化中心：

```python
# 源自 model/module.py 中的 BinarySphericalQuantizer
commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))
```

其中 `beta` 是commit损失的权重系数（默认0.05）。

#### 5.4.3 熵惩罚

熵惩罚用于提高码本的利用率，防止"码本坍塌"（codebook collapse）——即模型只使用码本中的一小部分。

BSQ使用了两种熵的计算方式：

**每样本熵（Per-sample Entropy）**：衡量每个样本在量化时的不确定性。高熵意味着样本的量化结果更均匀地分布在码本中。

**码本熵（Codebook Entropy）**：衡量整个码本的使用分布。高熵意味着码本被更均匀地使用。

```python
# 源自 model/module.py
if self.soft_entropy:
    persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
    entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
```

熵惩罚的公式为：

$$L_{entropy} = \gamma_0 \cdot H_{persample} - \gamma \cdot H_{codebook}$$

其中 $\gamma_0$（默认1.0）控制每样本熵的权重，$\gamma$（默认1.1）控制码本熵的权重。通过最大化码本熵（减去负号），模型被鼓励更均匀地使用码本。

#### 5.4.4 总损失

Tokenizer的总损失为：

$$L_{total} = \frac{1}{2}(L_{recon\_pre} + L_{recon\_all}) + L_{commit} + \zeta \cdot L_{entropy}$$

其中 $\zeta$（默认0.05）是熵惩罚的总权重。

### 5.5 软熵计算

BSQ的一个创新之处是使用了软熵（Soft Entropy）计算方法。传统的硬熵计算直接使用量化后的离散值，而软熵使用连续值的软分配概率：

```python
# 源自 model/module.py 中的 soft_entropy_loss
def soft_entropy_loss(self, z):
    group_code_book = self.group_codebook / (self.embed_dim ** 0.5 if self.l2_norm else 1)
    divided_z = rearrange(z, '... (g c) -> ... g c', c=self.group_size)
    
    # 计算子组与码本之间的距离
    distance = - 2 * torch.einsum('... g c, d c ->... g d', divided_z, group_code_book)
    prob = (-distance * self.inv_temperature).softmax(dim=-1)
```

软熵的优势在于：
1. **梯度更平滑**：避免了离散化带来的梯度不连续问题
2. **训练更稳定**：软分配使得损失函数的曲面更加平滑
3. **码本利用率更高**：软分配允许每个样本对多个码本条目产生梯度

### 5.6 分层Token的优势

Kronos的分层Token设计（s1和s2）相比传统的单一token设计有几个重要优势：

**信息分层**：s1 token编码K线的主要趋势信息（如涨跌方向、大致幅度），s2 token编码细节信息（如精确的价格位置）。这种分层与人类分析K线的方式一致——先判断大方向，再关注细节。

**词汇表效率**：如果使用单一token，要达到相同的量化精度，需要 $2^{20}$ 大小的词汇表。而分层设计将词汇表分解为两个 $2^{10}$ 大小的子词汇表，大大降低了建模的复杂度。

**条件生成**：在自回归推理时，模型可以先生成s1 token（确定大方向），再基于s1条件生成s2 token（确定细节）。这种条件生成使得预测更加稳定。

**多粒度预测**：在某些应用场景下，可能只需要粗粒度的预测（如趋势判断），此时可以只使用s1 token，提高推理效率。

---

## 第六章 核心架构：Decoder-only预测模型

### 6.1 架构选择

Kronos的预测模型采用了纯Decoder-only的Transformer架构，这一选择与GPT系列语言模型的设计哲学一致。相比编码器-解码器架构，Decoder-only架构有以下优势：

**简洁统一**：所有输入和输出共享同一个序列空间，不需要区分编码器和解码器。

**自回归生成**：天然支持自回归生成，每次预测一个token，逐步构建完整的预测序列。

**可扩展性**：Decoder-only架构在大规模模型上表现出色，更容易扩展到更大的参数规模。

**因果注意力**：通过因果掩码（Causal Mask），确保模型在预测每个位置时只能看到之前的信息，符合时间序列预测的因果性要求。

### 6.2 分层嵌入（HierarchicalEmbedding）

Kronos使用分层嵌入将s1和s2 token映射到连续的嵌入空间：

```python
# 源自 model/module.py 中的 HierarchicalEmbedding
class HierarchicalEmbedding(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model=256):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        
        vocab_s1 = 2 ** s1_bits
        vocab_s2 = 2 ** s2_bits
        
        self.emb_s1 = nn.Embedding(vocab_s1, d_model)
        self.emb_s2 = nn.Embedding(vocab_s2, d_model)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
```

嵌入计算过程：

```python
def forward(self, token_ids):
    if isinstance(token_ids, tuple) or isinstance(token_ids, list):
        s1_ids, s2_ids = token_ids
    else:
        s1_ids, s2_ids = self.split_token(token_ids, self.s2_bits)
    
    s1_emb = self.emb_s1(s1_ids) * math.sqrt(self.d_model)
    s2_emb = self.emb_s2(s2_ids) * math.sqrt(self.d_model)
    return self.fusion_proj(torch.cat([s1_emb, s2_emb], dim=-1))
```

关键设计点：

1. **独立嵌入表**：s1和s2各自有独立的嵌入表，词汇表大小分别为 $2^{s1\_bits}$ 和 $2^{s2\_bits}$
2. **缩放因子**：嵌入乘以 $\sqrt{d\_model}$ 进行缩放，这是Transformer的标准做法
3. **融合投影**：将两个嵌入拼接后通过线性层投影到 $d\_model$ 维度

### 6.3 旋转位置编码（RoPE）

Kronos使用旋转位置编码（Rotary Position Embedding, RoPE）来注入序列位置信息。RoPE最初由Su等人在2021年提出，相比传统的正弦位置编码和可学习位置编码，RoPE具有更好的外推能力。

```python
# 源自 model/module.py 中的 RotaryPositionalEmbedding
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, q, k):
        cos, sin = self._update_cos_sin_cache(q, q.shape[-2])
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )
    
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
```

RoPE的数学原理：对于位置 $m$ 处的查询向量 $q_m$ 和位置 $n$ 处的键向量 $k_n$，RoPE使得它们的内积只依赖于相对位置 $m-n$：

$$\langle f(q_m), f(k_n) \rangle = g(q_m, k_n, m-n)$$

这通过将向量的每对相邻维度进行旋转来实现。具体来说，对于第 $i$ 对维度 $(q_{2i}, q_{2i+1})$：

$$f(q_{2i}, q_{2i+1}, m) = (q_{2i}\cos(m\theta_i) - q_{2i+1}\sin(m\theta_i), q_{2i}\sin(m\theta_i) + q_{2i+1}\cos(m\theta_i))$$

其中 $\theta_i = 10000^{-2i/d}$ 是频率参数。

RoPE的优势：
1. **相对位置编码**：自然编码相对位置信息，无需显式存储绝对位置
2. **良好的外推能力**：在训练时未见过的序列长度上也能正常工作
3. **计算高效**：只需要逐元素的乘法和加法

### 6.4 RMSNorm归一化

Kronos使用RMSNorm（Root Mean Square Normalization）替代传统的LayerNorm：

```python
# 源自 model/module.py
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

RMSNorm与LayerNorm的区别：

**LayerNorm**：先减去均值，再除以标准差
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

**RMSNorm**：只除以均方根，不减去均值
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

其中 $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}$

RMSNorm的优势：
1. **计算更快**：不需要计算均值
2. **训练更稳定**：在某些任务上表现出更好的训练稳定性
3. **参数更少**：没有偏置项 $\beta$

### 6.5 SwiGLU前馈网络

Kronos使用SwiGLU（Swish-Gated Linear Unit）作为前馈网络的激活函数：

```python
# 源自 model/module.py
class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)
    
    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

SwiGLU的计算过程：

$$\text{SwiGLU}(x) = W_2 \cdot (\text{Swish}(W_1 x) \odot W_3 x)$$

其中 $\odot$ 表示逐元素乘法，$\text{Swish}(x) = x \cdot \sigma(x)$ 是Swish激活函数（也称为SiLU）。

SwiGLU的特点：
1. **门控机制**：$W_3 x$ 充当门控信号，控制信息的流动
2. **平滑激活**：Swish函数是平滑的、非单调的，相比ReLU有更好的梯度特性
3. **无偏置**：三个线性层都没有偏置项，减少参数量

### 6.6 因果自注意力

Kronos的自注意力机制使用因果掩码确保模型只能看到当前位置及之前的信息：

```python
# 源自 model/module.py 中的 MultiHeadAttentionWithRoPE
def forward(self, x, key_padding_mask=None):
    q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    
    q, k = self.rotary(q, k)  # 应用RoPE
    
    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=self.attn_dropout_p if self.training else 0.0,
        is_causal=True  # 因果掩码
    )
```

使用PyTorch的 `scaled_dot_product_attention` 函数，并设置 `is_causal=True` 来启用因果掩码。这意味着在计算注意力权重时，每个位置只能关注它自己和之前的位置。

### 6.7 Transformer Block

每个Transformer Block包含自注意力层和前馈层，使用残差连接和Pre-Norm设计：

```python
# 源自 model/module.py
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, ...):
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, ...)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ...)
    
    def forward(self, x, key_padding_mask=None):
        # 自注意力子层
        residual = x
        x = self.norm1(x)  # Pre-Norm
        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + attn_out  # 残差连接
        
        # 前馈子层
        residual = x
        x = self.norm2(x)  # Pre-Norm
        ffn_out = self.ffn(x)
        x = residual + ffn_out  # 残差连接
        return x
```

Pre-Norm设计（先归一化再计算）相比Post-Norm（先计算再归一化）有更好的训练稳定性，这在深层Transformer模型中尤为重要。

### 6.8 双头输出（DualHead）

Kronos使用双头输出结构，分别为s1和s2 token生成预测：

```python
# 源自 model/module.py
class DualHead(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model):
        self.vocab_s1 = 2 ** s1_bits
        self.vocab_s2 = 2 ** s2_bits
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)
    
    def forward(self, x):
        return self.proj_s1(x)  # s1 logits
    
    def cond_forward(self, x2):
        return self.proj_s2(x2)  # s2 logits（条件于s1）
```

s1和s2的预测是分步进行的：
1. 首先通过 `proj_s1` 生成s1的logits
2. 然后通过DependencyAwareLayer处理，再通过 `proj_s2` 生成s2的logits

损失计算：

```python
def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
    ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
    ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
    ce_loss = (ce_s1 + ce_s2) / 2
    return ce_loss, ce_s1, ce_s2
```

总损失是s1交叉熵和s2交叉熵的平均值。

### 6.9 依赖感知层（DependencyAwareLayer）

DependencyAwareLayer是Kronos的一个关键创新，它通过交叉注意力机制让s2的预测依赖于s1的预测结果：

```python
# 源自 model/module.py
class DependencyAwareLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, ...):
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads, ...)
        self.norm = RMSNorm(d_model)
    
    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        attn_out = self.cross_attn(
            query=sibling_embed,      # s1的嵌入作为query
            key=hidden_states,        # Transformer的输出作为key
            value=hidden_states,      # Transformer的输出作为value
            key_padding_mask=key_padding_mask
        )
        return self.norm(hidden_states + attn_out)
```

这个层的工作原理：
1. **输入**：Transformer的隐藏状态 `hidden_states` 和s1 token的嵌入 `sibling_embed`
2. **交叉注意力**：以s1嵌入为query，以Transformer输出为key和value
3. **残差连接**：将交叉注意力的输出加到原始隐藏状态上
4. **归一化**：通过RMSNorm进行归一化

在Kronos的前向传播中：

```python
# 源自 model/kronos.py 中的 Kronos.forward
s1_logits = self.head(x)  # 首先生成s1 logits

if use_teacher_forcing:
    sibling_embed = self.embedding.emb_s1(s1_targets)  # 使用真实标签
else:
    s1_probs = F.softmax(s1_logits.detach(), dim=-1)
    sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
    sibling_embed = self.embedding.emb_s1(sample_s1_ids)  # 使用采样结果

x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)  # 依赖感知处理
s2_logits = self.head.cond_forward(x2)  # 生成s2 logits
```

这种设计确保了s2的预测是在s1的条件下进行的，实现了真正的层次化生成。

### 6.10 时间嵌入（TemporalEmbedding）

Kronos使用专门的时间嵌入来编码K线数据的时间信息：

```python
# 源自 model/module.py
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, learn_pe):
        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        
        Embed = FixedEmbedding if not learn_pe else nn.Embedding
        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 0])
        hour_x = self.hour_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 3])
        month_x = self.month_embed(x[:, :, 4])
        return hour_x + weekday_x + day_x + month_x + minute_x
```

时间嵌入将5个时间特征（分钟、小时、星期几、日期、月份）分别编码后相加。当 `learn_pe=True` 时，使用可学习的嵌入表；否则使用固定的正弦/余弦嵌入。

这些时间嵌入被加到K线的token嵌入上，使模型能够感知时间的周期性模式（如开盘/收盘时间、周末效应、月末效应等）。

### 6.11 完整前向传播流程

Kronos的完整前向传播流程如下：

```python
# 源自 model/kronos.py 中的 Kronos.forward
def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, 
            use_teacher_forcing=False, s1_targets=None):
    # 1. 分层嵌入
    x = self.embedding([s1_ids, s2_ids])
    
    # 2. 添加时间嵌入
    if stamp is not None:
        time_embedding = self.time_emb(stamp)
        x = x + time_embedding
    
    # 3. Token Dropout
    x = self.token_drop(x)
    
    # 4. Transformer层处理
    for layer in self.transformer:
        x = layer(x, key_padding_mask=padding_mask)
    
    # 5. RMSNorm归一化
    x = self.norm(x)
    
    # 6. s1预测
    s1_logits = self.head(x)
    
    # 7. 依赖感知处理（用于s2预测）
    if use_teacher_forcing:
        sibling_embed = self.embedding.emb_s1(s1_targets)
    else:
        s1_probs = F.softmax(s1_logits.detach(), dim=-1)
        sample_s1_ids = torch.multinomial(...).view(s1_ids.shape)
        sibling_embed = self.embedding.emb_s1(sample_s1_ids)
    
    # 8. s2预测
    x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
    s2_logits = self.head.cond_forward(x2)
    
    return s1_logits, s2_logits
```

这个流程清晰地展示了数据从输入到输出的完整路径。

---

## 第七章 自回归推理机制

### 7.1 自回归生成原理

自回归（Autoregressive）生成是Kronos进行预测的核心机制。与一次性输出所有预测值的方法不同，自回归生成逐步地、一个token一个token地构建预测序列。

在每个时间步，模型根据已有的上下文（历史K线 + 已生成的预测）预测下一个K线的token，然后将新生成的token添加到上下文中，继续预测下一个。这个过程一直持续到生成指定数量的预测。

这种机制的优势在于：
1. **灵活性**：可以生成任意长度的预测序列
2. **质量**：每一步的预测都基于完整的上下文
3. **多样性**：通过采样参数可以生成多种可能的预测路径

### 7.2 滑动窗口缓冲区

由于Kronos模型有最大上下文长度限制（如512或2048），当序列长度超过这个限制时，需要使用滑动窗口来管理上下文：

```python
# 源自 model/kronos.py 中的 auto_regressive_inference
pre_buffer = x_token[0].new_zeros(batch_size, max_context)
post_buffer = x_token[1].new_zeros(batch_size, max_context)
buffer_len = min(initial_seq_len, max_context)

if buffer_len > 0:
    start_idx = max(0, initial_seq_len - max_context)
    pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
    post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]
```

在每个生成步骤中，缓冲区的更新逻辑为：

```python
if current_seq_len < max_context:
    # 序列长度未超过限制，直接追加
    pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
    post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
else:
    # 序列长度超过限制，使用滑动窗口
    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
    pre_buffer[:, -1] = sample_pre.squeeze(-1)
    post_buffer[:, -1] = sample_post.squeeze(-1)
```

滑动窗口通过 `torch.roll` 操作将缓冲区向左移动一个位置，然后在末尾添加新的token。这样，缓冲区始终包含最近的 `max_context` 个token。

### 7.3 Top-k和Top-p采样

Kronos使用Top-k和Top-p（Nucleus Sampling）两种采样策略来控制生成的多样性：

```python
# 源自 model/kronos.py 中的 top_k_top_p_filtering
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits
    
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits
```

**Top-k采样**：只保留概率最高的k个token，其余token的概率设为负无穷。当 `top_k=1` 时，等价于贪心搜索（总是选择概率最高的token）。

**Top-p采样（Nucleus Sampling）**：按概率从高到低排序，保留累积概率达到p的最小token集合。这种方法比Top-k更灵活，因为它会根据概率分布的形状动态调整候选token的数量。

### 7.4 温度控制

温度参数T控制采样的随机性：

```python
# 源自 model/kronos.py 中的 sample_from_logits
def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    probs = F.softmax(logits, dim=-1)
    
    if not sample_logits:
        _, x = torch.topk(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)
    return x
```

温度的影响：
- **T < 1.0**：分布变得更尖锐，模型更倾向于选择高概率的token，输出更确定性
- **T = 1.0**：保持原始分布不变
- **T > 1.0**：分布变得更平坦，模型更倾向于随机选择，输出更多样化

### 7.5 多样本平均

Kronos支持生成多条预测路径并取平均，以提高预测的稳定性和准确性：

```python
# 源自 model/kronos.py 中的 auto_regressive_inference
x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2))
x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(...)
y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(...)
```

当 `sample_count > 1` 时，输入数据被复制 `sample_count` 次，形成一个更大的batch。每条路径独立进行自回归生成，最后将所有路径的结果取平均：

```python
z = tokenizer.decode(input_tokens, half=True)
z = z.reshape(-1, sample_count, z.size(1), z.size(2))
preds = z.cpu().numpy()
preds = np.mean(preds, axis=1)  # 对所有路径取平均
```

多样本平均的原理：由于采样的随机性，每条路径可能不同。取平均可以平滑掉随机波动，得到更稳定的预测。这类似于集成学习的思想。

### 7.6 分层Token的解码

在自回归生成的每一步中，模型首先生成s1 token，然后基于s1生成s2 token：

```python
# 源自 model/kronos.py 中的 auto_regressive_inference 循环体
# 步骤1：解码s1
s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
s1_logits = s1_logits[:, -1, :]  # 只取最后一个位置的logits
sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p)

# 步骤2：基于s1解码s2
s2_logits = model.decode_s2(context, sample_pre)
s2_logits = s2_logits[:, -1, :]
sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p)
```

`decode_s1` 方法执行完整的Transformer前向传播，返回s1的logits和上下文表示：

```python
# 源自 model/kronos.py
def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
    x = self.embedding([s1_ids, s2_ids])
    if stamp is not None:
        time_embedding = self.time_emb(stamp)
        x = x + time_embedding
    x = self.token_drop(x)
    
    for layer in self.transformer:
        x = layer(x, key_padding_mask=padding_mask)
    
    x = self.norm(x)
    s1_logits = self.head(x)
    return s1_logits, x
```

`decode_s2` 方法使用依赖感知层，基于s1的嵌入和上下文生成s2的logits：

```python
def decode_s2(self, context, s1_ids, padding_mask=None):
    sibling_embed = self.embedding.emb_s1(s1_ids)
    x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
    return self.head.cond_forward(x2)
```

### 7.7 完整推理流程

KronosPredictor类封装了完整的推理流程：

```python
# 源自 model/kronos.py 中的 KronosPredictor.predict
def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, 
            sample_count=1, verbose=True):
    # 1. 数据验证和预处理
    df = df.copy()
    if self.vol_col not in df.columns:
        df[self.vol_col] = 0.0
        df[self.amt_vol] = 0.0
    
    # 2. 时间特征提取
    x_time_df = calc_time_stamps(x_timestamp)
    y_time_df = calc_time_stamps(y_timestamp)
    
    # 3. 数值提取和归一化
    x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    x = (x - x_mean) / (x_std + 1e-5)
    x = np.clip(x, -self.clip, self.clip)
    
    # 4. 添加batch维度
    x = x[np.newaxis, :]
    x_stamp = x_stamp[np.newaxis, :]
    y_stamp = y_stamp[np.newaxis, :]
    
    # 5. 自回归生成
    preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)
    
    # 6. 反归一化
    preds = preds.squeeze(0)
    preds = preds * (x_std + 1e-5) + x_mean
    
    # 7. 构建结果DataFrame
    pred_df = pd.DataFrame(preds, columns=self.price_cols + [self.vol_col, self.amt_vol], 
                           index=y_timestamp)
    return pred_df
```

这个流程展示了从原始数据到最终预测的完整路径。

### 7.8 批量预测

Kronos支持同时预测多个时间序列，通过批量处理提高GPU利用效率：

```python
# 源自 model/kronos.py 中的 KronosPredictor.predict_batch
def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, ...):
    # 验证所有序列的长度一致性
    if len(set(seq_lens)) != 1:
        raise ValueError("Parallel prediction requires all series to have consistent historical lengths")
    
    # 堆叠为batch
    x_batch = np.stack(x_list, axis=0).astype(np.float32)
    x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
    y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)
    
    # 批量生成
    preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, ...)
    
    # 分别反归一化
    pred_dfs = []
    for i in range(num_series):
        preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
        pred_df = pd.DataFrame(preds_i, columns=..., index=y_timestamp_list[i])
        pred_dfs.append(pred_df)
    
    return pred_dfs
```

批量预测的关键要求：
1. 所有序列必须有相同的回看窗口长度
2. 所有序列必须有相同的预测长度
3. 每个DataFrame必须包含必需的价格列

---

# 第三部分：工程篇

---

## 第八章 数据预处理与归一化

### 8.1 数据输入格式

Kronos接受pandas DataFrame作为输入，必须包含以下列：

```python
# 源自 model/kronos.py 中的 KronosPredictor
self.price_cols = ['open', 'high', 'low', 'close']
self.vol_col = 'volume'
self.amt_vol = 'amount'
```

必需列：`open`, `high`, `low`, `close`
可选列：`volume`, `amount`

当 `volume` 和 `amount` 缺失时，系统自动填充为0：

```python
if self.vol_col not in df.columns:
    df[self.vol_col] = 0.0
    df[self.amt_vol] = 0.0
```

### 8.2 实例级归一化

Kronos使用实例级归一化（Instance-level Normalization），即对每个样本独立进行归一化。归一化的统计量（均值和标准差）仅从回看窗口的数据计算，严格避免未来数据泄露：

```python
# 源自 model/kronos.py 中的 KronosPredictor.predict
x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
x = (x - x_mean) / (x_std + 1e-5)
x = np.clip(x, -self.clip, self.clip)
```

关键设计点：

1. **按特征维度计算**：均值和标准差是按特征维度（6维）分别计算的，保留了各特征的相对尺度信息

2. **防止除零**：`x_std + 1e-5` 中的小常数防止标准差为零时出现除零错误

3. **裁剪处理**：`np.clip(x, -self.clip, self.clip)` 将归一化后的值裁剪到 `[-clip, clip]` 范围内（默认clip=5），防止极端值的影响

4. **反归一化**：预测完成后，使用相同的统计量进行反归一化：
   ```python
   preds = preds * (x_std + 1e-5) + x_mean
   ```

### 8.3 时间特征提取

Kronos从时间戳中提取5个时间特征：

```python
# 源自 model/kronos.py
def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()
    time_df['minute'] = x_timestamp.dt.minute    # 分钟 (0-59)
    time_df['hour'] = x_timestamp.dt.hour        # 小时 (0-23)
    time_df['weekday'] = x_timestamp.dt.weekday  # 星期几 (0-6)
    time_df['day'] = x_timestamp.dt.day          # 日期 (1-31)
    time_df['month'] = x_timestamp.dt.month      # 月份 (1-12)
    return time_df
```

这些特征被转换为浮点数后输入模型：

```python
x_stamp = x_time_df.values.astype(np.float32)
y_stamp = y_time_df.values.astype(np.float32)
```

在模型内部，这些浮点数被转换为整数后用于查找嵌入表（参见TemporalEmbedding的实现）。

### 8.4 滑动窗口构建

在训练数据的构建中，Kronos使用滑动窗口方法从长时间序列中提取固定长度的样本：

```python
# 源自 finetune/dataset.py 中的 QlibDataset
self.window = self.config.lookback_window + self.config.predict_window + 1

for symbol in self.symbols:
    df = self.data[symbol].reset_index()
    series_len = len(df)
    num_samples = series_len - self.window + 1
    
    if num_samples > 0:
        for i in range(num_samples):
            self.indices.append((symbol, i))
```

窗口大小 = 回看窗口 + 预测窗口 + 1。其中+1是因为自回归训练需要输入和输出之间错开一个位置。

### 8.5 数据泄露防护

防止未来数据泄露是金融时间序列建模中的关键问题。Kronos在多个层面确保数据隔离：

**归一化隔离**：归一化只使用回看窗口内的数据，不包含未来的数据：

```python
# 源自 finetune/dataset.py
past_len = self.config.lookback_window
past_x = x[:past_len]
x_mean = np.mean(past_x, axis=0)
x_std = np.std(past_x, axis=0)
```

**数据集划分**：训练集、验证集、测试集按时间顺序严格划分，不允许交叉：

```python
# 源自 finetune/config.py
self.train_time_range = ["2011-01-01", "2022-12-31"]
self.val_time_range = ["2022-09-01", "2024-06-30"]
self.test_time_range = ["2024-04-01", "2025-06-05"]
```

注意验证集的起始时间早于训练集的结束时间，这是因为回看窗口需要历史数据作为上下文。

**随机采样控制**：训练数据的随机采样使用确定性的种子，确保可复现性：

```python
def set_epoch_seed(self, epoch: int):
    epoch_seed = self.config.seed + epoch
    self.py_rng.seed(epoch_seed)
```

---

## 第九章 训练策略与分布式训练

### 9.1 两阶段训练

Kronos采用两阶段训练策略：

**第一阶段：Tokenizer训练**。在这一阶段，KronosTokenizer在K线数据上训练，学习将连续的OHLCV数据编码为离散token。训练目标是最小化重建损失和量化损失。

**第二阶段：Predictor训练**。在这一阶段，Kronos预测模型在Tokenizer编码的token序列上训练，学习自回归地预测下一个token。Tokenizer的参数被冻结，只训练Predictor的参数。

这种两阶段设计的优势：
1. **解耦关注点**：Tokenizer专注于数据的表示学习，Predictor专注于序列建模
2. **灵活性**：训练好的Tokenizer可以被不同的Predictor复用
3. **训练稳定性**：分开训练避免了两个组件之间的不稳定性

### 9.2 优化器配置

Kronos使用AdamW优化器：

```python
# 源自 finetune/train_tokenizer.py
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['tokenizer_learning_rate'],
    weight_decay=config['adam_weight_decay']
)
```

AdamW是Adam优化器的改进版本，将权重衰减从梯度更新中解耦出来，提供了更好的正则化效果。

默认超参数：
- `adam_beta1=0.9`：一阶矩估计的衰减率
- `adam_beta2=0.95`：二阶矩估计的衰减率
- `adam_weight_decay=0.1`：权重衰减系数
- Tokenizer学习率：`2e-4`
- Predictor学习率：`4e-5`

### 9.3 学习率调度

Kronos使用OneCycleLR学习率调度器：

```python
# 源自 finetune/train_tokenizer.py
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=config['tokenizer_learning_rate'],
    steps_per_epoch=len(train_loader),
    epochs=config['epochs'],
    pct_start=0.03,
    div_factor=10
)
```

OneCycleLR策略的特点：
1. **预热阶段**：学习率从 `max_lr/div_factor` 逐渐增加到 `max_lr`
2. **退火阶段**：学习率从 `max_lr` 逐渐降低
3. **pct_start=0.03**：前3%的步骤用于预热
4. **div_factor=10**：初始学习率为最大学习率的1/10

### 9.4 分布式数据并行（DDP）

Kronos支持多GPU分布式训练，使用PyTorch的DistributedDataParallel（DDP）：

```python
# 源自 finetune/utils/training_utils.py
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank
```

DDP训练的启动方式：

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
```

DDP的关键配置：
- **后端**：使用NCCL（NVIDIA Collective Communications Library）作为通信后端
- **采样器**：使用DistributedSampler确保每个GPU处理不同的数据子集
- **梯度同步**：DDP自动在反向传播时同步梯度

```python
# 源自 finetune/train_tokenizer.py
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
```

`find_unused_parameters=False` 表示假设所有参数都参与了梯度计算，这可以提高通信效率。

### 9.5 梯度累积

当GPU内存不足以容纳期望的有效批次大小时，Kronos支持梯度累积：

```python
# 源自 finetune/train_tokenizer.py
for j in range(config['accumulation_steps']):
    start_idx = j * (ori_batch_x.shape[0] // config['accumulation_steps'])
    end_idx = (j + 1) * (ori_batch_x.shape[0] // config['accumulation_steps'])
    batch_x = ori_batch_x[start_idx:end_idx]
    
    zs, bsq_loss, _, _ = model(batch_x)
    z_pre, z = zs
    
    recon_loss_pre = F.mse_loss(z_pre, batch_x)
    recon_loss_all = F.mse_loss(z, batch_x)
    recon_loss = recon_loss_pre + recon_loss_all
    loss = (recon_loss + bsq_loss) / 2
    
    loss_scaled = loss / config['accumulation_steps']
    loss_scaled.backward()
```

梯度累积的工作原理：
1. 将一个大批次分成多个小批次
2. 每个小批次计算损失并反向传播，但不更新参数
3. 将损失除以累积步数，确保梯度的期望值与大批次一致
4. 累积完所有小批次后，统一更新参数

### 9.6 梯度裁剪

为了防止梯度爆炸，Kronos使用梯度裁剪：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
```

`clip_grad_norm_` 计算所有参数梯度的L2范数，如果超过 `max_norm`，则按比例缩小所有梯度。

Tokenizer训练使用 `max_norm=2.0`，Predictor训练使用 `max_norm=3.0`。

### 9.7 种子管理

为了确保实验的可复现性，Kronos在多个层面设置随机种子：

```python
# 源自 finetune/utils/training_utils.py
def set_seed(seed: int, rank: int = 0):
    actual_seed = seed + rank  # 每个进程使用不同的种子
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

注意每个rank使用不同的种子（`seed + rank`），以确保不同进程的数据加载器产生不同的随机序列。

### 9.8 验证和检查点保存

在每个epoch结束后，Kronos进行验证并根据验证损失保存最佳模型：

```python
# 源自 finetune/train_tokenizer.py
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    save_path = f"{save_dir}/checkpoints/best_model"
    model.module.save_pretrained(save_path)
```

验证损失通过跨进程规约（All-Reduce）计算全局平均值：

```python
val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
val_count_tensor = torch.tensor(val_sample_count_rank, device=device)
dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)
avg_val_loss = val_loss_sum_tensor.item() / val_count_tensor.item()
```

---

# 第四部分：实践篇

---

## 第十章 微调实践：Qlib与A股市场

### 10.1 Qlib简介

Qlib是微软开源的量化投资平台，提供了完整的数据管理、特征工程、模型训练和回测框架。Kronos的官方微调流程使用Qlib来准备中国A股市场的数据。

### 10.2 数据预处理

数据预处理脚本 `finetune/qlib_data_preprocess.py` 负责从Qlib加载原始数据并进行处理：

```python
# 源自 finetune/qlib_data_preprocess.py
class QlibDataPreprocessor:
    def __init__(self):
        self.config = Config()
        self.data_fields = ['open', 'close', 'high', 'low', 'volume', 'vwap']
    
    def load_qlib_data(self):
        data_fields_qlib = ['$' + f for f in self.data_fields]
        data_df = QlibDataLoader(config=data_fields_qlib).load(
            self.config.instrument, real_start_time, real_end_time
        )
```

数据处理流程：
1. 从Qlib加载原始OHLCV数据
2. 计算成交额：`amt = (open + high + low + close) / 4 * vol`
3. 按时间范围划分训练集、验证集、测试集
4. 保存为pickle文件

### 10.3 数据集类

QlibDataset类实现了PyTorch的Dataset接口，支持滑动窗口采样：

```python
# 源自 finetune/dataset.py
class QlibDataset(Dataset):
    def __init__(self, data_type: str = 'train'):
        self.data_path = f"{self.config.dataset_path}/{data_type}_data.pkl"
        self.window = self.config.lookback_window + self.config.predict_window + 1
        
        # 预计算所有可能的起始索引
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            num_samples = len(df) - self.window + 1
            for i in range(num_samples):
                self.indices.append((symbol, i))
    
    def __getitem__(self, idx):
        # 随机选择一个样本
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]
        
        # 提取滑动窗口
        win_df = df.iloc[start_idx:end_idx]
        x = win_df[self.feature_list].values.astype(np.float32)
        
        # 归一化（只使用回看窗口的数据）
        past_x = x[:past_len]
        x_mean = np.mean(past_x, axis=0)
        x_std = np.std(past_x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)
        
        return x_tensor, x_stamp_tensor
```

### 10.4 回测系统

Kronos提供了基于Qlib的回测系统：

```python
# 源自 finetune/qlib_test.py
class QlibBacktest:
    def run_single_backtest(self, signal_series):
        strategy = TopkDropoutStrategy(
            topk=self.config.backtest_n_symbol_hold,
            n_drop=self.config.backtest_n_symbol_drop,
            hold_thresh=self.config.backtest_hold_thresh,
            signal=signal_series,
        )
        # 执行回测
        portfolio_metric_dict, _ = backtest(strategy=strategy, **backtest_config)
        
        # 分析结果
        analysis = {
            "excess_return_without_cost": risk_analysis(report["return"] - report["bench"]),
            "excess_return_with_cost": risk_analysis(report["return"] - report["bench"] - report["cost"]),
        }
```

回测使用TopkDropoutStrategy：
- 持有预测信号最强的topk只股票
- 定期调仓，每次替换n_drop只股票
- 最短持有期为hold_thresh个交易日

### 10.5 信号生成

从模型预测到交易信号的转换：

```python
# 源自 finetune/qlib_test.py
# 获取预测
preds = auto_regressive_inference(tokenizer, model, ...)

# 生成多种信号
last_day_close = x[:, -1, 3].numpy()
signals = {
    'last': preds[:, -1, 3] - last_day_close,      # 最后一天预测收盘价的变化
    'mean': np.mean(preds[:, :, 3], axis=1) - last_day_close,  # 平均预测收盘价的变化
    'max': np.max(preds[:, :, 3], axis=1) - last_day_close,    # 最高预测收盘价的变化
    'min': np.min(preds[:, :, 3], axis=1) - last_day_close,    # 最低预测收盘价的变化
}
```

---

## 第十一章 微调实践：自定义CSV数据

### 11.1 CSV数据格式

Kronos支持在自定义的CSV数据上进行微调。CSV文件必须包含以下列：

```
timestamps, open, high, low, close, volume, amount
```

其中 `volume` 和 `amount` 可以全为0（如果没有这部分数据）。

### 11.2 配置系统

Kronos使用YAML配置文件管理微调参数：

```yaml
# 源自 finetune_csv/configs/config_ali09988_candle-5min.yaml
data:
  data_path: "/path/to/your/data.csv"
  lookback_window: 512
  predict_window: 48
  max_context: 512
  clip: 5.0
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0

training:
  tokenizer_epochs: 30
  basemodel_epochs: 20
  batch_size: 32
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001

model_paths:
  pretrained_tokenizer: "/path/to/Kronos-Tokenizer-base"
  pretrained_predictor: "/path/to/Kronos-base"
  exp_name: "my_experiment"
```

配置加载器 `CustomFinetuneConfig` 负责解析YAML文件并提供友好的访问接口：

```python
# 源自 finetune_csv/config_loader.py
class CustomFinetuneConfig:
    def __init__(self, config_path):
        self.loader = ConfigLoader(config_path)
        self._load_all_configs()
    
    def _load_all_configs(self):
        data_config = self.loader.get_data_config()
        self.data_path = data_config.get('data_path')
        self.lookback_window = data_config.get('lookback_window', 512)
        # ... 其他配置
```

### 11.3 自定义数据集类

CustomKlineDataset类实现了从CSV文件加载数据的功能：

```python
# 源自 finetune_csv/finetune_base_model.py
class CustomKlineDataset(Dataset):
    def _load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)
        
        # 提取时间特征
        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month
    
    def _split_data_by_time(self):
        # 按时间顺序划分数据集
        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))
        
        if self.data_type == 'train':
            self.data = self.data.iloc[:train_end]
        elif self.data_type == 'val':
            self.data = self.data.iloc[train_end:val_end]
        elif self.data_type == 'test':
            self.data = self.data.iloc[val_end:]
```

### 11.4 顺序训练

`train_sequential.py` 提供了端到端的顺序训练流程：

```python
# 源自 finetune_csv/train_sequential.py
class SequentialTrainer:
    def run_training(self):
        # 阶段1：训练Tokenizer
        if self.config.train_tokenizer:
            self.train_tokenizer_phase()
        
        # 阶段2：训练Predictor
        if self.config.train_basemodel:
            self.train_basemodel_phase()
```

使用方式：

```bash
# 完整训练
python train_sequential.py --config configs/config.yaml

# 跳过已存在的模型
python train_sequential.py --config configs/config.yaml --skip-existing

# 只训练Tokenizer
python train_sequential.py --config configs/config.yaml --skip-basemodel

# 只训练Predictor
python train_sequential.py --config configs/config.yaml --skip-tokenizer
```

### 11.5 DDP多卡训练

对于多GPU环境，可以使用DDP加速训练：

```bash
DIST_BACKEND=nccl \
torchrun --standalone --nproc_per_node=8 train_sequential.py --config configs/config.yaml
```

系统会自动检测分布式环境并进行相应的初始化：

```python
# 源自 finetune_csv/train_sequential.py
def _setup_distributed(self):
    if self.world_size > 1 and torch.cuda.is_available():
        backend = os.environ.get("DIST_BACKEND", "nccl").lower()
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
```

### 11.6 从零训练 vs 微调

Kronos支持两种训练模式：

**微调模式（默认）**：从预训练模型开始，在自定义数据上继续训练。这是推荐的方式，因为预训练模型已经学到了通用的K线模式。

**从零训练模式**：随机初始化模型参数，在自定义数据上从头训练。这需要大量的数据和计算资源。

配置方式：

```yaml
experiment:
  pre_trained: false           # 全局控制
  pre_trained_tokenizer: true  # 单独控制Tokenizer
  pre_trained_predictor: true  # 单独控制Predictor
```

当 `pre_trained=false` 时，系统会从配置文件中读取模型架构参数并随机初始化：

```python
# 源自 finetune_csv/finetune_base_model.py
if not config.pre_trained_predictor:
    cfg_path = os.path.join(config.pretrained_predictor_path, 'config.json')
    with open(cfg_path, 'r') as f:
        arch = json.load(f)
    model = Kronos(
        s1_bits=arch.get('s1_bits', 10),
        s2_bits=arch.get('s2_bits', 10),
        n_layers=arch.get('n_layers', 12),
        d_model=arch.get('d_model', 832),
        # ... 其他架构参数
    )
```

---

## 第十二章 回测系统设计

### 12.1 回测的意义

回测（Backtesting）是量化交易策略开发中不可或缺的环节。它通过在历史数据上模拟交易来评估策略的性能，帮助投资者在投入真实资金之前了解策略的风险和收益特征。

Kronos提供了两套回测系统：
1. **Qlib回测**：基于微软Qlib框架的专业回测系统
2. **独立回测**：不依赖Qlib的简化回测系统

### 12.2 独立回测系统

独立回测系统 `examples/run_backtest_kronos.py` 实现了完整的回测流程：

```python
# 源自 examples/run_backtest_kronos.py
class KronosBacktester:
    def calculate_trading_signals(self, hist_df, pred_df, threshold=0.02):
        # 计算预测收益率
        combined['pred_return'] = combined['predicted'].pct_change()
        
        # 生成交易信号
        combined['signal'] = np.where(
            combined['pred_return'] > threshold, 1,   # 买入信号
            np.where(combined['pred_return'] < -threshold, -1, 0)  # 卖出信号
        )
        
        # 持仓信号（保持上次信号直到新信号出现）
        combined['position'] = combined['signal'].replace(to_replace=0, method='ffill').fillna(0)
```

### 12.3 回测执行

回测的核心是模拟交易的执行过程：

```python
def run_backtest(self, combined_df):
    capital = self.initial_capital
    position = 0
    
    for i, (date, row) in enumerate(combined_df.iterrows()):
        current_price = row['actual'] if not pd.isna(row['actual']) else row['predicted']
        signal = row['position']
        
        if i > 0:
            prev_position = backtest_results['position'].iloc[i - 1]
            
            # 平仓信号
            if prev_position != 0 and signal == 0:
                capital = position * current_price
                position = 0
            
            # 开仓信号
            elif prev_position == 0 and signal != 0:
                shares = int(capital / current_price)
                if shares > 0:
                    position = shares * signal
                    capital -= shares * current_price
        
        portfolio_value = capital + position * current_price
```

### 12.4 性能指标

回测系统计算以下关键性能指标：

```python
def calculate_metrics(self, backtest_results, trades):
    # 总收益率
    total_return = (backtest_results['capital'].iloc[-1] - self.initial_capital) / self.initial_capital
    
    # 年化收益率
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    # 波动率
    volatility = returns.std() * np.sqrt(252)
    
    # 夏普比率（无风险利率3%）
    sharpe_ratio = (annual_return - risk_free_rate) / volatility
    
    # 最大回撤
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # 胜率
    win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
```

### 12.5 结果可视化

回测系统生成三张图表：

```python
def plot_backtest_results(self, backtest_results, metrics, stock_code, output_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. 资金曲线
    ax1.plot(backtest_results.index, backtest_results['capital'], label='策略资金曲线')
    
    # 2. 累计收益曲线（与基准对比）
    ax2.plot(backtest_results.index, cumulative_returns, label='策略累计收益')
    ax2.plot(backtest_results.index, benchmark_returns, label='基准收益（买入持有）')
    
    # 3. 回撤曲线
    ax3.fill_between(backtest_results.index, drawdown, 0, alpha=0.3, color='red')
```

---

## 第十三章 Web UI与可视化

### 13.1 Flask应用架构

Kronos提供了一个基于Flask的Web UI，支持交互式的预测和可视化：

```python
# 源自 webui/app.py
app = Flask(__name__)
CORS(app)

# 全局模型变量
tokenizer = None
model = None
predictor = None
```

### 13.2 API端点

Web UI提供以下API端点：

**数据管理**：
- `GET /api/data-files`：获取可用的数据文件列表
- `POST /api/load-data`：加载数据文件并返回数据信息

**模型管理**：
- `GET /api/available-models`：获取可用模型列表
- `POST /api/load-model`：加载指定的Kronos模型
- `GET /api/model-status`：获取当前模型状态

**预测**：
- `POST /api/predict`：执行预测并返回结果

### 13.3 模型配置

Web UI支持三种模型配置：

```python
AVAILABLE_MODELS = {
    'kronos-mini': {
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
    },
    'kronos-small': {
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
    },
    'kronos-base': {
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
    }
}
```

### 13.4 蜡烛图可视化

Web UI使用Plotly生成交互式的蜡烛图：

```python
def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None):
    fig = go.Figure()
    
    # 历史数据蜡烛图
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'],
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='历史数据',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # 预测数据蜡烛图
    fig.add_trace(go.Candlestick(
        x=pred_timestamps,
        open=pred_df['open'],
        high=pred_df['high'],
        low=pred_df['low'],
        close=pred_df['close'],
        name='预测数据',
        increasing_line_color='#66BB6A',
        decreasing_line_color='#FF7043'
    ))
```

### 13.5 预测结果保存

每次预测的结果都会自动保存到JSON文件：

```python
def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, 
                           input_data, prediction_params):
    save_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'file_path': file_path,
        'prediction_type': prediction_type,
        'prediction_params': prediction_params,
        'input_data_summary': {...},
        'prediction_results': prediction_results,
        'actual_data': actual_data,
        'analysis': {...}
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
```

---

## 第十四章 模型评估与测试

### 14.1 回归测试

Kronos包含严格的回归测试，确保模型在不同版本之间保持一致的预测行为：

```python
# 源自 tests/test_kronos_regression.py
@pytest.mark.parametrize("context_len", TEST_CTX_LEN)
def test_kronos_predictor_regression(context_len):
    # 加载输入数据和期望输出
    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    expected_df = pd.read_csv(expected_output_path, parse_dates=["timestamps"])
    
    # 加载模型（使用固定版本）
    tokenizer = KronosTokenizer.from_pretrained(
        "NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained(
        "NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    
    # 执行预测
    pred_df = predictor.predict(df=context_features, ...)
    
    # 验证结果
    np.testing.assert_allclose(obtained, expected, rtol=REL_TOLERANCE)
```

### 14.2 MSE评估

除了精确的回归测试外，Kronos还包含基于均方误差（MSE）的评估：

```python
@pytest.mark.parametrize("context_len, expected_mse", zip(MSE_CTX_LEN, MSE_EXPECTED))
def test_kronos_predictor_mse(context_len, expected_mse):
    # 在多个随机样本上计算MSE
    for row_idx in sample_indices:
        pred_df = predictor.predict(...)
        mse_values.append(float(np.mean((obtained - expected) ** 2)))
    
    mse = np.mean(mse_values).item()
    assert abs(mse - expected_mse) <= MSE_TOLERANCE
```

### 14.3 测试配置

回归测试使用固定的模型版本和种子来确保可复现性：

```python
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
SEED = 123
DEVICE = "cpu"
REL_TOLERANCE = 1e-5
```

---

# 第五部分：展望篇

---

## 第十五章 与同类模型对比

### 15.1 时间序列基础模型概览

近年来，时间序列基础模型领域涌现了多个重要工作。以下是与Kronos最具代表性的几个模型的对比。

### 15.2 TimesFM（Google）

TimesFM是Google在2024年发布的时间序列基础模型。它基于Decoder-only Transformer架构，使用Patch作为token单位。

**架构差异**：
- TimesFM使用Patch tokenization，将连续的时间序列值分组为Patch
- Kronos使用分层离散tokenization，将连续值量化为二进制码字

**数据差异**：
- TimesFM在多种时间序列数据上训练（天气、电力、交通等）
- Kronos专注于金融K线数据

**任务差异**：
- TimesFM主要针对点预测（point forecasting）
- Kronos支持完整的OHLCV预测

### 15.3 Chronos（Amazon）

Chronos是Amazon在2024年发布的时间序列基础模型。它将时间序列值量化为离散token，然后使用T5等语言模型架构进行训练。

**量化方法差异**：
- Chronos使用标量量化（bucketization），将连续值映射到固定的区间
- Kronos使用二进制球面量化（BSQ），保留了向量空间的几何结构

**架构差异**：
- Chronos使用编码器-解码器架构（T5）
- Kronos使用纯Decoder-only架构

**Token粒度差异**：
- Chronos每个时间步一个token
- Kronos每个时间步两个分层token（s1和s2）

### 15.4 MOMENT

MOMENT是2024年发布的通用时间序列基础模型，使用掩码自编码器（MAE）进行预训练。

**训练方式差异**：
- MOMENT使用掩码自编码（预测被掩码的部分）
- Kronos使用自回归预测（预测下一个token）

**通用性差异**：
- MOMENT追求通用性，在多种时间序列任务上训练
- Kronos专注于金融领域

### 15.5 对比总结

| 特性 | Kronos | TimesFM | Chronos | MOMENT |
|------|--------|---------|---------|--------|
| 架构 | Decoder-only | Decoder-only | Enc-Dec | MAE |
| Token化 | 分层BSQ | Patch | 标量量化 | 连续值 |
| 金融专注 | ✅ | ❌ | ❌ | ❌ |
| OHLCV支持 | ✅ | ❌ | ❌ | ❌ |
| 开源 | ✅ | ✅ | ✅ | ✅ |
| 预训练数据量 | 120亿条 | 未公开 | 未公开 | 未公开 |

---

## 第十六章 金融AI的前沿与展望

### 16.1 多模态金融AI

未来的金融AI系统可能会整合多种数据模态：

**文本信息**：新闻、公告、社交媒体、研报等文本信息对市场有重要影响。将K线分析与自然语言处理结合，可以实现更全面的市场理解。

**图结构数据**：公司之间的供应链关系、股权关系、行业分类等构成了复杂的图结构。图神经网络可以用于建模这些关系。

**替代数据**：卫星图像、信用卡消费数据、网络流量数据等替代数据源可以提供传统金融数据之外的信息。

### 16.2 强化学习与交易

强化学习（RL）为自动交易提供了另一种范式。与监督学习（如Kronos的预测）不同，RL直接优化交易策略的长期收益。

**深度Q网络（DQN）**：将交易动作（买入、卖出、持有）建模为Q值函数。

**策略梯度方法**：直接优化交易策略的参数，如PPO、A2C等。

**多智能体系统**：多个交易智能体之间的博弈和合作。

Kronos的预测能力可以与RL结合：使用Kronos的预测作为RL智能体的观测信息，帮助智能体做出更好的决策。

### 16.3 可解释性与监管

随着AI在金融领域的应用越来越广泛，可解释性和监管合规变得越来越重要。

**模型可解释性**：
- 注意力权重可视化：分析模型关注哪些历史时间步
- 特征重要性分析：了解哪些输入特征对预测最重要
- 反事实分析：改变输入观察输出的变化

**监管挑战**：
- 公平性：AI系统是否对不同市场、不同资产一视同仁？
- 透明度：监管机构需要理解AI系统的决策逻辑
- 稳定性：AI系统在极端市场条件下的行为是否可控？

### 16.4 数据质量与数据治理

金融数据的质量直接影响模型的性能。未来的方向包括：

**数据清洗**：自动检测和处理异常数据、缺失数据、错误数据。

**数据增强**：通过合成数据扩充训练集，提高模型的泛化能力。

**隐私保护**：在使用敏感金融数据时，需要考虑数据隐私和安全。

### 16.5 实时推理与部署

将Kronos部署到实际交易系统中需要考虑：

**延迟优化**：量化交易对延迟极为敏感，需要优化模型的推理速度。

**模型压缩**：通过知识蒸馏、量化、剪枝等技术减小模型体积。

**流式处理**：支持实时数据流的处理和预测。

**容错机制**：确保系统在异常情况下的稳定性。

---

## 第十七章 附录

### 17.1 API参考

#### KronosTokenizer

```python
class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """K线数据分层Tokenizer"""
    
    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers, 
                 ffn_dropout_p, attn_dropout_p, resid_dropout_p, s1_bits, s2_bits, 
                 beta, gamma0, gamma, zeta, group_size):
        """初始化Tokenizer
        Args:
            d_in: 输入维度（通常为6，对应OHLCV+amount）
            d_model: 模型内部维度
            n_heads: 注意力头数
            ff_dim: 前馈网络维度
            n_enc_layers: 编码器层数
            n_dec_layers: 解码器层数
            s1_bits: s1 token的位数
            s2_bits: s2 token的位数
            beta: commit loss权重
            gamma0: 每样本熵惩罚权重
            gamma: 码本熵惩罚权重
            zeta: 总熵惩罚权重
            group_size: 量化分组大小
        """
    
    def forward(self, x):
        """前向传播
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_in)
        Returns:
            tuple: ((z_pre, z), bsq_loss, quantized, z_indices)
        """
    
    def encode(self, x, half=False):
        """编码为token索引
        Args:
            x: 输入张量
            half: 是否返回分层索引
        Returns:
            token索引
        """
    
    def decode(self, x, half=False):
        """从token索引解码
        Args:
            x: token索引
            half: 是否为分层索引
        Returns:
            重建的连续数据
        """
```

#### Kronos

```python
class Kronos(nn.Module, PyTorchModelHubMixin):
    """K线序列预测模型"""
    
    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim, 
                 ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, 
                 learn_te):
        """初始化预测模型"""
    
    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, 
                use_teacher_forcing=False, s1_targets=None):
        """前向传播，返回 (s1_logits, s2_logits)"""
    
    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """解码s1 token，返回 (s1_logits, context)"""
    
    def decode_s2(self, context, s1_ids, padding_mask=None):
        """基于上下文解码s2 token"""
```

#### KronosPredictor

```python
class KronosPredictor:
    """K线预测器（封装了预处理和后处理）"""
    
    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
        """初始化预测器"""
    
    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, 
                top_p=0.9, sample_count=1, verbose=True):
        """单条预测
        Args:
            df: 包含OHLCV的DataFrame
            x_timestamp: 历史时间戳
            y_timestamp: 预测时间戳
            pred_len: 预测长度
            T: 温度参数
            top_k: Top-k采样参数
            top_p: Top-p采样参数
            sample_count: 采样路径数
        Returns:
            预测结果DataFrame
        """
    
    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, 
                      T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        """批量预测"""
```

### 17.2 配置参考

#### finetune/config.py 主要配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| qlib_data_path | ~/.qlib/qlib_data/cn_data | Qlib数据路径 |
| instrument | csi300 | 投资标的 |
| lookback_window | 90 | 回看窗口大小 |
| predict_window | 10 | 预测窗口大小 |
| max_context | 512 | 最大上下文长度 |
| clip | 5.0 | 归一化裁剪值 |
| epochs | 30 | 训练轮数 |
| batch_size | 50 | 批次大小 |
| tokenizer_learning_rate | 2e-4 | Tokenizer学习率 |
| predictor_learning_rate | 4e-5 | Predictor学习率 |
| adam_weight_decay | 0.1 | 权重衰减 |
| backtest_n_symbol_hold | 50 | 持有股票数 |
| inference_T | 0.6 | 推理温度 |
| inference_top_p | 0.9 | Top-p采样参数 |
| inference_sample_count | 5 | 采样路径数 |

### 17.3 常见问题

**Q: Kronos可以直接用于实际交易吗？**

A: Kronos是一个研究项目，其预测结果不应直接作为交易决策的唯一依据。实际的量化交易系统需要考虑更多的风险管理、仓位控制、交易成本等因素。建议将Kronos的预测作为交易信号的一个输入，结合其他分析方法综合判断。

**Q: 微调需要多少数据？**

A: 这取决于具体的应用场景。一般来说，至少需要数千根K线才能获得有意义的微调效果。对于日线级别，可能需要数年的数据；对于分钟线级别，几个月的数据可能就够了。

**Q: 模型支持哪些时间粒度？**

A: 理论上，Kronos支持任意时间粒度的K线数据。模型本身不假设特定的时间间隔，而是从时间戳中提取周期性特征。实际效果取决于训练数据中是否包含相应粒度的数据。

**Q: 如何选择合适的模型大小？**

A: 对于资源受限的环境，推荐使用Kronos-mini（4.1M参数）。对于大多数应用，Kronos-small（24.7M参数）是一个好的平衡点。如果对预测质量有更高要求且有足够的计算资源，可以使用Kronos-base（102.3M参数）。

**Q: 训练时需要GPU吗？**

A: 是的，训练需要GPU。Kronos的训练涉及大量的矩阵运算，GPU可以显著加速训练过程。推理也可以使用GPU，但对于小规模预测，CPU也可以工作（只是速度较慢）。

**Q: 如何处理缺失数据？**

A: Kronos要求输入数据不包含NaN值。如果数据中存在缺失值，建议先进行填充（如前向填充）或删除包含缺失值的行。

**Q: 模型的预测准确率如何？**

A: 金融预测的"准确率"很难定义，因为预测的是连续值而非离散类别。Kronos使用MSE等指标来评估预测质量。需要注意的是，金融市场的信噪比极低，任何模型都无法保证高准确率的预测。

---

## 补充章节A：Kronos的数学基础深度解析

### A.1 向量量化理论

向量量化（Vector Quantization, VQ）是信号处理和信息论中的经典技术，其核心思想是将连续的高维向量空间划分为有限数量的区域，每个区域用一个代表性的码字（codeword）来表示。所有码字的集合称为码本（codebook）。

在传统的向量量化中，给定一个训练数据集 $\{x_1, x_2, ..., x_N\}$，目标是找到一个码本 $C = \{c_1, c_2, ..., c_K\}$，使得量化误差最小化：

$$\min_C \sum_{i=1}^{N} \min_{k} \|x_i - c_k\|^2$$

这就是著名的Lloyd算法（也称为K-means算法）要解决的问题。

Kronos使用的BSQ与传统VQ有本质区别。BSQ不需要显式地存储码本，因为所有可能的码字由二进制向量空间 $\{-1, +1\}^d$ 中的所有元素构成。对于 $d$ 维的BSQ，码本大小为 $2^d$，这是一个指数级增长的集合。

然而，Kronos通过分层设计巧妙地控制了实际使用的码本大小。当使用 `s1_bits=10, s2_bits=10` 时，模型将20维的量化空间分解为两个10维的子空间，每个子空间的码本大小为 $2^{10}=1024$。这种分解使得自回归模型只需建模1024大小的词汇表，而不是百万级别的完整码本。

### A.2 直通估计器（Straight-Through Estimator）

BSQ中的二值化操作 $z \rightarrow \text{sign}(z)$ 是不可导的，因为sign函数在零点处的导数为无穷大（或者在实践中，除了零点外处处为零）。这意味着标准的反向传播无法直接用于训练。

直通估计器（STE）是解决这一问题的经典方法。STE的核心思想是：在前向传播中使用量化后的值，但在反向传播中假装量化操作不存在，直接将梯度传递给输入。

从Kronos的源码中可以清楚地看到STE的应用：

```python
def quantize(self, z):
    zhat = torch.where(z > 0,
                       torch.tensor(1, dtype=z.dtype, device=z.device),
                       torch.tensor(-1, dtype=z.dtype, device=z.device))
    return z + (zhat - z).detach()  # STE: 前向用zhat，反向用z
```

表达式 `z + (zhat - z).detach()` 的含义是：
- **前向传播**：结果为 `z + (zhat - z) = zhat`，即使用量化后的值
- **反向传播**：`.detach()` 使得 `(zhat - z)` 不参与梯度计算，梯度直接从输出传递到 `z`

STE的数学直觉是：虽然量化操作本身不可导，但我们假设"如果输入稍有变化，量化结果可能不变"，因此梯度近似为恒等映射。

### A.3 球面量化与L2归一化

BSQ中的"球面"（Spherical）指的是量化发生在单位球面上。在量化之前，输入向量被L2归一化到单位球面：

$$\hat{z} = \frac{z}{\|z\|_2}$$

这一步骤在Kronos的BSQuantizer中实现：

```python
def forward(self, z, half=False, collect_metrics=True):
    z = F.normalize(z, dim=-1)  # L2归一化
    quantized, bsq_loss, metrics = self.bsq(z, collect_metrics=collect_metrics)
```

L2归一化的作用：
1. **统一尺度**：不同样本的向量可能有不同的尺度，归一化将它们统一到同一尺度上
2. **简化量化**：在单位球面上，二值化操作等价于判断向量在每个维度上的方向
3. **提高稳定性**：归一化避免了量化误差被向量的模长放大

归一化后的量化操作数学上可以表示为：对于归一化后的向量 $\hat{z} \in \mathbb{R}^d$，量化结果为：

$$q_i = \begin{cases} +1 & \text{if } \hat{z}_i > 0 \\ -1 & \text{if } \hat{z}_i \leq 0 \end{cases}$$

量化后的向量 $q$ 也在单位球面上（准确地说，在超立方体 $\{-1, +1\}^d$ 的顶点上），其L2范数为 $\sqrt{d}$。为了保持一致的尺度，Kronos在量化后进行了缩放：

$$q_{scaled} = \frac{q}{\sqrt{d}}$$

### A.4 熵与码本利用率

熵（Entropy）是信息论中衡量不确定性的核心概念。在向量量化中，熵被用来衡量码本的利用率。

**离散熵**的定义为：

$$H(X) = -\sum_{i} p(x_i) \log p(x_i)$$

其中 $p(x_i)$ 是码本中第 $i$ 个码字被选中的概率。

**最大熵**发生在所有码字被均匀选中时，此时 $H_{max} = \log K$（$K$ 是码本大小）。**最小熵**为0，发生在只有一个码字被使用时。

Kronos使用两种熵的计算方式：

**每样本熵（Per-sample Entropy）**：对于每个训练样本，计算其量化结果在各组码字上的分布熵。高熵意味着样本的量化结果分布均匀，低熵意味着集中在少数码字上。

```python
def get_hard_per_sample_entropy(self, zb_by_sample):
    probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
    persample_entropy = - probs_per_dim * torch.log(probs_per_dim + 1e-8) \
                        - (1 - probs_per_dim) * torch.log(1 - probs_per_dim + 1e-8)
    return persample_entropy.sum(-1).mean()
```

**码本熵（Codebook Entropy）**：在整个训练集上统计每个码字被使用的频率，然后计算分布熵。高码本熵意味着码本被充分利用。

Kronos的熵惩罚公式为：

$$L_{entropy} = \gamma_0 \cdot H_{persample} - \gamma \cdot H_{codebook}$$

- **最大化 $H_{persample}$**：鼓励每个样本的量化结果更加多样化
- **最大化 $H_{codebook}$**：鼓励码本被更均匀地使用
- **$\gamma > \gamma_0$**：码本熵的权重更大，确保整体码本利用率

### A.5 交叉注意力机制

Kronos的DependencyAwareLayer使用交叉注意力（Cross-Attention）来建立s2与s1之间的依赖关系。交叉注意力与自注意力的区别在于：自注意力的Q、K、V来自同一个序列，而交叉注意力的Q来自一个序列，K和V来自另一个序列。

在Kronos中：
- **Query**：s1 token的嵌入（`sibling_embed`）
- **Key/Value**：Transformer的输出（`hidden_states`）

```python
class MultiHeadCrossAttentionWithRoPE(nn.Module):
    def forward(self, query, key, value, key_padding_mask=None):
        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rotary(q, k)  # 应用RoPE
        
        attn_output = F.scaled_dot_product_attention(q, k, v, ...)
```

交叉注意力的数学表达：

$$\text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q = W_Q \cdot \text{sibling\_embed}$，$K = W_K \cdot \text{hidden\_states}$，$V = W_V \cdot \text{hidden\_states}$。

这种设计的直觉是：s1 token的嵌入作为"查询"，去"检索"Transformer隐藏状态中与s1相关的信息，从而生成对s2的条件预测。

### A.6 缩放点积注意力

Kronos使用PyTorch的 `scaled_dot_product_attention` 函数，这是对标准注意力的高效实现：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M$ 是注意力掩码（包括因果掩码和填充掩码）。

缩放因子 $\sqrt{d_k}$ 的作用是防止当维度 $d_k$ 较大时，$QK^T$ 的值过大导致softmax函数进入饱和区（梯度接近零的区域）。

因果掩码确保每个位置只能关注它自己和之前的位置：

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

填充掩码用于处理变长序列，将填充位置的注意力权重设为负无穷：

```python
if key_padding_mask is not None:
    attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
    attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)
```

### A.7 SwiGLU的数学推导

SwiGLU是GLU（Gated Linear Unit）家族的一个变体，由Noam Shazeer在2020年提出。

**GLU的基本形式**：

$$\text{GLU}(x) = (W_1 x) \otimes \sigma(W_2 x)$$

其中 $\otimes$ 是逐元素乘法，$\sigma$ 是sigmoid函数。

**SwiGLU的改进**：用Swish函数替代sigmoid：

$$\text{SwiGLU}(x) = \text{Swish}(W_1 x) \otimes (W_3 x)$$

其中 $\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1+e^{-x}}$。

Swish函数的特点：
- **平滑**：处处可导
- **非单调**：在负半轴有一个小的负值区域
- **有界下界**：最小值约为 $-0.278$（在 $x \approx -1.278$ 处）
- **无界上界**：当 $x \to +\infty$ 时趋近于 $x$

SwiGLU相比ReLU的优势在于其平滑性和门控机制，这有助于训练更深的网络并提高模型的表达能力。

### A.8 OneCycleLR调度策略

OneCycleLR是Leslie Smith在2018年提出的学习率调度策略，其核心思想是在训练过程中先将学习率从小增加到最大值，然后再降低到非常小的值。

OneCycleLR的三个阶段：
1. **预热阶段（Warmup）**：学习率从 `max_lr/div_factor` 线性增加到 `max_lr`
2. **退火阶段（Annealing）**：学习率从 `max_lr` 按余弦函数降低
3. **微调阶段（Fine-tuning）**：学习率继续降低到非常小的值

Kronos的配置中 `pct_start=0.03` 表示前3%的步骤用于预热，`div_factor=10` 表示初始学习率为最大学习率的1/10。

OneCycleLR的数学表达（退火阶段）：

$$lr(t) = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})\left(1 + \cos\left(\pi \cdot \frac{t - T_{warmup}}{T_{total} - T_{warmup}}\right)\right)$$

这种调度策略的优势：
1. **快速收敛**：大学习率阶段帮助模型快速探索参数空间
2. **精细调优**：小学习率阶段帮助模型收敛到更好的局部最优
3. **正则化效果**：学习率的变化类似于隐式的正则化

---

## 补充章节B：金融量化实战知识

### B.1 量化交易概述

量化交易（Quantitative Trading）是利用数学模型、统计分析和计算机算法来进行交易决策的方法。与主观交易（依赖交易员的经验和直觉）不同，量化交易追求系统化、可重复、可验证的交易流程。

量化交易的核心流程：
1. **数据获取**：收集历史和实时的市场数据
2. **信号生成**：通过模型或规则产生交易信号
3. **投资组合构建**：根据信号和风险约束构建投资组合
4. **执行**：将交易指令发送到市场
5. **风险管理**：监控和控制风险敞口
6. **绩效评估**：评估策略的收益和风险特征

Kronos主要服务于第2步——信号生成。它通过预测未来的K线走势来产生交易信号。

### B.2 Alpha与风险因子

在量化投资中，Alpha指的是投资策略相对于基准的超额收益。寻找和捕获Alpha是量化投资的核心目标。

常见的风险因子包括：
- **市场因子（Beta）**：与市场整体走势相关的收益
- **规模因子（Size）**：小盘股与大盘股的收益差异
- **价值因子（Value）**：低估值与高估值股票的收益差异
- **动量因子（Momentum）**：近期表现好与差的股票的收益差异
- **波动率因子（Volatility）**：低波动与高波动股票的收益差异

Kronos的预测信号可能包含多种因子的暴露。在实际应用中，需要通过因子中性化来提取纯Alpha信号：

$$\alpha = r_{predicted} - \beta \cdot r_{market} - \sum_{i} \beta_i \cdot f_i$$

### B.3 风险管理

风险管理是量化交易中至关重要的一环。常见的风险管理方法包括：

**止损（Stop Loss）**：当损失达到预设阈值时强制平仓。

**仓位管理（Position Sizing）**：根据风险承受能力控制每个头寸的大小。常见的方法有固定比例法、凯利公式法等。

**分散化（Diversification）**：在多个资产、多个策略、多个时间框架上分散投资。

**风险价值（VaR）**：在给定的置信水平下，预期的最大损失。

**最大回撤控制**：设置最大可接受的回撤水平，超过时降低仓位或停止交易。

### B.4 常见评价指标

**年化收益率**：将总收益率折算为年化形式：

$$R_{annual} = (1 + R_{total})^{252/N} - 1$$

其中 $N$ 是交易日数，252是一年的交易日数。

**夏普比率**：风险调整后的收益指标：

$$SR = \frac{R_p - R_f}{\sigma_p}$$

其中 $R_p$ 是策略收益率，$R_f$ 是无风险利率，$\sigma_p$ 是策略收益率的标准差。

**索提诺比率**：只考虑下行风险的夏普比率：

$$Sortino = \frac{R_p - R_f}{\sigma_d}$$

其中 $\sigma_d$ 是下行标准差。

**信息比率**：超额收益与跟踪误差的比值：

$$IR = \frac{R_p - R_b}{\sigma_{p-b}}$$

其中 $R_b$ 是基准收益率，$\sigma_{p-b}$ 是超额收益的标准差。

**最大回撤**：从历史最高点到最低点的最大跌幅：

$$MDD = \max_{t} \left(\frac{\max_{s \leq t} P_s - P_t}{\max_{s \leq t} P_s}\right)$$

**卡尔马比率**：年化收益与最大回撤的比值：

$$Calmar = \frac{R_{annual}}{|MDD|}$$

### B.5 回测陷阱

回测是量化策略开发中不可或缺的环节，但也存在许多常见的陷阱：

**过拟合（Overfitting）**：策略在历史数据上表现很好，但在新数据上表现差。这通常是因为策略的参数过多或使用了未来数据。

**幸存者偏差（Survivorship Bias）**：只使用当前存在的股票进行回测，忽略了已经退市的股票。这会高估策略的历史表现。

**前视偏差（Look-ahead Bias）**：在回测中使用了在实际交易时尚不可用的信息。例如，使用收盘价数据在当天开盘时进行交易。

**交易成本忽略**：忽略佣金、滑点、冲击成本等交易成本，导致回测结果过于乐观。

**流动性假设**：假设可以以任意价格成交任意数量的股票，忽略了实际的市场流动性限制。

Kronos的微调流程中特别注意了数据泄露问题：
- 归一化只使用回看窗口内的数据
- 训练集、验证集、测试集按时间严格划分
- 使用确定性种子确保可复现性

### B.6 投资组合构建

投资组合构建是将交易信号转化为实际投资组合的过程。常见的方法包括：

**等权配置**：每个资产分配相同的权重。简单但不考虑风险。

**市值加权**：按市值比例分配权重。这是大多数指数的做法。

**风险平价**：使每个资产对组合风险的贡献相等。

**均值-方差优化（Markowitz）**：在给定收益目标下最小化风险，或在给定风险水平下最大化收益。

**Black-Litterman模型**：将投资者的观点与市场均衡相结合，生成更稳定的投资组合权重。

Kronos的回测系统使用TopkDropoutStrategy，这是一种简单的投资组合构建方法：选择预测信号最强的topk只股票等权持有。

### B.7 交易执行

从预测信号到实际交易之间，还需要考虑执行层面的问题：

**滑点（Slippage）**：实际成交价与预期价格之间的差异。滑点通常由市场波动和流动性不足引起。

**冲击成本（Market Impact）**：大额交易对市场价格的影响。交易量越大，冲击成本越高。

**订单类型**：市价单（立即成交但价格不确定）、限价单（价格确定但可能无法成交）等。

**交易时机**：开盘、收盘、盘中交易各有特点。开盘通常波动较大，收盘通常流动性较好。

Kronos的回测系统考虑了部分交易成本：

```python
backtest_config = {
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,   # 涨跌停限制
        "deal_price": "open",       # 以开盘价成交
        "open_cost": 0.001,         # 买入手续费
        "close_cost": 0.0015,       # 卖出手续费
        "min_cost": 5,              # 最低手续费
    },
}
```

---

## 补充章节C：深度学习在金融中的应用案例

### C.1 情绪分析与市场预测

自然语言处理（NLP）技术在金融领域的最直接应用是情绪分析。通过分析新闻、社交媒体、分析师报告等文本数据，可以提取市场情绪信号。

常见的情绪分析方法：
- **词典法**：使用预定义的情绪词典统计正面和负面词汇的比例
- **机器学习法**：训练分类器来判断文本的情绪倾向
- **深度学习法**：使用BERT、GPT等预训练语言模型进行情绪分析

情绪信号可以与Kronos的K线预测结合，形成更全面的市场分析。例如，当Kronos预测价格上涨且情绪分析也显示正面情绪时，信号的可信度更高。

### C.2 异常检测

金融市场的异常事件（如闪崩、暴涨、欺诈交易）的检测是另一个重要的应用领域。

常见的异常检测方法：
- **统计方法**：基于Z-score、IQR等统计量检测偏离正常范围的数据点
- **聚类方法**：将数据聚类，偏离所有簇的数据点被视为异常
- **自编码器**：训练自编码器重建正常数据，重建误差大的数据点可能是异常
- **孤立森林**：通过随机分割来隔离异常点

Kronos的预测误差本身也可以作为异常检测的信号：当实际走势与预测严重偏离时，可能意味着市场发生了异常事件。

### C.3 高频交易

高频交易（HFT）是在极短的时间尺度（毫秒甚至微秒）上进行交易的策略。HFT的特点是：
- **低延迟**：对硬件和网络延迟有极高要求
- **高换手**：每天可能进行数百万次交易
- **小利润**：每次交易的利润很小，但累积效果显著

Kronos目前的架构更适合中低频预测（分钟级到日级），但其技术思想可以延伸到高频领域。例如，可以训练专门的高频K线Tokenizer来捕捉微观结构信息。

### C.4 加密货币市场

加密货币市场为AI模型提供了独特的机遇：
- **24/7交易**：没有开盘/收盘的概念，数据量更大
- **高波动率**：价格波动剧烈，为预测模型提供了更大的alpha空间
- **市场效率较低**：相比传统金融市场，加密货币市场的效率较低，存在更多的定价偏差
- **数据透明**：区块链上的交易数据完全公开

Kronos的训练数据中包含了加密货币交易所的数据，这使得模型能够学习到加密货币K线的特殊模式。

### C.5 多资产类别

Kronos的设计使其能够处理多种资产类别的K线数据：

**股票**：个股和指数的K线数据
**期货**：商品期货、金融期货
**外汇**：货币对的汇率数据
**加密货币**：比特币、以太坊等加密货币
**债券**：债券收益率和价格

不同资产类别的K线具有不同的统计特性。例如，外汇市场的波动率通常低于加密货币市场。Kronos通过在多种资产上预训练，能够学习到这些差异。

---

## 补充章节D：Kronos的技术创新总结

### D.1 核心创新点

Kronos的技术创新可以总结为以下几点：

**1. 首个金融K线基础模型**：Kronos是第一个专门为金融K线数据设计的开源基础模型，填补了该领域的空白。

**2. 分层二进制球面量化**：将连续的OHLCV数据通过分层BSQ编码为离散token，既保留了数据的精细信息，又实现了高效的离散表示。

**3. 依赖感知的双头预测**：通过DependencyAwareLayer和DualHead实现s1和s2的条件生成，捕捉了K线数据的层次化结构。

**4. 时间感知的自回归架构**：通过专门的TemporalEmbedding和RoPE，模型能够感知K线数据的时间周期性。

**5. 海量数据预训练**：在45个全球交易所的120亿条K线上进行预训练，学习到跨市场、跨资产的通用模式。

### D.2 设计决策的权衡

Kronos的每个设计决策都涉及权衡：

**Decoder-only vs Encoder-Decoder**：
- Decoder-only的优势：自回归生成更自然，与语言模型的范式一致
- Encoder-Decoder的优势：可以双向编码历史信息
- Kronos的选择：Decoder-only，因为生成（预测）是核心任务

**分层Token vs 单一Token**：
- 分层Token的优势：词汇表更小，建模更容易，支持条件生成
- 单一Token的优势：信息损失更小
- Kronos的选择：分层Token，平衡了效率和精度

**BSQ vs 标量量化**：
- BSQ的优势：保留向量空间的几何结构，支持端到端训练
- 标量量化的优势：实现更简单
- Kronos的选择：BSQ，因为它更适合多维K线数据

**固定时间编码 vs 可学习时间编码**：
- 固定编码的优势：泛化能力更强
- 可学习编码的优势：可以捕捉特定的时间模式
- Kronos的选择：支持两种模式，通过 `learn_te` 参数控制

### D.3 未来改进方向

基于对Kronos源码的分析，以下是几个可能的改进方向：

**1. 更长的上下文**：当前Kronos-small和Kronos-base的最大上下文长度为512，限制了对长期依赖的建模。可以通过增加RoPE的外推能力或使用稀疏注意力来支持更长的上下文。

**2. 多分辨率Tokenization**：当前的Tokenizer只在单一时间粒度上工作。可以设计多分辨率的Tokenizer，同时捕捉不同时间尺度的模式。

**3. 更大的模型**：Kronos-large（499.2M参数）尚未开源。更大的模型可能具有更强的表达能力。

**4. 零样本能力**：当前的Kronos主要通过微调来适配特定任务。增强零样本能力可以使其在没有微调数据的情况下也能提供有用的预测。

**5. 多模态融合**：将K线数据与文本、新闻等其他模态的信息融合，提供更全面的市场分析。

---

## 参考文献

1. Shi, Y., Fu, Z., Chen, S., Zhao, B., Xu, W., Zhang, C., & Li, J. (2025). Kronos: A Foundation Model for the Language of Financial Markets. arXiv:2508.02739.

2. Yu, J., et al. (2024). Image and Video Tokenization with Binary Spherical Quantization. arXiv:2406.07548.

3. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.

4. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.

5. Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS 2019.

6. Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.

7. Holtzman, A., et al. (2019). The Curious Case of Neural Text Degeneration. ICLR 2020.

8. Das, A., et al. (2024). A Decoder-Only Foundation Model for Time-Series Forecasting. ICML 2024.

9. Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815.

10. Goswami, M., et al. (2024). MOMENT: A Family of Open Time-Series Foundation Models. ICML 2024.

---

## 致谢

感谢Kronos团队（Yu Shi, Zongliang Fu, Shuo Chen, Bohhan Zhao, Wei Xu, Changshui Zhang, Jian Li）开发并开源了这一重要项目。感谢所有为Kronos社区做出贡献的开发者和研究者。

---

*本书基于Kronos开源项目源码分析撰写，旨在为读者提供深入的技术解读。本书内容不构成任何投资建议。*
