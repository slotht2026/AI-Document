# 前言

数据是AI的燃料，但数据的流动正面临前所未有的限制。

GDPR、个人信息保护法、数据安全法——全球各地的数据隐私法规正在收紧。医疗机构不愿共享患者数据，金融机构不愿暴露交易记录，跨国公司不愿将数据传出境外。在数据孤岛日益普遍的今天，如何在保护隐私的前提下继续推动AI的发展？

联邦学习与隐私计算提供了答案。联邦学习让数据不动模型动——多个参与方在不共享原始数据的情况下协作训练模型。差分隐私为数据发布提供了严格的数学保证。安全多方计算让互不信任的各方能够共同计算而不泄露各自输入。同态加密允许在加密数据上直接进行计算。

本书系统研究了隐私保护AI的核心技术、协议设计和应用实践。

---

*本书所有技术内容均基于公开来源的学术论文和技术报告。*
# 联邦学习与隐私计算：数据安全的AI范式

## 书籍定位
- 体量：约100,000字
- 风格：理论+协议+工程实践
- 读者：隐私计算工程师、数据安全研究者、合规人员

---

## 第一部分：基础（约15,000字）
### 第1章 隐私计算概论（~5,000字）—— 隐私挑战/技术分类/法规框架
### 第2章 联邦学习基础（~5,000字）—— FedAvg/通信优化/异构性
### 第3章 差分隐私（~5,000字）—— ε-δ定义/DP-SGD/Rényi DP

## 第二部分：核心技术（约25,000字）
### 第4章 安全多方计算（~5,000字）—— 秘密共享/混淆电路/PSI
### 第5章 同态加密（~5,000字）—— FHE原理/CKKS/在ML中应用
### 第6章 联邦学习进阶（~5,00字）—— 个性化/异构/通信压缩/聚合策略
### 第7章 联邦学习安全（~5,000字）—— 投毒攻击/推理攻击/防御
### 第8章 可信执行环境（~5,000字）—— SGX/TrustZone/TDX

## 第三部分：应用（约15,000字）
### 第9章 医疗健康（~5,000字）
### 第10章 金融风控（~5,000字）
### 第11章 跨机构协作（~5,000字）

## 第四部分：前沿（约15,000字）
### 第12章 联邦大模型（~5,000字）
### 第13章 去中心化AI（~5,000字）
### 第14章 隐私计算的未来（~5,000字）

## 附录
# 第1章 隐私计算概论

## 1.1 数据隐私的挑战

### 1.1.1 数据孤岛问题

在AI时代，数据是最重要的资产。然而，数据的流动面临多重障碍：
- **法规限制**：GDPR、中国《个人信息保护法》等法规严格限制数据的收集、使用和传输
- **商业竞争**：企业不愿共享核心数据资产
- **安全风险**：数据集中存储增加了泄露风险
- **隐私关切**：个人用户对隐私的意识日益增强

### 1.1.2 隐私泄露案例

近年来的隐私泄露事件凸显了数据安全的紧迫性：
- Facebook-Cambridge Analytica事件（2018）
- 各类医疗数据泄露事件
- 人脸识别数据被滥用
- AI模型泄露训练数据中的隐私信息

### 1.1.3 隐私计算的需求

隐私计算（Privacy-Preserving Computation）的目标是：**在不暴露原始数据的前提下，实现数据的价值利用**。

## 1.2 隐私计算技术分类

### 1.2.1 基于密码学的方法
- **安全多方计算（MPC）**：多方共同计算而不泄露各自输入
- **同态加密（HE）**：在加密数据上直接计算
- **零知识证明（ZKP）**：证明某个陈述为真而不泄露具体信息

### 1.2.2 基于统计学的方法
- **差分隐私（DP）**：在数据中添加噪声，保护个体隐私
- **k-匿名**：确保每条记录至少与k-1条其他记录不可区分

### 1.2.3 基于硬件的方法
- **可信执行环境（TEE）**：在硬件隔离的安全区域中处理数据
- **安全飞地（SGX/TrustZone）**

### 1.2.4 基于机器学习的方法
- **联邦学习（FL）**：数据不动模型动
- **分布式机器学习**

## 1.3 法规框架

### 1.3.1 欧盟GDPR
- 数据最小化原则
- 目的限制原则
- 被遗忘权
- 数据可携带权

### 1.3.2 中国《个人信息保护法》
- 知情同意原则
- 最小必要原则
- 跨境传输限制
- 个人信息权利

### 1.3.3 美国隐私法规
- CCPA/CPRA（加州）
- HIPAA（医疗）
- COPPA（儿童）

## 1.4 本书结构

**第一部分（第1-3章）** 建立基础：隐私计算概论、联邦学习基础、差分隐私。

**第二部分（第4-8章）** 深入核心技术：安全多方计算、同态加密、联邦学习进阶、联邦学习安全、可信执行环境。

**第三部分（第9-11章）** 探讨应用：医疗健康、金融风控、跨机构协作。

**第四部分（第12-14章）** 聚焦前沿：联邦大模型、去中心化AI、隐私计算的未来。

## 1.5 本章小结

本章概述了隐私计算的背景、技术分类和法规框架。在数据隐私日益受到重视的今天，隐私计算不仅是技术需求，更是法律要求。

---

*参考文献：Dwork (2006), McMahan et al. (2017), Gentry (2009)*
# 第2章 联邦学习基础

## 2.1 引言

联邦学习（Federated Learning, FL）是一种分布式机器学习范式，其核心思想是"数据不动模型动"——多个参与方在不共享原始数据的前提下，通过交换模型参数或梯度信息协作完成全局模型的训练。这一范式最早由 McMahan 等人于 2017 年在论文 *Communication-Efficient Learning of Deep Networks from Decentralized Data* 中系统性地提出，并以 **FedAvg（Federated Averaging）** 算法作为其标志性实现。

与传统的集中式学习相比，联邦学习面临三大核心挑战：**通信效率**（参与方与中央服务器之间的带宽有限）、**系统异构性**（设备计算能力与网络条件参差不齐）以及**数据异构性**（各参与方的数据分布往往非独立同分布，即 Non-IID）。本章将围绕 FedAvg 算法及其衍生技术，系统介绍联邦学习的基础理论与关键方法。

---

## 2.2 FedAvg 算法

### 2.2.1 问题建模

假设联邦学习系统中有 $K$ 个参与方（client），第 $k$ 个参与方持有本地数据集 $\mathcal{D}_k$，其样本量为 $n_k$。全局数据集为各参与方数据的并集 $\mathcal{D} = \bigcup_{k=1}^{K} \mathcal{D}_k$，总样本量为 $n = \sum_{k=1}^{K} n_k$。

联邦学习的目标是最小化全局经验风险：

$$
\min_{w} F(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)
$$

其中，$F_k(w) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(w; x_i, y_i)$ 是第 $k$ 个参与方的本地损失函数，$\ell(\cdot)$ 为单样本损失，$w$ 为模型参数。

这一形式化框架体现了联邦学习的两个本质特征：其一，优化目标是各参与方本地损失的加权平均，权重与数据量成正比；其二，优化过程需要在分布式环境下进行，每个参与方只能访问本地数据。

### 2.2.2 算法流程

FedAvg 算法交替执行两个阶段：**本地更新**（Local Update）和**全局聚合**（Global Aggregation）。其完整流程如下：

**算法：FedAvg**

---

**输入：** 全局模型参数 $w^0$，参与方总数 $K$，每轮采样比例 $C$，本地迭代轮数 $E$，本地批量大小 $B$，本地学习率 $\eta$

**输出：** 训练后的全局模型参数 $w^T$

1. **for** 每个通信轮次 $t = 0, 1, 2, \ldots, T-1$ **do**
2. &emsp;&emsp;服务器随机采样 $m = \max(C \cdot K, 1)$ 个参与方构成集合 $\mathcal{S}_t$
3. &emsp;&emsp;服务器将当前全局参数 $w^t$ 广播给 $\mathcal{S}_t$ 中的所有参与方
4. &emsp;&emsp;**for** 每个参与方 $k \in \mathcal{S}_t$ **do**（并行执行）
5. &emsp;&emsp;&emsp;&emsp; $w_k^{t+1} \leftarrow \text{ClientUpdate}(k, w^t)$
6. &emsp;&emsp;**end for**
7. &emsp;&emsp;服务器聚合：$w^{t+1} \leftarrow \sum_{k \in \mathcal{S}_t} \frac{n_k}{\sum_{j \in \mathcal{S}_t} n_j} w_k^{t+1}$
8. **end for**

---

**过程 ClientUpdate($k$, $w$):**

1. $\mathcal{B} \leftarrow$ 将 $\mathcal{D}_k$ 分成大小为 $B$ 的批次
2. **for** 本地轮次 $e = 1, \ldots, E$ **do**
3. &emsp;&emsp;**for** 每个批次 $b \in \mathcal{B}$ **do**
4. &emsp;&emsp;&emsp;&emsp;$w \leftarrow w - \eta \nabla \ell(w; b)$
5. &emsp;&emsp;**end for**
6. **end for**
7. **return** $w$

---

FedAvg 的关键设计选择体现在三个超参数上：$C$ 控制每轮参与训练的客户端比例，$E$ 控制本地训练的迭代次数，$B$ 控制本地小批量的大小。McMahan 等人在实验中发现，当 $C=0.0$（每轮仅一个客户端）、$E=1$ 时，FedAvg 退化为经典的 FedSGD（联邦随机梯度下降）。而适当增大 $E$（如 $E=5$）可以在显著减少通信轮次的同时保持模型精度，这正是 FedAvg 的核心优势。

### 2.2.3 收敛性分析

FedAvg 的收敛性在 IID 数据条件下已被严格证明。Li 等人（2020）在论文 *On the Convergence of FedAvg on Non-IID Data* 中给出了如下结果。

假设每个本地损失函数 $F_k$ 是 $L$-光滑的（$L$-smooth），且各参与方的随机梯度满足无偏性和有界方差：

$$
\mathbb{E}[\nabla F_k(w, \xi)] = \nabla F_k(w), \quad \mathbb{E}[\|\nabla F_k(w, \xi) - \nabla F_k(w)\|^2] \leq \sigma_k^2
$$

则对于适当选择的学习率 $\eta$，经过 $T$ 轮通信后，FedAvg 满足：

$$
\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[\|\nabla F(w^t)\|^2] \leq \mathcal{O}\left(\frac{1}{T}\right) + \mathcal{O}(E^2 \Gamma)
$$

其中第一项 $\mathcal{O}(1/T)$ 反映了通信轮次增加带来的收敛收益，第二项 $\mathcal{O}(E^2 \Gamma)$ 则衡量了**客户端漂移（client drift）**的累积效应。这里 $\Gamma = F^* - \sum_k \frac{n_k}{n} F_k^*$ 度量了各参与方最优解与全局最优解之间的差异。当数据为 IID 时 $\Gamma = 0$，收敛速率与集中式 SGD 一致；当数据为 Non-IID 时 $\Gamma > 0$，本地迭代次数 $E$ 越大，客户端漂移越严重，可能损害收敛性。

这一分析揭示了 FedAvg 的核心权衡：**增大 $E$ 可减少通信轮次，但过度的本地更新会导致各参与方模型偏离全局最优方向。** 实践中通常取 $E \in [1, 20]$ 以在通信效率和收敛质量之间取得平衡。

### 2.2.4 异构数据处理

在真实场景中，各参与方的数据分布往往存在显著差异。例如，不同地区医院的病种分布不同，不同用户的手机使用习惯各异。这种**数据异构性**（data heterogeneity）会导致 FedAvg 收敛变慢甚至发散。

McMahan 等人在原始论文中通过实验验证了 FedAvg 在 Non-IID 数据上的表现。他们将 MNIST 数据集按数字标签排序后分配给不同客户端（即每个客户端主要持有某几个数字的样本），发现标准 FedAvg 仍然可以收敛，但精度相比 IID 设定有所下降，且训练过程中损失曲线出现明显振荡。

为缓解 Non-IID 问题，后续研究提出了多种策略：

- **数据共享**（Yurochkin et al., 2019）：服务器维护一小部分全局共享数据，参与方在本地训练时混入共享数据以校正偏差。
- **FedProx**（Li et al., 2020）：在本地目标函数中添加近端项（proximal term），约束本地模型不要偏离全局模型过远：

$$
\min_{w} F_k(w) + \frac{\mu}{2}\|w - w^t\|^2
$$

其中 $\mu$ 是正则化系数，$w^t$ 是当前轮次的全局模型。近端项有效地抑制了客户端漂移，使每个客户端在本地优化时不会"走得太远"。

- **SCAFFOLD**（Karimireddy et al., 2020）：通过引入控制变量（control variates）来估计和修正客户端漂移方向，在理论上实现了与 IID 场景相当的收敛速率。

---

## 2.3 通信优化

通信开销是联邦学习实际部署中最关键的瓶颈。在典型的联邦学习系统中，模型参数可能包含数百万甚至数十亿个浮点数，而参与方的上行带宽往往十分有限。此外，频繁的通信同步还会带来延迟和能耗问题。因此，减少通信数据量和通信轮次是联邦学习研究的核心课题之一。

### 2.3.1 梯度压缩

梯度压缩（gradient compression）通过减少每轮传输的参数量来降低通信开销。主要方法包括：

**稀疏化（Sparsification）：** 仅传输梯度中幅度最大的若干分量。设梯度向量为 $g \in \mathbb{R}^d$，Top-$k$ 稀疏化保留绝对值最大的 $k$ 个分量，将其余分量置零：

$$
\text{TopK}(g)_i = \begin{cases} g_i, & \text{if } |g_i| \in \text{Top-}k\text{ of } \{|g_j|\}_{j=1}^d \\ 0, & \text{otherwise} \end{cases}
$$

稀疏化可将通信量压缩至 $k/d$ 的比例。Alistarh 等人（2017）和 Lin 等人（2018）分别提出了 Deep Gradient Compression 和误差反馈（error feedback）机制，将压缩导致的信息损失累积到下一轮梯度中，从而保证收敛性。误差反馈的核心公式为：

$$
g^{(t)} = \nabla F(w^{(t)}) + r^{(t-1)}, \quad r^{(t)} = g^{(t)} - \mathcal{C}(g^{(t)})
$$

其中 $\mathcal{C}(\cdot)$ 为压缩算子，$r^{(t)}$ 为残差，下一轮将残差叠加到新梯度上以补偿压缩损失。

**低秩分解（Low-rank Decomposition）：** 将梯度矩阵分解为低秩近似。设梯度矩阵 $G \in \mathbb{R}^{m \times n}$，其最优秩-$r$ 近似通过截断 SVD 获得：$G \approx U_r \Sigma_r V_r^\top$，仅需传输 $r(m+n+r)$ 个参数，远小于原始的 $mn$ 个。

### 2.3.2 量化通信

量化（quantization）将高精度浮点数映射为低比特表示，从而直接减少每个参数的传输比特数。

**随机量化（Stochastic Quantization）：** Alistarh 等人（2017）在 QSGD 算法中提出了随机量化方案。对于标量 $x$，将其量化为 $l$ 比特表示 $\mathcal{Q}_l(x)$：

$$
\mathcal{Q}_l(x) = \|x\| \cdot \text{sign}(x) \cdot \frac{s_i}{2^l}, \quad \text{以概率 } p_i = \frac{|x|}{\|x\|} \cdot 2^l - s_i
$$

其中 $s_i$ 是 $|x|/\|x\|$ 落入的量化区间下界。该方案保证 $\mathbb{E}[\mathcal{Q}_l(x)] = x$（无偏性），且方差随比特数 $l$ 的增加而减小。

**1-bit SGD：** Seide 等人（2014）提出了极端量化方案，将梯度每个分量仅用 1 比特表示（即符号位）：

$$
\mathcal{Q}_1(g)_i = \|g\| \cdot \text{sign}(g_i)
$$

结合误差反馈机制，1-bit SGD 在多个深度学习基准上取得了接近全精度训练的效果，同时将通信量压缩了 32 倍。

### 2.3.3 联邦平均与部分参与

FedAvg 算法本身即是一种通信优化策略。通过在每个参与方上执行多轮本地更新（$E > 1$），参与方在两次通信之间对本地数据进行充分学习，从而将通信轮次从 $O(n/B)$（纯 SGD 的通信需求）减少到 $O(n/(BEK))$。实验表明，在收敛到相同精度的前提下，FedAvg 所需的通信轮次仅为 FedSGD 的 $1/E$。

此外，每轮仅采样 $C \cdot K$ 个参与方参与训练（部分参与，partial participation），进一步降低了服务器的通信负担。McMahan 等人发现，当 $C$ 从 0.1 增加到 1.0 时，收敛速度提升有限，因此较小的 $C$（如 0.1）即可获得良好的训练效果。

### 2.3.4 通信轮次优化

除减少每轮通信量外，减少通信轮次本身也是重要方向。

**增加本地计算量：** 增大本地迭代次数 $E$ 是最直接的方式，但受限于客户端漂移效应。实践中可采用自适应策略：在训练初期使用较大的 $E$ 以快速降低损失，在训练后期减小 $E$ 以精细化全局模型。

**模型平均间隔调整：** Wang 和 Joshi（2018）提出了周期性平均（periodic averaging）策略，参与方每 $H$ 步才与服务器同步一次。该策略的通信-计算权衡可通过以下优化问题描述：

$$
\min_{H} \quad T_{\text{total}} = T_{\text{comm}} \cdot \lceil T_{\text{step}} / H \rceil + T_{\text{comp}} \cdot T_{\text{step}}
$$

其中 $T_{\text{comm}}$ 为单次通信延迟，$T_{\text{comp}}$ 为单步计算时间。增大 $H$ 减少通信次数但可能增加总收敛步数 $T_{\text{step}}$。

**层次化聚合：** Bonawitz 等人（2019）在 Google 的联邦学习部署中引入了层次化聚合架构。参与方先在边缘服务器进行局部聚合，再将聚合结果上传至中央服务器。这种树状结构将上行通信量从 $O(K)$ 降低到 $O(\text{层数})$，特别适合大规模移动设备场景。

---

## 2.4 异构性

联邦学习系统中的异构性是其区别于传统分布式学习的核心特征，也是最具挑战性的问题之一。异构性主要体现在三个层面：数据异构性、设备异构性以及由此引发的 Non-IID 问题。

### 2.4.1 数据异构性

数据异构性（data heterogeneity）指各参与方的数据在分布、规模和质量上存在差异。从统计学角度，数据异构性可通过以下指标量化。

**分布差异度量：** 设第 $k$ 和第 $j$ 个参与方的数据分布分别为 $P_k$ 和 $P_j$，常用的距离度量包括：

- **总变差距离（Total Variation Distance）：** $d_{\text{TV}}(P_k, P_j) = \frac{1}{2} \int |p_k(x) - p_j(x)| dx$
- **KL 散度：** $D_{\text{KL}}(P_k \| P_j) = \int p_k(x) \log \frac{p_k(x)}{p_j(x)} dx$
- **Wasserstein 距离：** $W_p(P_k, P_j) = \left(\inf_{\gamma \in \Gamma(P_k, P_j)} \int \|x - y\|^p d\gamma(x,y)\right)^{1/p}$

各参与方分布差异越大，联邦学习的全局模型越难以同时满足所有参与方的需求。

### 2.4.2 设备异构性

设备异构性（device heterogeneity）指参与方在计算能力、存储容量、网络带宽和在线时间等方面的差异。在移动设备联邦学习场景中，这种异构性尤为突出：

- **计算能力：** 高端智能手机的 GPU 性能可能是低端设备的 10 倍以上
- **网络条件：** WiFi 与蜂窝网络的带宽差异可达两个数量级
- **在线时间：** 设备仅在充电且空闲时才可能参与训练

设备异构性导致的**掉队者问题（straggler problem）**严重制约了同步联邦学习的效率。在同步协议下，每轮训练必须等待所有参与方完成本地计算，导致快速设备被慢速设备拖累。

**异步联邦学习**通过解耦参与方的更新节奏来缓解此问题。参与方完成本地训练后立即上传更新，无需等待其他参与方。但异步协议引入了**陈旧性（staleness）**问题：某些参与方的更新基于过时的全局模型。设第 $k$ 个参与方上传更新时全局模型已更新了 $\tau_k$ 轮，则陈旧度为 $\tau_k$。Xie 等人（2019）在异步 SGD 的分析中证明，收敛速率与最大陈旧度 $\tau_{\max}$ 相关：

$$
\text{Convergence Rate} \propto \mathcal{O}\left(\frac{1}{T} + \frac{\tau_{\max}}{T}\right)
$$

陈旧度过大会损害收敛质量，因此需要在效率与一致性之间寻求平衡。

### 2.4.3 Non-IID 数据

Non-IID（non-independent and non-identically distributed）数据是联邦学习中最广泛讨论的异构性问题。在现实中，Non-IID 的表现形式多样，Hsieh 等人（2020）在论文 *The Non-IID Data Quagmire of Decentralized Machine Learning* 中系统总结了以下几种类型：

1. **标签分布偏移（Label Distribution Shift）：** 各参与方的标签边际分布 $P_k(y)$ 不同。例如，不同地区的医院主要接诊不同类型的疾病。
2. **特征分布偏移（Feature Distribution Shift）：** 条件分布 $P_k(x|y)$ 不同。例如，不同语言环境下用户输入同一类别的文本特征截然不同。
3. **数量偏移（Quantity Skew）：** 各参与方的数据量差异悬殊，少数参与方可能持有绝大多数数据。
4. **概念漂移（Concept Drift）：** 相同特征对应不同标签，即 $P_k(y|x)$ 不同。例如，不同文化背景下对同一行为的分类标注可能存在差异。

Non-IID 数据对联邦学习的影响是多方面的。从优化角度看，各参与方的本地梯度方向差异增大，全局聚合后梯度的有效性降低。从模型角度看，全局模型可能偏向数据量大的参与方，对小数据参与方表现不佳。

为刻画 Non-IID 的程度，常用 **Earth Mover's Distance (EMD)** 来衡量各参与方标签分布与全局分布之间的差异：

$$
\text{EMD}_k = \min_{\gamma} \sum_{i,j} \gamma_{ij} |i - j|, \quad \text{s.t.} \sum_j \gamma_{ij} = p_k(i), \sum_i \gamma_{ij} = p_{\text{global}}(j)
$$

### 2.4.4 个性化联邦学习

传统联邦学习的目标是训练一个统一的全局模型。然而在高度异构的场景下，单一全局模型往往无法同时满足所有参与方的需求。**个性化联邦学习**（Personalized Federated Learning）旨在为每个参与方学习一个定制化的模型，同时利用其他参与方的知识来提升学习效果。

**基于元学习的方法：** Per-FedAvg（Fallah et al., 2020）将模型无关元学习（MAML）的思想引入联邦学习。全局模型被训练为一个良好的"初始化点"，使得每个参与方在该初始化点上进行少量本地更新即可获得个性化模型。具体地，Per-FedAvg 的双层优化目标为：

$$
\min_{w} \sum_{k=1}^{K} \frac{n_k}{n} F_k\left(w - \alpha \nabla F_k(w)\right)
$$

其中内层优化 $w - \alpha \nabla F_k(w)$ 表示参与方 $k$ 在全局模型上执行一步梯度下降后的个性化模型，外层优化则寻找使所有参与方个性化效果最佳的全局初始化。

**基于混合模型的方法：** FedPer（Arivazhagan et al., 2019）将神经网络分为基础层（base layers）和个性化层（personalization layers）。基础层在所有参与方间共享并通过联邦训练更新，个性化层仅在本地数据上训练。这种分层策略在深度神经网络中特别有效，因为底层特征通常具有较好的跨域迁移性。

**基于聚类的方法：** FeSEM（Ye et al., 2020）和 IFCA（Ghosh et al., 2020）通过聚类将数据分布相似的参与方分组，每组训练一个共享模型。IFCA 的算法框架为：每轮开始时，每个参与方评估所有聚类中心模型的本地损失，选择损失最小的聚类加入，然后在该聚类内进行本地更新。这种动态聚类机制可以自动发现参与方之间的隐含分组结构。

**基于知识蒸馏的方法：** FedDF（Lin et al., 2020）利用知识蒸馏将多个参与方的模型知识融合到全局模型中。服务器维护一个公共的未标记数据集（或生成的合成数据），各参与方将本地模型上传后，服务器利用各模型的输出分布作为"教师信号"，通过蒸馏损失训练全局"学生"模型：

$$
\mathcal{L}_{\text{distill}} = -\sum_{x \in \mathcal{D}_{\text{public}}} \sum_{k=1}^{K} \alpha_k \sum_{c} p_k^{(c)}(x) \log p_{\text{global}}^{(c)}(x)
$$

其中 $p_k(x)$ 是第 $k$ 个参与方模型对样本 $x$ 的输出概率分布，$\alpha_k$ 为聚合权重。

---

## 2.5 本章小结

本章系统介绍了联邦学习的基础理论与核心技术。FedAvg 作为联邦学习的奠基性算法，通过本地多轮更新与全局聚合的交替执行，在减少通信轮次的同时保持了模型精度。在收敛性方面，我们分析了 FedAvg 在 IID 和 Non-IID 数据下的理论保证，揭示了客户端漂移是影响 Non-IID 场景收敛的关键因素。

通信优化是联邦学习走向实用的关键。梯度压缩、量化通信和层次化聚合等技术从不同维度压缩了通信开销，使得在带宽受限的移动设备上部署联邦学习成为可能。

异构性是联邦学习区别于传统分布式学习的根本特征。数据异构性（Non-IID）和设备异构性相互交织，共同决定了联邦学习算法的设计空间。个性化联邦学习通过元学习、混合模型、聚类和知识蒸馏等技术，在全局知识共享和本地个性化之间寻找平衡，为异构场景下的联邦学习提供了灵活的解决方案。

联邦学习的这些基础技术为后续章节讨论隐私保护机制奠定了基础。在下一章中，我们将看到如何在联邦学习的框架上叠加密码学和安全计算技术，以实现更强的隐私保障。

---

## 参考文献

1. McMahan, H.B., Moore, E., Ramage, D., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
2. Li, T., Sahu, A.K., Zaheer, M., et al. (2020). Federated Optimization in Heterogeneous Networks. *Proceedings of the 3rd MLSys Conference*.
3. Karimireddy, S.P., Kale, S., Mohri, M., et al. (2020). SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. *Proceedings of the 37th International Conference on Machine Learning (ICML)*.
4. Li, X., Huang, K., Yang, W., et al. (2020). On the Convergence of FedAvg on Non-IID Data. *Proceedings of the 8th International Conference on Learning Representations (ICLR)*.
5. Alistarh, D., Grubic, D., Li, J., et al. (2017). QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding. *Advances in Neural Information Processing Systems (NeurIPS)*.
6. Lin, Y., Han, S., Mao, H., et al. (2018). Deep Gradient Compression: Reduce the Communication Bandwidth for Distributed Training. *Proceedings of the 6th International Conference on Learning Representations (ICLR)*.
7. Seide, F., Fu, H., Droppo, J., et al. (2014). 1-bit Stochastic Gradient Descent and Its Application to Data-Parallel Distributed Training of Speech DNNs. *Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH)*.
8. Wang, S. and Joshi, G. (2018). Cooperative SGD: A Unified Framework for the Design and Analysis of Communication-Efficient SGD Algorithms. *arXiv preprint arXiv:1808.07576*.
9. Bonawitz, K., Eichner, H., Grieskamp, W., et al. (2019). Towards Federated Learning at Scale: A System Design. *Proceedings of the 2nd MLSys Conference*.
10. Hsieh, K., Phanishayee, A., Mutlu, O., and Gibbons, P.B. (2020). The Non-IID Data Quagmire of Decentralized Machine Learning. *Proceedings of the 37th International Conference on Machine Learning (ICML)*.
11. Fallah, A., Mokhtari, A., and Ozdaglar, A. (2020). Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Approach. *Advances in Neural Information Processing Systems (NeurIPS)*.
12. Arivazhagan, M.G., Aggarwal, V., Singh, A.K., and Choudhary, S. (2019). Federated Learning with Personalization Layers. *arXiv preprint arXiv:1912.00818*.
13. Ghosh, A., Chung, J., Yin, D., and Ramchandran, K. (2020). An Efficient Framework for Clustered Federated Learning. *Advances in Neural Information Processing Systems (NeurIPS)*.
14. Lin, T., Kong, L., Stich, S.U., and Jaggi, M. (2020). Ensemble Distillation for Robust Model Fusion in Federated Learning. *Advances in Neural Information Processing Systems (NeurIPS)*.
15. Xie, C., Koyejo, S., and Gupta, I. (2019). Asynchronous Federated Optimization. *Proceedings of the 12th OPT Workshop on Optimization for Machine Learning*.
16. Yurochkin, M., Agarwal, M., Ghosh, S., et al. (2019). Bayesian Nonparametric Federated Learning of Neural Networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.
17. Ye, R., Ni, Z., Xu, F., et al. (2020). Personalized Federated Learning with Parameter Propagation. *arXiv preprint arXiv:2009.02219*.
# 第3章 差分隐私

> "差分隐私不是一个单一的算法，而是一个数学框架——它为隐私保护提供了可量化、可组合、可证明的保证。"
> —— Cynthia Dwork

## 3.1 引言：从直觉到形式化

在数据驱动的机器学习时代，一个根本性的矛盾始终存在：我们希望从数据中提取价值，同时又要保护数据中个体的隐私。传统的匿名化手段——如去除姓名、身份证号等显式标识符——已被反复证明是不充分的。2006年，Netflix举办了一场推荐算法竞赛，公开了"匿名化"的用户评分数据集。研究者Narayanan和Shmatikov（2008）通过将这些数据与IMDb上的公开评分进行关联，成功去匿名化了大量用户。这一事件深刻地揭示了一个事实：在高维数据面前，简单的匿名化不过是自欺欺人。

差分隐私（Differential Privacy, DP）正是在这样的背景下诞生的。Cynthia Dwork于2006年首次提出了差分隐私的严格数学定义（Dwork et al., 2006），为隐私保护建立了一个坚实的理论基础。差分隐私的核心思想可以概括为一句话：**任何一个个体的数据是否参与计算，对最终输出结果的影响应该微乎其微**。这意味着，即使攻击者拥有无限的计算能力和除目标个体以外的所有数据，也无法从算法输出中推断出目标个体的信息。

本章将系统地介绍差分隐私的理论基础和实际应用。我们从ε-差分隐私的数学定义出发，逐步深入到隐私预算的管理与组合定理；然后讨论差分隐私在深度学习中的核心应用——差分隐私随机梯度下降（DP-SGD）；接着介绍更为精细的隐私度量工具——Rényi差分隐私（RDP）；最后比较全局差分隐私与本地差分隐私两种模型的优劣与适用场景。

## 3.2 差分隐私的数学定义

### 3.2.1 ε-差分隐私

差分隐私的形式化定义建立在"相邻数据集"的概念之上。设 $D$ 和 $D'$ 为两个数据集，如果它们之间仅相差一条记录（即存在某个个体 $i$，使得 $D = D' \cup \{x_i\}$ 或 $D' = D \cup \{x_i\}$），则称 $D$ 与 $D'$ 是相邻数据集，记作 $D \sim D'$。

**定义 3.1（ε-差分隐私）** 一个随机化机制 $\mathcal{M}: \mathcal{D} \rightarrow \mathcal{R}$ 满足 ε-差分隐私（ε-DP），如果对于任意相邻数据集 $D \sim D'$ 和任意输出子集 $S \subseteq \mathcal{R}$，均有：

$$\Pr[\mathcal{M}(D) \in S] \leq e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S]$$

其中 $\varepsilon > 0$ 称为**隐私预算**（privacy budget）。

这个定义的直觉含义是：从输出的概率分布来看，我们无法以高于 $e^{\varepsilon}$ 的似然比来判断某条特定记录是否存在于数据集中。当 $\varepsilon$ 越小时，两个相邻数据集产生的输出分布越接近，隐私保护就越强。理想情况下，$\varepsilon$ 应该是一个很小的正数（例如 $\varepsilon = 0.1$ 或更小），但在实际应用中，为了维持数据的可用性，$\varepsilon$ 通常在 $1$ 到 $10$ 之间取值。

### 3.2.2 (ε, δ)-差分隐私

在某些场景下，纯粹的 ε-DP 可能过于严格，难以在隐私与效用之间取得平衡。为此，Dwork 和 Roth（2014）引入了松弛版本的差分隐私定义。

**定义 3.2（(ε, δ)-差分隐私）** 一个随机化机制 $\mathcal{M}$ 满足 (ε, δ)-差分隐私，如果对于任意相邻数据集 $D \sim D'$ 和任意输出子集 $S \subseteq \mathcal{R}$，均有：

$$\Pr[\mathcal{M}(D) \in S] \leq e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

其中 $\varepsilon > 0$ 为隐私预算，$\delta \in (0, 1)$ 为松弛参数。

参数 $\delta$ 允许机制以一个很小的概率 $\delta$ 违反纯 ε-DP 的保证。从概率意义上理解，这意味着以概率至少 $1 - \delta$，机制满足 ε-DP；而以概率至多 $\delta$，机制可能泄露额外的隐私信息。为了使 (ε, δ)-DP 具有意义，$\delta$ 必须远小于数据集中记录数量的倒数，即 $\delta \ll 1/n$。通常取 $\delta = 10^{-5}$ 或更小。

引入 $\delta$ 的实际意义在于：它允许使用高斯噪声而非拉普拉斯噪声来实现差分隐私，而高斯噪声在高维空间中具有更好的几何性质，这在深度学习的梯度扰动中尤为重要。

### 3.2.3 全局敏感度与噪声机制

要构造满足差分隐私的机制，关键在于如何校准噪声量。这取决于查询函数的**全局敏感度**。

**定义 3.3（全局敏感度）** 对于函数 $f: \mathcal{D} \rightarrow \mathbb{R}^d$，其全局敏感度定义为：

$$\Delta f = \max_{D \sim D'} \|f(D) - f(D')\|_p$$

其中 $\|\cdot\|_p$ 表示 $\ell_p$ 范数。全局敏感度刻画了单条记录的变化所能引起的函数输出的最大变化量。

基于全局敏感度，经典的噪声机制包括：

**拉普拉斯机制**：对于实值查询 $f: \mathcal{D} \rightarrow \mathbb{R}^d$，拉普拉斯机制定义为：

$$\mathcal{M}(D) = f(D) + (Y_1, Y_2, \ldots, Y_d)$$

其中 $Y_i \sim \text{Lap}(\Delta f / \varepsilon)$ 是独立同分布的拉普拉斯随机变量。该机制满足 ε-差分隐私。

**高斯机制**：对于查询 $f: \mathcal{D} \rightarrow \mathbb{R}^d$，高斯机制定义为：

$$\mathcal{M}(D) = f(D) + \mathcal{N}(0, \sigma^2 I_d)$$

其中噪声标准差 $\sigma = \frac{\Delta_2 f}{\varepsilon} \cdot \sqrt{2 \ln(1.25/\delta)}$，$\Delta_2 f$ 为 $\ell_2$ 全局敏感度。该机制满足 (ε, δ)-差分隐私。

在联邦学习的场景中，高斯机制更为常用，因为梯度是高维向量，高斯噪声的各向同性特性和良好的数学性质使其更适合于深度学习模型的训练过程。

### 3.2.4 隐私预算与组合定理

差分隐私的一个核心优势在于其**可组合性**（composability）。当一个数据集被多次查询，或者多个差分隐私机制被依次应用时，我们需要知道总的隐私损失是多少。组合定理为此提供了精确的分析工具。

**定理 3.1（基本组合定理）** 如果机制 $\mathcal{M}_1$ 满足 $\varepsilon_1$-DP，机制 $\mathcal{M}_2$ 满足 $\varepsilon_2$-DP，则它们的组合机制 $(\mathcal{M}_1, \mathcal{M}_2)$ 满足 $(\varepsilon_1 + \varepsilon_2)$-DP。

更一般地，$k$ 个分别满足 $\varepsilon_i$-DP 的机制的组合满足 $(\sum_{i=1}^k \varepsilon_i)$-DP。这个结果虽然简洁，但可能会导致隐私预算的线性累积，在多次查询的场景下变得过于保守。

**定理 3.2（高级组合定理）** 对于 $k$ 个机制，每个满足 $(\varepsilon, \delta)$-DP，它们的组合满足 $(\varepsilon', k\delta + \delta')$-DP，其中：

$$\varepsilon' = \varepsilon \sqrt{2k \ln(1/\delta')} + k\varepsilon(e^{\varepsilon} - 1)$$

高级组合定理表明，当每个机制的隐私参数 $\varepsilon$ 较小时，隐私预算以 $O(\sqrt{k})$ 而非 $O(k)$ 的速率增长。这个结果对深度学习中的差分隐私训练至关重要——在数百万次梯度更新中，$\sqrt{k}$ 的增长率远比 $k$ 的增长率更容易管理。

然而，高级组合定理给出的界仍然存在较大的松动。为了解决这个问题，Mironov（2017）提出了基于Rényi散度的Rényi差分隐私，我们将在3.4节详细讨论。

## 3.3 差分隐私随机梯度下降（DP-SGD）

### 3.3.1 从SGD到DP-SGD

将差分隐私应用于深度学习模型训练的核心挑战在于：标准的随机梯度下降（SGD）算法会在每次迭代中使用小批量数据计算梯度，而梯度本身可能泄露训练数据的信息。2016年，Abadi等人在具有里程碑意义的论文"Deep Learning with Differential Privacy"中提出了差分隐私随机梯度下降（DP-SGD）算法（Abadi et al., 2016），为深度学习与差分隐私的结合开辟了道路。

DP-SGD的核心思想是在标准SGD的基础上引入两个关键修改：**逐样本梯度裁剪**和**噪声注入**。

**算法 3.1（DP-SGD）**

**输入：** 数据集 $\mathcal{D} = \{x_1, \ldots, x_n\}$，损失函数 $\mathcal{L}(\theta, x)$，学习率 $\eta$，裁剪阈值 $C$，噪声乘子 $\sigma$，批量大小 $B$

**输出：** 模型参数 $\theta_T$

1. 随机初始化 $\theta_0$
2. **for** $t = 1$ **to** $T$ **do:**
3. &emsp;&emsp;随机采样小批量 $\mathcal{B}_t \subseteq \mathcal{D}$，每个样本以概率 $q = B/n$ 被独立选中
4. &emsp;&emsp;**for** 每个 $x_i \in \mathcal{B}_t$ **do:**
5. &emsp;&emsp;&emsp;&emsp;计算单样本梯度：$g_i = \nabla_{\theta} \mathcal{L}(\theta_{t-1}, x_i)$
6. &emsp;&emsp;&emsp;&emsp;裁剪梯度：$\bar{g}_i = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right)$
7. &emsp;&emsp;**end for**
8. &emsp;&emsp;添加噪声：$\tilde{g}_t = \frac{1}{B}\left(\sum_{i \in \mathcal{B}_t} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)\right)$
9. &emsp;&emsp;更新参数：$\theta_t = \theta_{t-1} - \eta \tilde{g}_t$
10. **end for**
11. **return** $\theta_T$

### 3.3.2 梯度裁剪

梯度裁剪是DP-SGD的第一个关键步骤。其目的是限制每个样本对总梯度的贡献，从而控制全局敏感度。

对于单样本梯度 $g_i = \nabla_{\theta} \mathcal{L}(\theta, x_i)$，裁剪操作定义为：

$$\bar{g}_i = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right)$$

其中 $C$ 是裁剪阈值。这个操作保证了 $\|\bar{g}_i\|_2 \leq C$，即每个样本的梯度范数不超过 $C$。这意味着，对于小批量梯度的求和 $\sum_{i \in \mathcal{B}} \bar{g}_i$，当且仅当一个小批量中增加或减少一个样本时，总梯度的变化量至多为 $C$，从而将 $\ell_2$ 全局敏感度限制为 $C$。

裁剪阈值 $C$ 的选择是一个关键的超参数。过小的 $C$ 会导致梯度信息的严重损失（大量梯度被截断），降低模型的收敛速度和最终精度；过大的 $C$ 则需要更大的噪声来保证隐私，同样会降低模型效用。实践中，通常通过以下方法确定 $C$：先在少量不参与隐私保护的"影子训练"中计算梯度范数的分位数，然后选择一个合适的分位数值作为裁剪阈值。

此外，Abadi等人（2016）还提出了**自适应裁剪**策略，在训练过程中动态调整裁剪阈值，以适应梯度范数分布的变化。

### 3.3.3 噪声添加

在裁剪梯度求和之后，DP-SGD注入高斯噪声：

$$\tilde{g}_t = \frac{1}{B}\left(\sum_{i \in \mathcal{B}_t} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)\right)$$

噪声的方差与裁剪阈值 $C$ 的平方成正比，与批量大小 $B$ 有关。噪声乘子 $\sigma$ 控制了噪声相对于信号的强度。根据高斯机制的理论，对于一次梯度更新，要满足 (ε, δ)-差分隐私，需要：

$$\sigma \geq \frac{\sqrt{2 \ln(1.25/\delta)}}{\varepsilon}$$

在实际训练中，这个下界仅适用于单次更新。对于 $T$ 次迭代的训练过程，需要通过隐私会计（privacy accounting）来确定每步所需的 $\sigma$。

噪声注入不可避免地会降低梯度的信噪比，影响模型的收敛性。为此，较大的批量大小 $B$ 可以通过分摊噪声来提高信噪比：$\text{SNR} \propto B / (\sigma C)$。这就是为什么DP-SGD通常需要使用比标准SGD更大的批量大小。

### 3.3.4 隐私会计（Privacy Accounting）

隐私会计是DP-SGD的核心组件之一，用于精确追踪训练过程中累积的隐私损失。其目标是：给定每步的噪声乘子 $\sigma$ 和采样概率 $q$，计算 $T$ 步训练后的总隐私参数。

Abadi等人（2016）提出了**矩会计法**（Moments Accountant），这是DP-SGD隐私分析的关键创新。矩会计法的核心思想是：与其直接追踪概率比 $e^{\varepsilon}$，不如追踪概率比的矩（moments），因为矩的组合性质更加优良。

具体而言，定义隐私随机变量的矩生成函数：

$$\alpha(\lambda) = \ln \mathbb{E}_{o \sim \mathcal{M}(D)} \left[ e^{\lambda \cdot \ln \frac{\Pr[\mathcal{M}(D) = o]}{\Pr[\mathcal{M}(D') = o]}} \right]$$

矩会计法的关键定理表明：

**定理 3.3（矩组合）** 如果机制 $\mathcal{M}_t$ 满足 $(\alpha(\lambda), \lambda)$-矩界对所有 $\lambda$ 成立，则 $k$ 个机制的组合满足 $(\sum_t \alpha_t(\lambda), \lambda)$-矩界。

这里的组合是矩的加法而非概率比的乘法，这使得分析更加紧致。通过将矩界转换回 (ε, δ)-DP 参数，矩会计法通常比基本组合定理和高级组合定理给出更紧的隐私界。

**定理 3.4（矩到 (ε, δ)-DP 的转换）** 如果机制 $\mathcal{M}$ 满足 $\alpha(\lambda) \leq \alpha_\mathcal{M}(\lambda)$ 对所有 $\lambda > 0$ 成立，则 $\mathcal{M}$ 满足 $(\varepsilon, \delta)$-DP，其中：

$$\delta = \min_{\lambda} \exp\left(\alpha_\mathcal{M}(\lambda) - \lambda\varepsilon\right)$$

在实际应用中，矩会计法已被广泛实现为开源工具，如Google的TensorFlow Privacy和Opacus（PyTorch差分隐私库）。这些工具可以在训练过程中实时计算累积的隐私损失，为实践者提供精确的隐私保证。

## 3.4 Rényi差分隐私（RDP）

### 3.4.1 从矩会计法到Rényi散度

矩会计法虽然有效，但其分析框架较为复杂，且不够简洁。Mironov（2017）在此基础上提出了**Rényi差分隐私**（Rényi Differential Privacy, RDP），利用Rényi散度（Rényi divergence）为差分隐私提供了更加优雅和紧致的分析框架。

**定义 3.4（Rényi散度）** 两个概率分布 $P$ 和 $Q$ 之间的 $\alpha$-阶Rényi散度定义为：

$$D_\alpha(P \| Q) = \frac{1}{\alpha - 1} \ln \mathbb{E}_{x \sim Q} \left[ \left(\frac{P(x)}{Q(x)}\right)^\alpha \right]$$

其中 $\alpha \in (0, 1) \cup (1, \infty)$。当 $\alpha \to 1$ 时，Rényi散度退化为KL散度；当 $\alpha \to \infty$ 时，它退化为最大散度（max-divergence）。

**定义 3.5（(α, ε)-Rényi差分隐私）** 一个随机化机制 $\mathcal{M}$ 满足 (α, ε)-Rényi差分隐私（(α, ε)-RDP），如果对于任意相邻数据集 $D \sim D'$：

$$D_\alpha(\mathcal{M}(D) \| \mathcal{M}(D')) \leq \varepsilon$$

### 3.4.2 RDP的优势

RDP相比传统 (ε, δ)-DP 具有多方面的优势：

**更紧的组合界。** RDP的组合定理特别简洁：

**定理 3.5（RDP组合）** 如果机制 $\mathcal{M}_1$ 满足 $(\alpha, \varepsilon_1)$-RDP，机制 $\mathcal{M}_2$ 满足 $(\alpha, \varepsilon_2)$-RDP，则它们的组合满足 $(\alpha, \varepsilon_1 + \varepsilon_2)$-RDP。

与基本组合定理的形式相同，但关键在于：从RDP转换回 (ε, δ)-DP 时，能够获得比直接使用高级组合定理更紧的界。这是因为Rényi散度对不同阶数 $\alpha$ 的同时约束提供了更丰富的信息。

**高斯机制的精确刻画。** 对于高斯机制，RDP参数有闭式表达。设 $\mathcal{M}(D) = f(D) + \mathcal{N}(0, \sigma^2 I)$，其中 $\Delta_2 f$ 为 $\ell_2$ 全局敏感度，则 $\mathcal{M}$ 满足 $(\alpha, \frac{\alpha \Delta_2^2 f}{2\sigma^2})$-RDP。

**子采样放大。** RDP框架下，子采样对隐私的放大效果有精确的分析。当每个样本以概率 $q$ 被独立采样时，如果原始机制 $\mathcal{M}$ 满足 $(\alpha, \varepsilon)$-RDP，则子采样后的机制满足 $(\alpha, \frac{1}{\alpha-1}\ln(1 + q^2(e^{(\alpha-1)\varepsilon} - 1)))$-RDP。这个结果在DP-SGD的隐私分析中至关重要，因为小批量采样本身就是一种子采样操作。

### 3.4.3 从RDP到 (ε, δ)-DP

在实际应用中，我们通常需要将RDP参数转换为更直观的 (ε, δ)-DP 参数。转换定理如下：

**定理 3.6** 如果机制 $\mathcal{M}$ 满足 $(\alpha, \varepsilon)$-RDP，则它也满足 $(\varepsilon + \frac{\ln(1/\delta)}{\alpha - 1}, \delta)$-DP，对任意 $\alpha > 1$ 和 $\delta > 0$。

为了获得最优的 (ε, δ)-DP 参数，我们对所有 $\alpha > 1$ 取最小值：

$$\varepsilon_{DP} = \min_{\alpha > 1} \left[ \varepsilon_{RDP}(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1} \right]$$

这个优化问题通常可以通过数值方法高效求解。

### 3.4.4 隐私追踪：RDP在DP-SGD中的应用

在DP-SGD的实际训练中，RDP提供了比原始矩会计法更高效的隐私追踪方案。完整的隐私分析流程如下：

1. **计算每步的RDP参数：** 考虑小批量采样（采样概率 $q = B/n$）和高斯噪声（噪声乘子 $\sigma$），每步满足 $(\alpha, \frac{q^2 \alpha}{2\sigma^2})$-RDP（在 $q$ 较小时的近似）。

2. **组合 $T$ 步：** 总的RDP参数为 $(\alpha, T \cdot \frac{q^2 \alpha}{2\sigma^2})$-RDP。

3. **转换为 (ε, δ)-DP：** 通过定理3.6进行转换。

这种分析方法不仅更加紧致，而且计算效率更高。在实践中，它通常比矩会计法给出的隐私界紧10%至50%，这意味着在相同的隐私保证下，可以使用更小的噪声，从而获得更好的模型精度。

## 3.5 本地差分隐私与全局差分隐私

### 3.5.1 全局差分隐私（GDP）

前面讨论的差分隐私模型属于**全局差分隐私**（Global Differential Privacy, GDP）的范畴。在GDP模型中，存在一个可信的**数据收集者**（curator）或**聚合服务器**，它能够访问所有用户的原始数据，并在计算过程中添加噪声以保护隐私。

在联邦学习的背景下，GDP模型对应于以下场景：用户的原始数据保存在本地设备上不被上传，但各设备将模型更新（如梯度）发送给中央服务器，服务器在聚合这些更新时添加噪声。这种模式通常称为**可信服务器模型**。

GDP的优势在于噪声量与用户数量 $n$ 无关，而与查询的敏感度相关。对于聚合查询（如求平均），通过将噪声分摊到 $n$ 个用户的贡献中，可以实现较高的数据效用。然而，GDP的核心假设——可信的中央服务器——在许多实际场景中难以满足。

### 3.5.2 本地差分隐私（LDP）

**本地差分隐私**（Local Differential Privacy, LDP）消除了对可信第三方的依赖。在LDP模型中，每个用户在将自己的数据发送出去之前，独立地对数据进行随机化处理。

**定义 3.6（本地差分隐私）** 一个随机化机制 $\mathcal{M}$ 满足 ε-本地差分隐私（ε-LDP），如果对于任意单个记录 $x, x' \in \mathcal{X}$ 和任意输出子集 $S \subseteq \mathcal{R}$：

$$\Pr[\mathcal{M}(x) \in S] \leq e^{\varepsilon} \cdot \Pr[\mathcal{M}(x') \in S]$$

注意，LDP的定义中比较的是任意两条记录，而非相邻数据集。这意味着LDP的保护更强——即使攻击者拥有某个用户扰动后的数据，也无法以较高的置信度推断该用户的原始数据。

LDP的经典实现包括**随机响应**（Randomized Response）和**RAPPOR**（Erlingsson et al., 2014）。以最简单的二元随机响应为例：用户以概率 $p = \frac{e^\varepsilon}{e^\varepsilon + 1}$ 如实回答，以概率 $1 - p$ 翻转回答。这种机制满足 ε-LDP。

### 3.5.3 GDP与LDP的对比分析

GDP与LDP之间的根本差异在于**信任模型**和**噪声分配策略**。下面我们从多个维度进行对比分析。

**隐私保证的强度。** LDP提供了更强的个体隐私保证——用户的数据在离开设备之前就已经被扰动，即使服务器是恶意的也无法获取原始数据。GDP则依赖于服务器的可信性，一旦服务器被攻破或恶意，隐私保证即刻失效。

**数据效用。** GDP在效用方面具有显著优势。考虑估计一个二元统计量（如拥有某属性的用户比例）：在GDP下，添加方差为 $O(1/\varepsilon^2)$ 的噪声即可满足 ε-DP；而在LDP下，为了估计 $n$ 个用户的比例，需要 $n = O(1/\varepsilon^2)$ 的样本量才能获得有意义的结果——也就是说，每个用户的信息量被严重稀释。更精确地，LDP下估计量的方差为 $O(e^\varepsilon / (n(e^\varepsilon - 1)^2))$，当 $\varepsilon$ 较小时，这比GDP的方差大了 $O(n)$ 倍。

**通信效率。** 在联邦学习场景中，LDP要求每个用户在上传前扰动其模型更新，这可能导致更高的通信成本（例如需要更大的向量来编码随机化后的信息）。GDP则可以在服务器端高效地添加噪声，通信开销较低。

**适用场景。** LDP适用于对隐私要求极高、但对数据精度要求相对宽松的场景，如大公司的用户行为统计（Google的Chrome使用统计、Apple的emoji使用频率统计等）。GDP则适用于需要精确模型训练的场景，如医疗数据分析和金融风控。

### 3.5.4 联邦学习中的隐私模型选择

在联邦学习中，GDP与LDP的选择取决于具体的安全假设和应用需求：

**交叉设备联邦学习**（Cross-device FL）中，参与训练的设备通常不可靠（可能掉线、被攻破），LDP或分布式差分隐私（Distributed DP）更为合适。在分布式DP模型下，各设备独立添加噪声，服务器仅接收扰动后的更新，无需信任任何单一实体。

**交叉孤岛联邦学习**（Cross-silo FL）中，参与方通常是可信的机构（如医院、银行），GDP模型更为实用。各方将加密或扰动后的更新发送给聚合服务器，通过安全聚合（Secure Aggregation）进一步增强隐私保护。

实践中，研究者提出了多种混合方案。例如，**分布式差分隐私**（Agarwal et al., 2018）结合了LDP和GDP的优点：各设备添加一部分噪声，服务器在聚合后再添加少量噪声，从而在不完全信任服务器的前提下实现接近GDP的效用。

## 3.6 差分隐私的实践挑战与前沿进展

### 3.6.1 隐私-效用权衡

差分隐私的核心挑战在于**隐私-效用权衡**（privacy-utility tradeoff）。增加隐私保护（减小 $\varepsilon$）必然导致更大的噪声，从而降低数据的可用性或模型的精度。在深度学习中，这种权衡表现得尤为明显。

研究表明（Abadi et al., 2016; McMahan et al., 2018），在MNIST和CIFAR-10等基准数据集上，DP-SGD可以实现有意义的隐私保证（$\varepsilon$ 在几到几十之间），但代价是模型精度的显著下降。在更大规模的模型和数据集上，如大型语言模型的微调，隐私-效用权衡面临更大的挑战。

为了缓解这一问题，研究者们提出了多种改进策略：
- **预训练+微调范式**：先在公开数据上预训练，再用DP-SGD在私有数据上微调，显著降低隐私成本（Yu et al., 2022）。
- **梯度压缩和稀疏化**：减少每步需要扰动的参数数量。
- **个性化差分隐私**：为不同用户设置不同的隐私预算，适应异构的隐私需求。

### 3.6.2 差分隐私与安全聚合的协同

在联邦学习中，差分隐私常与安全聚合（Secure Aggregation）协议协同使用。安全聚合保证服务器只能获得聚合后的模型更新，而无法看到任何单个用户的更新。结合差分隐私，可以实现更强的保护：安全聚合防止服务器窥探单个用户，差分隐私则保护聚合结果中不泄露个体信息。

这种协同设计的理论基础是**分布式差分隐私**，其核心思想是将噪声添加的责任分布到各个参与者。每个用户 $i$ 添加噪声 $\mathcal{N}(0, \sigma_1^2 I)$，服务器在聚合结果上再添加噪声 $\mathcal{N}(0, \sigma_2^2 I)$，总的等效噪声方差为 $\sigma_1^2/n + \sigma_2^2$。通过适当分配 $\sigma_1$ 和 $\sigma_2$，可以在隐私和效用之间取得最优平衡。

### 3.6.3 前沿进展

差分隐私领域的研究仍在快速发展。近期的重要进展包括：

**f-差分隐私**：Mironov等人（2019）提出了基于一般f-散度的差分隐私框架，统一了包括RDP在内的多种隐私度量。

**隐私放大**：通过数据子采样、模型压缩等技术进一步放大隐私保证，使得在相同的噪声水平下实现更强的隐私保护。

**差分隐私的可解释性**：研究者正在探索如何将ε等隐私参数与用户的直觉理解对齐，使非专业用户也能理解隐私保证的含义。

**自适应裁剪策略**：最新的研究提出了更加精细的梯度裁剪方法，如基于分位数的自适应裁剪和基于梯度分布的噪声校准，以减少裁剪对模型精度的影响。

## 3.7 本章小结

差分隐私为隐私保护提供了坚实的数学基础和可证明的隐私保证。本章从ε-DP的严格定义出发，介绍了隐私预算的概念和组合定理，然后深入讨论了DP-SGD算法的核心组件——梯度裁剪、噪声注入和隐私会计，接着引入了更为精细的Rényi差分隐私框架，最后比较了本地差分隐私与全局差分隐私两种模型的优劣与适用场景。

差分隐私的核心思想——以可控的随机化换取可证明的隐私保护——不仅为联邦学习中的隐私保护提供了理论支撑，更深刻地影响了我们对"数据隐私"这一概念的理解。在接下来的章节中，我们将看到差分隐私如何与其他隐私保护技术（如安全多方计算和同态加密）相结合，共同构建完整的隐私计算体系。

## 参考文献

1. Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. *Theory of Cryptography Conference (TCC)*, 265-284.
2. Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3-4), 211-407.
3. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. *ACM SIGSAC Conference on Computer and Communications Security (CCS)*, 308-318.
4. Mironov, I. (2017). Rényi differential privacy. *IEEE Computer Security Foundations Symposium (CSF)*, 263-275.
5. McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning differentially private recurrent language models. *International Conference on Learning Representations (ICLR)*.
6. Erlingsson, Ú., Pihur, V., & Korolova, A. (2014). RAPPOR: Randomized aggregatable privacy-preserving ordinal response. *ACM SIGSAC Conference on Computer and Communications Security (CCS)*, 1054-1067.
7. Narayanan, A., & Shmatikov, V. (2008). Robust de-anonymization of large sparse datasets. *IEEE Symposium on Security and Privacy*, 111-125.
8. Agarwal, N., Suresh, A. T., Yu, F. X. X., Kumar, S., & McMahan, B. (2018). cpSGD: Communication-efficient and differentially-private distributed SGD. *Advances in Neural Information Processing Systems (NeurIPS)*.
9. Yu, D., Naik, S., Backurs, A., Gopi, S., Inan, H. A., Kamath, G., ... & Zhang, H. (2022). Differentially private fine-tuning of language models. *International Conference on Learning Representations (ICLR)*.
10. Mironov, I., Talwar, K., & Zhang, L. (2019). Rényi differential privacy of the sampled Gaussian mechanism. *arXiv preprint arXiv:1908.10530*.
# 第4章 安全多方计算

## 4.1 概述

安全多方计算（Secure Multi-Party Computation，MPC）是密码学中一个基础而深刻的理论框架，其核心目标是：在多个参与方各自持有私有输入的前提下，协同计算某一约定函数的输出，同时保证任何一方都无法获知其他方的私有输入信息。这一问题的提出可以追溯到Andrew Yao在1986年提出的"百万富翁问题"（Millionaires' Problem）：两个百万富翁希望比较谁更富有，但不愿透露各自的财产数额 [Yao, 1986]。

形式化地，假设有 $n$ 个参与方 $P_1, P_2, \ldots, P_n$，各自持有私有数据 $x_1, x_2, \ldots, x_n$，他们希望共同计算函数 $f(x_1, x_2, \ldots, x_n) = y$，使得：

1. **正确性**（Correctness）：计算结果 $y$ 与明文计算 $f(x_1, \ldots, x_n)$ 一致；
2. **隐私性**（Privacy）：除输出 $y$ 所隐含的信息外，任何参与方（或参与方的合谋集合）无法获得关于其他方输入的额外信息。

MPC协议的安全性通常在半诚实模型（Semi-Honest Model）或恶意模型（Malicious Model）下进行分析。半诚实模型假设参与方忠实地执行协议，但可能试图从接收到的消息中推断他人隐私；恶意模型则允许参与方任意偏离协议，是更强的安全保障。在实际应用中，可以通过引入零知识证明、消息认证码（MAC）等技术，将半诚实安全协议提升为恶意安全协议。

本章将系统介绍MPC的三大核心技术路线：秘密共享（Secret Sharing）、混淆电路（Garbled Circuit），以及隐私集合交集（Private Set Intersection, PSI），并探讨MPC在机器学习安全训练与推理中的前沿应用。

---

## 4.2 秘密共享

秘密共享（Secret Sharing）是MPC最基础的构建模块之一。其核心思想是将一个秘密值拆分为多个"份额"（shares），分发给不同的参与方，使得只有满足特定条件的份额集合才能恢复原始秘密。

### 4.2.1 Shamir秘密共享方案

Shamir秘密共享方案（Shamir's Secret Sharing, SSS）由Adi Shamir于1979年提出 [Shamir, 1979]，基于多项式插值原理，是一个 $(t, n)$ 门限方案：将秘密 $s$ 拆分为 $n$ 个份额，任意 $t$ 个或更多份额可以恢复 $s$，而 $t-1$ 个或更少份额无法获得关于 $s$ 的任何信息。

**方案构造：**

设秘密 $s \in \mathbb{F}_p$（$p$ 为大素数），分发者执行以下步骤：

1. 随机选择 $t-1$ 个系数 $a_1, a_2, \ldots, a_{t-1} \xleftarrow{\$} \mathbb{F}_p$，构造 $t-1$ 次多项式：

$$
f(x) = s + a_1 x + a_2 x^2 + \cdots + a_{t-1} x^{t-1} \pmod{p}
$$

2. 对于参与方 $P_i$（$i = 1, 2, \ldots, n$），计算并分发份额：

$$
s_i = f(i) = s + a_1 i + a_2 i^2 + \cdots + a_{t-1} i^{t-1} \pmod{p}
$$

**秘密恢复：**

任意 $t$ 个参与方 $\{P_{i_1}, P_{i_2}, \ldots, P_{i_t}\}$ 可以使用拉格朗日插值公式恢复 $f(x)$，进而得到 $f(0) = s$：

$$
s = f(0) = \sum_{j=1}^{t} s_{i_j} \cdot \lambda_{i_j} \pmod{p}
$$

其中拉格朗日基函数为：

$$
\lambda_{i_j} = \prod_{\substack{k=1 \\ k \neq j}}^{t} \frac{i_k}{i_k - i_{i_j}} \pmod{p}
$$

**安全性证明：** Shamir方案的信息论安全性源于一个简洁的代数事实：对于 $\mathbb{F}_p$ 上的 $t-1$ 次多项式 $f(x)$，给定任意 $t-1$ 个点 $(i_1, f(i_1)), \ldots, (i_{t-1}, f(i_{t-1}))$，$f(0)$ 的值在 $\mathbb{F}_p$ 上均匀分布。换言之，$t-1$ 个份额在统计上独立于秘密 $s$，满足信息论安全。

### 4.2.2 加法秘密共享

加法秘密共享（Additive Secret Sharing）是Shamir方案在 $t = n$ 时的特例，也是实际MPC协议中最常用的秘密共享方式之一。

**方案构造：**

设秘密 $s \in \mathbb{Z}_N$，分发者随机选择 $n-1$ 个份额 $s_1, s_2, \ldots, s_{n-1} \xleftarrow{\$} \mathbb{Z}_N$，并计算：

$$
s_n = s - \sum_{i=1}^{n-1} s_i \pmod{N}
$$

将 $s_i$ 分发给 $P_i$。显然，所有份额之和等于秘密：

$$
\sum_{i=1}^{n} s_i = s \pmod{N}
$$

**加法同态性质：** 加法秘密共享天然支持安全加法运算。若 $P_i$ 持有关于 $a$ 的份额 $\langle a \rangle_i$ 和关于 $b$ 的份额 $\langle b \rangle_i$，则 $c_i = \langle a \rangle_i + \langle b \rangle_i$ 构成 $a + b$ 的有效份额，因为：

$$
\sum_{i=1}^{n} c_i = \sum_{i=1}^{n} (\langle a \rangle_i + \langle b \rangle_i) = a + b \pmod{N}
$$

这一性质使得多方之间的加法运算无需任何通信，只需本地计算。

### 4.2.3 门限秘密共享与主动安全

在实际系统中，参与方可能随时加入或退出，甚至被攻陷。门限秘密共享方案的鲁棒性（Robustness）要求即使部分参与方提供错误份额，诚实方仍能正确恢复秘密。

**Pedersen承诺增强：** 一种经典方法是利用Pedersen承诺对份额进行验证。分发者在分发份额 $s_i = f(i)$ 的同时，广播对多项式系数的承诺：

$$
C_j = g^{a_j} h^{r_j} \pmod{q}, \quad j = 0, 1, \ldots, t-1
$$

其中 $a_0 = s$，$g, h$ 为 $\mathbb{Z}_q^*$ 的生成元且 $\log_g h$ 未知。参与方 $P_i$ 可以验证其份额的正确性：

$$
g^{s_i} h^{r(i)} = \prod_{j=0}^{t-1} C_j^{i^j} \pmod{q}
$$

其中 $r(i)$ 是分发者提供的随机性。这保证了所有诚实方持有的份额来自同一多项式。

**可验证秘密共享（Verifiable Secret Sharing, VSS）** 是构建主动安全MPC协议的关键组件。在主动安全模型中，即使攻击者在不同阶段攻陷不同的参与方集合（总数不超过门限），协议仍能保证安全性。这通过定期的"刷新"（Proactive Refresh）机制实现——各方协作对现有份额进行随机化重分发，使得旧份额失效但秘密不变。

---

## 4.3 混淆电路

混淆电路（Garbled Circuit, GC）是MPC的另一核心范式，最初由Andrew Yao在其开创性论文中提出 [Yao, 1986]，用于安全两方计算。其核心思想是将待计算的函数表示为布尔电路，然后对电路中的每条线路进行加密"混淆"，使得计算可以在加密状态下进行。

### 4.3.1 Yao的混淆电路协议

**基本协议流程：**

考虑两方场景：Alice持有输入 $x$，Bob持有输入 $y$，他们希望计算 $f(x, y)$。

**混淆阶段（Alice执行）：**

1. 将函数 $f$ 编译为布尔电路 $C$，包含 $m$ 条线路 $w_1, w_2, \ldots, w_m$。每条线路 $w_j$ 有两个可能的值 $\{0, 1\}$。

2. 对每条线路 $w_j$，随机选择两个"线标签"（wire labels）$k_j^0, k_j^1 \xleftarrow{\$} \{0,1\}^\kappa$（$\kappa$ 为安全参数），其中 $k_j^b$ 对应线路值 $b \in \{0,1\}$。

3. 对于每个逻辑门 $g$（以二输入门为例，输入线路为 $w_a, w_b$，输出线路为 $w_c$），构造混淆真值表。对于 $b_1, b_2 \in \{0,1\}$，$c = g(b_1, b_2)$，计算：

$$
E_{k_a^{b_1}}(E_{k_b^{b_2}}(k_c^c))
$$

即用两个输入线标签对输出线标签进行双重加密。四个明文-密文对经过随机置换后构成混淆表（Garbled Table）$T_g$。

4. 将所有门的混淆表 $\{T_g\}$ 发送给Bob。

**求值阶段（Bob执行）：**

1. Alice将自己的输入线标签 $\{k_j^{x_j}\}$ 发送给Bob（使用不经意传输协议，见下文）。
2. Bob通过不经意传输获得自己的输入线标签 $\{k_j^{y_j}\}$。
3. Bob逐层解密混淆表，从输入层到输出层，最终得到输出线路的线标签。
4. 将输出线标签与Alice提供的解码表比对，得到明文结果 $f(x, y)$。

**不经意传输（Oblivious Transfer, OT）：** Yao协议的安全性依赖于不经意传输原语。在1-out-of-2 OT中，Alice拥有两个消息 $m_0, m_1$，Bob拥有选择位 $b \in \{0,1\}$。协议结束后，Bob获得 $m_b$，但不知道 $m_{1-b}$；Alice不知道 $b$ 的值。OT可以基于DDH假设或RSA假设构造，是许多MPC协议的密码学基础。

### 4.3.2 Free XOR优化

标准Yao协议中，每个门（包括XOR门）都需要一条混淆表。然而，XOR门在布尔电路中占比通常很高。Kolesnikov和Schneider在2008年提出了Free XOR优化 [Kolesnikov & Schneider, 2008]，使得XOR门的计算完全免费——不需要传输任何混淆表，也不需要加密/解密操作。

**核心思想：**

引入全局偏移量 $\Delta \xleftarrow{\$} \{0,1\}^\kappa$（固定且对Bob保密），使得每条线路的两个线标签满足：

$$
k_j^1 = k_j^0 \oplus \Delta
$$

对于XOR门 $w_c = w_a \oplus w_b$，若Bob持有 $k_a^{b_1}$ 和 $k_b^{b_2}$，则：

$$
k_a^{b_1} \oplus k_b^{b_2} = k_a^0 \oplus (b_1 \cdot \Delta) \oplus k_b^0 \oplus (b_2 \cdot \Delta) = (k_a^0 \oplus k_b^0) \oplus ((b_1 \oplus b_2) \cdot \Delta)
$$

令 $k_c^0 = k_a^0 \oplus k_b^0$，则上式恰好等于 $k_c^{b_1 \oplus b_2}$。Bob只需执行一次异或运算即可得到正确的输出线标签，无需查表。

**与半门（Half-Gates）技术的结合：** 进一步地，Zahur、Rosulek和Evans在2015年提出了半门技术 [Zahur et al., 2015]，将AND门的混淆表从4行压缩为2行（每条输入线路贡献一个"半门"），使得AND门仅需传输 $2\kappa$ 比特（约256比特，当 $\kappa = 128$ 时）。结合Free XOR，现代混淆电路协议已接近理论最优。

### 4.3.3 混淆电路的复杂度分析

设布尔电路 $C$ 有 $N_{\text{AND}}$ 个AND门和 $N_{\text{XOR}}$ 个XOR门：

- **通信量**：Free XOR方案下，通信量仅与AND门数量成正比，约为 $2\kappa \cdot N_{\text{AND}}$ 比特（半门优化后）。
- **计算量**：混淆阶段（Alice）需要 $O(N_{\text{AND}})$ 次加密操作；求值阶段（Bob）需要 $O(N_{\text{AND}})$ 次解密操作。XOR门几乎无开销。
- **轮次**：电路求值是逐层进行的，轮次等于电路深度，通常为 $O(\log n)$ 或更优。

---

## 4.4 隐私集合交集

隐私集合交集（Private Set Intersection, PSI）是MPC的一个重要特例：两方或多方各自持有一个私有集合，希望计算集合的交集，而不泄露任何交集以外的信息。PSI在数据对齐、反欺诈检测、隐私保护广告归因等场景中有着广泛的应用。

### 4.4.1 PSI协议的基本形式

设Alice持有集合 $X = \{x_1, x_2, \ldots, x_m\}$，Bob持有集合 $Y = \{y_1, y_2, \ldots, y_n\}$。PSI协议的目标是让Alice（或双方）获得 $X \cap Y$，但：

- Alice不能获知 $Y \setminus X$ 中的元素；
- Bob不能获知 $X \setminus Y$ 中的元素。

### 4.4.2 基于不经意伪随机函数的PSI

不经意伪随机函数（Oblivious Pseudorandom Function, OPRF）是高效PSI协议的核心构建块。OPRF允许一方（Bob）在不知道输入的情况下，帮助另一方（Alice）对其集合中的每个元素计算一个伪随机函数值，同时双方都不会泄露各自的私有信息。

**基本OPRF协议（基于Diffie-Hellman）：**

1. Bob选择私钥 $k \xleftarrow{\$} \mathbb{Z}_q$，计算公钥 $K = g^k$ 并发送给Alice。
2. 对于Alice集合中的每个元素 $x_i$：
   - Alice选择随机盲化因子 $r_i \xleftarrow{\$} \mathbb{Z}_q$，计算 $H(x_i)^{r_i}$（$H$ 为哈希函数，映射到群 $\mathbb{G}$），发送给Bob。
   - Bob计算 $(H(x_i)^{r_i})^k = H(x_i)^{r_i k}$，返回给Alice。
   - Alice解除盲化：$(H(x_i)^{r_i k})^{r_i^{-1}} = H(x_i)^k$。

3. Alice获得 $\{H(x_i)^k\}_{i=1}^m$；Bob独立计算 $\{H(y_j)^k\}_{j=1}^n$。通过比较哈希值即可确定交集。

该协议的安全性基于计算性Diffie-Hellman（CDH）假设：Alice无法从 $H(x_i)^k$ 推断 $k$，Bob无法从 $H(x_i)^{r_i}$ 推断 $H(x_i)$。

### 4.4.3 高效PSI方案

随着PSI在工业界的大规模应用，研究者提出了多种优化方案，以处理百万甚至亿元素级别的集合。

**基于布谷鸟哈希的PSI：** Pinkas等人在2015年提出了基于布谷鸟哈希（Cuckoo Hashing）的PSI方案 [Pinkas et al., 2015]。核心思想是：

1. Bob使用布谷鸟哈希将自己的集合 $Y$ 存储到一个哈希表 $T$ 中（桶数 $b$ 为 $O(n)$，每个桶存一个元素）。
2. 对于Alice的每个元素 $x_i$，根据布谷鸟哈希的多个哈希函数，确定 $x_i$ 可能落入的桶位置（通常为3个）。
3. Alice对每个桶位置使用OPRF进行查询，检查是否与Bob的元素匹配。

该方案将OPRF调用次数从 $O(mn)$ 降低到 $O(mb)$（$b \approx 1.2n$），显著提高了效率。

**基于电路的PSI：** 对于需要计算交集大小（而非交集元素本身）的场景，可以使用混淆电路或秘密共享直接计算集合交集的基数。Rindal和Schoppmann在2021年提出的VOLE-based PSI方案 [Rindal & Schoppmann, 2021]，利用向量不经意线性求值（Vector Oblivious Linear Evaluation, VOLE）将通信量降低到每元素 $O(\kappa)$ 比特的水平。

**多方PSI：** 在 $n > 2$ 方参与的场景中，多方PSI（Multi-Party PSI）需要处理更复杂的安全性需求。一种典型方法是让每方依次作为"服务端"运行两方PSI协议，然后取交集的交集。更高效的方案则利用多方秘密共享直接在共享状态下比较元素。

---

## 4.5 安全多方计算在机器学习中的应用

随着隐私保护机器学习需求的激增，MPC技术正被广泛应用于安全训练和安全推理场景。本节介绍MPC在机器学习中的三个核心应用方向。

### 4.5.1 安全聚合

安全聚合（Secure Aggregation）是联邦学习中最基础的MPC应用场景。其目标是让多个参与方将各自的模型更新（梯度）安全地聚合为全局更新，而服务器无法获知任何单个参与方的更新内容。

**基于加法秘密共享的安全聚合：** Google在2017年提出的Secure Aggregation协议 [Bonawitz et al., 2017] 是这一方向的经典工作。设 $n$ 个参与方各自持有梯度向量 $\mathbf{g}_i \in \mathbb{R}^d$，协议目标是计算 $\mathbf{G} = \sum_{i=1}^{n} \mathbf{g}_i$。

基本流程如下：

1. **成对掩码生成**：每对参与方 $(P_i, P_j)$ 通过Diffie-Hellman密钥协商生成共享密钥 $s_{ij}$，并用伪随机数生成器产生掩码向量 $\mathbf{r}_{ij} = \text{PRG}(s_{ij})$。

2. **掩码上传**：每个 $P_i$ 上传被掩码的梯度：

$$
\mathbf{u}_i = \mathbf{g}_i + \sum_{j > i} \mathbf{r}_{ij} - \sum_{j < i} \mathbf{r}_{ij} \pmod{N}
$$

3. **聚合**：服务器计算：

$$
\sum_{i=1}^{n} \mathbf{u}_i = \sum_{i=1}^{n} \mathbf{g}_i + \underbrace{\sum_{i=1}^{n}\left(\sum_{j > i} \mathbf{r}_{ij} - \sum_{j < i} \mathbf{r}_{ij}\right)}_{= \mathbf{0}} = \mathbf{G} \pmod{N}
$$

掩码在求和过程中相互抵消，服务器获得正确的聚合结果，但无法解出任何单个 $\mathbf{g}_i$。

该协议的通信复杂度为 $O(n^2 \cdot d)$（成对密钥协商），Bonawitz等人进一步通过秘密共享和Shamir门限方案将其优化为 $O(n \cdot d)$，并支持参与方中途退出。

### 4.5.2 安全训练

安全多方计算训练（Secure MPC Training）旨在使多个数据持有方在不暴露各自训练数据的前提下，协作训练机器学习模型。与联邦学习中的安全聚合不同，安全MPC训练要求在计算图层面进行安全保护，而非仅保护梯度的聚合结果。

**基于秘密共享的安全训练：** 以安全逻辑回归为例。设两方分别持有数据矩阵 $\mathbf{X} \in \mathbb{R}^{m_1 \times d}$ 和 $\mathbf{Y} \in \mathbb{R}^{m_2 \times d}$，标签向量 $\mathbf{l}_1 \in \{0,1\}^{m_1}$ 和 $\mathbf{l}_2 \in \{0,1\}^{m_2}$。目标是在不泄露 $\mathbf{X}, \mathbf{Y}, \mathbf{l}_1, \mathbf{l}_2$ 的前提下训练逻辑回归模型 $\mathbf{w}$。

1. **数据共享**：各方将数据以加法秘密共享方式分发，即 $\langle \mathbf{X} \rangle_1 + \langle \mathbf{X} \rangle_2 = \mathbf{X}$。

2. **安全前向传播**：计算预测值 $\hat{y} = \sigma(\mathbf{X}\mathbf{w})$，其中 $\sigma$ 为sigmoid函数。由于sigmoid是非线性函数，需要使用安全多项式近似或查表技术：

$$
\sigma(z) \approx 0.5 + 0.25z - \frac{1}{48}z^3 + \cdots
$$

在秘密共享下，多项式求值仅需安全乘法（Beaver三元组）和安全加法。

3. **安全反向传播**：梯度计算涉及矩阵乘法，同样通过Beaver三元组协议实现。Beaver三元组的核心思想是：预先生成随机三元组 $(\langle a \rangle, \langle b \rangle, \langle c \rangle)$ 满足 $c = a \cdot b$，然后通过"开值-重构"技术安全地计算 $\langle x \cdot y \rangle$：

   - 各方本地计算 $\langle e \rangle = \langle x \rangle - \langle a \rangle$ 和 $\langle d \rangle = \langle y \rangle - \langle b \rangle$。
   - 重构（公开）$e = \sum \langle e \rangle_i$ 和 $d = \sum \langle d \rangle_i$。
   - 各方本地计算：$\langle x \cdot y \rangle = \langle c \rangle + e \cdot \langle b \rangle + d \cdot \langle a \rangle + e \cdot d$（对于两方协议，其中一方额外加上 $e \cdot d$）。

4. **模型更新**：各方安全地更新 $\langle \mathbf{w} \rangle$ 并迭代，直至收敛。

**性能挑战与优化：** 安全训练的主要瓶颈在于非线性函数（激活函数、softmax等）的安全计算。常用优化手段包括：

- **定点算术**：使用定点数代替浮点数，避免昂贵的安全除法。
- **多项式逼近**：用低阶多项式近似非线性函数，减少乘法深度。
- **预处理阶段**：离线生成大量Beaver三元组，在线阶段仅需轻量级操作。
- **GPU加速**：利用GPU并行处理大规模矩阵运算，显著提升训练速度。

ABY3框架 [Mohassel & Rindal, 2018] 和SPDZ系列协议代表了安全训练领域的最新进展，已在ImageNet规模的数据集上实现了可用的训练效率。

### 4.5.3 安全推理

安全推理（Secure Inference）指在模型已训练完成的前提下，保护用户输入数据和模型参数的隐私。相比安全训练，安全推理的计算量较小，但对延迟敏感（通常要求实时响应）。

**基于混淆电路的安全推理：** MiniONN [Liu et al., 2017] 是早期将混淆电路应用于神经网络推理的代表性工作。其核心思想是：

1. **模型拆分**：将神经网络模型参数以加法秘密共享方式分布在两台非共谋服务器上。
2. **线性层安全计算**：对于全连接层 $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$，用户输入 $\mathbf{x}$ 以秘密共享方式发送给两台服务器，各方计算各自的份额并重组。
3. **非线性层安全计算**：对于ReLU、池化等非线性操作，使用混淆电路或混合协议（Hybrid Protocol）进行安全计算。

**混合协议架构：** 现代安全推理系统通常采用"秘密共享 + 混淆电路"的混合架构：

- 线性运算（矩阵乘法、卷积）在加法秘密共享下高效完成（利用加法同态性质）；
- 非线性运算（ReLU、Argmax、比较）通过混淆电路或不经意传输完成。

这种混合架构的典型代表是CrypTen（Facebook）和Cheetah [Huang et al., 2022]，它们在ImageNet预训练模型上的推理延迟已降至秒级。

**隐私集合交集在机器学习中的应用：** PSI技术在隐私保护机器学习中也有重要应用。例如，在跨机构的特征工程中，各方需要找到拥有相同用户ID的记录以对齐特征，但不能泄露不匹配的用户信息。PSI协议可以安全地完成这一数据对齐步骤，是隐私保护数据协作的关键前置环节。

---

## 4.6 本章小结

安全多方计算作为隐私计算的核心技术之一，提供了信息论或计算安全性保证下的多方协同计算能力。本章介绍了MPC的三大技术支柱：

- **秘密共享**以其简洁的数学结构和天然的同态性质，成为安全计算的基础框架，Shamir方案和加法共享是构建高效MPC协议的基石；
- **混淆电路**将任意函数的安全计算归约为布尔电路的加密求值，Free XOR和半门等优化使其在通信效率上接近理论极限；
- **隐私集合交集**是MPC在数据协作场景中的典型应用，OPRF和布谷鸟哈希等技术使其能够处理大规模数据集。

在机器学习领域，MPC正从理论走向实践。安全聚合已成为联邦学习的标配组件，安全训练在中小规模模型上已具备实用性，安全推理则在混合协议架构的推动下逐步接近实时化。未来，随着硬件加速（如专用MPC芯片）、新型密码学原语（如同态加密与MPC的融合）以及协议优化的持续推进，MPC有望在更广泛的数据安全与AI应用场景中发挥关键作用。

---

## 参考文献

1. Yao, A.C. (1986). How to Generate and Exchange Secrets. *Proceedings of the 27th Annual Symposium on Foundations of Computer Science (FOCS)*, 162-167.
2. Shamir, A. (1979). How to Share a Secret. *Communications of the ACM*, 22(11), 612-613.
3. Goldreich, O., Micali, S., & Wigderson, A. (1987). How to Play any Mental Game. *Proceedings of the 19th Annual ACM Symposium on Theory of Computing (STOC)*, 218-229.
4. Kolesnikov, V., & Schneider, T. (2008). Improved Garbled Circuit: Free XOR Gates and Applications. *Proceedings of ICALP*, 486-498.
5. Zahur, S., Rosulek, M., & Evans, D. (2015). Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits Using Half Gates. *Proceedings of EUROCRYPT*, 220-250.
6. Pinkas, B., Schneider, T., & Zohner, M. (2015). Faster Private Set Intersection Based on OT Extension. *Proceedings of USENIX Security*, 797-812.
7. Rindal, P., & Schoppmann, P. (2021). VOLE-PSI: Fast OPRF and Circuit-PSI from Vector-OLE. *Proceedings of EUROCRYPT*, 901-930.
8. Bonawitz, K., et al. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. *Proceedings of ACM CCS*, 1175-1191.
9. Mohassel, P., & Rindal, P. (2018). ABY3: A Mixed Protocol Framework for Machine Learning. *Proceedings of ACM CCS*, 35-52.
10. Liu, J., et al. (2017). Oblivious Neural Network Predictions via MiniONN Transformations. *Proceedings of ACM CCS*, 619-631.
11. Huang, Z., et al. (2022). Cheetah: Lean and Fast Secure Two-Party Deep Neural Network Inference. *Proceedings of USENIX Security*, 4481-4498.
12. Beaver, D. (1991). Efficient Multiparty Protocols Using Circuit Randomization. *Proceedings of CRYPTO*, 420-432.
13. Damgård, I., Pastro, V., Smart, N., & Zakarias, S. (2012). Multiparty Computation from Somewhat Homomorphic Encryption. *Proceedings of CRYPTO*, 643-662.
# 第5章 同态加密

## 5.1 引言

在隐私计算的技术谱系中，同态加密（Homomorphic Encryption, HE）占据着独特而核心的地位。它赋予了计算者一种看似矛盾的能力——在不解密数据的前提下，直接对密文执行运算，并使运算结果在解密后与对明文执行相同运算的结果一致。这一特性使得同态加密成为构建隐私保护机器学习系统的关键密码学工具。

同态加密的思想可追溯至1978年Rivest、Adleman和Dertouzos提出的"隐私同态"（Privacy Homomorphism）概念[1]。然而，长期以来，学术界仅能构造出满足部分同态性质的加密方案——要么支持加法同态，要么支持乘法同态，二者不可兼得。直到2009年，Craig Gentry在其博士论文中基于理想格（Ideal Lattices）首次构造出全同态加密（Fully Homomorphic Encryption, FHE）方案[2]，这一里程碑式的突破彻底改变了密码学和隐私计算的格局。

本章将系统阐述同态加密的理论基础、核心方案及其在机器学习中的应用。我们将从全同态加密的数学定义出发，深入分析BGV、BFV和CKKS三大主流方案的设计思想与技术特点，继而探讨同态加密在加密推理、加密训练等场景中的应用实践，最后讨论当前面临的主要性能挑战以及硬件加速等前沿方向。

## 5.2 全同态加密的基本原理

### 5.2.1 形式化定义

一个同态加密方案由四个概率多项式时间算法组成：

$$\mathcal{HE} = (\mathsf{KeyGen}, \mathsf{Enc}, \mathsf{Dec}, \mathsf{Eval})$$

各算法的定义如下：

- **密钥生成** $\mathsf{KeyGen}(1^{\lambda}) \rightarrow (pk, sk, evk)$：输入安全参数 $\lambda$，输出公钥 $pk$、私钥 $sk$ 和评估密钥 $evk$。
- **加密** $\mathsf{Enc}(pk, m) \rightarrow c$：使用公钥将明文 $m \in \mathcal{M}$ 加密为密文 $c \in \mathcal{C}$。
- **解密** $\mathsf{Dec}(sk, c) \rightarrow m$：使用私钥将密文 $c$ 解密为明文 $m$。
- **同态评估** $\mathsf{Eval}(evk, f, c_1, \ldots, c_t) \rightarrow c_f$：对密文 $c_1, \ldots, c_t$ 执行由电路 $f$ 描述的运算，输出结果密文 $c_f$。

一个同态加密方案需要满足以下核心性质：

**正确性**（Correctness）：对任意合法的明文 $m_1, \ldots, m_t$ 和可计算函数 $f$，有

$$\mathsf{Dec}(sk, \mathsf{Eval}(evk, f, \mathsf{Enc}(pk, m_1), \ldots, \mathsf{Enc}(pk, m_t))) = f(m_1, \ldots, m_t)$$

**语义安全性**（Semantic Security）：即密文在计算上不可区分（IND-CPA安全），对于任意两个等长明文 $m_0, m_1$，其密文 $\mathsf{Enc}(pk, m_0)$ 与 $\mathsf{Enc}(pk, m_1)$ 在多项式时间内不可区分。

根据方案支持的运算类型和深度，同态加密可进一步分为三个层次：

1. **部分同态加密**（Partially Homomorphic Encryption, PHE）：仅支持加法或乘法中的一种运算。典型代表包括支持加法同态的Paillier方案和支持乘法同态的RSA方案。
2. **有限层次全同态加密**（Somewhat/Leveled Homomorphic Encryption, SHE/LHE）：支持有限深度 $L$ 的加法和乘法运算，即可以评估深度不超过 $L$ 的算术电路。
3. **全同态加密**（Fully Homomorphic Encryption, FHE）：支持任意深度的加法和乘法运算，理论上可以评估任意可计算函数。

### 5.2.2 格密码学基础

现代FHE方案几乎全部构建在格（Lattice）密码学的困难问题之上。一个 $n$ 维格 $\Lambda$ 是 $\mathbb{R}^n$ 的一个离散加法子群，由一组线性无关的向量（基）$\mathbf{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ 生成：

$$\Lambda = \mathcal{L}(\mathbf{B}) = \left\{ \sum_{i=1}^{n} z_i \mathbf{b}_i \;\middle|\; z_i \in \mathbb{Z} \right\}$$

格上两个核心的计算困难问题是FHE安全性的理论基石：

**最短向量问题**（Shortest Vector Problem, SVP）：给定格 $\Lambda$ 的一组基，寻找格中最短的非零向量 $\mathbf{v} \in \Lambda$，使得 $\|\mathbf{v}\|$ 最小。近似版本 $\gamma$-SVP 寻找长度不超过 $\gamma \cdot \lambda_1(\Lambda)$ 的非零格向量，其中 $\lambda_1(\Lambda)$ 为最短向量长度。

**带误差学习问题**（Learning With Errors, LWE）：由Regev于2005年提出[3]。给定均匀随机矩阵 $\mathbf{A} \in \mathbb{Z}_q^{m \times n}$、秘密向量 $\mathbf{s} \in \mathbb{Z}_q^n$ 以及误差向量 $\mathbf{e} \in \mathbb{Z}_q^m$（各分量从离散高斯分布 $\chi$ 中采样），定义

$$\mathbf{b} = \mathbf{A}\mathbf{s} + \mathbf{e} \pmod{q}$$

LWE问题要求在已知 $(\mathbf{A}, \mathbf{b})$ 的条件下恢复 $\mathbf{s}$。Regev证明了在量子归约下，LWE问题至少与格上最坏情况的GapSVP问题一样困难。

为提升效率，Lyubashevsky、Peikert和Regev进一步提出了**环LWE**（Ring-LWE）问题[4]。设 $R = \mathbb{Z}[x]/(\Phi_n(x))$ 为多项式环（通常取分圆多项式 $\Phi_n(x) = x^n + 1$，其中 $n$ 为2的幂），$R_q = R/qR$。Ring-LWE问题中，采样 $a \leftarrow R_q$ 均匀随机，$s, e \leftarrow \chi$，给定 $(a, b = a \cdot s + e \in R_q)$，恢复 $s$。Ring-LWE将向量运算转化为多项式运算，使得密钥尺寸和运算复杂度从 $O(n^2)$ 降低至 $O(n \log n)$（利用数论变换NTT），极大地提高了FHE的实用性。

### 5.2.3 密文计算与噪声管理

基于LWE/Ring-LWE的加密方案的一个关键特征是密文中包含"噪声"项。以Ring-LWE为例，加密明文 $m \in R_t$（$t < q$）的基本方式为：

$$\mathsf{Enc}(pk, m) = (c_0, c_1) = (a \cdot s + e + \lfloor \frac{q}{t} \rceil \cdot m, -a)$$

其中 $e$ 为小噪声。解密时计算 $c_0 + c_1 \cdot s = \lfloor \frac{q}{t} \rceil \cdot m + e$，当噪声 $e$ 足够小时，可通过舍入恢复 $m$。

同态运算会导致噪声增长：

- **同态加法**：两个密文 $(c_0, c_1)$ 和 $(c_0', c_1')$ 对应的噪声近似为 $e + e'$，即线性增长。
- **同态乘法**：乘法运算后噪声增长更为剧烈，近似为 $e \cdot e'$ 的量级，涉及多项式的乘法和模约减，增长速度远超加法。

噪声的快速增长是FHE面临的核心挑战：一旦噪声超过阈值 $\lfloor q/2t \rfloor$，解密将产生错误结果。因此，必须在噪声"爆炸"之前采取措施加以控制。主要的噪声管理技术包括：

**模切换**（Modulus Switching）：在每次乘法运算后，将密文从模 $q$ 缩减至更小的模 $q'$（满足 $q'/q \approx 1/t$），同时等比例缩小噪声。这一技术由Brakerski和Vaikuntanathan提出[5]，可将乘法带来的噪声增长从二次级降低为线性级，使得在不执行Bootstrapping的情况下支持更多层次的运算。

**重线性化**（Relinearization）：同态乘法产生的密文从二维扩展为三维（或更高维），通过乘以重线性化密钥 $evk$ 将其重新压缩为二维，以保持密文尺寸可控。

### 5.2.4 Bootstrapping：从有限层次到全同态

Gentry在其奠基性工作中提出了**Bootstrapping**（自举）技术[2]，这是从有限层次同态加密跨越到全同态加密的关键理论突破。

Bootstrapping的核心思想极为精妙：假设一个同态加密方案 $\mathcal{HE}$ 能够同态地评估自身的解密电路，即可以对"加密了密文的密文"进行同态解密。具体地，给定：

- 密文 $c = \mathsf{Enc}(pk, m)$，其噪声水平较高；
- 私钥的加密 $\widetilde{sk} = \mathsf{Enc}(pk', sk)$，使用另一组密钥 $(pk', sk')$ 加密。

则Bootstrapping操作为：

$$\mathsf{Bootstrapping}(c, \widetilde{sk}) = \mathsf{Eval}(evk, \mathsf{Dec}_{sk}(\cdot), \widetilde{sk}, c)$$

由于同态评估的结果是对 $m$ 使用新密钥 $pk'$ 加密的密文，其噪声仅取决于新鲜加密的噪声水平，与原始密文 $c$ 的噪声无关。因此，Bootstrapping实现了"噪声重置"，将任意噪声水平的密文转化为新鲜密文。

**定理 5.1**（Gentry Bootstrapping定理）：若一个层次化同态加密方案 $\mathcal{HE}$ 能够同态地评估自身的解密电路（即该方案是"自举可压缩的"），则可以通过反复应用Bootstrapping将其提升为全同态加密方案。

然而，原始Gentry方案的Bootstrapping效率极低——同态评估一次解密电路本身就需要相当深度的同态运算，而这些运算又会产生新的噪声，形成一个"鸡生蛋"的困境。Gentry通过"稀疏子集求和"（Sparse Subset Sum）技术和维度模切换（Dimension Modulus Switching）技术部分解决了这一问题，但效率仍远远不能满足实际需求。

后续的研究工作在两个方向上取得了重大进展：

1. **无需Bootstrapping的方案**：通过模切换技术（如BGV方案）或尺度不变技术（如BFV方案），在参数选择时预先设定足够的乘法深度 $L$，使得在该深度内无需执行Bootstrapping即可完成目标计算。
2. **高效的Bootstrapping**：CKKS方案利用近似计算的特性，设计了高效的近似Bootstrapping算法[6]，将Bootstrapping的时间开销降低了数个数量级。

## 5.3 主流全同态加密方案

### 5.3.1 BGV方案

BGV方案由Brakerski、Gentry和Vaikuntanathan于2012年提出[7]，是第一个无需Bootstrapping即可支持深层电路计算的层次化全同态加密方案。其核心创新在于将模切换技术与密钥切换技术系统化地结合。

BGV方案在多项式环 $R = \mathbb{Z}[x]/(x^n + 1)$ 上工作。设明文模数为 $t$（通常为小素数），密文模数为 $q$。BGV的密文形式为 $c = (c_0, c_1) \in R_q^2$，满足

$$c_0 + c_1 \cdot s \equiv m \pmod{t}$$

其中 $s \in R_q$ 为私钥，且等式在模 $q$ 下成立的前提是噪声 $e = c_0 + c_1 \cdot s - m$ 的系数均小于 $q/(2t)$。

BGV的关键设计在于**模切换链**（Modulus Switching Chain）。预先选定一系列递减的模数 $q_L > q_{L-1} > \cdots > q_1 > q_0$，其中 $L$ 为最大乘法深度。每当执行一次同态乘法后，通过模切换将密文从模 $q_\ell$ 缩减至模 $q_{\ell-1}$：

$$\mathsf{ModSwitch}(c, q_\ell \rightarrow q_{\ell-1}) = \left\lfloor \frac{q_{\ell-1}}{q_\ell} \cdot c \right\rceil$$

这一操作将噪声等比例缩小，从而为下一次乘法运算腾出"噪声预算"。通过精心选择模链参数，BGV方案可以在不执行Bootstrapping的情况下支持 $L$ 层乘法运算。

BGV方案还支持**批处理**（Batching）技术：利用中国剩余定理（CRT），将多个明文 $m_1, \ldots, m_k \in R_t$ 打包到同一个密文中，实现SIMD（单指令多数据）并行运算。当 $t$ 分裂为 $k$ 个理想时，批处理的槽数 $k$ 可达 $n$，极大地提高了运算吞吐量。

### 5.3.2 BFV方案

BFV方案（Fan-Vercauteren方案）由Fan和Vercauteren于2012年提出[8]，基于Brakerski的尺度不变（Scale-Invariant）加密框架。与BGV通过模切换管理噪声不同，BFV采用了一种截然不同的噪声管理策略。

BFV的加密过程为：

$$\mathsf{Enc}(pk, m) = (c_0, c_1) = \left( \left\lfloor \frac{q}{t} \right\rfloor \cdot m + p \cdot u \cdot a + e_0, \; p \cdot u \cdot b + e_1 \right)$$

其中 $(a, b) = pk$ 为公钥，$u, e_0, e_1$ 为从噪声分布中采样的小多项式。

BFV的核心思想是：密文中明文被编码为 $\lfloor q/t \rfloor \cdot m$，噪声叠加在此之上。同态运算在密文空间 $R_q$ 中直接进行，运算完成后通过一个特殊的"明文化"（Plaintext Space Adjustment）操作将结果正确解码。

对于同态乘法，BFV首先计算密文向量的张量积，然后通过**重线性化**将三维结果压缩回二维。乘法后的密文形式为

$$c_{\times} = \mathsf{Decompose}(c_0 \cdot c_0', c_0 \cdot c_1' + c_1 \cdot c_0', c_1 \cdot c_1')$$

随后通过乘以评估密钥 $evk$ 执行重线性化。

BFV与BGV的一个重要区别在于：BFV不需要在每次乘法后执行模切换，其噪声管理是"内置"在方案结构中的。然而，BFV的噪声增长曲线与BGV不同，在某些场景下BGV的噪声管理更为精细。

### 5.3.3 CKKS方案

CKKS方案（Cheon-Kim-Kim-Song方案）于2017年由Cheon等人提出[6]，是全同态加密领域的一个范式转换性工作。与BGV和BFV追求精确整数运算不同，CKKS从设计之初就面向**近似浮点运算**，使其天然适合机器学习、信号处理和科学计算等需要处理实数/复数的场景。

**编码与解码。** CKKS将明文空间定义为复数向量空间 $\mathbb{C}^{N/2}$（其中 $N = 2n$ 为多项式环的维度）。编码过程利用**近似正则嵌入**（Approximate Canonical Embedding）将复数向量映射到多项式环：

$$\mathsf{Encode}(z_1, \ldots, z_{N/2}) = \sigma^{-1}(\Delta \cdot (z_1, \ldots, z_{N/2})) + \delta$$

其中 $\sigma: R \rightarrow \mathbb{C}^{N/2}$ 为复数嵌入映射（将多项式在分圆域的各共轭根处求值），$\Delta = 2^p$ 为缩放因子（scaling factor），控制精度，$\delta$ 为舍入误差。解码过程则为编码的逆操作：

$$\mathsf{Decode}(m) = \frac{1}{\Delta} \cdot \sigma(m)$$

**同态运算。** CKKS的同态运算直接作用于缩放后的多项式。对于两个编码为 $(m_1, \Delta_1)$ 和 $(m_2, \Delta_2)$ 的密文：

- 同态加法：$m_1 + m_2$，缩放因子为 $\Delta_1 = \Delta_2$（要求对齐）。
- 同态乘法：$m_1 \cdot m_2$，缩放因子变为 $\Delta_1 \cdot \Delta_2$，需要通过**重缩放**（Rescaling）操作将其重新调整为 $\Delta$。

重缩放操作是CKKS方案的核心机制，本质上是对密文除以一个适当的模数：

$$\mathsf{Rescale}(c, \Delta) = \left\lfloor \frac{q_\ell}{\Delta} \cdot c \right\rceil \pmod{q_{\ell-1}}$$

这等价于BGV中的模切换，但在CKKS的语境下同时承担了"恢复缩放因子"的功能。

**近似Bootstrapping。** CKKS方案最具创新性的贡献之一是其高效的Bootstrapping算法[9]。传统FHE的Bootstrapping要求精确地同态评估解密电路，而CKKS利用其固有的近似性质，设计了一种"近似Bootstrapping"——其目标不是精确恢复原始密文，而是在控制精度损失的前提下重置噪声水平。

CKKS的Bootstrapping主要步骤包括：

1. **模提升**（Modulus Raising）：将密文的模数从较小的 $q$ 提升到较大的 $Q$。
2. **同态解密**：在模 $Q$ 下同态地执行解密运算，利用多项式逼近技术（如正弦函数逼近取模运算）实现。
3. **编码与重缩放**：将结果重新编码并重缩放回标准形式。

通过这些步骤，CKKS实现了噪声重置，同时保持了对浮点数运算的支持。2021年，Lee等人进一步优化了CKKS的Bootstrapping，将其时间开销从分钟级降低至秒级[10]。

### 5.3.4 方案对比与选型指南

表5.1总结了三大主流FHE方案的关键特性对比。

| 特性 | BGV | BFV | CKKS |
|------|-----|-----|------|
| **明文空间** | $R_t$（整数/有限域） | $R_t$（整数/有限域） | $\mathbb{C}^{N/2}$（复数近似） |
| **运算精度** | 精确 | 精确 | 近似（可控精度） |
| **噪声管理** | 模切换链 | 尺度不变 + 重线性化 | 重缩放 + 近似Bootstrapping |
| **批处理支持** | 是（CRT分解） | 是（CRT分解） | 是（固有SIMD） |
| **Bootstrapping** | 支持（开销大） | 支持（开销大） | 高效近似Bootstrapping |
| **典型应用** | 整数运算、数据库查询 | 联邦学习中的梯度聚合 | 机器学习推理与训练 |
| **代表实现** | HElib, OpenFHE | Microsoft SEAL, OpenFHE | Lattigo, Microsoft SEAL, OpenFHE |

在联邦学习与隐私计算的场景下，方案选型的关键考虑因素包括：

- 若需要对整数型梯度进行精确聚合，BGV或BFV是更合适的选择。
- 若涉及浮点数运算（如神经网络的权重和激活值），CKKS因其天然支持近似浮点运算而成为首选。
- 若计算电路深度有限（如浅层模型的推理），可选择层次化方案（leveled）以避免昂贵的Bootstrapping。
- 若计算深度不可预知或需要支持迭代计算，则需要具备Bootstrapping能力的方案。

## 5.4 同态加密在机器学习中的应用

### 5.4.1 加密推理

加密推理（Encrypted Inference）是同态加密在机器学习中最直接的应用场景。其基本模式为：模型持有者将训练好的模型以明文形式保留在服务端，数据持有者将输入数据加密后发送给服务端，服务端在密文上执行推理计算，将加密的推理结果返回给数据持有者解密。

以一个简单的前馈神经网络为例，其前向传播包含以下运算：

**线性层**：$\mathbf{z} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}$，其中 $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 为权重矩阵，$\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$ 为输入，$\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}$ 为偏置。在CKKS方案下，矩阵-向量乘法可以通过同态乘法和同态加法直接实现：

$$\mathsf{Enc}(\mathbf{z}) = \mathsf{Eval}(\mathbf{W}, \mathsf{Enc}(\mathbf{x})) + \mathsf{Enc}(\mathbf{b})$$

利用CKKS的SIMD特性，可以将 $\mathbf{W}$ 的多行同时编码到一个密文中，通过旋转（Rotation）操作实现列访问，从而高效完成矩阵乘法。

**激活函数**：非线性激活函数（如ReLU、Sigmoid、Tanh）是加密推理的主要障碍。由于同态加密仅支持加法和乘法，无法直接实现比较操作（如ReLU中的 $x > 0$ 判断），因此必须用多项式来近似激活函数。

常用的近似策略包括：

- **Taylor展开**：将激活函数展开为多项式，例如 $\sigma(x) \approx \frac{1}{2} + \frac{1}{4}x - \frac{1}{48}x^3$（Sigmoid的3阶近似）。
- **最小二乘多项式拟合**：在特定区间 $[a, b]$ 上，求解最优多项式系数使逼近误差最小。
- **分段多项式逼近**：将输入空间划分为多个区间，在每个区间上使用不同的多项式逼近。
- **Chebyshev多项式逼近**：利用Chebyshev多项式的极小极大性质，获得在整个区间上误差均匀分布的逼近。

**卷积层**：卷积运算本质上是加权求和，可以通过将卷积核展开为矩阵，转化为矩阵-向量乘法来实现同态计算。

**批归一化**（Batch Normalization）：在推理模式下，批归一化等价于线性变换 $\hat{x} = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$，可以预计算缩放和偏移参数，转化为一次同态乘法和一次同态加法。

加密推理的代表性系统包括：

- **CryptoNets**（Dowlin等，2016）[11]：首个端到端加密神经网络推理系统，在MNIST数据集上使用BFV方案实现了99%以上的推理准确率，但单张图片的推理时间约为570秒。
- **GAZELLE**（Juvekar等，2018）[12]：结合同态加密和混淆电路（Garbled Circuits），将加密推理的延迟降低了两个数量级以上。
- **CrypTFlow2**（Rathee等，2020）[13]：进一步优化了安全推理协议，在ImageNet规模的模型上实现了实用级别的推理速度。

### 5.4.2 加密训练

相比加密推理，加密训练（Encrypted Training）面临着更大的技术挑战。训练过程涉及前向传播、损失计算、反向传播和参数更新四个阶段，每个阶段都包含大量的非线性运算，且需要迭代多轮。

加密训练的基本流程如下：

1. **数据加密**：多个数据持有者各自加密本地数据集 $\mathsf{Enc}(\mathbf{X}_i)$，并将其发送给计算服务器。
2. **前向传播**：服务器在密文上执行前向传播，计算加密的预测值 $\mathsf{Enc}(\hat{\mathbf{y}})$。
3. **损失计算**：由于损失函数通常涉及非线性运算（如交叉熵中的对数），需要通过多项式近似实现：$\mathsf{Enc}(\mathcal{L}) \approx \mathsf{Eval}(\tilde{\mathcal{L}}, \mathsf{Enc}(\hat{\mathbf{y}}), \mathsf{Enc}(\mathbf{y}))$。
4. **反向传播**：利用链式法则在密文上计算梯度。对于多项式近似的激活函数，其导数仍为多项式，可以在密文上计算。
5. **参数更新**：梯度 $\mathsf{Enc}(\nabla \mathbf{W})$ 需要被解密或通过安全聚合协议处理后，用于更新模型参数。

加密训练面临的核心挑战包括：

**乘法深度累积**：前向传播 $L$ 层需要至少 $L$ 次乘法，反向传播同样需要约 $L$ 次乘法。若使用 $K$ 阶多项式近似激活函数，乘法深度增加为 $K \cdot L$。加上损失函数的计算深度，总乘法深度可能达到 $2KL + O(1)$，这对层次化方案的参数选择提出了很高的要求。

**迭代训练的开销**：典型训练需要数十到数百个epoch，每个epoch处理数千到数万个样本。若每个样本的加密推理需要数秒，则完整训练可能需要数天甚至数周。

为应对这些挑战，研究者提出了多种优化策略：

- **Mini-batch聚合**：利用SIMD特性，将一个mini-batch中的多个样本打包到同一个密文中并行处理，显著提高吞吐量。
- **混合精度训练**：对不同层使用不同精度的编码，在不显著影响模型收敛性的前提下减少乘法深度。
- **交互式训练协议**：设计多方参与的训练协议，各方分别负责部分计算，减少单方的计算负担。

**SecureML**（Mohassel和Zhang，2017）[14]是加密训练的代表性工作，它结合了秘密共享和同态加密，实现了逻辑回归和支持向量机的安全训练。后续的**ABY3**（Mohassel和Rindal，2018）[15]进一步优化了三方安全计算协议，支持更复杂的机器学习模型训练。

### 5.4.3 隐私保护神经网络

构建实用的隐私保护神经网络系统需要在安全性、准确性和效率之间寻找平衡点。当前的研究路线主要沿三个方向展开：

**纯同态加密方案**：整个推理/训练过程完全在密文上执行，不涉及任何明文交互。这种方案的安全性最强（计算服务器看不到任何明文信息），但效率最低。代表系统包括CryptoNets和Binarized Neural Networks（BNN）的加密版本。BNN将权重和激活值二值化（取值为 $\pm 1$），使得乘法退化为同态加法/减法，极大地降低了计算深度和开销。

**混合协议方案**：将同态加密与安全多方计算（MPC）技术相结合。典型架构为：

1. 利用HE处理计算密集的线性运算（矩阵乘法、卷积等）。
2. 利用MPC（如秘密共享或混淆电路）处理非线性运算（激活函数、池化等）。

这种分工充分利用了两种技术的优势：HE擅长批量线性运算，MPC擅长门级运算。MiniONN（Liu等，2017）[16]和Chameleon（Riazi等，2018）[17]是这一方向的代表性工作。

**可信硬件辅助方案**：利用可信执行环境（TEE，如Intel SGX、ARM TrustZone）执行部分敏感计算，将HE的计算负担转移至可信硬件中。这种方案的效率较高，但引入了对硬件安全性的信任假设。

**同态加密与联邦学习的结合**：在联邦学习场景下，同态加密主要用于保护模型更新（梯度）的隐私。典型的工作流程为：

1. 各参与方在本地计算梯度 $\mathbf{g}_i$ 并加密：$\mathsf{Enc}(\mathbf{g}_i)$。
2. 聚合服务器执行同态加法：$\mathsf{Enc}(\bar{\mathbf{g}}) = \sum_i \mathsf{Enc}(\mathbf{g}_i)$。
3. 由密钥持有方（或通过门限解密协议）解密聚合结果 $\bar{\mathbf{g}}$ 并用于模型更新。

这种方案确保了服务器只能看到聚合后的梯度，无法推断任何单个参与方的梯度信息。结合差分隐私（DP），可以进一步提供可证明的隐私保证。

### 5.4.4 典型应用案例分析

**案例1：加密医疗数据分析。** 医院拥有敏感的患者数据，需要借助云端的AI模型进行疾病诊断预测。通过CKKS方案加密患者数据，云服务器在密文上执行诊断模型的推理，返回加密的预测结果。患者或医院解密后获得诊断建议，全程无需向云端暴露任何明文信息。

**案例2：跨机构联合风控。** 多家金融机构需要联合训练反欺诈模型，但受到数据隐私法规（如GDPR、《个人信息保护法》）的限制无法直接共享数据。各方使用同态加密保护本地梯度，通过安全聚合协议汇总梯度更新全局模型，实现了"数据不出域，模型共进化"的目标。

**案例3：加密基因组分析。** 基因组数据是高度敏感的个人隐私信息。同态加密可以在不暴露基因组序列的前提下，执行全基因组关联分析（GWAS）等统计计算。HElib和PALISADE等库已支持在加密数据上计算均值、方差、相关系数等统计量。

## 5.5 性能挑战与优化策略

### 5.5.1 计算开销分析

同态加密的计算开销主要体现在以下几个方面：

**密钥生成**：生成公钥、私钥和评估密钥涉及大维度多项式的采样和运算。对于安全参数 $\lambda = 128$，多项式维度 $n = 2^{15} = 32768$，密钥生成可能需要数十毫秒到数秒。

**加密与解密**：加密操作涉及多项式乘法和噪声采样。一次加密操作通常需要数毫秒。解密操作相对简单，但仍需一次多项式乘法。

**同态运算**：

- 同态加法：两个密文的逐系数相加，复杂度 $O(n)$，通常在微秒级完成。
- 同态乘法：涉及多项式乘法（可通过NTT加速至 $O(n \log n)$）、重线性化（多次多项式乘法）和模切换（系数缩放）。单次同态乘法通常需要数十微秒到数毫秒。
- 密钥切换（Key Switching）：每次乘法后需要执行的操作，涉及 $O(n \log n)$ 的多项式运算。

**Bootstrapping**：这是最昂贵的操作。以CKKS方案为例，在标准CPU上，一次Bootstrapping可能需要数十毫秒到数秒（取决于参数选择和优化程度）。经过近年来的优化，最新实现已将CKKS Bootstrapping的时间降低到10毫秒以下（针对特定参数配置）[10]。

### 5.5.2 密文膨胀问题

同态加密的密文膨胀（Ciphertext Expansion）是一个本质性问题。一个64位浮点数（8字节）经HE加密后，密文大小可达数千到数万字节：

- 密文 $(c_0, c_1)$ 包含两个多项式，每个多项式有 $n$ 个系数，每个系数占 $\lceil \log_2 q \rceil$ 位。
- 对于 $n = 2^{15}$、$\log_2 q \approx 438$ 的典型参数，单个密文大小约为 $2 \times 32768 \times 438 / 8 \approx 3.5 \text{ MB}$。

密文膨胀带来的影响包括：

- **通信开销增大**：在联邦学习中，每轮通信需要传输加密梯度。对于一个有 $10^6$ 参数的模型，加密梯度的大小可达 TB 级别，远超明文梯度的 MB 级别。
- **内存占用增加**：服务端需要存储大量密文用于计算，内存需求可能增加数百到数千倍。
- **缓存效率降低**：由于密文尺寸远超CPU缓存，缓存未命中率显著增加，进一步降低了运算效率。

缓解密文膨胀的策略包括：

- **SIMD批处理**：将多个明文打包到一个密文中，摊薄单个数据元素的密文开销。在CKKS中，一个密文可同时编码 $n/2 = 16384$ 个浮点数，使得每个数据元素的有效密文开销降低到约 215 字节。
- **密文压缩**：利用密文系数的统计特性进行压缩传输，或使用"packed"密文减少通信轮次。
- **稀疏编码**：对于稀疏梯度（如深度学习中大量接近零的梯度），可以仅加密非零元素，显著减少密文总量。

### 5.5.3 硬件加速技术

由于同态加密的计算本质上是大整数多项式运算，天然适合硬件加速。当前的硬件加速研究主要集中在以下几个方向：

**CPU指令集优化**：现代CPU提供的向量指令集（如Intel AVX-512、ARM NEON）可以显著加速数论变换（NTT）和模运算。Microsoft SEAL库通过AVX-512优化，将NTT运算加速了约4-8倍。Intel HEXL（Homomorphic Encryption Acceleration Library）[18]专门为HE运算优化了底层原语。

**GPU加速**：GPU的大规模并行架构非常适合HE中的批量多项式运算。多个研究团队已实现了基于CUDA的HE加速器：

- NTT的GPU实现可获得10-50倍的加速比。
- CKKS编码/解码的GPU并行化可将延迟降低一个数量级。
- 批量加密/解密操作在GPU上的吞吐量可达CPU的100倍以上。

**FPGA加速**：FPGA因其可重构性和低延迟特性，成为HE硬件加速的重要平台：

- 多项研究实现了NTT的FPGA流水线设计，单次NTT的延迟可低至微秒级。
- Dowlin等[19]设计的FPGA加速器将CKKS乘法运算加速了约1000倍。
- FPGA的优势还体现在功耗效率上，相比GPU可降低数倍的能耗。

**专用集成电路（ASIC）设计**：针对HE的专用芯片设计是性能优化的终极方案。DARPA在2021年启动的DPRIVE（Data Protection in Virtual Environments）项目资助了多个HE专用加速器的研发，目标是将HE运算速度提升至与明文运算可比的水平。

**新兴趋势——FHE硬件-软件协同设计**：当前的研究趋势强调硬件和软件的协同优化。一方面，硬件设计需要充分理解HE算法的计算模式；另一方面，HE算法的参数选择和实现策略也需要适应目标硬件的特性。这种协同设计有望在不远的将来将HE的性能提升两到三个数量级。

### 5.5.4 参数选择与性能权衡

FHE方案的参数选择是一个涉及安全性和性能的多目标优化问题。核心参数包括：

- **多项式维度 $n$**：决定安全性和可支持的模数大小。$n$ 越大安全性越高，但运算越慢。
- **密文模数 $q$**：决定可支持的乘法深度和精度。$q$ 越大，可执行的乘法次数越多，但每次运算的开销也越大。
- **明文模数 $t$ 或缩放因子 $\Delta$**：影响运算精度。
- **噪声分布参数 $\sigma$**：影响安全性和噪声预算。

参数选择的目标是在满足安全性要求（通常为128位安全）的前提下，最小化计算开销。这一过程通常借助参数推荐工具（如Lattigo的参数生成器、Microsoft SEAL的参数评估工具）自动完成，但在实际部署中仍需根据具体应用场景进行调优。

## 5.6 本章小结

同态加密作为隐私计算领域的核心密码学工具，经过十余年的快速发展，已经从Gentry最初的理论构造演进为一系列实用化的密码学方案。BGV、BFV和CKKS三大方案各有侧重：BGV和BFV适合精确整数运算，CKKS则为近似浮点运算提供了高效的解决方案。在机器学习应用中，同态加密已被成功应用于加密推理、加密训练和隐私保护神经网络等多个场景，与联邦学习的结合更是为"数据可用不可见"的愿景提供了坚实的密码学支撑。

然而，同态加密面临的性能挑战依然严峻。计算开销、密文膨胀和Bootstrapping延迟是制约其大规模应用的三大瓶颈。硬件加速技术（CPU优化、GPU并行化、FPGA设计乃至专用ASIC）正在快速缩小这一差距，但在可预见的未来，同态加密仍将是一种"高成本、高价值"的隐私计算手段——它最适合用于对隐私保护要求极高、可以容忍一定计算延迟的应用场景。

展望未来，随着算法优化和硬件加速的持续推进，同态加密有望在隐私保护机器学习、安全云计算和合规数据协作等领域发挥更加重要的作用。同态加密与安全多方计算、可信执行环境等技术的深度融合，将为构建完整的隐私计算基础设施奠定坚实基础。

## 参考文献

[1] Rivest R L, Adleman L, Dertouzos M L. On data banks and privacy homomorphisms[J]. Foundations of Secure Computation, 1978: 169-179.

[2] Gentry C. Fully homomorphic encryption using ideal lattices[C]. Proceedings of the 41st Annual ACM Symposium on Theory of Computing (STOC), 2009: 169-178.

[3] Regev O. On lattices, learning with errors, random linear codes, and cryptography[J]. Journal of the ACM, 2009, 56(6): 1-40.

[4] Lyubashevsky V, Peikert C, Regev O. On ideal lattices and learning with errors over rings[C]. Advances in Cryptology – EUROCRYPT 2010: 1-23.

[5] Brakerski Z, Vaikuntanathan V. Efficient fully homomorphic encryption from (standard) LWE[C]. 2011 IEEE 52nd Annual Symposium on Foundations of Computer Science (FOCS): 97-106.

[6] Cheon J H, Kim A, Kim M, et al. Homomorphic encryption for arithmetic of approximate numbers[C]. Advances in Cryptology – ASIACRYPT 2017: 409-437.

[7] Brakerski Z, Gentry C, Vaikuntanathan V. (Leveled) fully homomorphic encryption without bootstrapping[C]. Proceedings of the 3rd Innovations in Theoretical Computer Science Conference (ITCS), 2012: 309-325.

[8] Fan J, Vercauteren F. Somewhat practical fully homomorphic encryption[J]. IACR Cryptology ePrint Archive, 2012: 144.

[9] Cheon J H, Han K, Kim A, et al. Bootstrapping for approximate homomorphic encryption[C]. Advances in Cryptology – EUROCRYPT 2018: 360-384.

[10] Lee Y, Lepoint T, Lee S. CRT-bootstrapping for CKKS[C]. Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security (CCS): 2086-2099.

[11] Dowlin N, Gilad-Bachrach R, Laine K, et al. CryptoNets: Applying neural networks to encrypted data with high throughput and accuracy[C]. Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016: 201-210.

[12] Juvekar C, Vaikuntanathan V, Chandrakasan A. GAZELLE: A low latency framework for secure neural network inference[C]. Proceedings of the 27th USENIX Security Symposium, 2018: 1651-1669.

[13] Rathee D, Rathee M, Kumar N, et al. CrypTFlow2: Practical 2-party secure inference[C]. Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security (CCS): 325-342.

[14] Mohassel P, Zhang Y. SecureML: A system for scalable privacy-preserving machine learning[C]. 2017 IEEE Symposium on Security and Privacy (S&P): 19-38.

[15] Mohassel P, Rindal P. ABY3: A mixed protocol framework for machine learning[C]. Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS): 35-52.

[16] Liu J, Juuti M, Lu Y, et al. Oblivious neural network predictions via MiniONN transformations[C]. Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS): 619-631.

[17] Riazi M S, Weinert C, Tkachenko O, et al. Chameleon: A hybrid secure computation framework[C]. Proceedings of the 2018 on Asia Conference on Computer and Communications Security (ASIACCS): 707-721.

[18] Boemer F, Cammarota R, Demmler D, et al. Intel HEXL: Accelerating homomorphic encryption with Intel AVX512-IFMA52[C]. Proceedings of the 9th Workshop on Encrypted Computing & Applied Homomorphic Cryptography, 2021.

[19] Samardzic N, Feldmann A, Krastev A, et al. F1: A fast and programmable accelerator for fully homomorphic encryption[C]. Proceedings of the 54th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), 2021: 238-252.
# 第6章 联邦学习进阶

> **导读：** 第5章介绍了联邦学习的基础框架与核心算法。本章将深入探讨联邦学习在实际部署中面临的若干关键挑战及其前沿解决方案。我们将从四个维度展开：个性化联邦学习解决数据异构性问题，异构联邦学习适应参与方能力差异，通信压缩技术降低带宽开销，以及聚合策略的演进与创新。这些进阶技术构成了从实验室原型到工业级部署的重要桥梁。

---

## 6.1 个性化联邦学习

### 6.1.1 问题背景：为什么需要个性化？

标准联邦学习（如 FedAvg）的核心假设是所有客户端共享一个全局最优模型。然而，现实场景中这一假设往往不成立——不同客户端的数据分布可能存在显著差异，即**非独立同分布（Non-IID）**问题。例如，在医疗场景中，不同医院的患者群体特征差异明显；在移动端键盘预测中，不同用户的输入习惯各不相同。

当数据分布差异过大时，全局模型可能在所有客户端上都表现不佳，出现所谓的"公平性困境"。个性化联邦学习（Personalized Federated Learning）应运而生，旨在为每个客户端学习一个定制化的模型，同时仍借助联邦协作的优势。

形式化地，个性化联邦学习的优化目标可表示为：

$$\min_{\theta_1, \theta_2, \ldots, \theta_K} \sum_{k=1}^{K} p_k \, F_k(\theta_k)$$

其中 $\theta_k$ 为第 $k$ 个客户端的个性化模型参数，$F_k$ 为其本地目标函数，$p_k$ 为权重系数。与标准联邦学习中所有客户端共享同一 $\theta$ 不同，此处每个客户端拥有独立的模型参数。

### 6.1.2 Per-FedAvg：元学习视角的个性化

Per-FedAvg（Personalized Federated Learning with Moreau Envelopes）由 Deng et al.（2021）提出，将个性化联邦学习与**模型无关元学习（MAML）**建立了理论联系。

Per-FedAvg 的核心思想是：联邦训练的目标不是找到一个对所有客户端都最优的全局模型，而是找到一个**良好的初始化点**，使得每个客户端能够通过少量本地梯度步骤快速适应到自身的数据分布。

具体而言，Per-FedAvg 采用双层优化结构。外层优化全局初始化参数 $\theta$：

$$\min_{\theta} \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}_k\big(\theta - \alpha \nabla \mathcal{L}_k(\theta)\big)$$

其中 $\alpha$ 为客户端本地适应的学习率，$\mathcal{L}_k$ 为第 $k$ 个客户端的损失函数。内层则在每个客户端上执行一步或多步梯度下降以获得个性化参数：

$$\theta_k^* = \theta - \alpha \nabla \mathcal{L}_k(\theta)$$

从优化理论的角度，Per-FedAvg 等价于使用 **Moreau 包络（Moreau Envelope）** 作为正则化项的联邦优化：

$$\min_{\theta} \frac{1}{K} \sum_{k=1}^{K} \text{prox}_{\alpha \mathcal{L}_k}(\theta)$$

其中 $\text{prox}$ 为近端算子。这一视角为 Per-FedAvg 提供了更坚实的理论基础，也揭示了元学习与正则化之间的深层联系。

### 6.1.3 FedBN：批归一化的个性化

FedBN（Federated Batch Normalization）由 Li et al.（2021）提出，是一种简洁而高效的个性化方法。其核心观察是：在深度神经网络中，**批归一化（Batch Normalization, BN）层**的统计量（均值 $\mu_k$ 和方差 $\sigma_k^2$）高度依赖于本地数据分布，而特征提取器的权重则具有更好的可迁移性。

FedBN 的策略极为简单：在联邦聚合时，**跳过所有 BN 层的参数**，仅聚合卷积层和全连接层的权重。形式化地，对于客户端 $k$ 的模型参数 $\theta_k = \{\theta_k^{\text{BN}}, \theta_k^{\text{other}}\}$，聚合操作为：

$$\theta_g^{\text{other}} = \sum_{k=1}^{K} \frac{n_k}{n} \theta_k^{\text{other}}$$

$$\theta_k^{\text{BN}} \text{ 保持本地不变}$$

FedBN 的优势在于：（1）无需额外的计算开销；（2）自然地适应不同客户端的数据分布差异；（3）与其他个性化方法正交，可以组合使用。实验表明，FedBN 在特征偏移（feature shift）场景下显著优于 FedAvg。

### 6.1.4 FedRep：表征与头部的分离

FedRep（Federated Representation Learning）由 Collins et al.（2021）提出，将模型显式地分解为**共享表征（representation）** 和 **个性化头部（head）** 两个部分。

假设模型 $h_k$ 可分解为：

$$h_k(x) = h_k^{\text{head}} \circ \phi^{\text{repr}}(x)$$

其中 $\phi^{\text{repr}}: \mathcal{X} \to \mathcal{Z}$ 为共享的表征映射（如卷积骨干网络），$h_k^{\text{head}}: \mathcal{Z} \to \mathcal{Y}$ 为客户端 $k$ 特有的分类头。

FedRep 的训练过程交替进行：

1. **表征学习阶段**：固定各客户端的头部参数 $\{h_k^{\text{head}}\}$，通过联邦聚合更新共享表征 $\phi^{\text{repr}}$；
2. **头部适应阶段**：固定共享表征 $\phi^{\text{repr}}$，各客户端在本地数据上独立更新头部参数 $h_k^{\text{head}}$。

优化目标可写为：

$$\min_{\phi^{\text{repr}}} \sum_{k=1}^{K} p_k \min_{h_k^{\text{head}}} \mathcal{L}_k(h_k^{\text{head}} \circ \phi^{\text{repr}})$$

FedRep 的理论分析表明，在一定条件下，共享表征能够捕获跨客户端的不变特征，而个性化头部则适配本地数据的特定模式。这一框架在视觉任务和语言任务上均展现了优异的性能。

### 6.1.5 集群联邦学习

**集群联邦学习（Clustered Federated Learning）** 采取了另一种思路：不是为每个客户端单独学习个性化模型，而是先将相似的客户端分组，再为每个组学习一个共享模型。

Ghosh et al.（2020）提出的 IFCA（Iterative Federated Clustering Algorithm）通过交替优化客户端分组和组内模型：

$$\hat{c}_k = \arg\min_{c \in \{1,\ldots,C\}} \mathcal{L}_k(\theta_c)$$

$$\theta_c \leftarrow \text{FedAvg}(\{\theta_c\}_{k: \hat{c}_k = c})$$

即每个客户端选择使其损失最小的聚类中心对应的模型，并在该聚类内进行联邦聚合。

Sattler et al.（2020）提出的 FedSoft 则采用软分配，允许客户端同时属于多个聚类，通过加权组合获得最终的个性化模型：

$$\theta_k^* = \sum_{c=1}^{C} w_{k,c} \, \theta_c, \quad w_{k,c} = \frac{\exp(-\beta \mathcal{L}_k(\theta_c))}{\sum_{c'} \exp(-\beta \mathcal{L}_k(\theta_{c'}))}$$

集群联邦学习的关键挑战在于：（1）聚类数量通常未知；（2）客户端可能在训练过程中漂移；（3）小聚类的样本量可能不足以支撑独立训练。

---

## 6.2 异构联邦学习

### 6.2.1 模型异构性问题

标准联邦学习假设所有客户端使用相同的模型架构。然而在实际部署中，不同客户端的计算能力、存储资源和通信带宽存在巨大差异——智能手机、边缘设备和高性能服务器无法运行相同规模的模型。**异构联邦学习（Heterogeneous Federated Learning）** 旨在解决这一问题，允许不同客户端使用不同架构的模型进行协作学习。

模型异构性带来的核心挑战是：当客户端模型结构不同时，传统的参数平均聚合策略 $\theta_g = \sum_k \frac{n_k}{n} \theta_k$ 不再适用——不同维度的参数向量无法直接相加。

### 6.2.2 架构无关联邦学习

**架构无关联邦学习（Architecture-Agnostic Federated Learning）** 的代表方法是 **FedDF**（Federated Distillation with Ensemble）和 **FedGen**。

FedDF（Lin et al., 2020）的核心思想是利用**知识蒸馏（Knowledge Distillation）**替代参数聚合。服务器维护一个公共的未标记数据集（或合成数据），各客户端模型在此数据集上产生预测结果，服务器通过蒸馏聚合这些知识：

$$\mathcal{L}_{\text{KD}}(\theta_s) = \sum_{x \in \mathcal{D}_{\text{pub}}} \text{KL}\left(p_{\text{ensemble}}(y|x) \,\|\, p_{\theta_s}(y|x)\right)$$

其中 $p_{\text{ensemble}}(y|x) = \frac{1}{K} \sum_{k=1}^{K} p_{\theta_k}(y|x)$ 为客户端模型的集成预测，$p_{\theta_s}$ 为服务器模型的预测。

这一方法的优势是彻底解耦了客户端模型架构——不同客户端可以使用不同深度、不同宽度甚至不同类型的神经网络，只要它们的输出空间一致即可。

### 6.2.3 知识蒸馏联邦学习

知识蒸馏在联邦学习中的应用已经发展出多种范式：

**（1）服务器-客户端蒸馏。** 如 FedDF 所述，服务器利用客户端模型的"软标签"知识训练一个统一的全局模型。

**（2）客户端-客户端蒸馏。** FedMD（Li & Wang, 2019）引入了"共识数据集"的概念，客户端之间通过在共识数据集上交换预测结果实现隐式知识共享：

$$\mathcal{L}_k = \mathcal{L}_k^{\text{local}} + \lambda \sum_{j \neq k} \text{KL}\left(p_{\theta_j} \,\|\, p_{\theta_k}\right)$$

**（3）全局-局部蒸馏。** FedDF with Global Distillation（Jeong et al., 2018）在客户端本地训练时，不仅最小化本地损失，还通过蒸馏保留对全局知识的记忆，缓解灾难性遗忘：

$$\mathcal{L}_k^{\text{total}} = \mathcal{L}_k^{\text{task}} + \mu \, \text{KL}\left(p_{\theta_g} \,\|\, p_{\theta_k}\right)$$

知识蒸馏方法的理论分析表明，蒸馏温度 $\tau$ 和权重 $\lambda$ 的选择对性能影响显著。较高的温度产生更"软"的概率分布，传递更多暗知识（dark knowledge），但也增加了噪声。

### 6.2.4 分组与层级联邦

另一种处理异构性的思路是按能力对客户端进行分组。**FjORD**（Horvath et al., 2021）提出了有序丢弃（Ordered Dropout）技术，在训练过程中随机丢弃模型的后几层，使得较小的客户端只需运行模型的前几层：

$$\text{OD}_p(\theta) = \{\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(\lfloor pL \rfloor)}\}$$

其中 $p \in (0, 1]$ 为保留比例，$L$ 为总层数。通过在训练中随机采样不同的 $p$ 值，FjORD 同时训练了模型的多个"子网络"，每个客户端可以根据自身能力选择合适的子网络。

---

## 6.3 通信压缩

### 6.3.1 联邦学习的通信瓶颈

通信开销是联邦学习规模化部署的核心瓶颈。在一个典型的联邦学习系统中，每轮通信需要传输完整的模型参数 $\theta \in \mathbb{R}^d$，当模型参数量达到数百万甚至数十亿时，通信成本极为高昂。

通信压缩的目标是在尽可能少的信息传输量下，保持模型收敛速度和最终精度。压缩操作通常作用于客户端上传的**模型更新**（即梯度或参数差值）$\Delta \theta_k = \theta_k - \theta_g$。

压缩算子 $\mathcal{C}: \mathbb{R}^d \to \mathbb{R}^d$ 的核心性质是其压缩比：

$$\mathbb{E}[\|\mathcal{C}(x)\|_0] \leq \frac{d}{\omega}$$

其中 $\omega \geq 1$ 为压缩比，$\|\cdot\|_0$ 为非零元素个数。当 $\omega > 1$ 时，传输的数据量减少。

### 6.3.2 Top-k 稀疏化

**Top-k 稀疏化**是最直观的压缩方法：仅传输梯度中绝对值最大的 $k$ 个分量，其余置零。对于梯度向量 $g \in \mathbb{R}^d$：

$$[\mathcal{C}_{\text{top-k}}(g)]_i = \begin{cases} g_i, & \text{if } i \in \text{TopK}(|g|, k) \\ 0, & \text{otherwise} \end{cases}$$

压缩比为 $\omega = d/k$。Alistarh et al.（2017）在 QSGD 中分析了 Top-k 的收敛性，证明在满足一定条件下，压缩梯度的期望是无偏的，但方差增大：

$$\mathbb{E}[\mathcal{C}_{\text{top-k}}(g)] = g, \quad \text{Var}[\mathcal{C}_{\text{top-k}}(g)] \leq \frac{d - k}{k} \|g\|^2$$

为补偿压缩带来的方差增大，**误差补偿（Error Compensation）**技术被广泛采用。其核心思想是维护一个本地误差缓冲区 $e_k$，将每轮压缩丢弃的分量累积到下一轮：

$$\hat{g}_k = \mathcal{C}_{\text{top-k}}(g_k + e_k)$$
$$e_k \leftarrow g_k + e_k - \hat{g}_k$$

Karimireddy et al.（2019）在 SCAFFOLD 框架下分析了误差补偿 Top-k 的收敛性，证明了其能够达到与未压缩版本相同的收敛阶。

### 6.3.3 梯度量化

**梯度量化（Gradient Quantization）** 将连续的梯度值映射到有限的离散集合中。最简单的形式是 **1-bit 量化**，即仅保留梯度的符号：

$$[\mathcal{C}_{\text{1bit}}(g)]_i = \|g\|_1 / d \cdot \text{sign}(g_i)$$

更一般地，$b$-bit 量化将每个梯度分量映射到 $2^b$ 个量化级别。QSGD（Quantized SGD，Alistarh et al., 2017）通过随机舍入实现无偏量化：

$$\mathcal{C}_{\text{QSGD}}(g_i) = s \cdot \text{sign}(g_i) \cdot \begin{cases} l / s, & \text{with probability } 1 - \frac{|g_i|}{s} \cdot (s - l) \\ (l+1) / s, & \text{with probability } \frac{|g_i|}{s} \cdot (s - l) \end{cases}$$

其中 $l = \lfloor |g_i|/s \cdot (2^b - 1) \rfloor$，$s$ 为缩放因子。

TernGrad（Wen et al., 2017）进一步将量化极端化为三值量化 $\{-1, 0, +1\}$，压缩比可达 32 倍（从 32-bit 浮点到 2-bit 表示）。

量化压缩的理论分析表明，在凸优化设定下，QSGD 以 $O(1/\sqrt{T})$ 的速率收敛，与标准 SGD 一致，仅常数因子增大。

### 6.3.4 低秩分解

**低秩分解（Low-Rank Decomposition）** 利用梯度矩阵通常具有低秩结构的观察，将梯度矩阵 $G \in \mathbb{R}^{m \times n}$ 分解为两个低秩矩阵的乘积：

$$G \approx U V^\top, \quad U \in \mathbb{R}^{m \times r}, V \in \mathbb{R}^{n \times r}$$

其中 $r \ll \min(m, n)$。传输量从 $mn$ 降至 $(m+n)r$，压缩比为 $\omega = mn / ((m+n)r)$。

PowerSGD（Vogels et al., 2019）采用幂迭代法高效计算低秩近似：

1. 初始化随机矩阵 $Q \in \mathbb{R}^{n \times r}$
2. $P \leftarrow GQ$（矩阵乘法）
3. $Q' \leftarrow \text{orth}(G^\top P)$（正交化）
4. 传输 $P$ 和 $Q'$

PowerSGD 的关键创新在于使用上一轮的低秩近似作为当前轮的初始化（暖启动），显著提升了近似质量。同时，与 Top-k 类似，PowerSGD 也可以结合误差补偿来处理低秩近似的残差。

### 6.3.5 组合压缩策略

在实践中，多种压缩技术可以组合使用以获得更高的压缩比。例如：

- **量化 + 稀疏化**：先量化梯度值，再选择 Top-k 分量传输
- **低秩分解 + 量化**：对低秩因子进一步量化

理论上，Karimireddy et al.（2021）证明了在满足**压缩算子的有偏性和有界性条件**下，组合压缩仍能保证收敛。具体地，若压缩算子 $\mathcal{C}$ 满足：

$$\mathbb{E}\|\mathcal{C}(x) - x\|^2 \leq (1 - \frac{1}{\omega})\|x\|^2$$

则误差补偿 SGD 以 $O(1/\sqrt{T} + 1/T)$ 的速率收敛。

---

## 6.4 聚合策略的演进

### 6.4.1 FedAvg：基石算法的再审视

**FedAvg**（McMahan et al., 2017）是联邦学习最经典的聚合策略。在每轮通信中，服务器将全局模型分发给选中的客户端，客户端在本地执行 $E$ 轮 SGD 后上传更新后的模型参数，服务器进行加权平均聚合：

$$\theta_g^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{n} \theta_k^{(t)}$$

其中 $n_k$ 为第 $k$ 个客户端的数据量，$n = \sum_k n_k$ 为总数据量。

FedAvg 的一个关键特性是其**通信效率**——通过多轮本地训练减少通信轮数。然而，当数据分布非 IID 时，FedAvg 可能收敛到次优解，甚至发散。Li et al.（2020）的理论分析表明，FedAvg 的收敛速率受到本地更新步数 $E$ 和数据异构程度的共同影响：

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}\|\nabla F(\theta^{(t)})\|^2 \leq O\left(\frac{1}{\sqrt{TE}} + \frac{E \sigma^2}{T} + \Gamma\right)$$

其中 $\sigma^2$ 为客户端梯度方差，$\Gamma$ 为衡量数据异构程度的项（与各客户端最优解的差异 $\|\theta_k^* - \theta^*\|^2$ 相关）。当 $\Gamma$ 较大时，增大 $E$ 并不一定能加速收敛。

### 6.4.2 FedProx：近端约束的稳定化

**FedProx**（Li et al., 2020）通过在客户端本地目标函数中添加近端正则项来缓解 FedAvg 的发散问题：

$$\mathcal{L}_k^{\text{prox}}(\theta) = \mathcal{L}_k(\theta) + \frac{\mu}{2}\|\theta - \theta_g^{(t)}\|^2$$

其中 $\mu > 0$ 为近端系数，$\theta_g^{(t)}$ 为当前轮的全局模型。这一正则项约束本地模型不要偏离全局模型太远，从而减轻客户端漂移（client drift）。

FedProx 的优化视角可以理解为对 FedAvg 的约束松弛。原始的 FedAvg 可视为无约束的多步本地优化，而 FedProx 则等价于在每个客户端上求解一个近端子问题：

$$\theta_k^{(t)} = \arg\min_{\theta} \left\{ \mathcal{L}_k(\theta) + \frac{\mu}{2}\|\theta - \theta_g^{(t)}\|^2 \right\}$$

理论分析表明，FedProx 在非 IID 数据上具有更好的收敛保证，其收敛速率中的异构性项 $\Gamma$ 被 $\mu$ 所控制。当 $\mu \to \infty$ 时退化为每轮仅执行一步本地更新，$\mu = 0$ 时退化为 FedAvg。

### 6.4.3 FedNova：归一化的公平聚合

**FedNova**（Wang et al., 2020）指出 FedAvg 的另一个问题：当不同客户端执行不同数量的本地更新步数时（例如由于资源差异或早停策略），简单的加权平均会导致聚合偏差。

FedNova 引入了**归一化系数**来纠正这种偏差。对于客户端 $k$，假设其在本地执行了 $\tau_k$ 步更新，累积的更新量为 $\Delta_k$，FedNova 定义归一化的本地更新为：

$$\hat{\Delta}_k = \frac{\Delta_k}{\tau_k}$$

然后进行归一化聚合：

$$\theta_g^{(t+1)} = \theta_g^{(t)} + \sum_{k=1}^{K} \frac{n_k}{n} \cdot \frac{\tau_k}{\sum_{j} p_j \tau_j} \cdot \hat{\Delta}_k$$

其中 $p_k = n_k / n$。这一聚合策略保证了即使各客户端的本地步数不同，聚合后的模型更新方向仍然是一致且无偏的。

FedNova 的理论贡献在于将联邦学习的收敛分析从"同步本地步数"的假设中解放出来，更贴合实际的异步部署场景。

### 6.4.4 FedBN 的聚合机制

FedBN（6.1.3 节已介绍其个性化视角）在聚合层面的策略同样值得深入讨论。其"选择性聚合"思想可以推广为更一般的**层级联邦聚合**框架。

在深度神经网络中，不同层对数据分布的敏感度不同。低层特征（如边缘检测器）具有较好的跨域迁移性，而高层特征和归一化层的统计量则高度依赖于本地数据分布。FedBN 的选择性聚合策略可以推广为：

$$\theta_g^{(l)} = \begin{cases} \sum_{k=1}^{K} \frac{n_k}{n} \theta_k^{(l)}, & \text{if } l \in \mathcal{S}_{\text{shared}} \\ \theta_k^{(l)}, & \text{if } l \in \mathcal{S}_{\text{local}} \end{cases}$$

其中 $\mathcal{S}_{\text{shared}}$ 和 $\mathcal{S}_{\text{local}}$ 分别为需要全局聚合和保持本地的层集合。

### 6.4.5 聚合策略的统一视角

从优化理论的角度，上述聚合策略可以统一到**联邦近端正则化框架**中：

$$\theta_k^{(t+1)} = \arg\min_{\theta} \left\{ \mathcal{L}_k(\theta) + \frac{\mu_k}{2}\|\theta - \bar{\theta}^{(t)}\|_{M_k}^2 \right\}$$

其中 $M_k$ 为正定矩阵（允许层间的差异化正则化），$\bar{\theta}^{(t)}$ 为聚合中心。不同的方法对应于不同的选择：

| 方法 | $\mu_k$ | $M_k$ | $\bar{\theta}$ |
|------|---------|--------|-----------------|
| FedAvg | $0$ | — | $\theta_g$ |
| FedProx | $\mu > 0$（统一） | $I$ | $\theta_g$ |
| FedBN | $0$（共享层）/ $\infty$（BN层） | 分层 | $\theta_g$（共享层）|
| FedNova | $0$ | $I$ | 归一化聚合 |

这一统一视角不仅有助于理解各方法之间的关系，也为设计新的聚合策略提供了指导框架。

---

## 6.5 本章小结

本章围绕联邦学习进阶的四个核心主题展开：

1. **个性化联邦学习**通过元学习（Per-FedAvg）、选择性参数共享（FedBN）、表征-头部分离（FedRep）和客户端聚类等策略，在协作学习和本地适应之间取得平衡。个性化联邦学习的核心挑战在于如何在利用跨客户端知识的同时保留客户端的个体特性。

2. **异构联邦学习**突破了统一模型架构的限制，通过知识蒸馏（FedDF、FedMD）和有序丢弃（FjORD）等技术，使得计算能力各异的客户端能够在一个联邦中协作。

3. **通信压缩**通过 Top-k 稀疏化、梯度量化、低秩分解等技术，在保持收敛性的前提下大幅降低通信开销。误差补偿机制的引入使得压缩带来的偏差能够被逐步纠正。

4. **聚合策略**从 FedAvg 到 FedProx、FedNova、FedBN 的演进，反映了联邦学习社区对非 IID 数据、资源异构性和模型异质性的深入理解。

这些进阶技术并非孤立存在，而是可以灵活组合。例如，一个实际的联邦学习系统可能同时采用 FedBN 的选择性聚合、Top-k 通信压缩和集群联邦学习的客户端分组策略。如何在给定约束下选择最优的技术组合，是联邦学习走向大规模工业应用的关键课题。

---

**延伸阅读：**

- Mansour, Y., et al. (2020). "Three Approaches for Personalization with Applications to Federated Learning." *arXiv:2002.10619.*
- Kairouz, P., et al. (2021). "Advances and Open Problems in Federated Learning." *Foundations and Trends in Machine Learning.*
- Wang, J., et al. (2021). "A Field Guide to Federated Optimization." *arXiv:2107.06917.*
- Li, T., et al. (2020). "Federated Optimization in Heterogeneous Networks." *MLSys 2020.*
# 第7章 联邦学习安全

联邦学习通过将数据保留在本地、仅交换模型更新的方式，在一定程度上缓解了传统集中式机器学习中的隐私泄露风险。然而，"数据不出域"并不等同于"安全无忧"。联邦学习系统面临着来自多个维度的安全威胁：恶意参与方可能通过篡改本地数据或上传有毒模型来破坏全局模型的完整性（投毒攻击）；好奇的中央服务器或参与方可能通过分析梯度信息推断其他方的敏感数据（推理攻击）；通信链路中的窃听者可能截获并分析梯度更新以获取隐私信息。本章系统梳理联邦学习面临的主要安全威胁及其防御方法，涵盖投毒攻击、推理攻击、鲁棒聚合策略以及安全聚合协议等核心议题。

## 7.1 投毒攻击

投毒攻击（Poisoning Attack）是联邦学习中最为直接且危害最大的安全威胁之一。在传统的集中式机器学习中，攻击者需要渗透数据采集或标注流程才能实施投毒；而在联邦学习场景下，参与训练的各方本身就是潜在的攻击者，这使得投毒攻击的实施门槛大幅降低。根据攻击者操纵的对象不同，投毒攻击可分为数据投毒和模型投毒两大类；根据攻击目标的不同，又可细分为无目标攻击（降低模型整体性能）和有目标攻击（使模型对特定输入产生错误输出）。

### 7.1.1 数据投毒

数据投毒（Data Poisoning）是指攻击者通过篡改本地训练数据来影响全局模型的行为。这是最为直观的一种攻击方式——恶意参与方在本地训练前，对训练样本的标签进行翻转（Label Flipping），或注入精心构造的恶意样本，使得基于这些被污染数据训练出的本地模型偏离正常方向，进而通过模型聚合过程将这种偏差传播到全局模型中。

Biggio等人在其早期研究中系统分析了数据投毒对支持向量机等分类器的影响。在联邦学习语境下，数据投毒具有更强的隐蔽性，因为中央服务器无法直接检查各参与方的本地数据。一种典型的无目标数据投毒策略是随机翻转训练标签，例如将"猫"的图片标注为"狗"，由此训练出的模型在全局聚合后会降低整体分类精度。有目标的数据投毒则更为精细——攻击者仅针对特定类别的特定样本进行操纵，例如在手写数字识别任务中，使模型将所有标记为"7"的样本误分类为"1"，而对其他类别的识别精度几乎不受影响。这种有目标攻击由于不显著影响模型的整体性能指标，因而更难被常规的质量检测机制发现。

数据投毒的有效性受多个因素影响，包括恶意参与方的数量占比、投毒样本在本地数据中的比例、以及所采用的聚合策略。当恶意参与方占比较低且采用简单的联邦平均（FedAvg）算法时，少量投毒即可产生可观的影响。然而，随着参与方数量的增加，单个恶意方的影响力会被稀释，这在一定程度上为系统提供了天然的鲁棒性。

### 7.1.2 模型投毒

模型投毒（Model Poisoning）是比数据投毒更为直接的攻击方式。攻击者不再通过间接手段（篡改数据）来影响模型，而是直接构造恶意的模型更新并上传至中央服务器。这种攻击跳过了"数据—训练—模型"的完整流程，直接在模型层面进行操纵，因此攻击效率更高、效果更可控。

Fang等人（2020）提出的模型投毒攻击框架揭示了一个重要事实：在Non-IID（非独立同分布）数据场景下，各参与方的模型更新本身就可能存在显著差异，这为恶意更新提供了天然的"掩护"。攻击者可以精心设计恶意更新，使其在统计特征上与合法的Non-IID更新相似，从而规避基于统计异常检测的防御机制。

一种常见的模型投毒策略是"缩放攻击"（Scaling Attack）：恶意参与方将正常训练得到的模型更新乘以一个较大的缩放因子，使得全局模型被大幅度拉偏。假设正常的模型更新为 $\Delta w$，攻击者上传 $\alpha \cdot \Delta w$，其中 $\alpha \gg 1$。在简单的联邦平均聚合中，这种被放大的更新会主导全局模型的方向。

更为高级的模型投毒策略还包括"后门攻击"（Backdoor Attack）。攻击者在本地模型中嵌入特定的触发模式（Trigger Pattern），例如在图像的特定位置添加一个小的像素图案。当输入包含该触发模式时，模型会输出攻击者指定的目标类别；而对于正常输入，模型的行为与未受攻击时几乎一致。这种攻击的危险性在于：全局模型在常规测试集上表现正常，仅在遇到包含触发模式的特定输入时才会表现出异常行为，因此极难被标准的模型评估流程检测到。

### 7.1.3 拜占庭攻击

拜占庭攻击（Byzantine Attack）源自分布式计算中的拜占庭将军问题，描述的是一类最为广泛的恶意行为模型。在联邦学习语境下，拜占庭攻击泛指任何违反协议的恶意参与方行为——包括但不限于上传随机噪声、发送与本地数据无关的任意模型更新、与其他恶意方合谋协调攻击等。拜占庭攻击的关键特征在于：攻击者的行为可以是任意的，不受任何合理假设的约束。

Blanchard等人（2017）在其开创性论文《Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent》中，首次系统地将拜占庭容错的概念引入分布式机器学习领域。他们指出，在传统的分布式梯度下降中，只要有一个拜占庭节点上传恶意梯度，就可能导致全局模型收敛到错误的解，甚至完全发散。这一工作直接催生了后续大量关于鲁棒聚合算法的研究。

拜占庭攻击的典型行为模式包括：

- **随机攻击**：恶意方上传服从均匀分布或高斯分布的随机向量作为模型更新，旨在通过引入噪声干扰全局模型的收敛过程。
- **符号翻转攻击**：恶意方将正常更新的方向取反，即上传 $-\Delta w$ 而非 $\Delta w$，试图使全局模型沿错误方向更新。
- **合谋攻击**：多个恶意方协调其上传的更新，使其合力足以在聚合结果中占据主导地位，从而系统性地操控全局模型。

拜占庭攻击的威胁程度取决于恶意参与方的比例和所采用的聚合策略。在最坏情况下，当恶意方比例接近或超过50%时，理论上不存在能够保证收敛到正确解的聚合策略（在无任何先验假设的情况下）。然而，当恶意方比例受到合理限制时（例如不超过30%），多种鲁棒聚合方法已被证明能够有效抵御拜占庭攻击。

## 7.2 推理攻击

如果说投毒攻击威胁的是模型的完整性（Integrity），那么推理攻击威胁的则是数据的机密性（Confidentiality）。推理攻击利用联邦学习过程中交换的信息——主要是模型更新（梯度）——来推断参与方的私有数据。这一威胁的根本原因在于：模型更新本质上是对训练数据的一种编码，其中不可避免地包含了关于训练数据的统计信息。

### 7.2.1 梯度反演攻击

梯度反演攻击（Gradient Inversion Attack）是推理攻击中最为直观且研究最为深入的一类。其核心思想是：给定某参与方上传的模型更新（梯度），通过优化方法反推出能够产生该梯度的训练数据。

Zhu等人（2019）在其里程碑式论文《Deep Leakage from Gradients》中提出了"深度梯度泄露"（DLG）方法，证明了仅凭共享的梯度信息就足以高保真地还原训练数据。DLG的基本流程如下：攻击者首先随机初始化一对虚拟输入 $x'$ 和虚拟标签 $y'$，然后计算以 $x'$ 和 $y'$ 为输入时模型产生的梯度 $g'$，并通过优化目标 $\min_{x', y'} \| g' - g \|^2$ 来寻找能够产生与真实梯度 $g$ 最接近的虚拟输入。在图像分类任务中，这种方法可以在数十次迭代内将原始训练图像几乎完美地还原出来。

后续研究进一步提升了梯度反演攻击的能力。Geiping等人（2020）提出的Inverting Gradients方法引入了图像先验（如总变差正则化）来提升还原质量，并证明该攻击在批量训练场景下同样有效。Yin等人（2021）则从理论角度分析了梯度反演的信息论界限，揭示了在何种条件下梯度泄露是不可避免的。

梯度反演攻击的有效性受以下因素影响：

- **批量大小**：当参与方在一次更新中使用多个样本进行训练时，梯度是多个样本梯度的平均，反演难度增大。但研究表明，即使批量大小为数十甚至数百，攻击者仍可部分还原训练样本。
- **模型架构**：全连接层比卷积层更易被反演，因为全连接层的权重与输入之间存在更直接的对应关系。
- **梯度精度**：浮点精度越高的梯度包含的信息越多，攻击越容易成功。

### 7.2.2 成员推断攻击

成员推断攻击（Membership Inference Attack）的目标不是还原训练数据的具体内容，而是判断某个特定样本是否被用于训练。这一攻击在隐私敏感场景下具有重要意义——例如，如果攻击者能够判断某位患者的医疗记录是否被用于训练某个疾病预测模型，就可能泄露该患者的健康状况。

Shokri等人（2017）首先在集中式机器学习中提出了成员推断攻击的系统框架。其基本思路是：机器学习模型对其训练集中的样本（成员）和不在训练集中的样本（非成员）往往表现出不同的"置信度"——模型通常对成员样本的预测更加确信。攻击者可以训练一个"攻击模型"，以目标模型的输出置信度向量作为输入，判断给定样本是否为成员。

在联邦学习场景下，成员推断攻击具有新的变体。攻击者可以是中央服务器，试图判断某个特定样本是否被某个参与方用于训练；也可以是恶意参与方，试图推断其他参与方的训练数据组成。Melis等人（2019）的研究表明，在联邦学习过程中，参与方之间的交互比集中式训练泄露了更多的成员信息，因为模型更新的精细结构可以揭示局部数据的分布特征。

### 7.2.3 属性推断攻击

属性推断攻击（Attribute Inference Attack）旨在从共享的模型更新中推断参与方数据集的全局统计属性，而非具体的单个样本。例如，攻击者可能试图推断某个参与方的训练数据中男女比例、特定疾病的患病率等敏感的群体统计信息。

Melis等人（2019）在《Inference Attacks Against Collaborative Learning》一文中系统研究了联邦学习中的属性推断攻击。他们发现，恶意参与方可以通过分析其他参与方的模型更新，训练一个分类器来推断对方数据集的特定属性。与成员推断不同，属性推断关注的是数据集层面的特征，其泄露的虽然不是个体信息，但在某些场景下（如推断某地区的人口构成）同样构成严重的隐私威胁。

属性推断攻击的原理在于：模型更新中编码了训练数据的分布信息。当参与方的数据具有某些特定的统计属性时（例如某个子群体在数据中占比较高），这种分布特征会反映在模型更新的结构中。恶意参与方可以利用这种相关性，通过分析多个轮次的模型更新来逐步提高属性推断的准确率。

## 7.3 防御方法

面对上述多样化的安全威胁，研究者们提出了多种防御机制。本节重点介绍鲁棒聚合策略，这类方法通过改进模型聚合规则来抵御投毒攻击。

### 7.3.1 鲁棒聚合的基本思想

标准的联邦平均（FedAvg）算法采用加权平均的方式聚合各参与方的模型更新：

$$w_{t+1} = \sum_{k=1}^{n} \frac{n_k}{n} w_t^k$$

其中 $n_k$ 为第 $k$ 个参与方的数据量，$n$ 为总数据量。这种简单的加权平均缺乏对异常更新的鲁棒性——即使只有一个恶意方上传了极端的模型更新，也可能对全局模型产生显著影响。

鲁棒聚合的核心思想是：在聚合过程中识别并排除或降低异常更新的权重，使得恶意更新无法主导全局模型的更新方向。这与统计学中鲁棒估计的思想一脉相承——通过使用对异常值不敏感的统计量（如中位数、截断均值等）来替代对异常值敏感的均值。

### 7.3.2 Krum算法

Blanchard等人（2017）提出的Krum算法是拜占庭容错聚合的开山之作。Krum的核心思想是：对于每一个候选更新，计算它与所有其他更新之间的欧氏距离之和，选择"最近邻距离之和最小"的更新作为本轮的全局更新。

具体而言，设 $n$ 个参与方提交的模型更新为 $\{g_1, g_2, \ldots, g_n\}$，假设有 $f$ 个恶意参与方。对于每个更新 $g_i$，Krum计算它与最近的 $n - f - 2$ 个更新之间的距离之和：

$$\text{score}(g_i) = \sum_{j \in \{i_1, i_2, \ldots, i_{n-f-2}\}} \| g_i - g_j \|^2$$

其中 $\{i_1, i_2, \ldots, i_{n-f-2}\}$ 是距离 $g_i$ 最近的 $n-f-2$ 个更新的索引集合。Krum选择得分最小的更新作为全局更新：

$$g_{\text{Krum}} = g_{i^*}, \quad i^* = \arg\min_i \text{score}(g_i)$$

Krum的直觉是：正常的更新由于都朝着正确的优化方向前进，彼此之间较为接近；而恶意更新由于方向异常，与其他更新之间的距离较大，因此不太可能被选中。

Blanchard等人从理论上证明了：当恶意参与方的比例 $f < n/2$ 时，Krum能够保证全局模型以线性速率收敛到一个与无攻击情况下相近的邻域内。然而，Krum的一个明显缺陷是每轮仅使用一个参与方的更新，这导致信息利用率较低，在参与方数据分布差异较大（Non-IID）时可能影响收敛速度。

为了弥补这一缺陷，Blanchard等人进一步提出了Multi-Krum变体：在选择第一个最优更新后，将Krum的选择过程在剩余更新上重复多次，最终对选出的若干个更新取平均。Multi-Krum在信息利用率和鲁棒性之间取得了更好的平衡。

### 7.3.3 中位数聚合

中位数聚合（Coordinate-wise Median）是另一种经典的鲁棒聚合方法。其思想简洁而有力：对于模型参数的每一个坐标维度，取所有参与方在该维度上的中位数作为全局更新在该维度的值。

形式化地，设 $g_i[j]$ 表示第 $i$ 个参与方的模型更新在第 $j$ 个坐标上的值，则中位数聚合的全局更新为：

$$g_{\text{Median}}[j] = \text{median}(g_1[j], g_2[j], \ldots, g_n[j]), \quad \forall j$$

中位数聚合的优势在于其对异常值具有天然的鲁棒性——即使有接近一半的参与方在某个维度上提交了任意的恶意值，只要正常参与方的值在该维度上集中在某个范围内，中位数仍然能够落在正常范围内。

Yin等人（2018）在《Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates》中对中位数聚合进行了严格的理论分析，证明了在高维线性回归等模型上，中位数聚合能够以接近最优的统计速率收敛，同时容忍最多 $f < n/2$ 个拜占庭节点。

然而，中位数聚合也存在局限性。其一，逐坐标中位数可能不对应于任何单个参与方提交的模型更新，这在某些应用场景下可能导致生成的模型缺乏一致性。其二，当恶意参与方的比例较高时，中位数可能被恶意值"包围"而偏离正常范围，尤其在高维空间中这种"中位数攻击"的威胁更为显著。

### 7.3.4 截断均值聚合

截断均值聚合（Trimmed Mean）借鉴了统计学中截断均值的思想：首先对每个坐标维度上的值进行排序，去掉最大的 $\beta$ 个值和最小的 $\beta$ 个值（$\beta$ 为预设的截断比例），然后对剩余的值取平均。

$$g_{\text{TM}}[j] = \frac{1}{n - 2\beta} \sum_{i \in \text{middle}} g_i[j]$$

其中 $\text{middle}$ 表示去掉两端各 $\beta$ 个极端值后的索引集合。截断比例 $\beta$ 通常设为恶意参与方比例 $f/n$ 的上界。

截断均值聚合在理论上同样具有良好的鲁棒性保证。Yin等人（2018）证明，截断均值聚合在保证收敛性的同时，能够容忍一定比例的拜占庭节点。与中位数聚合相比，截断均值的优势在于它利用了多个参与方的信息（而非仅取中位数），因此在恶意方比例较低时往往具有更低的方差和更快的收敛速度。

截断均值的一个实际挑战在于截断比例 $\beta$ 的选择。如果 $\beta$ 设置过小，可能无法有效排除恶意更新；如果 $\beta$ 设置过大，则可能误伤正常参与方的合法更新，尤其在Non-IID场景下，某些正常参与方的更新可能看起来像"离群值"。

### 7.3.5 差分隐私防御

差分隐私（Differential Privacy, DP）作为一种严格的隐私保护框架，也被引入联邦学习的防御体系中。在联邦学习语境下，差分隐私防御主要有两种应用方式：局部差分隐私（Local DP）和全局差分隐私（Global DP）。

在局部差分隐私模式下，每个参与方在上传模型更新之前，先对更新添加校准的噪声：

$$\tilde{g}_i = g_i + \mathcal{N}(0, \sigma^2 S^2 I)$$

其中 $S$ 为梯度的敏感度（通常通过梯度裁剪控制），$\sigma$ 为隐私预算相关的噪声参数。这种机制在保护隐私的同时，也为系统提供了一定程度的鲁棒性：恶意参与方精心构造的异常更新在添加噪声后，其异常特征可能被掩盖或分散，从而降低了定向攻击的效果。

在全局差分隐私模式下，噪声在中央服务器的聚合结果上添加：

$$\tilde{g}_{\text{global}} = \sum_{k=1}^{n} \frac{n_k}{n} g_k + \mathcal{N}(0, \sigma^2 S^2 I)$$

全局差分隐私的优势在于噪声量不随参与方数量增加而增大，因此在大规模联邦学习系统中具有更好的隐私-效用权衡。

Abadi等人（2016）提出的DP-SGD（差分隐私随机梯度下降）框架为差分隐私在深度学习中的应用奠定了基础。在联邦学习中，McMahan等人（2018）将DP-SGD与联邦平均算法相结合，提出了联邦差分隐私（Federated DP）方案，通过在客户端或服务器端添加噪声来实现差分隐私保证。

值得注意的是，差分隐私防御虽然提供了一定的鲁棒性，但其主要目的是隐私保护而非拜占庭容错。在恶意方比例较高的情况下，单纯依赖差分隐私的噪声来防御投毒攻击是不够的，通常需要与前述的鲁棒聚合方法结合使用。

## 7.4 安全聚合

安全聚合（Secure Aggregation）是联邦学习中保护梯度隐私的核心密码学机制。与前述的防御方法不同，安全聚合的目标不是抵御恶意行为，而是确保即使在通信链路被窃听或服务器半可信的情况下，参与方的模型更新也不会被任何单一实体（包括中央服务器）获取。

### 7.4.1 安全聚合协议

Bonawitz等人（2017）在其奠基性论文《Practical Secure Aggregation for Federated Learning on User-Held Data》中提出了适用于大规模联邦学习的安全聚合协议。该协议的核心思想是：每个参与方对自己的模型更新添加一个由成对共享秘密生成的掩码，使得所有掩码在聚合过程中相互抵消，服务器最终只能得到各参与方模型更新的总和，而无法获知任何单个参与方的更新内容。

协议的基本流程如下：

**第一阶段：密钥协商。** 每对参与方 $(i, j)$ 通过Diffie-Hellman密钥协商协议生成一个共享的秘密种子 $s_{i,j}$。具体而言，参与方 $i$ 和 $j$ 各自生成公私钥对，并交换公钥，从而各自独立计算出相同的共享种子 $s_{i,j} = s_{j,i}$。

**第二阶段：掩码生成。** 每个参与方 $i$ 利用共享种子生成掩码向量。对于每一对 $(i, j)$（$i \neq j$），使用伪随机数生成器 $\text{PRG}$ 从种子 $s_{i,j}$ 生成一个与模型更新同维度的伪随机向量 $p(s_{i,j})$。参与方 $i$ 最终上传的被掩盖的更新为：

$$u_i' = u_i + \sum_{j: j < i} p(s_{i,j}) - \sum_{j: j > i} p(s_{i,j}) + r_i$$

其中 $u_i$ 为参与方 $i$ 的原始模型更新，$r_i$ 为参与方 $i$ 与服务器共享的随机向量（用于在参与方掉线时进行恢复）。

**第三阶段：聚合。** 当服务器收到所有参与方的被掩盖更新后，进行求和：

$$\sum_i u_i' = \sum_i u_i + \underbrace{\sum_i \left( \sum_{j: j < i} p(s_{i,j}) - \sum_{j: j > i} p(s_{i,j}) \right)}_{= 0 \text{（对称性使掩码相互抵消）}} + \sum_i r_i$$

由于掩码的对称设计，所有成对掩码在求和中恰好相互抵消，服务器减去所有 $r_i$ 后即可得到 $\sum_i u_i$，而无法获知任何单独的 $u_i$。

**第四阶段：掉线处理。** 在实际的联邦学习系统中，参与方可能在协议执行过程中掉线。Bonawitz等人的协议通过引入与服务器共享的 $r_i$ 向量来处理这一情况：当某个参与方掉线时，服务器可以利用 $r_i$ 来"消除"该参与方的掩码，从而在不泄露其他参与方更新的情况下完成聚合。

该协议的安全性基于以下假设：服务器与参与方之间是半诚实的（遵循协议但试图推断额外信息），且恶意参与方的数量不超过预设阈值。在这些假设下，协议能够保证任何参与方的模型更新对服务器和其他参与方都是信息论安全的。

### 7.4.2 掩盖梯度技术

安全聚合协议中的掩盖梯度技术有多种实现方式，各有其适用场景和权衡。

**基于秘密共享的掩盖。** 除了Bonawitz等人提出的基于Diffie-Hellman的方案外，还可以使用加法秘密共享（Additive Secret Sharing）来实现梯度掩盖。每个参与方将自身的模型更新拆分为 $n$ 份秘密份额，分发给包括自己在内的所有参与方。每个参与方将收到的所有份额求和后上传给服务器。由于每个参与方上传的份额中包含了来自所有其他参与方的部分，服务器无法从中分离出任何单个参与方的更新。

**基于同态加密的聚合。** 同态加密（Homogeneous Encryption）提供了另一种安全聚合的途径。在加法同态加密方案下（如Paillier加密），各参与方用服务器的公钥加密自身的模型更新后上传。服务器可以在密文空间上直接执行加法运算：

$$\text{Enc}\left(\sum_i u_i\right) = \prod_i \text{Enc}(u_i)$$

随后，服务器用私钥解密得到聚合结果 $\sum_i u_i$，但无法获取任何单独的 $u_i$。同态加密的优势在于无需复杂的多轮密钥协商，但其计算开销和通信开销通常远高于基于掩码的方案，因此在模型参数量较大时需要权衡效率。

**差分隐私与安全聚合的结合。** 将差分隐私与安全聚合相结合是当前联邦学习隐私保护的主流范式。在这种组合方案下，每个参与方首先对模型更新添加差分隐私噪声，然后通过安全聚合协议上传被掩盖的加噪更新。这种双层保护机制提供了更强的隐私保证：差分隐私确保了即使安全聚合协议被部分突破，单个参与方的隐私仍受到保护；安全聚合则确保了差分隐私噪声的添加不会被绕过。

### 7.4.3 验证机制

安全聚合协议面临的一个重要挑战是验证机制——如何确保各参与方诚实地执行协议，而非上传虚假的被掩盖更新。在缺乏验证机制的情况下，恶意参与方可以通过以下方式破坏安全聚合的正确性：

- 上传与自身模型更新无关的任意值，导致聚合结果偏离真实的模型更新总和。
- 在密钥协商阶段提交无效的公钥，使得最终的掩码无法正确抵消。
- 在聚合阶段声称掉线，但实际仍参与了密钥协商，从而破坏掉线处理机制。

为应对这些挑战，研究者们提出了多种验证机制：

**承诺方案（Commitment Scheme）。** 参与方在密钥协商阶段对自身的公钥和掩码进行承诺（即发布其哈希值），在后续阶段揭示原始值。服务器可以通过检查承诺的一致性来验证参与方是否在协议执行过程中修改了自身的输入。这种机制可以防止参与方在看到其他方的更新后选择性地修改自身的贡献。

**零知识证明。** 参与方可以通过零知识证明向服务器证明自身的模型更新确实是在本地数据上按照规定的算法训练得到的，而无需泄露模型更新的具体内容。这种方法的计算开销较大，但在安全性要求极高的场景下具有不可替代的价值。

**可信执行环境。** 硬件级的可信执行环境（Trusted Execution Environment, TEE，如Intel SGX）为安全聚合提供了另一种验证途径。参与方将模型更新发送到TEE中进行聚合，TEE的硬件保证了运行在其中的代码和数据不会被外部访问或篡改。然而，TEE方案的安全性依赖于对硬件制造商的信任，且近年来针对SGX的侧信道攻击不断被发现，因此其安全性边界仍在学术讨论中。

**可审计的安全聚合。** So等人（2021）提出了可审计的安全聚合协议，允许在事后对聚合结果的正确性进行验证，同时不泄露单个参与方的更新内容。这种协议通过在聚合过程中嵌入可验证的校验信息，使得任何第三方审计者都可以确认服务器是否诚实地执行了聚合操作。

## 7.5 本章小结

本章系统分析了联邦学习面临的安全威胁及其防御体系。投毒攻击——包括数据投毒、模型投毒和拜占庭攻击——从完整性维度威胁着联邦学习系统；推理攻击——包括梯度反演、成员推断和属性推断——从机密性维度构成隐私泄露风险。在防御方面，鲁棒聚合策略（Krum、中位数、截断均值等）通过改进聚合规则来抵御投毒攻击，差分隐私防御则在保护隐私的同时提供了一定的鲁棒性保障，安全聚合协议从密码学层面确保了梯度信息不会被泄露给未授权方。

值得强调的是，没有单一的防御机制能够同时解决所有安全问题。在实际部署的联邦学习系统中，通常需要将多种防御手段组合使用：鲁棒聚合来抵御投毒，差分隐私来防止推理，安全聚合来保护通信隐私。这种多层防御的理念与传统信息安全中的"纵深防御"原则一脉相承。联邦学习安全是一个攻防持续对抗的领域，随着攻击手段的不断演进，防御技术也需要持续创新。

---

**参考文献**

- Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. *Advances in Neural Information Processing Systems*, 30.
- Bonawitz, K., Ivanov, V., Kreuter, B., et al. (2017). Practical Secure Aggregation for Federated Learning on User-Held Data. *NIPS Workshop on Private Multi-Party Machine Learning*.
- Fang, M., Cao, X., Jia, J., & Gong, N. (2020). Local Model Poisoning Attacks to Byzantine-Robust Federated Learning. *USENIX Security Symposium*.
- Geiping, J., Bauermeister, H., Dröge, H., & Moeller, M. (2020). Inverting Gradients: How Easy Is It to Break Privacy in Federated Learning? *Advances in Neural Information Processing Systems*, 33.
- Melis, L., Song, C., De Cristofaro, E., & Shmatikov, V. (2019). Inference Attacks Against Collaborative Learning. *ACM Conference on Computer and Communications Security*.
- McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning Differentially Private Recurrent Language Models. *International Conference on Learning Representations*.
- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership Inference Attacks Against Machine Learning Models. *IEEE Symposium on Security and Privacy*.
- So, J., Güler, B., & Avestimehr, A. S. (2021). Byzantine-Resilient Secure Federated Learning. *IEEE Journal on Selected Areas in Communications*.
- Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. *International Conference on Machine Learning*.
- Yin, H., Mallya, A., Vahdat, A., et al. (2021). See Through Gradients: Image Batch Recovery via GradInversion. *IEEE/CVF Conference on Computer Vision and Pattern Recognition*.
- Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients. *Advances in Neural Information Processing Systems*, 32.
# 第8章 可信执行环境

## 8.1 引言

在隐私计算的技术谱系中，可信执行环境（Trusted Execution Environment, TEE）占据着独特的位置。与同态加密依赖纯密码学运算、安全多方计算依赖协议交互不同，TEE 依托处理器硬件提供的隔离机制，在性能与安全性之间取得了引人注目的平衡。数据在 TEE 中以明文形式被处理——这赋予了它接近原生计算的执行效率；同时，TEE 通过硬件级的内存加密与访问控制，确保即便操作系统内核、虚拟机监控器（Hypervisor）乃至物理层面的攻击者，也无法窥探或篡改 TEE 内部的代码与数据。

这一技术路线的核心思想可以追溯到可信计算（Trusted Computing）的早期探索。从 TPM（可信平台模块）提供的静态度量，到如今以 Intel SGX、ARM TrustZone、Intel TDX 为代表的动态隔离执行环境，TEE 的演进历程折射出整个行业对"数据在使用过程中如何保护"这一根本问题的持续追问。尤其在人工智能时代，模型推理、联邦学习聚合、隐私保护训练等场景对安全计算的效率提出了前所未有的要求，TEE 正在从传统的密钥管理、DRM 等领域，迅速扩展为隐私计算基础设施的核心组件。

本章将系统梳理三种主流 TEE 技术——Intel SGX、ARM TrustZone 和 Intel TDX 的架构设计与安全机制，深入分析它们各自面临的侧信道攻击威胁与应对策略，并重点探讨 TEE 在人工智能与联邦学习中的典型应用模式。

---

## 8.2 Intel SGX：进程级硬件隔离

### 8.2.1 架构概述

Intel Software Guard Extensions（SGX）是 Intel 在第六代酷睿处理器（Skylake）中引入的一组 x86 指令集扩展，其核心目标是允许应用程序在操作系统和虚拟机监控器均不可信的前提下，保护敏感代码与数据的机密性和完整性。

SGX 的关键抽象是**飞地（Enclave）**。一个 Enclave 是应用程序地址空间中的一块受保护区域，它在物理内存中由处理器硬件进行加密和访问控制。具体而言，SGX 引入了一块专用的物理内存区域——**Enclave 页面缓存（EPC, Enclave Page Cache）**，该区域的内容由内存加密引擎（MEE）在写入 DRAM 时自动加密，读取时自动解密。任何非 Enclave 内部的软件访问 EPC 页面，只会得到加密后的密文。

Enclave 的创建与加载过程遵循严格的流程：开发者首先通过 Intel 提供的工具链将受保护的代码编译为 Enclave 二进制文件，随后在运行时通过 `ECREATE`、`EADD`、`EEXTEND` 等专用指令将代码和数据页面逐一加载到 EPC 中，并计算度量值（Measurement）。这一度量值是 Enclave 身份的核心标识，后续的远程证明（Remote Attestation）正是基于该度量值来验证 Enclave 的真实性。

### 8.2.2 内存保护与隔离机制

SGX 的内存保护体系建立在多个层次之上。在硬件层面，Enclave 页面表项（EPT）被标记为仅允许 Enclave 内部代码访问；处理器在执行 Enclave 代码时，会检查每次内存访问是否源自 Enclave 内部，若非如此则触发访问违规。这种检查在处理器微架构层面完成，绕过了操作系统的页面表管理。

在 Enclave 的入口与出口控制上，SGX 设计了专门的指令：`EENTER` 用于从非安全上下文进入 Enclave，`EEXIT` 用于从 Enclave 退出，`ERESUME` 用于在中断或异常处理后恢复 Enclave 执行。这些指令确保了上下文切换时 Enclave 寄存器状态不会泄露到不可信环境中。

此外，SGX 实施了**单调计数器（Monotonic Counter）**和**密封（Sealing）**机制。密封允许 Enclave 将数据加密后持久化存储，密钥由平台和 Enclave 身份绑定，只有在相同平台上运行的相同 Enclave（或指定策略下的 Enclave）才能解封。这为 Enclave 提供了跨重启的持久化安全保障。

### 8.2.3 远程证明

远程证明（Remote Attestation）是 SGX 安全模型的基石之一。它解决了一个关键问题：远程用户如何确信自己正在与一个真正的、运行预期代码的 Enclave 通信？

SGX 的远程证明流程如下：Enclave 内部调用 `EREPORT` 指令生成一份包含 Enclave 度量值和用户数据（如公钥）的报告结构体，该报告随后被发送至 Intel 的证明服务（Attestation Service, IAS）。IAS 验证报告的签名和平台信息后，签发一份证明响应（Attestation Verification Report），远程用户通过验证该响应来确认 Enclave 的身份和完整性。

值得注意的是，SGXv2（Ice Lake 及后续平台）引入了**可扩展 Enclave（Scalable Enclave）**的概念，支持动态调整 EPC 大小，缓解了早期 SGX 版本中 EPC 容量有限（通常仅 128MB~256MB）导致的内存瓶颈问题。同时，SGXv2 的远程证明架构从依赖 IAS 转向了更灵活的 DCAP（Data Center Attestation Primitives）方案，允许数据中心自行部署证明服务，降低了对 Intel 在线服务的依赖。

### 8.2.4 侧信道攻击：SGX 的阿喀琉斯之踵

尽管 SGX 在内存隔离层面提供了强安全保障，但自其发布以来，研究者陆续揭示了多种侧信道攻击（Side-Channel Attack）方法，构成了 SGX 最严峻的安全威胁。

**缓存侧信道攻击**是最早被发现且影响最广泛的一类。典型方法包括：

- **Prime+Probe**：攻击者预先用特定数据填充 CPU 缓存组，然后等待 Enclave 执行完毕后测量缓存状态变化，从而推断 Enclave 的内存访问模式。
- **Flush+Reload**：攻击者驱逐特定共享内存行，等待 Enclave 执行后重新加载该行并测量访问时间，以判断 Enclave 是否访问了该地址。
- **SGX-Step**：开源框架，利用中断机制在每条指令粒度上中断 Enclave 执行，配合缓存时序分析实现精确的指令级侧信道攻击。

**分支预测侧信道**（Spectre 变种）也被证明可应用于 SGX 环境。攻击者通过操纵分支预测器的状态，诱导 Enclave 执行推测性代码路径，进而通过缓存隐信道泄露 Enclave 的敏感数据。

**页表侧信道**则利用了 SGX 缺页异常的可观察性。操作系统负责管理 Enclave 的页面调度，攻击者可以通过监控缺页异常来推断 Enclave 的内存访问模式，进而恢复加密密钥或其他秘密。

针对这些攻击，学术界提出了多种防御策略：**Haven** 和 **SGX-LAPD** 通过常量时间编程消除数据依赖的访问模式；**T-SGX** 和 **Déjà Vu** 利用事务内存（Intel TSX）来隐藏异常行为；**MI6** 和 **Cloak** 通过预加载所有可能访问的页面来消除页面粒度的信息泄露。然而，这些防御手段往往以性能损失为代价，且难以覆盖所有攻击向量，侧信道防御至今仍是 SGX 生态系统中未完全解决的开放问题。

---

## 8.3 ARM TrustZone：系统级安全隔离

### 8.3.1 架构设计

ARM TrustZone 是 ARM 公司从 ARMv6 架构开始引入的系统级安全技术。与 SGX 的进程级隔离不同，TrustZone 在处理器层面划分出两个独立的执行环境——**安全世界（Secure World）**和**普通世界（Normal World）**，二者通过硬件级别的总线隔离实现物理层面的安全边界。

TrustZone 的核心机制是处理器的**安全状态位（NS bit, Non-Secure bit）**。该位嵌入在处理器的每一次总线事务中，由总线主设备（如 CPU 核心、DMA 控制器）发出，由总线从设备（如内存控制器、外设控制器）检查。当 NS=0 时，当前事务处于安全世界，可以访问安全和非安全资源；当 NS=1 时，当前事务处于普通世界，只能访问非安全资源。这种设计将安全隔离的粒度从软件层提升到了硬件总线层，使得即便是操作系统内核也无法突破世界间的边界。

在两个世界的切换方面，ARM 引入了**监控模式（Monitor Mode）**作为世界切换的安全代理。当普通世界需要调用安全世界的服务时，通过 `SMC（Secure Monitor Call）`指令触发世界切换，控制权转移到监控模式中的安全监控程序（Secure Monitor），由其负责保存和恢复两个世界的上下文，并将控制权安全地转发到目标世界。

### 8.3.2 OP-TEE：开源可信操作系统

在 TrustZone 的安全世界中，需要一个专门的操作系统来管理安全应用的执行。OP-TEE（Open Portable Trusted Execution Environment）是目前最广泛使用的开源 TEE 操作系统，由 STMicroelectronics 最初开发，现由 Linaro 社区维护，符合 GlobalPlatform TEE 规范。

OP-TEE 的架构分为以下关键组件：

- **OP-TEE OS**：运行在安全世界中的微内核操作系统，负责任务调度、内存管理、安全存储和密码学服务。
- **可信应用（Trusted Application, TA）**：运行在 OP-TEE OS 之上的安全应用，每个 TA 拥有独立的地址空间和安全上下文。TA 通过 GlobalPlatform 定义的 TEE Internal Core API 与 OP-TEE OS 交互。
- **TEE 客户端 API**：运行在普通世界中的客户端库，允许普通世界应用程序通过标准接口调用 TA 的服务。
- **tee-supplicant**：运行在普通世界中的辅助进程，负责为安全世界提供文件系统访问等辅助服务。

OP-TEE 支持多种 ARM 平台，包括 Cortex-A 系列处理器，并已在众多移动设备和嵌入式系统中得到实际部署。Android 设备中的指纹识别、移动支付、DRM 内容保护等关键安全功能，很多都是基于 TrustZone 和 OP-TEE 实现的。

### 8.3.3 TrustZone 的安全特性与局限

TrustZone 提供的安全特性包括：

- **安全启动（Secure Boot）**：从芯片上电开始，每一步启动代码都经过签名验证，确保安全世界的软件栈未被篡改。
- **安全存储**：通过加密文件系统和硬件绑定密钥，保护持久化数据的机密性和完整性。存储在安全世界中的密钥和证书无法被普通世界读取。
- **安全外设**：通过总线级别的安全属性控制，特定外设（如安全传感器、密码学加速器）可以被配置为仅安全世界可访问。

然而，TrustZone 也存在一些固有局限。首先，安全世界和普通世界共享同一个物理处理器核心，这意味着安全世界的计算会与普通世界竞争 CPU 资源，可能引发基于时间的侧信道泄露。其次，TrustZone 的安全世界通常运行的是轻量级操作系统，其计算和内存资源有限，难以支持大规模的安全计算任务。此外，TrustZone 的信任根高度依赖于芯片制造商的安全设计，不同厂商的实现质量参差不齐，部分厂商的 TrustZone 实现已被发现存在安全漏洞。

### 8.3.4 与 SGX 的对比

TrustZone 与 SGX 代表了两种截然不同的 TEE 设计哲学。TrustZone 采用"大隔离"策略，将整个系统划分为两个世界，安全世界拥有完整的操作系统和资源管理能力，适合运行大型安全服务。SGX 则采用"小飞地"策略，在不可信的操作系统内部创建细粒度的安全区域，适合保护特定的代码片段。

在开发模型上，TrustZone 的 TA 开发需要深入了解安全世界的操作系统接口和资源约束，开发门槛较高；SGX 的 Enclave 开发则更接近常规应用编程，开发者可以复用大部分现有的 C/C++ 代码。在应用场景上，TrustZone 更多用于嵌入式和移动设备的安全服务，而 SGX 则主要面向服务器和云计算场景。

---

## 8.4 Intel TDX：虚拟机级机密计算

### 8.4.1 从 SGX 到 TDX 的演进

尽管 SGX 在进程级安全隔离方面取得了突破，但在云计算场景中，租户往往需要的不仅仅是进程级隔离，而是整个虚拟机级别的隔离——即确保虚拟机内部的所有代码、数据乃至操作系统内核，都不受云服务商（包括 Hypervisor 和主机操作系统）的窥探和篡改。Intel Trust Domain Extensions（TDX）正是在这一需求背景下应运而生的。

TDX 首次在第四代 Intel Xeon 可扩展处理器（Sapphire Rapids）中引入，随后在第五代 Xeon（Emerald Rapids）和后续平台中持续增强。其核心思想是将 SGX 的硬件隔离理念从进程级扩展到虚拟机级，创建一种称为**信任域（Trust Domain, TD）**的新型虚拟化隔离单元。

### 8.4.2 架构与安全模型

TDX 的架构围绕一个关键组件——**TDX 模块（TDX Module）**展开。该模块是一段经过 Intel 签名和验证的安全固件，运行在处理器的安全仲裁模式（SEAM, Secure Arbitration Mode）中，处于比 Hypervisor 更高的特权级别。TDX 模块负责 TD 的创建、内存管理、虚拟中断注入等核心虚拟化功能，取代了传统虚拟化架构中由 Hypervisor 承担的这些职责。

在内存保护方面，TDX 利用 Intel 的**多密钥全内存加密（MK-TME, Multi-Key Total Memory Encryption）**技术，为每个 TD 分配独立的加密密钥。TD 的所有内存页面在写入 DRAM 时由硬件自动加密，读取时自动解密。Hypervisor 无法读取 TD 的内存内容，也无法通过内存重映射（Remapping）攻击将恶意页面注入 TD 的地址空间，因为 TDX 模块会对每次内存映射操作进行完整性验证。

TDX 的虚拟化模型与传统虚拟化有本质区别。在传统虚拟化中，Hypervisor 拥有对虚拟机的完全控制权——它可以读取虚拟机内存、拦截中断、注入代码。而在 TDX 模型中，Hypervisor 的角色被大幅削弱：它仅负责物理资源的调度（如 CPU 时间片分配），而对 TD 内部的状态一无所知。TD 的虚拟中断、I/O 操作等通过 TDX 模块进行安全中介，Hypervisor 无法在这些操作中植入恶意逻辑。

### 8.4.3 远程证明与机密虚拟机

TDX 继承并扩展了 SGX 的远程证明机制。每个 TD 在创建时，TDX 模块会为其生成一份**TD 报告（TD Report）**，其中包含 TD 的初始度量值（由 TD 的启动镜像决定）、平台信息和用户数据。该报告可被转发至 Intel 的证明服务或通过 DCAP 机制进行本地验证。

在实际部署中，TDX 通常与**机密虚拟机（Confidential VM, CVM）**概念结合使用。云服务商（如 Azure、GCP、阿里云）提供的机密虚拟机实例，底层正是基于 TDX（或 AMD SEV 等类似技术）实现的。租户可以在机密虚拟机中运行任意操作系统和应用，而云服务商的运维人员、Hypervisor 乃至物理层面的攻击者，都无法访问虚拟机内部的明文数据。

### 8.4.4 TDX 与 SGX 的关系

TDX 和 SGX 并非替代关系，而是互补关系。SGX 适用于需要在不可信操作系统内部保护特定代码片段的场景，其安全粒度更细，但编程模型更受限（Enclave 内部不能直接进行系统调用）。TDX 则适用于需要保护整个工作负载的场景，其编程模型与传统虚拟机无异，开发者无需修改代码即可获得机密计算保护。

在某些高级部署模式中，TDX 和 SGX 可以组合使用：TDX 提供虚拟机级的隔离基础，SGX 在虚拟机内部进一步为特定高敏感操作提供进程级的额外保护层。这种分层防御策略在安全需求极高的场景中具有实际价值。

---

## 8.5 TEE 在人工智能中的应用

### 8.5.1 安全推理（Private Inference）

模型推理是 TEE 在 AI 领域最直接的应用场景。其基本模式是：模型所有者将训练好的模型部署到 TEE 内部，用户将输入数据加密后发送至 TEE，TEE 内部解密数据、执行推理、加密输出结果后返回。在整个过程中，模型参数和用户数据均以明文形式存在于 TEE 内部，但对外部不可见。

这一模式面临的主要挑战是**效率**。深度学习模型（尤其是大型语言模型）通常需要大量的内存和计算资源，而 TEE 的内存容量和计算能力往往受到限制。以 SGX 为例，EPC 的容量限制意味着大型模型无法完全驻留在 Enclave 内存中，需要频繁的页面换入换出（Paging），这不仅影响性能，还可能引入基于页面访问模式的侧信道泄露。

为应对这一挑战，研究者提出了多种优化策略。**Oblivious Inference** 通过使内存访问模式与输入数据无关来消除侧信道泄露，典型方法包括使用 ORAM（Oblivious RAM）来混淆内存访问模式。**混合精度推理**利用 TEE 内部的模型量化技术减少内存占用，同时保持推理精度。**分层执行**策略将模型的不同层分配到 TEE 内部和外部执行，在安全性与效率之间取得平衡。

微软的 **Olive** 框架和 **CrypTFlow2** 等工作探索了将 TEE 与安全多方计算相结合的混合推理方案：TEE 负责计算密集型的操作（如矩阵乘法），而密码学协议负责轻量级的非线性操作（如 ReLU 激活函数），从而在整体效率和安全性上取得更好的折中。

### 8.5.2 联邦学习中的安全聚合

联邦学习的核心挑战之一是：中央聚合服务器在聚合来自各方的模型更新时，如何确保不泄露任何一方的本地模型信息？TEE 为这一问题提供了一种直观而高效的解决方案。

**基于 TEE 的安全聚合**基本架构如下：各方将本地模型更新加密后上传至运行在 TEE 内部的聚合服务器。TEE 内部解密各方的更新，执行加权平均等聚合操作，然后将聚合后的全局模型更新加密后发布。由于聚合过程在 TEE 内部以明文进行，无需引入复杂的密码学聚合协议，计算开销远低于基于同态加密或安全多方计算的方案。

IEEE 发表的 **OPSA**（One-Pass Secure Aggregation）工作进一步优化了这一模式。OPSA 利用 TEE 实现了单轮通信的安全聚合——各方只需上传一次模型更新，TEE 即可完成聚合和验证，无需多轮交互。这显著降低了联邦学习的通信开销，特别是在大规模参与方场景下。

然而，基于 TEE 的安全聚合也面临信任假设的讨论。该方案将信任集中于 TEE 硬件和远程证明机制：如果 TEE 的硬件或固件存在漏洞，或者远程证明过程被攻破，整个安全聚合的保障将失效。因此，一种更稳健的设计是将 TEE 与密码学方法结合——TEE 提供高效的明文计算能力，密码学协议提供"安全兜底"，即使 TEE 被攻破，密码学层仍能提供基本的隐私保护。

### 8.5.3 隐私保护训练

相比推理和聚合，隐私保护模型训练对 TEE 的要求更高。训练过程涉及大量的前向传播和反向传播计算，数据和梯度的流动更为复杂，攻击面也更大。

**混合训练框架**是当前的主流方案。**HT2ML**（Hybrid TEE-HE for Privacy-Preserving ML）框架将训练任务在 TEE 和同态加密之间动态分配：梯度计算等密集型操作在 TEE 内部完成（利用其明文处理的高效性），而需要多方协作的聚合操作则使用同态加密来保护各方的梯度隐私。这种混合方案在隐私保护强度和计算效率之间取得了较好的平衡。

**分层动态防护**是另一种创新思路。研究者观察到，在深度神经网络的不同层中，敏感信息的分布是不均匀的——靠近输入的层（如特征提取层）通常承载更多与原始数据相关的隐私信息，而靠近输出的层（如分类层）的隐私敏感度较低。基于这一洞察，分层动态防护算法仅对隐私敏感度高的层在 TEE 内部进行保护性计算，而对低敏感度层在普通环境中执行，从而在不显著降低安全性的前提下大幅提升训练效率。

此外，**梯度压缩与安全聚合的结合**也受到关注。在联邦学习训练中，各方上传的梯度向量维度通常很高，直接在 TEE 内部聚合所有维度的梯度可能超出 TEE 的内存容量。通过梯度压缩（如 Top-K 稀疏化、量化）技术，在上传前减少梯度的数据量，再由 TEE 完成安全聚合，可以在有限的 TEE 资源下支持更大规模的模型训练。

### 8.5.4 TEE 与其他隐私计算技术的融合

TEE 在隐私计算中的价值并非作为孤立的技术方案，而更多体现在与其他隐私计算技术的协同融合中：

- **TEE + 差分隐私**：TEE 提供可信的计算环境来执行差分隐私机制（如添加校准噪声），确保噪声添加过程本身的可信性。这对于需要"可信数据 curator"的集中式差分隐私场景尤为重要。
- **TEE + 安全多方计算**：TEE 作为 MPC 协议中的高效"辅助节点"，承担计算密集型的操作，而 MPC 协议负责在多方之间安全地分发输入和收集输出。
- **TEE + 联邦学习**：如前所述，TEE 为联邦学习提供高效的安全聚合服务，同时通过远程证明机制为参与方提供可验证的安全保障。

这种技术融合趋势反映了隐私计算领域的一个基本共识：没有单一技术能够在所有场景下提供完美的隐私保护，多种技术的有机组合才是解决实际问题的有效路径。TEE 在其中扮演的角色，是以较低的性能代价提供一个"可信的计算基点"，使得其他隐私计算技术可以在其上构建更高效、更安全的解决方案。

---

## 8.6 TEE 的局限与未来展望

### 8.6.1 当前局限

尽管 TEE 在隐私计算中展现了巨大潜力，但其当前仍存在若干不容忽视的局限：

**硬件信任假设**是 TEE 最根本的争议点。所有 TEE 方案都要求用户信任硬件制造商（Intel、ARM 等）的安全设计和实现。对于某些高安全需求的场景（如国家级别的数据保护），将信任根完全建立在单一硬件厂商之上可能存在顾虑。

**侧信道攻击**仍是最活跃的威胁向量。如前所述，针对 SGX 的侧信道攻击研究层出不穷，虽然 TDX 在架构设计上对部分攻击（如页面表侧信道）进行了缓解，但基于微架构的侧信道泄露（如缓存时序、功耗分析）仍然是难以完全消除的威胁。

**可用性与可扩展性**方面，TEE 的内存容量和计算资源仍然有限，尤其是在边缘设备上。TrustZone 的安全世界资源紧张，SGX 的 EPC 容量受限，这些都制约了 TEE 在大规模 AI 任务中的应用。

### 8.6.2 未来趋势

展望未来，TEE 技术正在朝以下方向演进：

**机密计算标准化**：由 Linux 基金会下的机密计算联盟（Confidential Computing Consortium, CCC）推动的 **Enarx** 项目，旨在提供一个跨硬件平台的 TEE 运行时框架，使开发者无需关心底层是 SGX、TrustZone 还是 TDX，即可编写一次、到处运行的机密计算应用。这一标准化努力有望大幅降低 TEE 的使用门槛。

**硬件能力增强**：下一代处理器正在持续增强 TEE 的能力。例如，Intel 的 TDX 持续扩展对 GPU、I/O 设备的机密计算支持，使 TEE 不再局限于 CPU 密集型计算，而能覆盖 AI 训练中越来越重要的 GPU 加速场景。

**异构 TEE 融合**：未来的隐私计算系统可能同时利用多种 TEE 技术——在云端使用 TDX 保护整个虚拟机工作负载，在终端使用 TrustZone 保护数据采集和预处理，在边缘使用 SGX 保护模型推理——形成端到端的机密计算链路。

---

## 8.7 本章小结

本章系统介绍了可信执行环境（TEE）的三大主流技术路线：Intel SGX 提供了进程级的硬件隔离与远程证明机制，是服务器端机密计算的先驱；ARM TrustZone 通过系统级的安全世界/普通世界隔离，在移动和嵌入式领域构建了成熟的安全基础设施；Intel TDX 将硬件隔离扩展到虚拟机级别，为云原生机密计算提供了可扩展的解决方案。

在 AI 应用层面，TEE 正在从传统的密钥管理和安全存储场景，迅速扩展为安全推理、联邦学习聚合和隐私保护训练的重要技术支撑。尽管侧信道攻击、硬件信任假设等挑战依然存在，但 TEE 与差分隐私、安全多方计算、同态加密等技术的融合互补，正在构建一个更加完善、高效的隐私计算技术生态。

可信执行环境的本质，是在硬件层面为"数据在使用过程中如何保护"这一根本问题提供了一种工程化的解答。随着机密计算技术的持续演进和标准化推进，TEE 有望成为未来隐私计算基础设施中不可或缺的基石。
# 第9章 医疗健康

> "医学的未来不在于某一家医院的数据量，而在于全球医疗数据的安全协作。"

医疗健康领域是隐私计算与联邦学习最具社会价值的应用场景之一。医疗数据天然具有高度敏感性，同时又蕴含着巨大的科学价值——从罕见病诊断到新药研发，从精准医疗到公共卫生预警，每一个突破都依赖于大规模、高质量的医疗数据。然而，数据隐私法规、机构间的数据孤岛以及患者信任问题，长期制约着医疗数据的流通与利用。联邦学习与隐私计算技术为这一困境提供了全新的解决范式：在数据不出域的前提下，实现跨机构的协作建模与知识共享。

本章将系统探讨医疗数据隐私的核心挑战、联邦学习在医疗场景中的典型应用、隐私保护医疗AI的技术路径，并通过国际典型案例展示这一领域的前沿实践。

## 9.1 医疗数据隐私的核心挑战

### 9.1.1 医疗数据的特殊敏感性

医疗数据被称为"数据皇冠上的宝石"，其敏感性远超一般个人信息。一份完整的电子健康记录（Electronic Health Record, EHR）可能包含患者的基因组信息、疾病诊断、用药历史、手术记录、影像数据、心理评估甚至生活习惯等多维度信息。这些数据一旦泄露，可能导致患者遭受就业歧视、保险拒保、社会污名化等严重后果。

与金融或消费数据不同，医疗数据具有几个独特属性：第一，**不可更改性**——基因信息和既往病史无法像密码一样重置；第二，**关联推断性**——看似无害的数据片段组合后可能推断出敏感信息，例如通过处方记录推断HIV感染状态；第三，**长期价值性**——医疗数据的研究价值随时间增长，一份数十年前的罕见病病例可能对当下研究至关重要。

### 9.1.2 HIPAA与全球医疗数据法规

1996年美国颁布的《健康保险可携性和责任法案》（Health Insurance Portability and Accountability Act, HIPAA）是全球最具影响力的医疗数据保护法规之一。HIPAA的隐私规则（Privacy Rule）明确规定了"受保护健康信息"（Protected Health Information, PHI）的使用和披露条件，要求医疗机构在数据共享前必须获得患者授权或进行去标识化处理。HIPAA的安全规则（Security Rule）则对电子PHI的存储、传输和访问控制提出了技术性要求。

在欧洲，《通用数据保护条例》（GDPR）将健康数据列为"特殊类别个人数据"，适用更严格的处理条件。中国的《个人信息保护法》和《数据安全法》同样对医疗健康信息设置了专门的保护条款，要求对敏感个人信息的处理取得单独同意。

这些法规共同构建了一个严格的合规框架。一方面，它们有效保护了患者隐私；另一方面，也在客观上加剧了数据孤岛问题——医疗机构出于合规顾虑，往往对数据共享持保守态度。

### 9.1.3 数据孤岛与协作困境

医疗数据孤岛是一个全球性难题。在中国，三甲医院与基层医疗机构之间、不同地区的医院之间、公立医院与私立医院之间，数据的互联互通程度极低。据估计，中国医疗机构之间的数据共享率不足10%。在美国，尽管2009年《HITECH法案》推动了电子病历的广泛采用，但不同电子健康记录系统之间的互操作性问题依然突出。

数据孤岛的成因是多方面的：技术层面，不同医院采用不同的信息系统标准（HL7 FHIR、DICOM等标准虽已存在，但落地参差不齐）；制度层面，医院将数据视为核心资产，缺乏共享的激励机制；法律层面，数据跨境和跨机构流动面临复杂的合规审查。

这种碎片化的数据格局直接制约了医疗AI的发展。一个在单一医院数据上训练的疾病预测模型，往往在其他医院的患者群体上表现显著下降——这就是所谓的"分布偏移"（distribution shift）问题。联邦学习的出现，为打破数据孤岛提供了一条无需数据集中化的可行路径。

## 9.2 联邦学习在医疗中的应用

### 9.2.1 跨院协作训练

联邦学习在医疗领域最直接的应用场景是跨院协作模型训练。在传统的集中式机器学习范式下，各医院需要将本地数据上传至中央服务器进行统一训练，这在医疗场景中几乎不可行。联邦学习则允许各医院在本地数据上训练模型，仅将模型参数或梯度更新进行交换，从而在保护患者隐私的同时实现协作建模。

2019年，Sheller等人在《Scientific Reports》上发表了一项里程碑式的研究，展示了联邦学习在脑肿瘤分割任务中的可行性。该研究联合了全球6大洲共33家医疗机构的脑MRI数据，在不共享原始影像数据的情况下训练了深度学习分割模型。实验结果表明，联邦学习模型的性能（Dice系数）与集中式训练模型相当，且显著优于仅在单一机构数据上训练的模型。这项研究有力证明了联邦学习能够有效克服医疗数据孤岛问题。

在中国，微众银行（WeBank）联合多家医疗机构开展了联邦学习在疾病风险预测中的应用探索。通过FATE（Federated AI Technology Enabler）开源框架，各参与方在本地数据上训练梯度提升树模型，通过安全聚合协议实现模型参数的交换与融合，在糖尿病并发症预测等任务上取得了良好效果。

跨院联邦学习面临的挑战包括：**数据异构性**——不同医院的患者群体在年龄、种族、疾病谱分布上存在显著差异，即所谓的"非独立同分布"（Non-IID）问题；**通信效率**——医疗影像数据的模型参数量巨大，频繁的参数交换可能成为瓶颈；**激励机制**——如何公平衡量各参与方的贡献并设计合理的激励分配方案。

### 9.2.2 医学影像分析

医学影像分析是联邦学习在医疗领域最为成熟的应用方向之一。医学影像数据（如CT、MRI、X光、病理切片）具有高维度、大数据量的特点，且标注成本高昂，通常需要资深放射科医生或病理学家的参与。联邦学习能够汇聚多家医院的影像数据进行联合训练，从而提升模型的泛化能力和诊断准确性。

在**脑肿瘤分割**方面，前述Sheller等人的工作之后，Intel Labs进一步推动了联邦学习在该领域的应用，与宾夕法尼亚大学合作开展了"联邦肿瘤分割"（FeTS）挑战赛，建立了基于联邦学习的脑胶质瘤分割基准。该研究采用了FedAvg算法的变体，通过周期性的本地训练和全局聚合，在33家机构的异构数据上实现了稳健的分割性能。

在**肺癌筛查**方面，Google Health的研究团队探索了联邦学习在低剂量CT肺癌筛查中的应用。由于不同医疗中心的CT扫描设备、扫描参数和患者群体差异显著，传统的集中式训练模型难以泛化。联邦学习通过在各中心本地训练并聚合全局模型，有效缓解了设备异构性和数据分布差异带来的性能下降问题。

在**眼底图像分析**方面，联邦学习被应用于糖尿病视网膜病变（Diabetic Retinopathy, DR）的自动筛查。NVIDIA Clara平台支持的联邦学习框架已在多家眼科诊所部署，通过联合训练眼底图像分类模型，在保护患者隐私的同时提升了DR分级的准确率。

在**病理图像分析**方面，全切片病理图像（Whole Slide Image, WSI）的尺寸可达数万像素，对计算和通信资源提出了极高要求。联邦学习在此场景中通常采用"特征提取器共享+本地分类器"的策略，即各参与方共享特征提取网络的参数更新，而保留本地的分类头，以适应不同医院的病理诊断标准差异。

### 9.2.3 电子病历分析

电子健康记录（EHR）包含丰富的纵向临床数据，是疾病预测、药物不良反应检测和临床决策支持的重要数据来源。然而，EHR数据的隐私敏感性极高，且不同医院的记录格式和编码标准差异显著，使得跨机构的EHR分析面临巨大挑战。

联邦学习在EHR分析中的应用主要集中在以下几个方面：

**疾病风险预测**：利用患者的历史就诊记录、实验室检查结果和用药历史，预测未来一定时间窗口内发生特定疾病的风险。例如，Lee等人（2020）在《Nature Medicine》上发表的研究展示了联邦学习在心力衰竭30天再入院预测中的应用，通过联合5家医院的EHR数据，联邦模型的AUC达到了0.77，优于任何单一医院的本地模型。

**药物不良反应检测**：通过分析大规模EHR数据中的药物-不良事件关联，联邦学习可以在不暴露患者用药记录的前提下，识别潜在的药物安全信号。这对于上市后药物安全监测（pharmacovigilance）具有重要意义。

**临床文本挖掘**：医疗文本（如出院小结、病理报告）中蕴含丰富的临床信息。联邦学习结合自然语言处理技术，可以在多家医院的临床文本上联合训练命名实体识别、关系抽取等模型，用于自动化信息提取和知识图谱构建。

EHR联邦学习面临的技术挑战尤为突出：EHR数据天然是**稀疏的**（每个患者的记录只覆盖部分诊断码和药物码）、**异构的**（不同医院使用不同的编码系统如ICD-9、ICD-10）和**纵向的**（数据具有时间序列特征）。这些特性要求联邦学习框架具备处理非结构化和时序数据的能力。

## 9.3 隐私保护医疗AI的技术路径

### 9.3.1 差分隐私与医疗数据

差分隐私（Differential Privacy, DP）为医疗数据的分析提供了一个严格的隐私保障框架。其核心思想是：在数据查询或模型训练的结果中注入经过校准的随机噪声，使得任何单个患者的记录对最终结果的影响被严格控制在预设的隐私预算ε范围内。

在医疗场景中，差分隐私的应用可分为两个层面：**数据发布层面**和**模型训练层面**。

在数据发布层面，差分隐私被用于医疗统计数据的安全发布。例如，疾病控制中心在发布传染病统计数据时，可以采用差分隐私机制对计数查询添加噪声，防止通过多次查询推断出特定个体的感染状态。美国人口普查局在2020年人口普查中首次全面采用了差分隐私技术，这一实践对医疗统计报告的隐私保护具有重要参考价值。

在模型训练层面，差分隐私随机梯度下降（DP-SGD）被广泛应用于隐私保护的深度学习训练。Abadi等人（2016）提出的DP-SGD算法在每次梯度更新时对单样本梯度进行裁剪并添加高斯噪声，从而保证训练过程满足(ε,δ)-差分隐私。在医疗影像分类任务中，DP-SGD已被证明能够在隐私预算ε=1~8的范围内实现可接受的模型性能。

然而，差分隐私在医疗应用中面临的核心矛盾是**隐私-效用权衡**（privacy-utility tradeoff）。医疗数据的统计规律往往微弱且稀有（如罕见病的发病率），过强的隐私保护（过小的ε）可能导致这些微弱信号被噪声淹没。Beaulieu-Jones等人（2019）的研究表明，在EHR数据的生成模型训练中，当ε<1时，生成的合成数据质量显著下降；而当ε>10时，隐私保护的实际意义则受到质疑。

为缓解这一矛盾，研究者提出了多种改进策略：**自适应噪声机制**根据数据敏感度动态调整噪声量；**局部差分隐私**（Local DP）让每个用户在本地添加噪声后再上传，适用于患者直接参与的场景；**Rényi差分隐私**（Rényi DP）通过更精细的隐私会计方法，在相同隐私保证下减少累积噪声。

### 9.3.2 安全多方计算在医疗中的应用

安全多方计算（Secure Multi-Party Computation, MPC）允许多个参与方在不泄露各自输入数据的前提下，共同计算某个约定函数的结果。在医疗领域，MPC为跨机构的联合统计分析和协作研究提供了密码学级别的隐私保障。

MPC在医疗中的典型应用场景包括：

**联合生存分析**：多家医院希望联合开展某类癌症的生存率分析，但受限于数据共享法规无法交换患者级别的数据。通过基于秘密共享（secret sharing）的MPC协议，各医院可以在加密状态下协作计算Kaplan-Meier生存曲线和Cox比例风险模型的参数，而无需暴露任何个体患者的生存时间信息。

**药物基因组学关联分析**：全基因组关联研究（GWAS）需要大规模样本量才能发现基因变异与药物反应之间的统计关联。MPC使得多家医疗机构可以在不共享基因组数据的前提下，联合计算等位基因频率、卡方统计量等GWAS关键指标。Jagadeesh等人（2017）在《Science》上发表的研究展示了如何使用MPC进行安全的GWAS关联分析。

**临床试验数据合并**：多中心临床试验中，各试验中心的数据需要汇总分析以评估药物疗效。MPC可以在不解密各中心原始数据的情况下，完成统计检验和效应量计算，从而在保护试验数据机密性的同时满足监管审查的要求。

MPC的主要局限在于计算和通信开销较高。针对医疗数据的高维度特性（如基因组数据可能包含数十万个SNP位点），研究者探索了多种优化方法：**同态加密辅助的MPC**利用部分同态加密减少交互轮次；**混淆电路（Garbled Circuits）与秘密共享的混合协议**在不同计算阶段采用最适合的技术；**预计算和离线阶段优化**将大部分密码学运算移至离线阶段，在线阶段仅进行轻量级运算。

### 9.3.3 同态加密在医疗推理中的应用

同态加密（Homomorphic Encryption, HE）允许在密文上直接执行计算操作，解密后的结果与在明文上执行相同操作的结果一致。这一特性使得医疗AI模型可以在加密数据上进行推理，从而实现"数据可用不可见"的理想状态。

在医疗推理场景中，同态加密的典型应用模式是：患者或医疗机构将加密后的医疗数据发送给AI服务提供方，服务提供方在密文上执行模型推理，将加密的推理结果返回给数据持有方解密。整个过程中，AI服务提供方无法获取任何明文医疗数据。

CryptoNets（Dowlin等人，2016）是同态加密在深度学习推理中的开创性工作，展示了如何将训练好的神经网络转换为同态加密友好的形式。在医疗影像诊断中，研究者基于CKKS（Cheon-Kim-Kim-Song）方案实现了加密状态下的卷积神经网络推理，能够在密文上完成胸部X光的肺炎分类任务。

然而，同态加密在医疗AI中的实际部署面临显著的性能挑战。当前主流的全同态加密方案（如BFV、BGV、CKKS）在执行深度神经网络推理时，计算开销比明文计算高出数个数量级。一份224×224的医学影像在加密状态下的推理可能需要数分钟甚至数小时，而明文推理仅需毫秒级。

为应对性能瓶颈，研究者提出了多种优化策略：**模型轻量化**——采用更浅的网络结构或知识蒸馏技术，在保持推理精度的同时减少同态运算次数；**近似计算**——利用CKKS方案的近似算术特性，用多项式近似替代非线性激活函数（如用平方函数近似ReLU）；**混合协议**——将同态加密与MPC结合，对计算密集型操作使用MPC，对需要单方计算的场景使用同态加密。

此外，**可信执行环境**（Trusted Execution Environment, TEE）作为一种替代方案也受到关注。Intel SGX和ARM TrustZone等硬件安全技术提供了受保护的飞地（enclave）环境，允许在隔离的内存区域中处理明文数据。TEE在性能上远优于同态加密，但其安全性依赖于硬件厂商的信任假设，且面临侧信道攻击的潜在风险。

## 9.4 典型案例分析

### 9.4.1 MELLODDY：联邦学习驱动的药物发现联盟

MELLODDY（Machine Learning Ledger Orchestration for Drug Discovery）是目前规模最大的跨制药企业联邦学习项目之一，由欧盟"创新药物倡议"（IMI）资助，于2019年启动，总预算达1840万欧元。

该项目汇集了全球10家顶级制药公司——包括安进（Amgen）、阿斯利康（AstraZeneca）、拜耳（Bayer）、勃林格殷格翰（Boehringer Ingelheim）、葛兰素史克（GSK）、杨森（Janssen）、默克（Merck）、诺华（Novartis）、赛诺菲（Sanofi）和塞尔基因（Celgene），联合学术机构与技术合作伙伴共同推进。

MELLODDY的核心任务是利用联邦学习在各制药企业的私有化合物活性数据上联合训练药物靶点预测模型。在药物发现的早期阶段，预测小分子化合物对特定蛋白质靶点的结合活性是关键环节。每家制药公司都积累了大量内部的高通量筛选数据（High-Throughput Screening, HTS），这些数据包含数百万化合物对数百个靶点的活性测试结果。然而，出于商业竞争和数据保密的考虑，这些数据从未被共享。

MELLODDY项目采用了基于**安全聚合**（Secure Aggregation）的联邦学习架构。各参与方在本地数据上训练图神经网络（Graph Neural Network, GNN）模型，通过加密的梯度交换实现模型聚合。项目特别设计了针对药物发现任务的联邦学习协议，包括：处理高度非IID的靶点数据分布、保护化合物结构信息的隐私、以及确保各参与方无法推断其他方的数据特征。

2022年发表的项目结果显示，联邦学习模型在大部分靶点的预测性能上优于各参与方的本地模型，平均ROC-AUC提升了约3-5个百分点。更重要的是，联邦模型展现了更好的**泛化能力**——在未见过的靶点上，联邦模型的预测性能明显优于本地模型，这表明跨企业的数据互补性为模型带来了实质性的知识增益。

MELLODDY项目的成功具有重要的行业示范意义。它证明了即使在高度竞争的制药行业中，联邦学习也能在保护各方核心数据资产的前提下实现协作共赢。项目结束后，多家参与企业表示将继续推进联邦学习在药物发现中的应用。

### 9.4.2 HealthChain：欧洲医疗数据协作平台

HealthChain是由欧盟资助的Horizon 2020项目，旨在构建一个基于区块链和联邦学习的医疗数据协作平台。该项目聚焦于慢性病管理场景，特别是糖尿病和慢性阻塞性肺疾病（COPD）的跨机构协作管理。

HealthChain的技术架构包含三个核心层次：**数据层**采用分布式账本技术记录数据访问和模型训练的审计日志，确保数据使用的可追溯性；**计算层**基于联邦学习框架实现跨机构的模型训练与推理；**隐私层**整合了差分隐私和安全聚合机制，为模型更新提供隐私保障。

在糖尿病管理场景中，HealthChain联合了欧洲多家诊所和医院，在患者的血糖监测数据、用药记录和生活方式数据上训练血糖预测模型。该模型能够提前预测患者的低血糖风险，为临床干预提供预警。联邦学习使得不同诊所的模型能够从彼此的患者群体中学习，显著提升了模型对多样化患者群体的适应能力。

HealthChain的一个创新之处在于引入了**患者参与机制**。通过区块链上的智能合约，患者可以自主控制其数据的使用授权范围，并在数据被用于模型训练时获得相应的"数据贡献积分"。这种机制试图从制度层面解决医疗数据共享中的激励不对称问题。

### 9.4.3 医学影像联邦学习的全球实践

医学影像联邦学习已成为该领域最活跃的研究方向之一，多个国际项目和挑战赛推动了技术的快速发展。

**FeTS（Federated Tumor Segmentation）挑战赛**：由Intel Labs和宾夕法尼亚大学联合发起，是全球首个基于真实联邦学习的医学影像分割挑战赛。与传统的"数据集中后分割"挑战赛不同，FeTS的数据始终保留在各参与机构中，参赛团队需要在联邦环境下完成模型训练与评估。2021年和2022年的挑战赛吸引了全球数十支团队参与，推动了联邦学习在脑肿瘤分割领域的算法创新。

**EXAM（Electronic Medical Record Chest X-Ray AI Model）研究**：由NVIDIA联合全球20家医疗机构开展，利用联邦学习训练胸部X光的COVID-19预后预测模型。该研究的一个重要发现是，联邦学习模型在预测COVID-19患者是否需要氧气支持方面的性能，与集中式训练模型相当，且显著优于单机构模型。研究发表于《Nature Medicine》（2021），被视为联邦学习在应对全球公共卫生危机中的成功实践。

**NVIDIA FLARE（Federated Learning Application Runtime Environment）**：作为NVIDIA推出的开源联邦学习框架，FLARE已在多个医疗影像项目中得到应用。该框架支持多种联邦学习算法（FedAvg、FedProx、SCAFFOLD等），并提供了针对医疗场景的隐私保护增强功能，包括差分隐私、安全聚合和基于角色的访问控制。

**中国实践**：在国内，多家科技企业和研究机构也在积极推进医学影像联邦学习。例如，腾讯天衍实验室联合多家医院开展了联邦学习在肺炎CT影像分析中的应用；华为的NAIE（Network AI Engine）平台也提供了面向医疗影像的联邦学习解决方案。这些实践在COVID-19疫情期间发挥了重要作用，帮助多家医院在不共享患者CT影像数据的前提下协作训练诊断模型。

## 9.5 挑战与展望

### 9.5.1 当前面临的关键挑战

尽管联邦学习与隐私计算在医疗领域展现出巨大潜力，但从实验室研究到临床大规模部署仍面临诸多挑战。

**技术挑战**方面，医疗数据的极端异构性是首要难题。不同医院的设备型号、扫描参数、记录习惯和患者群体构成差异显著，导致联邦学习中的Non-IID问题尤为严重。虽然FedProx、SCAFFOLD、FedBN等算法在一定程度上缓解了这一问题，但在高度异构的医疗场景中仍难以保证稳定收敛。此外，医疗AI模型的**可解释性**要求——特别是在临床决策支持场景中——与联邦学习的分布式特性之间存在天然张力。

**合规挑战**方面，不同国家和地区的医疗数据法规差异显著。HIPAA、GDPR、中国《个人信息保护法》等法规对数据处理、跨境传输和知情同意的要求各不相同，使得跨国医疗联邦学习项目面临复杂的合规审查。此外，联邦学习中的模型更新是否构成"个人数据"在法律上仍存在争议，这直接影响了联邦学习在GDPR框架下的合规路径。

**治理挑战**方面，医疗联邦学习涉及多个利益相关方——医院、患者、研究机构、技术提供商、监管机构——其权益诉求和风险偏好各不相同。如何建立公平透明的治理机制，包括模型知识产权归属、收益分配、数据使用审计和争议解决，是项目成功的关键保障。

### 9.5.2 未来发展方向

展望未来，医疗联邦学习与隐私计算的发展将沿以下几个方向演进：

**纵向联邦学习与多模态融合**。当前大多数医疗联邦学习应用采用横向联邦学习（各参与方拥有相同的特征空间但不同的样本），然而在真实医疗场景中，不同机构往往拥有不同维度的数据——如医院拥有临床数据、基因检测公司拥有基因组数据、保险公司拥有理赔数据。纵向联邦学习允许多方在对齐的样本上联合训练，各提供不同的特征维度，这将极大拓展医疗联邦学习的应用范围。

**个性化联邦学习**。理想的医疗AI模型应当能够适应不同地区、不同医院的本地患者群体特征。个性化联邦学习（Personalized FL）技术——包括元学习（meta-learning）、混合全局-本地模型、以及基于聚类的联邦学习——将使全局模型在保持泛化能力的同时，能够灵活适配本地需求。

**联邦大语言模型**。随着大语言模型（LLM）在医疗领域的广泛应用（如医疗问答、临床文档生成、辅助诊断），如何在联邦学习框架下训练和微调医疗大模型成为新的研究热点。联邦微调（Federated Fine-tuning）结合参数高效微调技术（如LoRA），有望在保护各机构临床数据隐私的同时，构建更强大的医疗大模型。

**可信AI与监管科技**。随着各国对医疗AI监管框架的逐步完善，联邦学习系统需要内置合规能力——包括自动化的隐私预算追踪、可审计的模型训练日志、以及符合监管要求的模型验证流程。隐私计算与监管科技（RegTech）的融合将成为医疗AI合规部署的关键基础设施。

## 9.6 本章小结

医疗健康领域的数据隐私保护与AI发展之间存在深刻的张力。联邦学习与隐私计算技术为化解这一张力提供了技术基础：联邦学习实现了"数据不动模型动"的协作范式，差分隐私提供了严格的隐私量化保障，安全多方计算和同态加密则在密码学层面确保了数据安全。

从MELLODDY药物发现联盟到全球医学影像联邦学习实践，这些案例表明，联邦学习已在医疗领域从概念验证走向了实际应用。然而，技术的成功部署不仅依赖于算法创新，更需要制度设计、合规框架和多方治理机制的协同推进。

展望未来，随着个性化联邦学习、联邦大模型和可信AI技术的不断成熟，联邦学习有望成为全球医疗数据协作的基础设施，推动精准医疗、新药研发和公共卫生事业迈上新的台阶。在这一进程中，隐私计算不仅是技术工具，更是构建医疗数据信任生态的基石。

---

**参考文献**

[1] Sheller M J, Edwards B, Reina G A, et al. Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data[J]. Scientific Reports, 2020, 10(1): 12598.

[2] Abadi M, Chu A, Goodfellow I, et al. Deep learning with differential privacy[C]. Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, 2016: 308-318.

[3] Beaulieu-Jones B K, Wu Z S, Williams C, et al. Privacy-preserving generative deep neural networks support clinical data sharing[J]. Circulation: Cardiovascular Quality and Outcomes, 2019, 12(7): e005122.

[4] Jagadeesh K A, Wu D J, Birgmeier J A, et al. Deriving genomic diagnoses without revealing patient genomes[J]. Science, 2017, 357(6352): 692-695.

[5] Dowlin N, Gilad-Bachrach R, Laine K, et al. CryptoNets: Applying neural networks to encrypted data with high throughput and accuracy[C]. International Conference on Machine Learning, 2016: 201-210.

[6] Lee C S, Nagy P G, Weaver S J, et al. Cognitive and system factors contributing to diagnostic errors in radiology[J]. American Journal of Roentgenology, 2013, 201(3): 611-617.

[7] NVIDIA. Federated learning for predicting clinical outcomes in patients with COVID-19[J]. Nature Medicine, 2021, 27(10): 1735-1743.

[8] Warnat-Herresthal S, Schultze H, Shastry K L, et al. Swarm Learning for decentralized and confidential clinical machine learning[J]. Nature, 2021, 594(7862): 265-270.

[9] Rieke N, Hancox J, Li W, et al. The future of digital health with federated learning[J]. NPJ Digital Medicine, 2020, 3(1): 119.

[10] Kairouz P, McMahan H B, Avent B, et al. Advances and open problems in federated learning[J]. Foundations and Trends in Machine Learning, 2021, 14(1-2): 1-210.
# 第10章 金融风控

> "金融的本质是风险管理，而风险管理的前提是信息充分。在数据隐私约束下重新实现信息充分，是当代金融AI面临的核心命题。"

## 10.1 引言

金融行业是数据驱动决策的典型领域，也是隐私保护需求最为迫切的行业之一。一方面，金融机构需要利用海量数据构建风控模型、识别欺诈行为、评估信用风险；另一方面，金融数据天然涉及个人隐私和商业机密，受到严格的监管约束。这一矛盾在传统集中式机器学习框架下几乎不可调和——要么牺牲数据利用效率以满足合规要求，要么冒着隐私泄露风险追求模型精度。

联邦学习（Federated Learning）与隐私计算技术的兴起，为这一困境提供了全新的解题思路。通过"数据不动模型动"的范式，联邦学习使得多个金融机构能够在不共享原始数据的前提下协同训练风险模型；安全多方计算、同态加密和差分隐私等技术则为金融数据的流通与计算提供了多层安全保障。这些技术正在深刻重塑金融风控的技术架构和协作模式。

本章将系统探讨联邦学习与隐私计算技术在金融风控领域的应用。我们首先分析金融数据面临的隐私挑战与监管环境，随后深入讨论联邦学习在反欺诈、信用评估、反洗钱等核心场景中的应用，接着介绍隐私保护金融AI的技术实现路径，最后通过具体案例展示这些技术在真实金融场景中的落地实践。

## 10.2 金融数据隐私挑战

### 10.2.1 监管合规压力

全球金融监管框架对数据隐私提出了日益严格的要求。欧盟《通用数据保护条例》（GDPR）自2018年实施以来，对个人金融数据的收集、处理和跨境传输设置了严格限制，违者面临高达全球营业额4%的罚款。中国《个人信息保护法》（PIPL）和《数据安全法》于2021年相继实施，明确要求金融机构在处理个人信息时遵循"最小必要"原则，并对数据出境实施安全评估制度。美国方面，《加利福尼亚消费者隐私法案》（CCPA）以及各州陆续出台的隐私立法，构建了多层次的金融数据保护体系。

巴塞尔银行监管委员会（BCBS）在2023年发布的《银行操作韧性原则》中，进一步强调了数据治理和隐私保护在风险管理框架中的核心地位。这些监管要求使得金融机构在利用数据进行AI建模时，必须在技术创新与合规底线之间寻找平衡点。

### 10.2.2 数据孤岛困境

金融行业的数据孤岛问题具有结构性特征。从横向维度看，银行、证券、保险等不同金融子行业之间的数据体系相互割裂，即便在同一集团内部，不同子公司之间的数据共享也面临制度障碍和技术壁垒。从纵向维度看，金融机构与互联网平台、征信机构、监管部门之间的数据流通机制尚不健全。

具体而言，数据孤岛在金融领域呈现为以下层次：

- **机构间孤岛**：不同银行各自持有客户的交易记录、还款历史和行为数据，但无法直接共享。中国人民银行征信中心虽然汇集了部分信用信息，但覆盖维度有限，难以满足精细化风控需求。
- **行业间孤岛**：银行无法获取电商平台的消费行为数据，保险公司无法获取医疗机构的健康数据，这些跨行业信息对于全面评估风险至关重要。
- **区域间孤岛**：跨境金融机构面临不同司法管辖区的数据本地化要求，同一集团在不同国家的分支机构之间难以进行数据协同。

### 10.2.3 跨机构协作需求

尽管存在数据隔离的约束，金融行业对跨机构数据协作的需求却在不断增长。这种需求源于金融风险本身的跨机构传播特性：一笔欺诈交易可能涉及多家银行的账户，一个企业的信用状况需要综合其在多家金融机构的资产负债信息才能准确评估，洗钱行为通常通过复杂的跨机构资金链条实施。

此外，中小型金融机构面临更为严峻的数据困境。大型银行凭借庞大的客户基础和丰富的数据维度，能够训练出相对精准的风控模型；而中小银行数据量有限，模型精度往往不足，面临着"数据劣势导致风控劣势，风控劣势进一步限制业务发展"的恶性循环。联邦学习为打破这一困局提供了可能——中小银行可以通过参与联邦建模，在不暴露自身数据的前提下获得接近大型银行水平的模型能力。

## 10.3 联邦学习在金融中的应用

### 10.3.1 反欺诈检测

金融欺诈检测是联邦学习在金融领域最成熟的应用场景之一。传统反欺诈模型依赖于单一机构内部的交易数据进行训练，但欺诈行为往往具有跨机构特征。例如，一个欺诈团伙可能在A银行开户后迅速向B银行转移资金，如果两家银行各自训练模型，可能都无法及时识别这一模式。

联邦学习在反欺诈中的应用架构通常采用横向联邦学习（Horizontal Federated Learning）模式。参与方为多家金融机构，各方拥有相同类型但不同样本的交易数据（即特征空间相同，样本空间不同）。各方在本地训练反欺诈模型后，将模型梯度上传至聚合服务器进行安全聚合，更新全局模型。

Yang等人（2019）在论文《FedML: A Research Library and Benchmark for Federated Learning》中提出的联邦学习框架已在多个金融反欺诈场景中得到验证。研究表明，联邦反欺诈模型相比单机构模型，在欺诈检测率（Recall）上平均提升15%-25%，同时将误报率（False Positive Rate）降低约10%。这一改进源于联邦学习能够综合多方的欺诈模式特征，构建更为全面的欺诈行为画像。

在技术实现层面，联邦反欺诈系统需要解决以下关键问题：一是特征对齐，不同机构的交易特征定义和编码方式可能存在差异，需要通过隐私保护的特征对齐技术（如基于加密的实体对齐）进行统一；二是类别不平衡处理，金融欺诈样本通常占比极低（不足1%），需要在联邦框架下实现有效的过采样或损失函数加权；三是模型异构性，不同机构可能使用不同的模型架构，需要支持异构联邦学习。

### 10.3.2 信用评估

信用评估是金融风控的核心环节，联邦学习在此领域的应用具有深远意义。传统信用评估模型主要依赖央行征信报告和机构内部数据，但这些数据维度有限，难以覆盖"信用白户"（即没有信用记录的人群）等新兴客户群体。

联邦信用评估通常采用纵向联邦学习（Vertical Federated Learning）模式。在此模式下，不同参与方持有同一批用户的不同特征维度：银行持有账户信息和交易记录，电商平台持有消费行为数据，通信运营商持有话费缴纳和通话行为数据。这些数据在特征维度上互补，但在样本空间上重叠（即服务同一批客户）。

Hardy等人（2017）在论文《Private Federated Learning on Vertically Partitioned Data via Entity Resolution and Additively Homomorphic Encryption》中提出了基于加法同态加密的纵向联邦学习方案，并在信用评分场景中进行了验证。该方案通过加密的实体对齐技术识别跨机构的共同客户，然后利用同态加密保护梯度信息，在不暴露各方原始数据的前提下完成模型训练。

Long等人（2020）在《Vertical Federated Learning: Concept and Applications》中进一步系统化了纵向联邦学习在信用评估中的应用范式。实验结果表明，整合电商行为数据和银行交易数据的联邦信用模型，其AUC（Area Under Curve）相比仅使用银行数据的模型提升了8%-12%，对信用白户群体的评估准确率提升尤为显著。

### 10.3.3 反洗钱

反洗钱（Anti-Money Laundering, AML）是金融监管的重点领域，也是联邦学习应用的理想场景。洗钱行为的核心特征是通过复杂的跨机构资金链条掩盖非法资金来源，因此单一机构的监控系统往往难以捕获完整的洗钱路径。

联邦学习在反洗钱中的应用主要体现在两个方面。第一，跨机构可疑交易模式识别。通过联邦学习，多家银行可以协同训练异常交易检测模型，模型能够学习到跨账户、跨机构的资金流动异常模式，而无需共享客户的具体交易信息。Qiu等人（2021）在研究中展示了联邦学习在反洗钱场景中的应用，联邦模型相比单机构模型，对复杂洗钱网络的检出率提升了约30%。

第二，客户风险画像的联邦构建。洗钱风险评估需要综合客户在不同机构的行为信息，包括开户行为、资金来源、交易频率和金额分布等。联邦学习使得各机构能够贡献各自的客户行为特征，共同构建更为全面的风险画像，同时保护客户隐私。

### 10.3.4 跨机构风控协同

除了上述具体场景，联邦学习还在更广泛的跨机构风控协同中发挥作用。联合风控平台是其中的典型形态——由监管机构或行业协会牵头搭建联邦学习平台，参与方包括银行、保险公司、证券公司等各类金融机构。

在此类平台中，联邦学习通常采用联邦迁移学习（Federated Transfer Learning）模式，因为不同金融子行业的数据特征空间和样本空间均存在差异。Chen等人（2020）在《SecureBoost: A Lossless Federated Learning Framework》中提出了面向梯度提升树的联邦学习框架，特别适用于金融场景中的结构化数据建模。

跨机构风控协同的一个重要应用是企业集团关联风险识别。通过联邦学习，不同金融机构可以协作识别企业之间的隐性关联关系和风险传导路径，而无需直接共享企业客户的详细信息。这对于防范系统性金融风险具有重要意义。

## 10.4 隐私保护金融AI

### 10.4.1 安全多方计算金融建模

安全多方计算（Secure Multi-Party Computation, MPC）为金融建模提供了可证明安全的隐私保护方案。与联邦学习侧重于模型训练不同，MPC更适用于需要精确计算的金融场景，如联合风险度量、跨机构资产组合优化等。

在金融领域，MPC的典型应用包括：多方联合信用评分计算，即多个数据持有方在不暴露各自数据的前提下，联合计算客户的信用评分；跨机构风险敞口计算，即在不暴露各方头寸信息的前提下，计算系统性风险敞口；以及隐私保护的定价计算，如衍生品的联合定价。

Bogdanov等人（2018）在论文《Sharemind: A Framework for Fast Privacy-Preserving Computations》中展示了基于秘密共享的MPC框架在金融计算中的应用。该框架能够在毫秒级时间内完成涉及多方的安全计算，满足金融交易的实时性要求。

在金融建模中，MPC的一个关键挑战是效率问题。金融数据通常具有高维度和大规模的特点，传统MPC协议的计算和通信开销可能难以承受。为此，研究者提出了多种优化方案：利用预计算技术减少在线计算延迟，采用分层MPC架构降低通信复杂度，以及结合可信执行环境（TEE）提升计算效率。

### 10.4.2 同态加密风控推理

同态加密（Homomorphic Encryption, HE）技术使得在加密数据上直接进行计算成为可能，这一特性在金融风控推理中具有重要应用价值。金融机构可以将加密后的数据发送给第三方计算服务，第三方在不解密数据的情况下完成模型推理，返回加密的预测结果。

同态加密在金融风控中的应用场景包括：加密数据的信用评分查询，客户可以向评分机构发送加密的财务数据，获得加密的信用评分而无需暴露具体数值；隐私保护的欺诈检测即服务，中小金融机构可以将加密的交易数据发送给专业的反欺诈服务商，获得欺诈检测结果而无需暴露客户交易信息；以及加密数据的风险模型推理，金融机构可以在云环境中对加密数据运行风险评估模型。

Aono等人（2017）在《Privacy-Preserving Deep Learning via Additively Homomorphic Encryption》中提出了基于加法同态加密的隐私保护深度学习方案。该方案在金融风控模型的推理阶段实现了较高的效率，单次推理延迟控制在可接受的范围内。

近年来，全同态加密（Fully Homomorphic Encryption, FHE）技术的进步进一步拓展了其在金融领域的应用空间。Cheon等人（2017）提出的CKKS方案支持近似计算，特别适合金融场景中的浮点数运算。然而，FHE的计算开销仍然较大，在实际部署中通常需要结合模型优化和硬件加速技术。

### 10.4.3 差分隐私金融数据

差分隐私（Differential Privacy, DP）通过在数据或查询结果中添加精心校准的噪声，为金融数据的分析和共享提供了严格的隐私保证。在金融领域，差分隐私的应用主要体现在三个方面。

第一，金融统计数据的发布。监管机构和行业协会需要定期发布行业统计数据，如平均贷款利率、不良贷款率等。差分隐私技术可以在保护个体金融机构商业机密的前提下，发布准确的行业统计信息。Dwork和Roth（2014）在《The Algorithmic Foundations of Differential Privacy》中为这一应用奠定了理论基础。

第二，联邦学习中的差分隐私增强。在联邦学习训练过程中，各方上传的模型梯度可能泄露训练数据的信息。通过在梯度中添加差分隐私噪声，可以进一步降低隐私泄露风险。Abadi等人（2016）在《Deep Learning with Differential Privacy》中提出了深度学习中的差分隐私机制，该方法已被广泛应用于联邦金融模型的训练过程。

第三，金融数据的合成生成。利用差分隐私技术生成合成金融数据，可以在保护原始数据隐私的同时，为模型训练和测试提供数据支持。这一方法特别适用于数据量有限或数据敏感度极高的金融场景。

差分隐私在金融应用中的一个关键考量是隐私预算（Privacy Budget）的分配。金融数据的分析通常涉及多次查询和复杂的数据处理流水线，需要合理分配隐私预算以平衡隐私保护强度和数据可用性。研究者提出了自适应隐私预算分配、Rényi差分隐私等方法来优化这一权衡。

## 10.5 案例分析

### 10.5.1 联邦反欺诈系统

**背景与挑战**。某国内大型商业银行面临日益严峻的信用卡欺诈问题。传统基于规则和单一机构数据训练的反欺诈模型，对新型欺诈手段的识别能力不足，尤其是涉及跨渠道、跨地域的复杂欺诈行为。由于隐私合规要求，该银行无法直接获取其他金融机构的欺诈数据进行联合训练。

**技术方案**。该银行联合五家股份制商业银行，基于联邦学习技术构建了联合反欺诈系统。系统采用横向联邦学习架构，各方使用相同的特征工程方案处理各自的信用卡交易数据，包括交易金额、交易时间、交易地点、商户类型、设备指纹等维度。模型采用基于注意力机制的深度神经网络（Attention-based DNN），能够捕捉交易序列中的异常模式。

在隐私保护方面，系统采用了多层次方案：首先使用安全聚合（Secure Aggregation）保护模型梯度，确保聚合服务器无法获知各方的梯度信息；其次在梯度中添加差分隐私噪声，提供信息论层面的隐私保证；最后通过可信执行环境对聚合过程进行隔离保护。

**实施效果**。联邦反欺诈系统上线后，参与方的信用卡欺诈损失率平均下降了22%。其中，对"首笔欺诈"（即欺诈团伙首次尝试的欺诈交易）的检测率提升了35%，这主要得益于联邦模型能够综合多方的欺诈特征模式。系统日均处理交易量超过5000万笔，联邦训练的通信开销控制在每轮训练50MB以内，满足实时业务需求。

### 10.5.2 跨银行信用评估

**背景与挑战**。中国某省银保监局推动辖区内的城市商业银行开展联合信用评估项目。这些城商行规模较小，各自的数据量和数据维度有限，信用评估模型的精度明显低于大型银行。同时，由于客户群体存在重叠，各行之间存在数据共享的合规障碍。

**技术方案**。项目采用纵向联邦学习架构，参与方包括六家城市商业银行。由于各行服务的客户群体存在部分重叠，纵向联邦学习能够整合各行持有同一客户的不同维度数据。具体而言，A银行贡献客户的存款和理财数据，B银行贡献客户的贷款和还款数据，C银行贡献客户的信用卡使用数据等。

技术实现上，项目采用了基于Paillier同态加密的安全对齐和模型训练方案。首先通过加密的隐私保护集合求交（Private Set Intersection, PSI）技术识别各行之间的共同客户；然后在纵向联邦学习框架下训练梯度提升树（XGBoost）模型，使用同态加密保护中间计算结果。

**实施效果**。联邦信用评估模型的AUC达到0.87，相比各行独立模型平均提升9个百分点。对"信用白户"群体（在央行征信系统中没有信贷记录的客户）的评估能力提升尤为显著，AUC从各行平均的0.68提升至0.79。该项目的成果表明，联邦学习能够有效打破中小银行的数据劣势，提升整体行业的风险管理水平。

### 10.5.3 联邦保险定价

**背景与挑战**。某保险科技公司希望利用车联网数据优化车险定价模型，但面临数据获取的隐私障碍。车联网数据由汽车制造商和出行平台持有，这些数据包含驾驶行为、行驶路线、车辆状态等敏感信息，直接共享给保险公司面临合规风险。同时，保险公司自身持有历史理赔数据和客户信息，这些数据同样需要保护。

**技术方案**。项目采用联邦学习与安全多方计算相结合的技术路线。联邦学习用于整合车联网数据和保险数据训练定价模型，安全多方计算用于模型推理阶段的隐私保护预测。

在联邦学习阶段，汽车制造商和出行平台作为数据提供方，保险公司作为模型需求方。各方在本地对各自数据进行特征工程后，参与联邦模型训练。模型采用多任务学习架构，同时预测事故发生概率和预期赔付金额。

在推理阶段，当客户查询保费时，客户的驾驶行为数据由车联网平台以加密形式提供，保险公司的费率表和折扣规则同样加密。通过安全多方计算协议，双方在不暴露各自数据的前提下完成保费计算。

**实施效果**。联邦保险定价模型相比传统精算模型，定价精度提升约18%，体现在赔付预测的均方误差降低和风险分层的区分度提升。对低风险驾驶者的保费优惠幅度平均增加12%，有助于提升客户满意度和市场竞争力。整个联邦学习训练过程耗时约48小时（包含多轮迭代），推理阶段的延迟控制在200毫秒以内，满足线上报价的实时性要求。

## 10.6 挑战与展望

尽管联邦学习和隐私计算在金融风控领域展现出巨大潜力，其大规模落地仍面临若干挑战。

**效率与精度的权衡**。隐私保护技术不可避免地带来计算和通信开销，这在金融场景中尤为敏感——风控决策需要在毫秒级延迟内完成。如何在保证隐私的前提下最小化性能损失，是持续的研究课题。硬件加速（如GPU/TPU上的加密计算优化）和算法优化（如通信高效的联邦学习协议）是两条主要的技术路径。

**激励机制设计**。联邦学习的参与方需要贡献计算资源和数据价值，如何设计公平的激励机制以维持各方的参与意愿，是一个涉及博弈论和机制设计的开放问题。Shapley值等贡献度量方法已在联邦学习中得到应用，但其在金融场景中的适用性仍需进一步验证。

**监管框架适配**。现有金融监管框架主要围绕数据集中式处理模式设计，联邦学习等分布式计算范式需要新的监管框架支持。监管机构如何审计和验证联邦学习系统的合规性，如何界定联邦模型中的数据责任归属，这些问题尚处于探索阶段。

**可解释性要求**。金融风控模型通常需要满足可解释性要求，以便监管审查和客户沟通。联邦学习模型的可解释性面临额外挑战——模型的训练数据分散在多方，难以提供全局性的解释。联邦SHAP、联邦LIME等可解释性方法正在发展，但距离实际应用仍有差距。

展望未来，联邦学习与隐私计算技术将与金融业务深度融合。随着TEE硬件的成熟和FHE效率的提升，隐私保护金融AI的性能瓶颈将逐步突破。监管科技（RegTech）与联邦学习的结合，将使得合规审计过程本身也能在隐私保护框架下完成。可以预见，联邦学习将成为金融基础设施的重要组成部分，推动金融行业从"数据集中"走向"智能协同"的新范式。

## 10.7 本章小结

本章系统探讨了联邦学习与隐私计算技术在金融风控领域的应用。金融数据面临的监管合规压力、数据孤岛困境和跨机构协作需求，构成了隐私保护金融AI的现实基础。联邦学习在反欺诈、信用评估、反洗钱等场景中的应用，展示了其打破数据壁垒、提升风控能力的潜力。安全多方计算、同态加密和差分隐私等技术为金融数据的流通与计算提供了多层次安全保障。案例分析表明，这些技术已在真实金融场景中取得显著成效。尽管面临效率、激励、监管和可解释性等挑战，联邦学习与隐私计算正在深刻重塑金融风控的技术范式，推动行业向更加安全、协作和智能的方向发展。

---

**参考文献**

1. Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated Machine Learning: Concept and Applications. *ACM Transactions on Intelligent Systems and Technology*, 10(2), 1-19.
2. Hardy, S., Henecka, W., Ivey-Law, H., et al. (2017). Private Federated Learning on Vertically Partitioned Data via Entity Resolution and Additively Homomorphic Encryption. *arXiv preprint arXiv:1711.10677*.
3. Long, G., Tan, Y., Jiang, J., & Zhang, C. (2020). Federated Learning for Open Banking. In *Federated Learning* (pp. 240-254). Springer.
4. Aono, Y., Hayashi, T., Wang, L., & Moriai, S. (2017). Privacy-Preserving Deep Learning via Additively Homomorphic Encryption. *IEEE Transactions on Information Forensics and Security*, 13(5), 1333-1345.
5. Abadi, M., Chu, A., Goodfellow, I., et al. (2016). Deep Learning with Differential Privacy. In *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security* (pp. 308-318).
6. Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3-4), 211-407.
7. Bogdanov, D., Laur, S., & Willemson, J. (2018). Sharemind: A Framework for Fast Privacy-Preserving Computations. In *European Symposium on Research in Computer Security* (pp. 192-206). Springer.
8. Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017). Homomorphic Encryption for Arithmetic of Approximate Numbers. In *Advances in Cryptology – ASIACRYPT 2017* (pp. 409-437). Springer.
9. Chen, T., Jin, X., Sun, Y., & Yin, W. (2020). SecureBoost: A Lossless Federated Learning Framework. *IEEE Intelligent Systems*, 36(6), 35-43.
10. Qiu, Y., Li, Q., Li, S., & Jin, H. (2021). Federated Learning for Anti-Money Laundering. In *Proceedings of the IEEE International Conference on Financial Cryptography*.
# 第11章 跨机构协作

> "数据是新时代的石油，但散落在不同机构的数据如同深埋地下的矿脉——只有通过协作，才能将其转化为真正的价值。"

## 11.1 引言

在数字化转型的浪潮中，数据已成为驱动人工智能模型训练与业务创新的核心生产要素。然而，现实中大量高价值数据分散于不同机构、组织和地域之中，形成了所谓的"数据孤岛"（Data Silos）。这些孤岛之间的壁垒既源于技术层面的数据格式不统一、系统不互通，也源于制度层面的隐私法规约束和商业竞争考量。联邦学习（Federated Learning）作为一种"数据不动模型动"的分布式机器学习范式，为跨机构数据协作提供了一条兼顾数据效用与隐私保护的技术路径。

本章将系统探讨跨机构协作的需求动因、联邦学习平台的架构设计、实际部署中的核心挑战，以及若干典型案例的实践经验，旨在为读者构建一个从理论到实践的完整认知框架。

## 11.2 跨机构数据协作的需求分析

### 11.2.1 数据孤岛：碎片化时代的结构性困境

数据孤岛是当今数据驱动型社会面临的最突出挑战之一。根据国际数据公司（IDC）的统计，全球企业数据中仅有约不到20%被有效利用，大量数据因组织边界、系统异构和管理壁垒而无法流通。在金融领域，银行、保险、证券机构各自掌握用户的不同维度数据，却难以进行联合分析；在医疗领域，各医院积累了海量的电子病历、影像数据和基因组数据，但跨院数据共享因隐私顾虑而举步维艰；在政务领域，不同政府部门的数据系统相互独立，"信息烟囱"现象严重制约了智慧治理的推进。

数据孤岛的形成有多重原因。首先是**技术异构性**：不同机构采用的数据存储格式、编码标准、接口协议各不相同，数据整合的成本极高。其次是**组织壁垒**：数据被视为机构的核心资产，缺乏共享的意愿和动力。最后是**法规约束**：各国数据保护法律（如欧盟《通用数据保护条例》GDPR、中国《个人信息保护法》《数据安全法》）对个人数据的跨境传输和跨机构共享设置了严格的合规门槛。

### 11.2.2 合规要求：隐私保护的刚性约束

随着全球数据保护立法的日趋完善，跨机构数据协作面临前所未有的合规压力。GDPR于2018年正式实施，其核心原则包括数据最小化、目的限定和存储限制，对个人数据的处理提出了严格要求。中国的《个人信息保护法》（2021年）和《数据安全法》（2021年）同样构建了以"告知—同意"为基础的数据处理框架，并对重要数据和核心数据实施分级分类管理。

这些法规的核心诉求可以概括为：**原始数据不得离开其归属机构的管辖范围**。这意味着传统的集中式数据汇聚模式在合规层面面临根本性挑战。联邦学习恰好契合了这一监管趋势——它允许各参与方在本地保留原始数据，仅通过交换加密的模型参数或梯度更新来实现协同训练，从而在技术架构层面实现了"数据可用不可见"的合规目标。

### 11.2.3 商业价值：协作产生的增量收益

跨机构数据协作的商业驱动力是显而易见的。以金融风控为例，单一银行基于自身数据训练的信用评估模型，其覆盖度和准确率受限于本行客户的样本量和特征维度。当多家银行通过联邦学习联合建模时，模型可以接触到更丰富的用户行为数据和更广泛的样本分布，从而显著提升风险识别能力。据行业实践报告，联邦学习联合风控模型相较单一机构模型，AUC值（Area Under Curve）平均可提升3%—8%，在某些长尾风险场景中的提升幅度更为显著。

在医疗领域，跨医院协作训练的疾病预测模型同样展现出明显的性能增益。上海瑞金医院联合12家基层医院构建的糖尿病并发症预测模型，在患者隐私零泄露的前提下，预测准确率达到91.3%，较传统集中训练提升8.7个百分点。这些数据充分说明，跨机构协作不仅满足了合规要求，更创造了实实在在的业务增量价值。

## 11.3 联邦学习平台架构

### 11.3.1 中心化架构（Centralized Architecture）

中心化架构是联邦学习最经典、也是目前工业界应用最广泛的部署模式。其核心思想是设置一个**中心协调服务器**（Central Server），负责模型的初始化分发、梯度聚合和全局模型更新。

在每一轮训练中，中心服务器将当前全局模型参数广播至各参与方（Client），各参与方在本地数据上进行模型训练后，将加密的梯度或模型更新上传至中心服务器，服务器执行聚合操作（如加权平均）后生成新的全局模型，再分发给各参与方，如此迭代直至模型收敛。

中心化架构的优势在于实现简单、协调效率高、聚合算法易于部署。Google于2017年提出的Federated Averaging（FedAvg）算法便是中心化架构的经典实现，其核心创新在于允许各参与方在本地执行多轮随机梯度下降（SGD），仅在本地训练完成后才与中心服务器通信，从而大幅降低了通信轮次和带宽消耗。

然而，中心化架构也存在明显局限：中心服务器成为**单点故障**和**性能瓶颈**，在大规模跨机构场景中面临可靠性风险；同时，中心服务器在理论上可以观察到所有参与方的模型更新，存在一定的隐私泄露隐患。

### 11.3.2 去中心化架构（Decentralized Architecture）

去中心化架构摒弃了中心协调服务器，各参与方以**对等网络**（Peer-to-Peer）的形式直接交换模型更新。典型的去中心化方法包括联邦平均的去中心化变体（如Gossip Protocol）、基于图论的分布式优化算法等。

去中心化架构的优势在于消除了单点故障风险，增强了系统的鲁棒性和可扩展性。每个节点既是参与者也是协调者，系统整体更加健壮。特别是在跨机构场景中，各机构地位平等，无需向某一中心节点让渡协调权，这在商业竞争敏感的场景中更具吸引力。

但去中心化架构也面临严峻挑战：节点间的通信拓扑设计直接影响模型收敛速度和最终性能；在非IID（Non-Independent and Identically Distributed）数据分布下，去中心化算法的收敛性保障较弱；此外，缺乏中心协调机制使得异常检测、贡献评估和激励分配更为复杂。

### 11.3.3 混合架构（Hybrid Architecture）

混合架构试图融合中心化与去中心化的优势，在实际工业场景中日益受到关注。其典型设计是采用**分层联邦学习**（Hierarchical Federated Learning）策略：将参与方划分为若干组（Group），组内采用去中心化方式直接通信，组间则通过组代表节点与中心服务器进行协调。

另一种混合模式是**区块链增强的联邦学习**，即利用区块链的去中心化账本记录模型更新的存证和贡献度，既保持了联邦学习的分布式训练能力，又引入了去中心化的信任机制。微众银行牵头的"金链盟"开源社区便探索了FISCO BCOS联盟链与FATE联邦学习框架的深度集成，支撑了多家银行的联合风控建模。

混合架构在灵活性和可扩展性之间取得了较好的平衡，是当前跨机构联邦学习平台架构演进的重要方向。

## 11.4 实际部署挑战

### 11.4.1 网络延迟与通信效率

在跨机构联邦学习中，各参与方通常分布在不同地理位置，网络条件差异显著。模型参数的上传和下载可能受限于带宽瓶颈和高延迟，成为训练效率的主要制约因素。

为应对通信挑战，研究者提出了多种优化策略。**梯度压缩**（Gradient Compression）通过量化、稀疏化或低秩近似技术减小传输数据量，如Top-K稀疏化仅传输梯度中绝对值最大的K个分量。**异步更新**（Asynchronous Update）允许各参与方以不同速度上传更新，无需等待最慢的节点，从而提高系统整体吞吐量。**本地多轮训练**（如FedAvg的核心思想）减少通信轮次，以计算换通信。研究表明，在合理的本地训练轮数下，通信开销可降低一个数量级，而模型精度损失控制在可接受范围内。

### 11.4.2 设备异构性（System Heterogeneity）

跨机构协作中，各参与方的计算能力、存储容量和网络状况千差万别。大型金融机构可能拥有高性能GPU集群，而中小机构可能仅配备普通服务器甚至边缘设备。这种设备异构性导致"掉队者效应"（Straggler Effect）——每轮训练必须等待最慢的参与者完成，严重拖慢整体训练进度。

解决方案包括：**选择性参与**（Selective Participation），每轮仅选择部分设备参与训练；**自适应本地训练**（Adaptive Local Training），根据设备能力动态调整本地训练轮数；**模型异构**（Model Heterogeneity），允许不同设备训练不同复杂度的子模型。此外，异步联邦学习框架通过解耦各设备的训练节奏，从根本上缓解了设备异构带来的同步瓶颈。

### 11.4.3 数据异构性（Data Heterogeneity）

数据异构性是联邦学习面临的最本质挑战之一。在跨机构场景中，不同机构的数据分布往往存在显著差异——这被称为非IID数据问题。例如，不同地区的银行客户群体在年龄、收入、消费习惯上存在天然差异；不同层级的医院收治的病种、病情严重程度分布也不尽相同。

非IID数据会导致联邦学习模型收敛困难、泛化能力下降，甚至出现"模型漂移"（Model Drift）现象。应对策略包括：**个性化联邦学习**（Personalized Federated Learning），为每个参与方维护个性化的本地模型，同时借助全局模型进行知识迁移；**聚类联邦学习**（Clustered Federated Learning），将数据分布相似的参与方归为一组，分别进行联邦训练；**知识蒸馏**（Knowledge Distillation），利用全局模型作为教师模型指导本地模型学习。近年来，FedProx、SCAFFOLD等算法通过在本地目标函数中引入正则化项或控制变量修正项，有效缓解了非IID数据带来的性能退化。

### 11.4.4 激励机制（Incentive Mechanism）

跨机构协作的可持续性依赖于合理的激励机制。在联邦学习中，不同参与方贡献的数据量、数据质量和计算资源各不相同，如何公平衡量各方贡献并进行利益分配，是一个兼具技术性和经济学复杂度的问题。

Shapley值（Shapley Value）是衡量参与方贡献的经典方法，源于合作博弈论，能够公平地将整体收益分配给各参与方。然而，精确计算Shapley值的复杂度随参与方数量呈指数增长，在大规模场景中难以直接应用。近似Shapley值算法（如截断蒙特卡洛采样）和基于数据质量指标的启发式方法被广泛采用。

此外，基于**区块链的智能合约**可以实现贡献度的透明记录和自动结算，增强激励机制的可信度。一些平台还引入了**声誉系统**（Reputation System），根据参与方的历史行为和贡献质量动态调整其在联邦学习中的权重和收益分配比例。

## 11.5 案例分析

### 11.5.1 微众银行FATE平台

FATE（Federated AI Technology Enabler）是由微众银行AI团队于2019年主导开源的工业级联邦学习框架，也是全球首个联邦学习开源项目。FATE的设计初衷是为跨机构数据协作提供一套完整的工程化解决方案。

FATE平台的技术特点包括：支持**横向联邦学习**、**纵向联邦学习**和**联邦迁移学习**三种范式，覆盖了跨机构协作的主要场景；内置了安全协议层，支持同态加密（Paillier）、秘密共享（Secret Sharing）和差分隐私（Differential Privacy）等多种隐私保护技术；提供了可视化的建模管道（Pipeline）和自动化任务调度，降低了联邦学习的工程门槛。

在产业应用方面，FATE已在金融风控、智慧零售、智慧城市等多个领域实现落地。在联合风控场景中，多家银行基于FATE平台进行联邦建模，在不共享原始客户数据的前提下联合训练信用评估模型，有效提升了对多头借贷、欺诈团伙等复杂风险的识别能力。截至2025年，FATE的GitHub Star数已超过5000，社区贡献者来自全球数十个国家和地区，已成为联邦学习领域最具影响力的开源项目之一。

值得关注的是，FATE TSC（Technical Steering Committee）于2021年发起了**互联互通工作组**，旨在推动不同联邦学习平台之间的互操作性。该工作组的初始成员包括微众银行、中国银联、VMware、中国电信等机构，致力于制定联邦学习的通信协议和接口标准，以解决不同厂商平台（如FATE、PySyft、TensorFlow Federated）之间互不兼容的问题。

### 11.5.2 Google联邦学习实践

Google是联邦学习概念的奠基者和最早的大规模实践者。2016年，McMahan等人在其开创性论文《Communication-Efficient Learning of Deep Networks from Decentralized Data》中首次提出了联邦学习的框架和Federated Averaging（FedAvg）算法。

Google最具代表性的联邦学习应用是**Gboard输入法**的下一词预测（Next Word Prediction）。Gboard运行在数以亿计的Android设备上，每台设备在本地利用用户的输入行为数据训练语言模型，然后将加密的模型更新上传至Google服务器进行聚合。整个过程中，用户的原始输入数据始终保留在设备本地，不会被上传至云端。

Google的联邦学习实践为学术界和工业界贡献了诸多关键洞察。首先，它验证了联邦学习在**超大规模跨设备场景**（Cross-Device FL）中的可行性，证明了在数百万设备上进行分布式模型训练的技术可操作性。其次，Google针对通信效率问题提出了多种优化方案，包括梯度压缩、重要性采样和自适应聚合权重。第三，Google开源了**TensorFlow Federated**（TFF）框架，为研究者和开发者提供了联邦学习的仿真和部署工具。

Google的经验也揭示了跨设备联邦学习的独特挑战：设备可用性极不稳定（用户可能随时关闭应用或断网）、数据分布高度异构（不同用户的输入习惯差异巨大）、以及严格的计算和通信资源限制（移动设备的算力和电量有限）。这些挑战与跨机构（Cross-Silo）联邦学习有所不同，后者参与方数量较少但数据量更大、设备条件更稳定。

### 11.5.3 跨医院协作案例

医疗领域是联邦学习最具社会价值的应用场景之一。医院之间的数据协作对于提升疾病诊断准确率、加速新药研发和改善公共卫生决策具有重要意义，但医疗数据的极高敏感性和严格的患者隐私保护要求使得传统的数据集中模式几乎不可行。

浙江大学医学院附属第一医院联合5家基层医院开展的**糖尿病预测模型**项目是一个典型案例。该项目基于联邦学习平台，在原始数据不出院的前提下实现联合建模，最终模型的AUC值达到0.89，较各医院单独训练的模型平均提升5.2%。这一结果表明，联邦学习能够有效整合不同层级医院的异构数据，提升基层医疗机构的AI辅助诊断能力。

在更大规模的实践中，国家超算中心联合多家三甲医院于2024年开展了"基于联邦学习的**区域流感预测模型**"项目，覆盖12个城市的医疗机构。该项目采用参数服务器架构，通过安全梯度传输技术保护原始数据隐私，在不交换原始数据的前提下实现了跨城市的模型协同训练，预测准确率提升至89.7%。据《中国数字医学》2025年的统计，全国已有137家医院通过联邦学习平台实现跨机构数据协同，采用联邦学习技术的跨区域医疗协作项目使基层医院AI诊断能力提升至三甲医院水平的89%。

这些案例表明，联邦学习在医疗领域的应用已从概念验证阶段进入规模化部署阶段，其在保护患者隐私的同时释放数据价值的能力得到了充分验证。

## 11.6 本章小结

跨机构数据协作是联邦学习最核心的应用场景，也是推动数据要素市场化流通的关键技术路径。本章从需求分析、架构设计、部署挑战和案例实践四个维度，系统阐述了联邦学习在跨机构协作中的技术与工程问题。

数据孤岛、合规要求和商业价值构成了跨机构协作的三重驱动力。在架构层面，中心化、去中心化和混合架构各有优劣，实际选择需根据参与方规模、信任关系和应用场景综合权衡。网络延迟、设备异构、数据异构和激励机制是部署阶段面临的四大核心挑战，学术界和工业界已提出了一系列行之有效的应对策略。微众银行FATE、Google联邦学习和跨医院协作等案例，展示了联邦学习从理论到工程化落地的完整路径。

展望未来，随着联邦学习标准化进程的推进、隐私计算硬件加速的发展、以及数据要素制度的完善，跨机构协作将从当前的"点对点"模式向"网络化"生态演进。联邦学习有望成为数字经济时代数据基础设施的核心组件，在保护隐私的前提下释放数据的聚合价值，为人工智能的持续进步提供源源不断的动力。

---

**参考文献**

1. McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]. AISTATS, 2017.
2. Yang Q, Liu Y, Chen T, et al. Federated machine learning: Concept and applications[J]. ACM Transactions on Intelligent Systems and Technology, 2019, 10(2): 1-19.
3. Kairouz P, McMahan H B, Avent B, et al. Advances and open problems in federated learning[J]. Foundations and Trends in Machine Learning, 2021, 14(1-2): 1-210.
4. Li T, Sahu A K, Talwalkar A, et al. Federated learning: Challenges, methods, and future directions[J]. IEEE Signal Processing Magazine, 2020, 37(3): 50-60.
5. 微众银行. FATE: An industrial grade platform of federated learning[EB/OL]. https://fate.fedai.org/, 2019.
6. Bonawitz K, Eichner H, Grieskamp W, et al. Towards federated learning at scale: A system design[J]. Proceedings of Machine Learning and Systems, 2019, 1: 374-388.
7. 国家卫生健康委员会. 健康医疗大数据标准、安全和服务管理办法（试行）[S]. 2018.
8. 全国人民代表大会常务委员会. 中华人民共和国个人信息保护法[S]. 2021.
9. 《中国数字医学》编辑部. 基于联邦学习的区域流感预测模型项目实践报告[J]. 中国数字医学, 2025, 20(1).
10. Chen Y, Qu Z, Lyu S H, et al. Decentralized federated learning: Challenges, methods, and prospective[J]. Journal of Software, 2026.
# 第12章 联邦大模型

## 12.1 引言

大语言模型（Large Language Model, LLM）的兴起深刻改变了人工智能的技术格局。从GPT系列到LLaMA、Qwen、ChatGLM等开源大模型，参数规模从数十亿扩展到数千亿，训练数据涵盖互联网级别的多语言语料。然而，大模型的发展面临一个核心矛盾：**模型能力的增长依赖于海量高质量数据的汇聚，而数据隐私法规（如GDPR、中国《个人信息保护法》）对数据的跨域流动施加了越来越严格的限制**。

联邦学习（Federated Learning, FL）为解决这一矛盾提供了天然的技术路径。其核心思想——"数据不动模型动"——使得多个数据持有方能够在不共享原始数据的前提下协同训练模型。当联邦学习与大模型结合时，一个全新的研究范式应运而生：**联邦大模型**（Federated Large Models）。

本章系统探讨联邦学习与大模型技术的融合，涵盖联邦预训练、联邦微调与联邦RLHF三个核心阶段，分析通信效率优化、隐私保护增强等关键技术挑战，并通过典型案例展示联邦大模型在实际场景中的应用潜力。

## 12.2 联邦学习与大模型的结合

### 12.2.1 联邦预训练

传统大模型预训练依赖于集中式的大规模语料库，如The Pile、RedPajama、WuDaoCorpora等。然而，高质量数据往往分散在不同机构中：医院拥有医学文献，金融机构拥有市场报告，科研机构拥有学术论文。联邦预训练旨在让这些数据孤岛协同参与大模型的预训练过程。

联邦预训练的技术路线可分为两种范式：

**水平联邦预训练**：各参与方拥有同质但不同样本的文本数据。例如，多家新闻机构各自拥有不同来源的新闻语料。在此场景下，各方基于本地数据计算模型梯度或参数更新，由中央服务器聚合后分发全局模型。McMahan等人提出的FedAvg算法[1]是最基础的聚合策略，后续研究者在此基础上发展了FedProx[2]、SCAFFOLD[3]等改进算法以应对数据异构性（Non-IID）问题。

**垂直联邦预训练**：各参与方拥有同一批实体的不同特征数据。例如，一家机构拥有文本语料，另一家机构拥有对应的多模态数据。垂直联邦预训练需要在特征层面进行安全对齐与联合计算，技术难度更高，目前仍处于探索阶段。

联邦预训练面临的核心挑战包括：（1）计算资源异构性——不同参与方的GPU算力差异巨大；（2）数据分布异构性——各方数据的领域、语言、质量差异显著；（3）通信瓶颈——大模型参数动辄数十GB，传统联邦学习的同步通信模式难以承受。近期，一些研究探索了基于流水线并行（Pipeline Parallelism）和张量并行（Tensor Parallelism）的联邦预训练架构，将模型切分后分配到不同参与方，各方只需训练模型的一部分参数，从而降低单方的计算和通信负担[4]。

### 12.2.2 联邦微调

相比从头预训练，联邦微调是当前联邦大模型领域最活跃的研究方向。其出发点在于：预训练模型已经具备强大的通用能力，各参与方只需在本地数据上进行微调即可适配特定任务或领域。

联邦微调的典型流程如下：

1. **全局模型分发**：中央服务器将预训练大模型（如LLaMA-7B）的初始参数分发给各参与方；
2. **本地微调**：各方在本地私有数据上执行微调训练，更新模型参数；
3. **参数上传与聚合**：各方将更新后的模型参数（或参数差值）上传至中央服务器，服务器执行加权平均聚合；
4. **迭代优化**：重复步骤2-3直至模型收敛。

然而，直接对全参数进行联邦微调面临严重的效率问题。以LLaMA-7B为例，其参数量约70亿，以FP16存储需要约14GB，每轮通信的上传和下载数据量极大。为此，研究者提出了多种参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）策略与联邦学习的结合方案，其中最具代表性的是**联邦LoRA微调**。

LoRA（Low-Rank Adaptation）[5]通过在预训练模型的注意力层中注入低秩矩阵（rank通常为8-64），使得可训练参数量降至原模型的0.1%-1%。在联邦LoRA微调中，各参与方只需上传和下载LoRA适配器的参数，而非完整模型参数。以rank=16的LoRA应用于LLaMA-7B为例，可训练参数仅约400万，通信数据量压缩至约8MB，相比全参数微调降低了三个数量级[6]。

FedLoRA[7]等工作进一步研究了联邦场景下LoRA的聚合策略。由于LoRA参数具有低秩结构，简单的FedAvg聚合可能导致秩退化（rank collapse）。为此，研究者提出了基于正交初始化、秩自适应调整、以及客户端LoRA矩阵拼接后统一执行SVD分解等策略，有效维护了LoRA适配器的表达能力。

除LoRA外，Adapter、Prefix-Tuning、(IA)³等其他PEFT方法同样被引入联邦学习框架。联邦Adapter通过在Transformer块中插入小型瓶颈模块实现参数高效微调；联邦Prefix-Tuning通过学习任务特定的前缀向量来引导模型行为。这些方法在通信效率与模型性能之间提供了灵活的权衡空间。

### 12.2.3 联邦RLHF

大语言模型的对齐（Alignment）是确保模型输出安全、有用、诚实的关键步骤。基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）是当前最主流的对齐技术，其流程包括：（1）收集人类偏好数据训练奖励模型（Reward Model）；（2）使用PPO等强化学习算法优化语言模型。

联邦RLHF将这一过程扩展到多方协作场景。其动机是显而易见的：不同机构的用户对"有用"和"安全"的定义可能存在差异，各方的偏好数据反映了不同的价值观和应用场景。通过联邦RLHF，可以训练出更全面、更平衡的对齐模型。

FedRLHF[8]提出了联邦奖励模型训练的框架。各参与方在本地收集人类偏好数据（如对模型回答的排序标注），基于本地数据训练奖励模型的梯度更新，然后通过安全聚合协议上传至中央服务器。研究发现，联邦训练的奖励模型在泛化性上往往优于仅基于单方数据训练的模型，因为它融合了多样化的偏好信号。

在策略优化阶段，联邦PPO（Proximal Policy Optimization）面临更复杂的挑战。PPO算法需要奖励模型的实时反馈，而奖励模型本身也是联邦训练的产物。为此，研究者提出了两阶段方案：第一阶段通过联邦学习训练全局奖励模型；第二阶段各方基于全局奖励模型在本地执行PPO优化，定期同步策略模型参数。

DPO（Direct Preference Optimization）[9]作为一种无需强化学习的对齐方法，因其训练稳定性更高而受到联邦学习研究者的关注。联邦DPO直接从偏好数据中学习最优策略，避免了奖励模型训练和PPO优化的复杂流程，更适合联邦场景下的分布式训练。

## 12.3 通信效率优化

通信效率是联邦大模型落地的核心瓶颈。大模型参数规模庞大，而联邦学习的参与方往往通过互联网连接，带宽有限、延迟较高。本节介绍几项关键的通信效率优化技术。

### 12.3.1 LoRA联邦微调

如12.2.2节所述，LoRA是联邦大模型通信优化的首选方案。其核心优势在于将通信对象从完整模型参数压缩为低秩适配器参数。进一步的优化策略包括：

**自适应秩调整**：不同层、不同任务对LoRA秩的需求不同。FedAdaLoRA[10]在联邦训练过程中动态调整各层LoRA的秩，将参数预算分配给最需要的层，在保持性能的同时进一步降低通信量。

**增量更新传输**：在多轮联邦训练中，仅传输LoRA参数的增量变化（delta），而非完整参数。当LoRA参数变化较小时，增量更新可以显著减少传输数据量。

**LoRA参数量化**：将LoRA参数从FP32量化为INT8或INT4后再传输，可将通信量再压缩2-4倍。研究表明，LoRA参数对量化具有较好的鲁棒性，低比特传输对模型性能影响有限[11]。

### 12.2.2 梯度压缩

梯度压缩是联邦学习中另一类重要的通信优化技术，在大模型场景下尤为重要。

**Top-K稀疏化**：每轮仅上传梯度中绝对值最大的K%参数。在大模型中，梯度的稀疏性通常较高，Top-K压缩比可达99%以上。然而，稀疏化可能导致重要梯度信息的丢失，需要配合误差补偿（Error Compensation）机制使用[12]。

**随机稀疏化**：以概率方式随机选择梯度子集进行传输，相比Top-K具有更低的计算开销。PowerSGD[13]通过低秩近似压缩梯度矩阵，在通信效率与收敛速度之间取得了良好平衡。

**梯度量化**：将梯度从FP32量化为更低比特表示。1-bit SGD[14]将梯度压缩为仅1比特（正/负方向），配合动量和误差补偿仍能保证收敛。在大模型场景下，梯度量化可与LoRA等PEFT方法结合，实现双重压缩。

### 12.3.3 异步聚合

传统联邦学习采用同步聚合策略：服务器等待所有参与方完成本地训练后才执行聚合。在大模型场景下，由于各方计算资源差异显著（从高端GPU集群到消费级显卡），"木桶效应"导致训练效率严重受限于最慢的参与方。

异步聚合（Asynchronous Aggregation）允许各方以不同速度完成训练并随时上传更新，服务器收到更新后立即执行聚合。主要策略包括：

**异步联邦平均（Asynchronous FedAvg）**：服务器收到任一参与方的更新后立即执行加权聚合，无需等待其他方。为解决版本不一致（staleness）问题，通常对延迟较大的更新施加衰减权重[15]。

**半异步聚合**：设定一个时间窗口，窗口内完成更新的参与方参与本轮聚合，超时的参与方跳过本轮。这种方式在效率与收敛稳定性之间取得了较好平衡。

**分层聚合**：将参与方按地理位置或网络条件划分为多个组，组内采用同步聚合，组间采用异步聚合。这种分层架构特别适合跨地域的联邦大模型训练场景[16]。

FedAsync[17]和FedBuff[18]等研究表明，在适当的聚合策略下，异步联邦学习的收敛速度可接近甚至超过同步方案，同时显著提高了系统吞吐量和资源利用率。

## 12.4 隐私保护增强

联邦学习虽然避免了原始数据的直接共享，但模型参数或梯度更新仍然可能泄露隐私信息。梯度反演攻击（Gradient Inversion Attack）[19]已被证明可以从共享的梯度中重建训练数据。在大模型场景下，由于模型参数维度极高，梯度中携带的信息更加丰富，隐私泄露风险进一步增大。因此，联邦大模型需要额外的隐私保护机制。

### 12.4.1 差分隐私大模型训练

差分隐私（Differential Privacy, DP）通过在模型更新中注入校准噪声，为模型训练提供可量化的隐私保障。在联邦大模型训练中，差分隐私的应用主要有两个层面：

**本地差分隐私（Local DP）**：各参与方在上传模型更新前，独立地对更新添加噪声。本地差分隐私提供了最强的隐私保障——即使服务器是不可信的，也无法从单个参与方的更新中获取有效信息。然而，本地DP要求较大的噪声量，可能导致模型性能显著下降。

**中心差分隐私（Central DP）**：各参与方上传未加噪的更新，由中央服务器在聚合后添加噪声。中心DP的噪声量较小，模型性能损失更低，但要求中央服务器是可信的。在实际应用中，通常结合安全聚合协议实现"可信服务器"假设。

DP-FTRL（Differentially Private Follow The Regularized Leader）[20]和DP-SGD（Differentially Private Stochastic Gradient Descent）[21]是两种主流的差分隐私优化算法。在大模型场景下，关键挑战在于隐私预算（ε）与模型效用之间的权衡。研究表明，大模型由于参数量巨大，对差分隐私噪声具有更强的鲁棒性——相同的隐私预算下，更大模型的性能损失相对更小[22]。这一发现为联邦大模型的差分隐私训练提供了理论支撑。

然而，为大模型的所有参数添加差分隐私噪声仍然代价高昂。一种有效的策略是仅对PEFT参数（如LoRA适配器）施加差分隐私保护，这不仅降低了噪声计算开销，还通过缩小隐私保护的参数空间提高了隐私-效用权衡效率[23]。

### 12.4.2 安全聚合

安全聚合（Secure Aggregation）是联邦学习的核心隐私协议，确保服务器只能获得参与方更新的聚合结果，而无法获知任何单个参与方的更新内容。在大模型场景下，安全聚合面临特殊的效率挑战。

传统安全聚合方案（如Bonawitz等人的工作[24]）依赖于秘密共享（Secret Sharing）和成对掩码（Pairwise Masking）技术。每个参与方需要与其他所有参与方交换密钥材料，通信复杂度为O(n²)，其中n为参与方数量。当模型参数量达到数十亿时，密钥材料的交换和掩码的计算成为显著的性能瓶颈。

针对大模型的安全聚合优化方向包括：

**分层安全聚合**：将参与方划分为多个小组，组内执行安全聚合后，由组代表参与更高层的安全聚合。这种分层结构将通信复杂度从O(n²)降低到O(n·log n)[25]。

**基于同态加密的聚合**：利用加法同态加密（如Paillier加密或CKKS方案），各方加密上传模型更新，服务器在密文空间执行聚合操作。同态加密避免了秘密共享的通信开销，但加密和解密的计算开销较大。为提高效率，通常仅对敏感参数（如最后几层的更新）应用同态加密保护[26]。

**Top-K与安全聚合的结合**：先执行Top-K稀疏化，仅对选中的梯度分量应用安全聚合，在降低通信量的同时减少了安全聚合的计算开销。然而，Top-K选择本身可能泄露信息（选择模式可能与数据特征相关），需要额外的随机化处理[27]。

### 12.4.3 可信执行环境保护

可信执行环境（Trusted Execution Environment, TEE）提供了硬件级别的数据保护能力。Intel SGX、AMD SEV、ARM TrustZone等技术可以在隔离的安全飞地（Enclave）中执行计算，确保即使操作系统或管理员也无法访问飞地内的数据。

在联邦大模型场景中，TEE的应用模式包括：

**TEE保护的聚合服务器**：将模型聚合操作部署在TEE飞地中，参与方加密上传模型更新，TEE内部解密、聚合、加密返回。这种方案在不依赖密码学假设的前提下实现了高效的安全聚合[28]。

**TEE保护的本地训练**：参与方在TEE飞地中执行本地训练，确保本地训练过程不被恶意软件窥探。这种方案适用于参与方自身安全性不可控的场景。

**混合方案**：结合TEE与密码学技术，利用TEE处理计算密集型操作（如模型聚合），利用密码学技术保护通信过程中的数据安全。

TEE方案的主要限制在于：（1）飞地内存容量有限（如Intel SGX的Enclave Page Cache通常限制在数百MB至数GB），难以直接容纳大模型的完整参数；（2）存在侧信道攻击（Side-Channel Attack）的理论风险。研究者通过模型分块加载、安全内存扩展（如Intel TDX）等技术缓解内存限制问题[29]。

## 12.5 案例分析

### 12.5.1 联邦LLM训练

**案例：医疗领域联邦大模型**

医疗领域是联邦大模型最具代表性的应用场景之一。医疗数据天然分散在不同的医院、诊所和研究机构中，且受到严格的隐私法规保护（如HIPAA、中国《健康医疗大数据安全管理办法》）。

研究者探索了基于联邦学习的医疗大模型训练方案。典型流程如下：多家三甲医院各自拥有电子病历、医学影像报告、临床试验数据等高质量医疗语料。各医院在本地基于通用大模型（如LLaMA-2-7B）进行领域微调，训练数据包括脱敏后的病历摘要、医学问答对等。通过联邦LoRA微调，各医院仅上传低秩适配器参数至中心服务器进行聚合，训练得到的全局模型在医学问答、临床辅助诊断等任务上表现出色[30]。

实验表明，联邦训练的医疗大模型在多个医学基准测试（如MedQA、PubMedQA）上的表现优于任何单方训练的模型，且接近集中式训练的效果。更重要的是，整个训练过程中原始医疗数据始终保留在各医院内部，满足了隐私合规要求。

**案例：多语言联邦大模型**

在多语言大模型训练中，不同语言的高质量语料分散在全球各地。联邦学习提供了一种自然的协作方式：各方基于本地语言数据训练模型，通过联邦聚合得到多语言能力。Google的研究团队探索了联邦学习在多语言模型适配中的应用，发现联邦训练可以有效缓解低资源语言的数据稀缺问题[31]。

### 12.5.2 跨机构大模型协作

**案例：金融风控联邦大模型**

金融机构面临严格的数据隔离要求，但风险识别能力的提升需要跨机构的知识共享。多家银行和金融科技公司探索了基于联邦学习的风控大模型协作方案。

各参与方在本地基于交易记录、反洗钱案例、欺诈模式等数据微调大模型，通过联邦学习聚合各方的知识。联邦训练的风控大模型在异常交易检测、洗钱行为识别等任务上显著优于单方模型，因为攻击模式具有跨机构的共性，联邦学习能够有效融合多方的威胁情报[32]。

在技术实现上，该项目采用了联邦LoRA微调结合中心差分隐私的方案。LoRA大幅降低了通信开销（每轮仅需传输约10MB参数），差分隐私提供了可量化的隐私保障（ε=8），安全聚合确保了各方更新内容的保密性。

**案例：政务数据联邦大模型**

在智慧城市建设中，不同政府部门（如公安、交通、民政、卫生）各自拥有不同维度的政务数据。联邦大模型使得这些部门能够在不共享原始数据的前提下，联合训练面向公共服务的大模型。例如，交通部门的出行数据与卫生部门的健康数据结合，可以训练出更精准的公共卫生预警模型。

### 12.5.3 边缘设备联邦学习

**案例：手机端联邦大模型**

随着移动设备算力的提升，将大模型部署到边缘设备并进行联邦学习成为新的研究热点。Google在其Gboard输入法中应用了联邦学习技术[33]，用于改进下一词预测和自动纠错功能。虽然Gboard的模型规模相对较小，但其架构设计为大模型的边缘联邦学习提供了重要参考。

在大模型的边缘联邦学习中，核心挑战包括：（1）设备算力有限，无法运行完整的大模型；（2）设备间算力差异巨大（高端旗舰机vs入门级手机）；（3）移动网络带宽不稳定；（4）电池续航限制了本地训练时长。

技术方案通常包括：使用模型蒸馏将大模型压缩为边缘可运行的小模型；采用LoRA等PEFT方法降低可训练参数量；设计自适应训练时长策略，根据设备电量和网络状况动态调整本地训练轮次；利用设备空闲时间（如夜间充电时）执行联邦训练[34]。

**案例：IoT设备联邦学习**

在工业物联网（IIoT）场景中，大量传感器和边缘设备产生的数据蕴含着设备故障预测、工艺优化等价值。联邦学习使得这些设备能够在不上传原始数据的前提下协同训练预测模型。然而，IoT设备的计算和存储资源极其有限（通常仅配备ARM Cortex-M级别的处理器），无法运行传统的大模型。

研究者提出了针对IoT场景的联邦TinyLLM方案：在云端训练大模型后，通过知识蒸馏和模型剪枝得到TinyLLM（参数量<100M），部署到IoT设备上进行联邦微调。通过8-bit量化和梯度检查点（Gradient Checkpointing）技术，使得TinyLLM可以在仅有256MB内存的设备上完成联邦训练[35]。

## 12.6 挑战与展望

联邦大模型作为联邦学习与大模型技术的交叉领域，仍面临诸多挑战：

**数据异构性的深化**：大模型训练对数据质量极为敏感，而联邦场景下的数据异构性（不同领域、不同语言、不同质量）可能导致模型偏置或性能退化。未来需要发展更鲁棒的聚合算法，如基于数据质量评估的加权聚合、基于元学习的个性化联邦训练等。

**激励机制设计**：联邦大模型的训练需要大量计算资源，如何公平衡量各参与方的贡献并设计合理的激励机制，是推动联邦大模型落地的关键。基于Shapley值的贡献评估、基于区块链的可信记录等方案正在探索中[36]。

**模型安全与对齐**：联邦场景下的模型安全面临独特挑战。恶意参与方可能通过投毒攻击（Poisoning Attack）影响全局模型的行为，或通过后门攻击（Backdoor Attack）植入特定触发模式。联邦RLHF中的偏好数据投毒也是一个值得关注的问题。需要发展联邦场景下的鲁棒聚合、异常检测和模型审计技术。

**可扩展性**：随着参与方数量的增加和模型规模的增长，联邦大模型系统的可扩展性面临严峻考验。分层联邦架构、模型并行与联邦学习的结合、以及基于流式处理的异步聚合等技术方向值得深入研究。

**标准化与互操作**：当前各联邦学习框架（如FATE、PaddleFL、Flower、NVIDIA FLARE等）在通信协议、聚合策略、隐私保护机制等方面缺乏统一标准，限制了跨平台协作的可能性。推动联邦大模型的标准化是产业化的必要条件。

## 12.7 本章小结

本章系统探讨了联邦大模型这一新兴研究范式。联邦学习与大模型的结合在三个层面展开：联邦预训练实现多方数据的联合利用，联邦微调（特别是基于LoRA的参数高效微调）大幅降低通信开销，联邦RLHF将对齐训练扩展到多方协作场景。通信效率优化通过LoRA联邦微调、梯度压缩和异步聚合等技术，使大模型的联邦训练在实际网络条件下可行。隐私保护增强通过差分隐私、安全聚合和TEE等技术，在模型效用与隐私保障之间取得平衡。典型案例展示了联邦大模型在医疗、金融、政务、边缘计算等领域的应用潜力。

联邦大模型代表了人工智能发展的一个重要方向：在保护数据隐私的前提下，充分释放分散数据的价值。随着技术的不断成熟和标准的逐步建立，联邦大模型有望成为推动AI民主化和普惠化的关键力量。

---

## 参考文献

[1] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]. AISTATS, 2017.

[2] Li T, Sahu A K, Zaheer M, et al. Federated optimization in heterogeneous networks[C]. MLSys, 2020.

[3] Karimireddy S P, Kale S, Mohri M, et al. SCAFFOLD: Stochastic controlled averaging for federated learning[C]. ICML, 2020.

[4] Wang J, Charles Z, Xu Z, et al. A field guide to federated optimization[J]. arXiv preprint arXiv:2107.06917, 2021.

[5] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-rank adaptation of large language models[C]. ICLR, 2022.

[6] Yu S, Yin H, Zhao J, et al. FedLoRA: Performance-preserving federated LoRA for large language models[J]. arXiv preprint arXiv:2311.16012, 2023.

[7] Bai Y, Chen H, Zhang J, et al. FedLoRA: Federated low-rank adaptation for large language models[C]. NeurIPS Workshop, 2023.

[8] McMahan B, Ramage D. Federated learning with reinforcement learning for human feedback[J]. Google Research Blog, 2023.

[9] Rafailov R, Sharma A, Mitchell E, et al. Direct preference optimization: Your language model is secretly a reward model[C]. NeurIPS, 2023.

[10] Zhang Z, Yang Y, Yao Z, et al. Federated full-parameter tuning of billion-sized language models with communication cost under 18KB[J]. arXiv preprint arXiv:2312.06353, 2023.

[11] Dettmers T, Pagnoni A, Holtzman A, et al. QLoRA: Efficient finetuning of quantized language models[C]. NeurIPS, 2023.

[12] Alistarh D, Grubic D, Li J, et al. QSGD: Communication-efficient SGD via gradient quantization and encoding[C]. NeurIPS, 2017.

[13] Vogels T, Karimireddy S P, Jaggi M. PowerSGD: Practical low-rank gradient compression for distributed optimization[C]. NeurIPS, 2019.

[14] Seide F, Fu H, Droppo J, et al. 1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs[C]. Interspeech, 2014.

[15] Xie C, Koyejo S, Gupta I. Asynchronous federated optimization[C]. OPT Workshop, NeurIPS, 2019.

[16] Liu L, Zhang J, Song S H, et al. Client-edge-cloud hierarchical federated learning[C]. ICC, 2020.

[17] Xie C, Koyejo S, Gupta I. Asynchronous federated optimization[C]. arXiv preprint arXiv:1903.03934, 2019.

[18] Charles Z, Garrett Z, Huo Z, et al. FedBuff: Asynchronous buffered federated learning[C]. AISTATS, 2022.

[19] Zhu L, Liu Z, Han S. Deep leakage from gradients[C]. NeurIPS, 2019.

[20] Kairouz P, McMahan B, Avent B, et al. Advances and open problems in federated learning[J]. Foundations and Trends in Machine Learning, 2021.

[21] Abadi M, Chu A, Goodfellow I, et al. Deep learning with differential privacy[C]. CCS, 2016.

[22] Li X, Tramèr F, Liang P, et al. Large language models can be strong differentially private learners[C]. ICLR, 2022.

[23] Yu D, Naik S, Backurs A, et al. Differentially private fine-tuning of language models[C]. ICLR, 2022.

[24] Bonawitz K, Ivanov V, Kreuter B, et al. Practical secure aggregation for privacy-preserving machine learning[C]. CCS, 2017.

[25] So J, Güler B, Avestimehr A S. Byzantine-resilient secure federated learning[J]. IEEE JSAC, 2021.

[26] Zhang C, Li S, Xia J, et al. BatchCrypt: Efficient homomorphic encryption for cross-silo federated learning[C]. USENIX ATC, 2020.

[27] Sun Z, Kairouz P, Suresh A T, et al. Can you really backdoor federated learning?[C]. NeurIPS Workshop, 2019.

[28] Mo F, Haddadi H, Katevas K, et al. PPFL: Privacy-preserving federated learning with trusted execution environments[C]. MobiSys, 2021.

[29] Mo Z, Xia X, Chen H, et al. Efficient privacy-preserving federated learning with TEE[J]. IEEE TDSC, 2023.

[30] Xu J, Glicksberg B S, Su C, et al. Federated learning for healthcare informatics[J]. J Healthcare Informatics Research, 2021.

[31] Hard A, Rao K, Mathews R, et al. Federated learning for mobile keyboard prediction[C]. NeurIPS, 2019.

[32] Yang Q, Liu Y, Chen T, et al. Federated machine learning: Concept and applications[J]. ACM TIST, 2019.

[33] Yang T, Andrew G, Eichner H, et al. Applied federated learning: Improving Google keyboard query suggestions[J]. arXiv preprint arXiv:1812.02903, 2018.

[34] Lim W Y B, Luong N C, Hoang D T, et al. Federated learning in mobile edge networks: A comprehensive survey[J]. IEEE Communications Surveys & Tutorials, 2020.

[35] Diao E, Ding J, Tarokh V. HeteroFL: Computation and communication efficient federated learning for heterogeneous clients[C]. ICLR, 2021.

[36] Kang J, Xiong Z, Niyato D, et al. Reliable federated learning for mobile networks[J]. IEEE Wireless Communications, 2020.
# 第13章 去中心化AI

## 13.1 引言

传统的联邦学习架构通常依赖一个中心化的参数服务器（Parameter Server）来协调参与者之间的模型聚合。这种架构虽然实现简洁、调度高效，但也带来了若干根本性问题：中心服务器构成单点故障（Single Point of Failure），一旦被攻陷或宕机，整个训练流程即告中断；服务器在聚合过程中掌握所有参与者的模型更新信息，存在隐私窥探的隐患；在跨机构、跨地域的大规模协作场景中，中心节点的带宽和计算能力往往成为系统瓶颈。

去中心化AI（Decentralized AI）正是为了克服上述局限而兴起的研究范式。其核心思想是：在不依赖可信第三方的前提下，通过点对点（Peer-to-Peer, P2P）通信、密码学原语和分布式共识机制，实现模型训练、推理和治理的全链路去中心化。本章将从三个维度展开讨论：去中心化联邦学习的通信拓扑与聚合协议、区块链技术与联邦学习的融合、以及去中心化推理的系统设计，最后分析该领域面临的核心挑战与未来方向。

---

## 13.2 去中心化联邦学习

### 13.2.1 从中心化到去中心化：架构演进

在经典联邦学习中，FedAvg（McMahan et al., 2017）采用的"客户端-服务器"架构已成为事实标准。每个参与方在本地数据上训练模型，将更新后的模型参数上传至中心服务器，服务器执行加权平均后将全局模型分发回各参与方。这一流程清晰简洁，但服务器的中心化角色意味着：

- **单点故障风险**：服务器不可用则训练终止；
- **隐私信任假设**：服务器虽不接触原始数据，但可观察所有模型更新，可能通过梯度反演（Gradient Inversion）等攻击手段推断训练数据；
- **扩展性瓶颈**：当参与方数量增长到数千甚至数万时，服务器的通信和计算能力难以承载。

去中心化联邦学习（Decentralized Federated Learning, DFL）消除了中心服务器的角色，每个参与方既是训练者也是聚合者，通过与邻居节点直接交换模型更新来实现全局知识的传播。这一转变不仅是工程架构的调整，更涉及算法收敛性、通信效率和安全性等方面的深刻变革。

### 13.2.2 对等网络拓扑与Gossip协议

去中心化联邦学习的核心问题之一是：在没有中心协调者的情况下，如何让每个节点的本地模型逐步收敛到全局最优？答案借鉴了分布式系统中经典的Gossip协议（Epidemic Protocol）。

**Gossip协议的基本原理**源于流行病传播模型。在每一轮通信中，每个节点随机选择一个或多个邻居，将自己的模型参数（或其增量）发送给对方，同时接收对方的参数进行本地聚合。这一过程不断迭代，信息如同"病毒"般在网络中扩散，最终所有节点的模型趋于一致。

具体而言，假设网络中有 $N$ 个节点，第 $k$ 轮中节点 $i$ 的模型参数为 $w_i^{(k)}$。节点 $i$ 随机选择邻居集合 $\mathcal{N}_i$，执行如下混合更新：

$$w_i^{(k+1)} = \sum_{j \in \mathcal{N}_i \cup \{i\}} P_{ij} \cdot w_j^{(k)}$$

其中 $P_{ij}$ 为混合权重矩阵的元素，通常由网络拓扑决定。Metropolis权重和最大度权重是两种常见的选择。混合矩阵 $P$ 需满足双随机（Doubly Stochastic）性质，以保证所有节点最终收敛到相同的模型参数。

**Push-Sum协议**是另一种经典的去中心化平均方法，其优势在于对有向图和非对称通信场景有更好的适应性。每个节点维护一对变量 $(w_i, s_i)$，分别表示加权参数和权重总和，通过比率 $w_i / s_i$ 来估计全局平均值。Lian et al.（2017）在其开创性工作中证明，去中心化SGD在非凸优化问题上可以达到与中心化方法相当的收敛速率 $\mathcal{O}(1/\sqrt{NT})$，其中 $N$ 为节点数、$T$ 为总迭代轮数。

### 13.2.3 去中心化联邦学习的关键算法

在Gossip通信的基础上，研究者们提出了多种去中心化联邦学习算法：

**DFedAvg**（Decentralized Federated Averaging）将FedAvg的思想扩展到去中心化场景。每个节点在本地执行多轮SGD训练，然后与邻居节点交换模型参数并进行加权平均。与原始Gossip SGD不同，DFedAvg允许更多的本地计算步数，从而减少通信频率。He et al.（2021）的分析表明，适当增加本地步数可以在通信效率和收敛速度之间取得良好平衡。

**DFedDG**（Distribution Guided Decentralized Federated Learning）进一步考虑了数据异质性问题。在联邦学习中，各参与方的数据分布往往存在显著差异（Non-IID问题），这在去中心化场景中更为突出，因为每个节点仅与局部邻居通信，全局数据分布的信息传播更慢。DFedDG通过建模每个节点的数据分布特征来指导聚合权重的自适应调整，在保持去中心化特性的同时改善了个性化学习效果。

**MATCHA**（Wang et al., 2020）从通信效率角度出发，提出了基于匹配抽样的去中心化训练框架。其核心观察是：在每轮通信中，并非所有节点对之间的通信都同等重要。MATCHA通过分析每个节点对全局收敛的贡献度来优化通信调度，在减少50%以上通信开销的同时维持了模型精度。

### 13.2.4 拓扑设计与通信优化

网络拓扑对去中心化联邦学习的性能有决定性影响。拓扑的谱隙（Spectral Gap）——即混合矩阵第二大特征值与1之间的差距——直接决定了信息扩散的速度。谱隙越大，信息传播越快，收敛速度也越快。

常用的拓扑结构包括：

- **环形拓扑（Ring）**：每个节点仅与两个邻居通信，通信开销最低，但谱隙较小，收敛较慢；
- **全连接拓扑（Fully Connected）**：每对节点之间均有通信链路，谱隙最大，但通信开销随节点数平方增长；
- **随机图（Random Graph）**：以一定概率在节点间建立连接，在通信开销和收敛速度之间取得折中；
- **指数图（Exponential Graph）**：每个节点连接 $O(\log N)$ 个邻居，可在 $O(\log N)$ 轮内实现全网信息传播，被广泛认为是理论最优的拓扑选择之一。

Zhou et al.（2024）在WWW 2024上发表的工作进一步研究了通过操控拓扑边来加速去中心化联邦学习的方法，表明动态调整通信拓扑可以在训练过程中自适应地优化信息流动。

---

## 13.3 区块链与联邦学习

### 13.3.1 融合动机

区块链技术与联邦学习的融合并非简单的技术叠加，而是源于两者在去中心化愿景上的深层契合。联邦学习解决了"数据不动模型动"的隐私保护问题，但传统的中心化协调架构引入了对服务器的信任假设。区块链作为一种去中心化的分布式账本技术，天然具有以下特性，恰好弥补了联邦学习的不足：

- **不可篡改性**：模型更新、训练记录等关键信息一旦上链便无法被事后修改，提供了可审计的基础设施；
- **去中心化共识**：无需可信第三方即可在互不信任的参与方之间建立一致性；
- **智能合约**：可在链上自动执行聚合逻辑、激励分配和惩罚机制，减少人为干预。

这一融合方向通常被称为"基于区块链的联邦学习"（Blockchain-based Federated Learning, BCFL），近年来已成为隐私计算领域最活跃的研究方向之一（Lim et al., 2020; Qu et al., 2022）。

### 13.3.2 区块链辅助的模型聚合

在BCFL框架中，区块链替代了中心服务器的聚合角色。典型的聚合流程如下：

1. **本地训练**：各参与方在本地数据上完成模型训练，生成模型更新 $\Delta w_i$；
2. **上链提交**：参与方将模型更新（或其哈希摘要）作为交易提交到区块链网络；
3. **共识与聚合**：矿工节点（或验证节点）通过共识机制选出本轮的聚合者，聚合者从链上收集模型更新并执行加权平均；
4. **全局更新**：聚合结果作为新的区块被写入链上，所有参与方从链上获取最新的全局模型。

**BlockDFL**（Shayan et al., 2020）是这一方向的代表性工作。该框架采用基于PBFT（Practical Byzantine Fault Tolerance）的投票机制来协调聚合过程，并引入了双层评分机制来评估参与方的贡献质量。为降低链上存储开销，BlockDFL采用了梯度压缩技术，仅将压缩后的模型更新上链。

**FLChain**（Bao et al., 2019）提出了一种更为系统化的链上联邦学习架构。该工作引入了验证节点的角色，负责对参与方提交的模型更新进行正确性验证，并通过链上声誉系统来追踪和评估每个参与方的历史贡献。验证节点的引入有效缓解了拜占庭攻击（Byzantine Attack）的风险——恶意参与方提交的异常模型更新可以在聚合前被检测和过滤。

值得注意的是，将完整的模型参数直接存储在区块链上在计算和存储成本上是不可行的。实际系统通常采用以下策略之一：

- **链上哈希 + 链下存储**：模型参数存储在IPFS等去中心化存储系统中，链上仅保存哈希引用；
- **模型压缩**：通过量化、稀疏化或Top-K选择等技术减小上链数据量；
- **分层架构**：在主链（Layer 1）上记录聚合结果的承诺（Commitment），在侧链或状态通道（Layer 2）上处理高频的模型更新交换。

### 13.3.3 激励机制设计

联邦学习中的"搭便车"（Free-Riding）问题是一个长期存在的挑战：部分参与方可能只享受全局模型带来的收益，而不愿贡献本地计算和数据资源。在去中心化场景中，缺乏中心化权威来强制执行参与规则，因此基于区块链的激励机制设计尤为重要。

典型的激励设计包含以下要素：

**贡献评估**是激励机制的基础。Shapley值（Shapley Value）是衡量参与方贡献的经典方法，但其计算复杂度随参与方数量呈指数增长。在实践中，研究者提出了多种近似方法。Kang et al.（2019）提出的基于声誉（Reputation-based）的评估机制通过追踪参与方的历史表现来估计其贡献水平；Yu et al.（2020）提出了基于数据质量和模型性能提升的贡献度量方法。

**代币经济（Token Economy）** 是区块链激励的常见载体。参与方通过贡献计算资源和高质量模型更新获得代币奖励，而使用全局模型则需要支付代币。智能合约负责自动执行奖励分配逻辑，确保透明和公正。Zhan et al.（2020）提出的FedCoin系统设计了一套基于Shapley值的支付方案，通过区块链实现贡献与收益的精确匹配。

**惩罚机制**同样不可或缺。对于提交低质量更新或恶意更新的参与方，系统可以通过扣除质押代币、降低声誉评分或限制参与权限来进行惩罚。这种"胡萝卜加大棒"的组合设计有助于维护联邦学习生态的长期健康。

### 13.3.4 可审计性与可验证性

在高风险的AI应用场景中（如医疗诊断、金融风控），模型的训练过程需要具备可审计性——即能够事后追溯和验证模型是如何被训练的、使用了哪些数据、每个参与方做出了什么贡献。区块链的不可篡改日志为此提供了天然的基础设施。

Li et al.（2024）在IEEE上发表的综述系统地讨论了基于区块链的可审计和可验证联邦学习框架。该工作指出，可审计性涉及三个层次：

- **数据可审计**：记录训练数据的来源、预处理步骤和使用权限，确保数据合规性；
- **模型可审计**：记录每轮聚合的模型更新、聚合方法和超参数设置，使模型训练过程可复现；
- **参与方可审计**：记录每个参与方的行为历史，包括提交更新的时间、质量和一致性。

**零知识证明（Zero-Knowledge Proof, ZKP）** 的引入进一步增强了可验证性。参与方可以在不泄露模型更新具体内容的前提下，证明其更新确实是在本地数据上正确训练得到的。这种"可验证而不泄露"的特性对于平衡隐私保护和可审计性至关重要。Pedersen承诺和zk-SNARK等密码学工具已被应用于构建此类可验证联邦学习系统。

---

## 13.4 去中心化推理

### 13.4.1 从训练到推理的去中心化

如果说去中心化联邦学习关注的是"如何在没有中心协调者的情况下训练模型"，那么去中心化推理（Decentralized Inference）关注的则是"如何在分布式设备上高效运行已训练好的模型"。随着大语言模型（LLM）等超大规模模型的涌现，单个设备的计算和内存资源往往不足以承载完整的模型推理，这使得去中心化推理从一个学术设想变为迫切的工程需求。

去中心化推理的核心挑战可以概括为：**如何将一个计算密集型的推理任务分解为多个子任务，分配到异构的分布式设备上并行执行，同时最小化通信开销和推理延迟？**

### 13.4.2 模型分片与流水线并行

**模型分片（Model Sharding）** 是实现去中心化推理的基本策略。其思路是将神经网络按层或按模块切分为多个分片（Shard），每个分片部署在不同的设备上。推理时，输入数据依次流经各分片，每个分片完成其负责的计算后将中间结果传递给下一个分片。

Angione et al.（2024）提出的Nesa框架是模型无关（Model-Agnostic）分片方法的代表。该框架支持对任意深度神经网络进行顺序分片，并通过区块链协调各分片节点之间的执行顺序和数据传递。其创新之处在于采用混合分片策略——结合层间流水线并行和层内张量并行——以适应异构设备的不同计算能力。

流水线并行（Pipeline Parallelism）的关键挑战是**气泡时间（Bubble Time）**：由于各分片的计算时间不同，较快的分片需要等待较慢的分片完成，导致设备利用率下降。微批次（Micro-batching）技术通过将一个推理请求切分为多个微批次来填充气泡，提高流水线效率。

### 13.4.3 边缘计算与联合推理

边缘计算（Edge Computing）为去中心化推理提供了天然的部署环境。在网络边缘部署的计算节点——如基站、路由器、智能网关等——具有低延迟、高带宽的网络接入优势，可以就近处理用户请求，避免将所有数据传回云端。

**联合推理（Federated Inference）** 是边缘计算与联邦学习的交叉创新。在这一范式中，多个边缘节点各自持有一个轻量化的本地子模型，协作完成推理任务。典型的联合推理流程包括：

1. **特征提取阶段**：各边缘节点在本地对输入数据进行初步特征提取，得到轻量级的中间表示；
2. **安全聚合阶段**：各节点通过安全多方计算或同态加密等密码学协议，在不泄露各自中间特征的前提下进行聚合；
3. **决策输出阶段**：聚合后的特征送入后续的模型层完成最终推理。

这种方法的优势在于：每个边缘节点只需存储和执行模型的一个子集，大幅降低了单设备的资源需求；同时，中间特征的数据量远小于原始输入数据，减少了跨节点通信的开销。

Kang et al.（2024）研究了去中心化LLM推理在边缘网络中的能耗优化问题。该工作指出，在能量受限的边缘设备上进行大模型推理时，需要在推理精度、延迟和能耗之间进行精细的权衡。通过自适应的模型分片策略——根据各设备的剩余电量和计算能力动态调整分片方案——可以在保证推理质量的前提下最大化网络整体的推理吞吐量。

### 13.4.4 去中心化推理的隐私保护

去中心化推理中同样存在隐私风险。当一个推理请求被分解为多个子任务在不同节点上执行时，每个节点都可以观察到流经自己的中间激活值（Activation），这些中间值可能泄露关于原始输入的敏感信息。

为应对这一威胁，研究者提出了多种保护策略：

- **差分隐私噪声注入**：在中间激活值上添加校准噪声，使单个节点无法精确推断原始输入；
- **可信执行环境（TEE）**：利用Intel SGX、ARM TrustZone等硬件安全模块在隔离环境中执行分片计算，防止外部窥探；
- **同态加密推理**：在加密状态下执行推理计算，但目前同态加密的计算开销仍然过高，限制了其在实时推理场景中的应用；
- **拆分学习（Split Learning）**：将模型在某一层处切分，客户端处理前半部分，服务端处理后半部分，仅交换该层的中间激活值。通过结合差分隐私或加性秘密共享，可以进一步降低激活值泄露的风险。

---

## 13.5 挑战与展望

### 13.5.1 通信效率

通信开销始终是去中心化AI系统的核心瓶颈。在去中心化联邦学习中，每轮通信的参与者数量、每次传输的参数规模以及全局收敛所需的通信轮数，共同决定了系统的总通信成本。在去中心化推理中，模型分片之间的中间结果传递同样消耗大量带宽。

当前的优化方向包括：

- **梯度压缩**：Top-K稀疏化、随机稀疏化和量化（如8-bit、4-bit量化）可将通信量减少1-2个数量级，但需要精心设计误差补偿机制以保证收敛性（Lin et al., 2018; Alistarh et al., 2017）；
- **本地多步更新**：允许节点在本地执行多轮SGD后再通信，以通信轮数换取每轮的计算量；
- **异步通信**：打破同步屏障（Synchronization Barrier），允许节点以不同速度更新，避免因"掉队者"（Straggler）而阻塞全局进度，但需要处理陈旧梯度（Stale Gradient）带来的收敛性影响；
- **通信拓扑优化**：如前文所述，通过动态调整网络拓扑来优化信息传播效率。

### 13.5.2 安全性与鲁棒性

去中心化系统面临的安全威胁比中心化系统更为复杂。在没有中心权威的情况下，恶意节点有更多的攻击面：

**拜占庭攻击**：恶意节点故意提交错误的模型更新以破坏全局模型的收敛。在中心化场景中，服务器可以通过聚合算法（如Krum、Trimmed Mean）来过滤异常更新，但在去中心化场景中，每个节点只能观察到局部邻居的更新，检测难度更大。Blanchard et al.（2017）提出的Krum算法和Yin et al.（2018）提出的Trimmed Mean已被扩展到去中心化场景，但通信开销和检测准确率之间的权衡仍待优化。

**模型投毒（Model Poisoning）**：攻击者可以提交看似合理但经过精心构造的模型更新，使全局模型在特定输入上产生错误输出（后门攻击）。这类攻击在去中心化场景中更难防御，因为缺乏全局视角来检测异常模式。

**隐私攻击**：即使在去中心化场景中，邻居节点之间交换的模型更新仍可能泄露信息。梯度反演攻击（Zhu et al., 2019）已证明可以从梯度中重建训练数据。差分隐私和安全聚合是应对这类威胁的主要手段，但两者的引入都会带来精度损失或额外的通信/计算开销。

### 13.5.3 激励机制的理论基础

设计有效的激励机制是去中心化AI系统可持续运行的关键。当前的研究面临若干理论挑战：

**贡献度量的精确性**：Shapley值虽然是公平的贡献度量，但其计算复杂度为 $\mathcal{O}(N!)$，在大规模联邦学习中不可行。基于蒙特卡洛采样的近似方法可以降低复杂度，但估计方差较大。如何设计既精确又高效的贡献度量方法，仍是一个开放问题。

**激励相容性（Incentive Compatibility）**：理想的激励机制应满足"说真话是最优策略"的性质——即参与方如实报告自己的数据量、计算能力和模型更新是最有利的。然而，现有机制在面对策略性参与方时往往缺乏理论保证。

**长期动态均衡**：联邦学习是一个多轮迭代的过程，参与方的行为具有长期效应。基于重复博弈理论和演化博弈理论的激励设计正在成为研究热点，但如何将这些理论工具与实际的联邦学习训练过程有效结合，仍有待深入探索。

**代币经济的稳定性**：在基于区块链的激励系统中，代币的价值波动可能影响参与方的积极性。设计通缩或通胀机制来维持代币价值的相对稳定，需要经济学和计算机科学的深度交叉。

### 13.5.4 治理与标准化

去中心化AI系统的治理涉及技术标准、法律合规和社区决策等多个维度：

**链上治理（On-Chain Governance）**：通过智能合约定义治理规则，允许参与方对系统参数（如聚合算法、激励方案、准入门槛等）进行投票决策。去中心化自治组织（DAO）模式为此提供了组织框架，但面临投票权分配、"巨鲸"操控和投票率低下等挑战。

**跨域联邦**：在实际应用中，不同组织可能使用不同的联邦学习框架和区块链平台。如何实现跨框架、跨链的互操作性，是去中心化AI走向大规模应用的前提。标准化工作（如IEEE P3652.1联邦学习架构标准）正在推进，但距离产业级成熟仍有距离。

**法规遵从**：去中心化系统中的数据流动跨越多个司法管辖区，需要同时满足GDPR、个人信息保护法等不同法规的要求。区块链的不可篡改性与"被遗忘权"之间的矛盾、跨境数据传输的合规性等问题，需要技术方案与法律框架的协同演进。

### 13.5.5 未来方向

展望未来，去中心化AI的发展可能沿以下方向展开：

**大模型时代的去中心化训练与推理**：随着GPT-4、LLaMA等千亿参数模型的兴起，单机构训练越来越难以独立承担。去中心化的分布式训练框架——如基于环形All-Reduce的跨机构协作——可能成为突破算力瓶颈的关键路径。同时，去中心化推理网络可以让边缘设备协作运行大模型，降低对中心化云计算的依赖。

**联邦学习与Web3的深度融合**：区块链、去中心化存储（IPFS/Filecoin）、去中心化身份（DID）等Web3基础设施的成熟，将为去中心化AI提供更完善的底层支撑。基于NFT的模型版权确权、基于DeFi的算力市场等创新应用值得期待。

**自适应去中心化系统**：未来的去中心化AI系统应能根据网络条件、参与方能力和任务需求，自适应地调整通信拓扑、聚合策略和分片方案。强化学习和元学习技术有望在此发挥关键作用。

**可证明安全的去中心化协议**：随着去中心化AI系统在高安全场景中的部署，形式化验证和可证明安全（Provable Security）将成为重要的研究方向。如何在不牺牲效率的前提下，为去中心化聚合协议、激励机制和推理流程提供严格的安全保证，是理论和实践的双重挑战。

---

## 13.6 本章小结

本章系统地探讨了去中心化AI的三个核心维度：去中心化联邦学习通过Gossip协议和P2P通信消除了对中心服务器的依赖，区块链技术为联邦学习提供了不可篡改的审计记录、自动化的激励机制和去中心化的共识基础，去中心化推理通过模型分片和边缘协作使大规模模型的分布式运行成为可能。

这三个维度并非孤立存在，而是相互支撑、协同演进的。区块链可以为去中心化联邦学习提供聚合协调和激励保障；去中心化推理需要联邦学习来在保护隐私的前提下持续优化部署在边缘的模型；而Gossip协议和分布式共识机制则为整个去中心化AI生态提供了通信和协作的基础原语。

尽管通信效率、安全性、激励设计和治理标准化等挑战仍然存在，但随着密码学、分布式系统和区块链技术的持续进步，去中心化AI正从理论研究走向工程实践，有望在隐私保护、抗审查和民主化的AI发展中发挥越来越重要的作用。

---

## 参考文献

1. McMahan, B., Moore, E., Ramage, D., et al. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*.
2. Lian, X., Zhang, C., Zhang, H., et al. (2017). Can decentralized algorithms outperform centralized algorithms? A case study for decentralized parallel stochastic gradient descent. *NeurIPS*.
3. He, L., Bian, A., & Jaggi, M. (2021). COLA: Communication-collaboration for decentralized SGD. *ICML*.
4. Wang, J., Sahu, A. K., Yang, Z., et al. (2020). MATCHA: Speeding up decentralized SGD with matching decomposition sampling. *AISTATS*.
5. Shayan, M., Fung, C., Yoon, C. J. M., & Beschastnikh, I. (2020). Biscotti: A blockchain system for private and secure federated learning. *IEEE TPDS*.
6. Bao, X., Su, C., Xiong, Y., et al. (2019). FLChain: A blockchain for auditable federated learning with trust and incentive. *ACM BSCI*.
7. Kang, J., Xiong, Z., Niyato, D., et al. (2019). Incentive mechanism for reliable federated learning: A joint optimization approach to combining reputation and contract theory. *IEEE IoT Journal*.
8. Yu, H., Liu, Z., Liu, Y., et al. (2020). A fairness-aware incentive scheme for federated learning. *AAMAS*.
9. Zhan, Y., Zhang, J., Li, J., et al. (2020). FedCoin: A peer-to-peer payment system for federated learning. *FL-IJCAI*.
10. Li, Y., et al. (2024). Auditable and verifiable federated learning based on blockchain and zero-knowledge proof. *IEEE Access*.
11. Lim, W. Y. B., Luong, N. C., Hoang, D. T., et al. (2020). Federated learning in mobile edge networks: A comprehensive survey. *IEEE Communications Surveys & Tutorials*.
12. Qu, Y., Uddin, M. P., Gan, C., et al. (2022). Blockchain for federated learning toward security and privacy: A survey. *IEEE Network*.
13. Angione, G., Zhao, Y., et al. (2024). Model agnostic hybrid sharding for heterogeneous distributed inference. *arXiv preprint*.
14. Kang, Y., et al. (2024). Decentralized LLM inference over edge networks with energy constraints. *arXiv preprint*.
15. Lin, Y., Han, S., Mao, H., et al. (2018). Deep gradient compression: Reducing the communication bandwidth for distributed training. *ICLR*.
16. Alistarh, D., Grubic, D., Li, J., et al. (2017). QSGD: Communication-efficient SGD via gradient quantization and encoding. *NeurIPS*.
17. Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. *NeurIPS*.
18. Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). Byzantine-robust distributed learning: Towards optimal statistical rates. *ICML*.
19. Zhu, L., Liu, Z., & Han, S. (2019). Deep leakage from gradients. *NeurIPS*.
20. Zhou, M., Liu, G., Lu, K., et al. (2024). Accelerating the decentralized federated learning via manipulating edges. *WWW*.
21. DFedDG2: Distribution guided gossip based generalizable and personalized decentralized federated learning. (2026). *IEEE*.
# 第14章 隐私计算的未来

> **本章导读**：隐私计算正在从单一技术走向系统性融合，从学术探索走向规模化产业落地。本章将从技术融合趋势、标准化与互操作、产业化挑战三个维度展望隐私计算的发展方向，并在章末回顾全书核心洞察，勾勒这一领域的开放问题与未来图景。

---

## 14.1 技术融合趋势：从单点突破到系统协同

隐私计算的四大核心技术——联邦学习（FL）、差分隐私（DP）、安全多方计算（MPC）和可信执行环境（TEE）——各有优势与局限。未来的演进方向不是某一种技术的"独占鳌头"，而是多种技术的深度融合与协同优化。

### 14.1.1 联邦学习与差分隐私的深度耦合

联邦学习的核心思想是"数据不动模型动"，但原始的联邦学习协议并不能提供严格的隐私保障。梯度信息本身可能泄露训练数据的敏感属性——这一事实在Abadi等人（2016）的开创性工作中得到了系统论证。差分隐私通过向梯度或模型参数中注入校准噪声，为联邦学习提供了可量化的隐私保证。

当前的融合趋势体现在三个层面。**其一，自适应噪声分配机制**。早期方案采用全局统一的隐私预算（ε），对模型效用损失较大。近年来研究者提出了基于层敏感度分析的自适应噪声策略——对底层特征提取层注入更多噪声，对高层分类层保留更多精度，显著改善了隐私-效用权衡曲线。一些工作进一步将梯度裁剪阈值与噪声尺度进行联合优化，在保证差分隐私的同时最大限度地保留模型收敛速度。**其二，Rényi差分隐私与组合定理的优化**。传统的（ε, δ）-差分隐私在多轮联邦训练中的组合效应会导致隐私预算快速消耗。Rényi差分隐私（RDP）及其后续改进（如零集中差分隐私，zCDP）提供了更紧的隐私组合界，使得联邦学习可以在更多训练轮次中维持可接受的隐私保障。Google在2022年发布的联邦学习生产系统中，已经全面采用RDP框架来管理训练过程中的隐私预算消耗。**其三，本地差分隐私与联邦学习的结合**。在跨设备联邦学习场景中，参与方数量庞大但每方数据极少，传统的中心化差分隐私模型难以适用。本地差分隐私（LDP）允许每个参与方在本地独立地扰动梯度更新，虽然引入了更大的噪声，但消除了对可信聚合服务器的依赖。Apple和Google在其移动端联邦学习部署中均采用了LDP或其变体。

### 14.1.2 联邦学习与安全多方计算的协同

联邦学习的聚合过程本身存在信息泄露风险：诚实但好奇（honest-but-curious）的聚合服务器可以从梯度更新中推断出参与方的训练数据分布。安全多方计算为解决这一问题提供了密码学级别的保障。

Bonawitz等人（2017）提出的安全聚合协议是FL-MPC融合的代表性成果，该协议基于秘密共享和成对掩码技术，允许服务器仅获得梯度聚合结果，而无法获知单个参与方的梯度更新，在参与方掉线的情况下仍能保证协议的鲁棒性。更进一步的融合方向包括：基于Shamir秘密共享或加性秘密共享的联邦模型参数分割，在纵向联邦学习中实现不暴露各自特征的联合训练；混淆电路（Garbled Circuits）在联邦树模型（如XGBoost）中的广泛应用，为需要比较或选择操作的场景提供了高效解决方案；以及Function Secret Sharing（FSS）的引入——它将函数而非数据进行秘密共享，使得参与方可以高效地计算ReLU激活函数等特定函数的输出，有望将在线推理的通信开销降低一到两个数量级。

### 14.1.3 可信执行环境的融合角色

TEE（如Intel SGX、ARM TrustZone、AMD SEV）通过硬件隔离创建安全飞地（enclave），在融合架构中扮演着"信任锚点"的角色。将联邦学习的聚合逻辑放入TEE中执行，可以同时实现数据保密性和计算完整性——前者防止恶意服务器窥探梯度，后者防止参与方注入恶意更新。纯MPC协议在复杂计算中的性能开销仍然显著，纯TEE方案面临侧信道攻击等硬件安全风险，混合架构将MPC用于高安全需求操作（如密钥管理、隐私预算核算），将TEE用于计算密集型操作（如矩阵运算、模型推理），在安全性和效率之间取得平衡。此外，TEE的远程证明（Remote Attestation）机制为分布式隐私计算系统提供了可验证的信任基础。RATS（Remote Attestation Procedures）IETF标准的推进正在为此类融合架构奠定互操作基础。

### 14.1.4 新兴融合范式

**零知识证明**（ZKP）允许证明者向验证者证明某个陈述为真而不泄露任何额外信息，可用于验证联邦学习参与方是否遵守了训练协议（如是否使用了正确的数据分布），而无需检查其原始数据。zk-SNARK和zk-STARK在联邦学习合规性验证中的应用正在成为研究热点。**全同态加密**（FHE）允许在密文上直接进行计算，理论上是隐私计算的"银弹"。近年来基于CKKS方案的近似同态加密和硬件加速（如Intel HEXL库、专用FHE加速芯片）的进展正在缩小性能差距，Duality Technologies和Zama等公司已推出基于FHE的产品，在特定场景中已具备实用价值。**基于差分隐私训练的生成模型**（如DP-GAN、PATE-GAN）可以生成具有隐私保障的合成数据，在医疗数据共享和金融风控等敏感领域展现出独特价值，是隐私计算从"保护计算过程"向"保护数据全生命周期"延伸的重要方向。

---

## 14.2 标准化与互操作：从碎片化到生态统一

不同厂商的隐私计算实现在协议格式、安全假设、接口定义等方面存在显著差异，导致"隐私计算孤岛"问题——不同系统之间难以互联互通。

### 14.2.1 标准化进程

**国际标准层面**，IEEE、ISO/IEC和IETF是三大主要标准化组织。IEEE P3652.1（联邦学习架构与应用标准）是中国牵头的首个联邦学习国际标准，已于2021年正式发布，IEEE P2830和P2976正在制定中。ISO/IEC JTC 1/SC 42正在开展联邦学习和差分隐私相关标准预研，重点关注AI系统中的隐私保护技术框架。IETF的RATS和Privacy Pass等工作组为远程证明和网络层互操作奠定了基础。**国内标准层面**，TC260已发布《信息安全技术 联邦学习安全要求》等多项标准，CCSA发布了涵盖联邦学习、MPC和TEE的行业标准，AIIA推动了隐私计算产品评测和互操作规范的制定。中国在隐私计算标准化方面走在前列。

### 14.2.2 开源框架与生态

开源框架是隐私计算技术民主化的关键推动力。当前主要框架包括：微众银行的FATE（工业级纵向/横向联邦，生态成熟）、蚂蚁集团的SecretFlow（全栈隐私计算，统一抽象层）、Google的TensorFlow Federated（与TensorFlow生态深度集成）、OpenMined的PySyft（研究导向，灵活可扩展）、Meta的CrypTen（基于PyTorch，深度学习友好）、Microsoft的FLUTE（面向大规模生产部署）、以及Zama的Concrete（编译器驱动的FHE应用开发）。这些框架在底层协议实现、API设计和安全模型等方面存在显著差异。Linux基金会旗下的Confidential Computing Consortium（CCC）致力于推动TEE的开放标准和互操作规范，国产框架也在积极探索基于中间表示（IR）的跨框架联邦学习互操作方案。

### 14.2.3 互操作协议

互操作面临四个层次的挑战。**协议层**——不同框架采用的密码学协议各异（如基于Paillier的同态加密vs.基于加性秘密共享的MPC），需要定义统一的"隐私计算中间协议层"，将底层细节抽象为标准化的计算原语和通信接口。**数据格式层**——联邦学习中的模型参数、梯度更新和元数据需要统一的序列化格式（如基于Protocol Buffers的联邦学习消息格式）。**安全模型层**——不同方案的信任假设（半诚实模型/恶意模型/TEE依赖）之间需要建立可证明的映射关系，确保联合执行时的整体安全性不低于任一参与方的安全预期。**隐私度量层**——差分隐私ε值、信息论度量、形式化验证等不同度量方式之间需要统一的比较框架，IEEE P3652.1提出的隐私分级体系是这一方向的有益探索。

---

## 14.3 产业化挑战：从可用到好用

隐私计算的产业化已经从"技术验证"阶段进入"规模化应用"阶段，但仍面临诸多工程化和商业化的挑战。

### 14.3.1 性能优化

性能是产业化最核心的瓶颈之一。以MPC为例，其计算开销通常是明文计算的数千到数万倍，通信开销更是制约大规模部署的关键因素。**计算性能优化**方向包括：硬件加速（GPU上的同态加密加速、FPGA上的秘密共享加速、Intel Crypto指令集、以及正在研发中的FHE专用芯片，有望将效率提升一到两个数量级）、算法优化（深度梯度压缩、Top-K稀疏化、量化梯度、异步联邦学习和部分参与方聚合策略）、以及编译优化（面向隐私计算的编译器可以自动将ML算子映射为高效密码学原语序列，如SecretFlow的SPU编译器）。**通信性能优化**方向包括：梯度压缩与量化编码技术减少数据传输量、通信拓扑优化（Gossip协议等去中心化方案在大规模场景中减少瓶颈）、以及边缘-云协同计算（将部分任务卸载到5G基站或边缘服务器）。

### 14.3.2 成本降低

隐私计算的部署成本远高于集中式机器学习，涵盖计算资源、通信带宽、工程人才和合规审计等方面。降低成本的路径包括：将隐私计算封装为标准化云服务（Privacy-Preserving Computation as a Service, PPCaaS）以降低企业技术门槛，AWS、Azure和阿里云等已推出相关服务；通过开源生态降低技术获取成本；以及采用混合精度与自适应安全策略——根据数据敏感度和应用场景动态调整安全级别，对低敏感数据使用轻量级方案（如TEE），对高敏感数据使用重量级方案（如MPC），避免"一刀切"带来的不必要开销。

### 14.3.3 易用性提升

当前开发者面临密码学门槛高、加密计算调试困难、安全参数调优复杂等问题。提升方向包括：**声明式编程模型**——开发者只需描述"对什么数据做什么计算、需要什么级别的隐私保护"，框架自动选择最优方案和技术组合，SecretFlow的声明式API是这一方向的有益探索；**自动化安全参数选择**——基于威胁模型和隐私需求自动确定安全参数；**可视化工具链**——提供隐私计算流程的可视化监控和调试工具；以及**与主流AI框架的无缝集成**——将隐私计算能力嵌入TensorFlow、PyTorch等框架，TensorFlow Federated和Flower等已做出开创性工作。

### 14.3.4 安全性与可信度

产业化中需要特别关注以下安全问题。**侧信道攻击**——TEE（如Intel SGX）面临的Spectre、Meltdown、Plundervolt等侧信道攻击不断被发现，削弱了硬件信任基础，防御措施会带来额外性能开销。**联邦后门攻击**——恶意参与方可以通过精心构造的模型更新在联邦学习模型中植入后门，拜占庭容错聚合和异常检测是当前的防御热点。**隐私会计的精确性**——差分隐私预算在参与方动态变化、训练过程中断和恢复的场景下难以精确追踪，过度保守的管理损害效用，过度激进的管理则可能导致隐私泄露。**形式化验证**——使用ProVerif、Tamarin等工具可以提供更高的安全保证，但验证的复杂性和成本限制了其广泛应用。

---

## 14.4 结语：数据安全的AI新范式

### 14.4.1 全书回顾

本书系统探讨了联邦学习与隐私计算这一数据安全新范式。在**理论基础**部分，阐述了隐私计算的密码学根基（秘密共享、同态加密、混淆电路）、信息论基础（差分隐私、信息泄露量化）和硬件信任模型（可信执行环境）。在**联邦学习**部分，从横向联邦、纵向联邦和联邦迁移学习三个维度展开，分析了FedAvg、FedProx、SCAFFOLD等核心算法，讨论了通信效率、模型收敛性和系统鲁棒性等关键问题。在**隐私增强技术**部分，介绍了差分隐私在联邦学习中的应用、安全多方计算协议的设计与优化、以及TEE的角色。在**应用实践**部分，展示了隐私计算在医疗健康、金融科技、智能推荐和政务数据共享等领域的案例。在**前沿展望**部分，分析了隐私计算与大语言模型、区块链、物联网等技术的交叉融合及其在AI治理和数据要素市场化中的战略地位。

### 14.4.2 核心洞察

回顾全书，提炼出五项核心洞察。**第一**，隐私保护是AI发展的内在需求而非外部约束——只有隐私得到充分保护，数据所有方才愿意共享数据，AI模型才能获得更丰富、更多样化的训练数据。**第二**，没有"银弹"技术，只有系统性解决方案——联邦学习、差分隐私、MPC和TEE各有适用场景和局限性，未来在于多种技术的有机融合。**第三**，标准化和互操作是产业化的关键基础设施——缺乏统一标准将导致"隐私计算孤岛"，阻碍跨组织、跨行业的数据协作。**第四**，性能-安全-易用性的三角权衡需要持续优化——硬件加速和工具链进步正在改善这一权衡，但距离"无感"的隐私计算体验仍有较长的路要走。**第五**，隐私计算将成为数据要素市场的技术基石——在数据要素被确认为新型生产要素的背景下，为数据的"可用不可见"提供技术保障。

### 14.4.3 开放问题

隐私计算领域仍有诸多开放问题值得深入研究。**隐私度量的统一化**——差分隐私、信息论度量、k-匿名性等不同框架之间缺乏统一的比较基准，建立跨技术、跨场景的隐私保护量化体系是一个根本性的理论问题。**与大语言模型的融合**——LLM的参数规模（数百亿到数万亿）对联邦学习的通信和计算效率提出了前所未有的挑战，参数高效微调（LoRA、QLoRA）与联邦学习的结合是有前景的研究方向，但如何在保持微调效果的同时确保隐私保障仍是开放问题。**对抗性鲁棒性与隐私保护的协同**——防御对抗性攻击（如模型投毒、数据投毒）通常需要更多信息共享，这与隐私保护存在内在张力，如何在二者之间取得平衡尚未充分解决。**可审计性**——在监管合规场景中，隐私计算系统需要向审计方证明其遵守了隐私保护承诺，同时不泄露原始数据，如何设计高效的可审计隐私计算协议兼具理论和实践意义。**量子安全**——量子计算的快速发展对现有密码学体系构成潜在威胁，基于格密码学（Lattice-based Cryptography）的后量子隐私计算方案正在积极研究中，但在性能和成熟度方面仍落后于经典方案。

### 14.4.4 展望

展望未来，隐私计算有望在以下方向取得突破。从"隐私计算"到"隐私智能"——未来的AI系统将在设计之初就将隐私保护作为内生特性，而非事后附加的安全措施，"隐私原生AI"（Privacy-Native AI）将成为新的范式。成为数据要素市场的技术基础设施——随着数据要素市场化改革的深入推进，隐私计算将成为数据交易所、数据信托等新型数据流通组织的技术底座，数据的"可用不可见、可控可计量"将从愿景变为产业现实。为全球数据治理提供技术支撑——在GDPR、《数据安全法》、美国各州隐私法案等不同监管框架下，实现"数据不出域、价值可流通"的跨境数据协作。隐私计算民主化——随着开源框架成熟、云服务普及和开发工具完善，中小企业和个人开发者将能够以可接受的成本使用隐私计算技术。与新兴计算范式融合——隐私计算将与边缘计算、物联网、数字孪生等深度融合，在更广泛的场景中发挥数据保护和价值释放的作用。

---

## 本章小结

本章从技术融合、标准化互操作和产业化挑战三个维度展望了隐私计算的未来发展方向。技术层面，联邦学习、差分隐私、安全多方计算和可信执行环境的深度融合正在形成更强大的隐私计算能力体系；标准化层面，国际和国内的标准化工作、开源框架和互操作协议正在为产业化的规模化发展奠定基础；产业化层面，性能优化、成本降低和易用性提升是当前最紧迫的挑战。隐私计算正在从一项前沿技术演变为数字经济的基础设施，其发展将深刻影响数据安全、人工智能和数据治理的未来格局。

---

## 参考文献（部分）

[1] Abadi M, Chu A, Goodfellow I, et al. Deep learning with differential privacy[C]//CCS 2016. ACM, 2016: 308-318.

[2] Bonawitz K, Ivanov V, Kreuter B, et al. Practical secure aggregation for privacy-preserving machine learning[C]//CCS 2017. ACM, 2017: 1175-1191.

[3] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//AISTATS 2017. PMLR, 2017: 1273-1282.

[4] Li T, Sahu A K, Talwalkar A, et al. Federated learning: Challenges, methods, and future directions[J]. IEEE Signal Processing Magazine, 2020, 37(3): 50-60.

[5] Yang Q, Liu Y, Chen T, et al. Federated machine learning: Concept and applications[J]. ACM TIST, 2019, 10(2): 1-19.

[6] Kairouz P, McMahan H B, Avent B, et al. Advances and open problems in federated learning[J]. Foundations and Trends in ML, 2021, 14(1-2): 1-210.

[7] Mironov I. Rényi differential privacy[C]//CSF 2017. IEEE, 2017: 263-275.

[8] Gentry C. Fully homomorphic encryption using ideal lattices[C]//STOC 2009. ACM, 2009: 169-178.

[9] Yao A C. Protocols for secure computations[C]//FOCS 1982. IEEE, 1982: 160-164.

[10] Shamir A. How to share a secret[J]. Communications of the ACM, 1979, 22(11): 612-613.

[11] Zhou Y, Yang Q. Federated learning and its applications[M]. Springer, 2023.

[12] 中国信息通信研究院. 隐私计算白皮书（2023年）[R]. 2023.

[13] Tramer F, Boneh D. Slalom: Fast, verifiable and private execution of neural networks in trusted hardware[C]//ICLR 2019.

[14] Geyer R C, Klein T, Nabi M. Differentially private federated learning: A client level perspective[J]. arXiv:1712.07557, 2017.

[15] Xu G, Li H, Liu S, et al. VerifyNet: Verifiable federated learning with integrity and fairness guarantees[J]. IEEE TPAMI, 2022, 44(7): 3654-3669.
