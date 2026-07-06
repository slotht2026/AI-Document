# 前言

具身智能（Embodied AI）是人工智能的下一个前沿。

如果说大语言模型让AI获得了"思考"的能力，具身智能则让AI获得了"行动"的能力。一个具身智能体不仅能理解世界，还能在物理世界中移动、操控、与人类和环境交互。

2024-2025年，具身智能迎来了爆发期。特斯拉的Optimus人形机器人开始在工厂中试运行，Figure的Figure 02展示了令人惊叹的对话式操控能力，Google DeepMind的RT系列证明了大规模预训练在机器人领域的可行性，NVIDIA的Isaac平台为机器人仿真提供了强大的基础设施。

本书系统研究了具身智能的核心技术，从感知到控制，从仿真到真实世界部署。

---

*本书所有技术内容均基于公开来源的学术论文和技术报告。*
# 具身智能：让AI走进物理世界

## 书籍定位
- 体量：约100,000字
- 风格：技术原理+工程实践+前沿研究
- 读者：机器人工程师、AI研究者、自动驾驶从业者

---

## 第一部分：基础（约15,000字）

### 第1章 具身智能概论（~5,000字）
1.1 什么是具身智能 / 1.2 从虚拟到物理 / 1.3 核心挑战 / 1.4 本书结构

### 第2章 机器人学基础（~5,000字）
2.1 运动学与动力学 / 2.2 传感器与感知 / 2.3 控制理论基础 / 2.4 仿真环境

### 第3章 视觉感知（~5,000字）
3.1 目标检测与分割 / 3.2 深度估计 / 3.3 6DoF位姿估计 / 3.4 视觉SLAM

## 第二部分：学习与控制（约20,000字）

### 第4章 模仿学习（~5,000字）
4.1 行为克隆 / 4.2 DAgger / 4.3 扩散策略 / 4.4 从人类示范学习

### 第5章 强化学习（~5,000字）
5.1 模型无关RL / 5.2 基于模型的RL / 5.3 Sim-to-Real迁移 / 5.4 安全强化学习

### 第6章 基础模型与机器人（~5,000字）
6.1 视觉语言动作模型VLA / 6.2 RT系列 / 6.3 世界模型 / 6.4 大模型作为规划器

### 第7章 抓取与操控（~5,000字）
7.1 抓取规划 / 7.2 灵巧操控 / 7.3 接触丰富任务 / 7.4 工具使用

## 第三部分：移动与导航（约15,000字）

### 第8章 自主导航（~5,000字）
8.1 地图构建 / 8.2 路径规划 / 8.3 语义导航 / 8.4 视觉导航

### 第9章 四足与人形机器人（~5,000字）
9.1 四足运动控制 / 9.2 人形机器人 / 9.3 步态学习 / 9.4 全身控制

### 第10章 自动驾驶（~5,000字）
10.1 感知系统 / 10.2 决策规划 / 10.3 端到端驾驶 / 10.4 仿真与测试

## 第四部分：前沿与展望（约15,000字）

### 第11章 多机器人协作（~5,000字）
11.1 多机通信 / 11.2 任务分配 / 11.3 协作操控 / 11.4 群体机器人

### 第12章 Sim-to-Real（~5,000字）
12.1 域随机化 / 12.2 系统辨识 / 12.3 Teacher-Student / 12.4 仿真平台

### 第13章 具身智能的未来（~5,000字）
13.1 通用机器人 / 13.2 人机协作 / 13.3 伦理与安全 / 13.4 结语

## 附录
# 第1章 具身智能概论

## 1.1 什么是具身智能

### 1.1.1 定义

具身智能是指拥有物理身体、能够与物理环境进行交互的智能系统。与纯软件的AI系统不同，具身智能体必须在真实世界的约束下工作——重力、摩擦力、碰撞、时间延迟、传感器噪声、执行器误差。

这个概念可以追溯到1950年代。控制论先驱Norbert Wiener和W. Ross Ashby认为，智能必须通过与环境的交互来体现。1990年代，MIT的Rodney Brooks提出了"包容体系结构"（Subsumption Architecture），主张"身体是智能的基础"。

### 1.1.2 与虚拟AI的区别

| 维度 | 虚拟AI（LLM等） | 具身智能（机器人） |
|------|----------------|------------------|
| 环境 | 数字世界 | 物理世界 |
| 交互 | 文本/图像 | 力/运动/触觉 |
| 反馈 | 即时 | 有延迟 |
| 约束 | 计算资源 | 物理定律 |
| 成本 | 边际成本低 | 硬件成本高 |
| 安全 | 信息安全 | 物理安全 |

### 1.1.3 核心挑战

具身智能面临的核心挑战包括：

**感知不确定性**：真实世界的传感器数据充满噪声和不确定性。光照变化、遮挡、运动模糊都会影响感知质量。

**物理交互复杂性**：抓取一个物体需要精确的力控制、接触建模和滑动检测。这比在屏幕上点击一个按钮复杂得多。

**实时性要求**：机器人必须在毫秒级时间内做出决策，否则可能导致碰撞或任务失败。

**安全性**：机器人的错误可能造成物理伤害，安全要求远高于软件系统。

**泛化能力**：在实验室训练的策略部署到真实环境时，往往因为环境差异而失败（Sim-to-Real gap）。

## 1.2 从虚拟到物理

### 1.2.1 具身智能的三个层次

**感知层**：理解周围环境——物体在哪里、是什么、状态如何。

**决策层**：规划行动——做什么、怎么做、按什么顺序。

**执行层**：精确执行——控制关节力矩、保持平衡、施加适当力度。

这三个层次必须紧密协作。感知错误会导致决策错误，决策错误会导致执行失败。

### 1.2.2 从规则到学习

传统机器人学依赖于精确的数学模型和手工设计的控制律。但真实世界的复杂性使得这种方法难以扩展。

近年来，学习方法成为具身智能的主流：
- **模仿学习**：从人类示范中学习
- **强化学习**：通过试错学习
- **基础模型**：利用预训练的大模型进行规划和理解

### 1.2.3 从专用到通用

早期的机器人系统是专用的——焊接机器人只会焊接，搬运机器人只会搬运。具身智能的目标是构建**通用机器人**——能够执行各种任务、适应各种环境的智能体。

## 1.3 核心技术栈

### 1.3.1 硬件平台

- **机械臂**：工业协作臂（UR、Franka）、灵巧手（Shadow Hand、LEAP Hand）
- **移动机器人**：轮式、四足（Unitree Go2、Boston Dynamics Spot）、人形（Optimus、Figure 02）
- **无人机**：四旋翼、固定翼
- **自动驾驶车辆**：乘用车、卡车

### 1.3.2 传感器

- **视觉**：RGB相机、深度相机（RealSense、Azure Kinect）、事件相机
- **触觉**：力/力矩传感器、触觉皮肤（GelSight、DIGIT）
- **本体感受**：关节编码器、IMU
- **距离传感**：激光雷达、超声波

### 1.3.3 软件栈

- **ROS/ROS2**：机器人操作系统
- **仿真器**：Isaac Sim、MuJoCo、PyBullet、Gazebo
- **深度学习框架**：PyTorch、JAX
- **视觉模型**：SAM、CLIP、DINO

## 1.4 本书结构

本书按照"基础→学习→应用→前沿"的逻辑组织：

**第一部分（第1-3章）** 建立基础：具身智能概论、机器人学基础、视觉感知。

**第二部分（第4-7章）** 深入学习与控制：模仿学习、强化学习、基础模型与机器人、抓取与操控。

**第三部分（第8-10章）** 探讨移动与导航：自主导航、四足与人形机器人、自动驾驶。

**第四部分（第11-13章）** 聚焦前沿与展望：多机器人协作、Sim-to-Real迁移、具身智能的未来。

## 1.5 本章小结

本章建立了具身智能的基本概念框架。具身智能是让AI走进物理世界的关键技术，它融合了机器人学、计算机视觉、机器学习和控制理论。理解具身智能的核心挑战（感知不确定性、物理交互复杂性、实时性、安全性、泛化能力）是深入学习的前提。

---

*参考文献：Brooks (1991), Dautenhahn (2007), Bisk et al. (2020)*
# 第2章 机器人学基础

> "机器人学是连接数字智能与物理世界的桥梁。" —— Oussama Khatib

具身智能的核心命题是让智能体通过物理身体与真实世界交互。要理解具身智能系统如何感知、决策和行动，首先需要掌握机器人学的基础理论。本章将系统介绍机器人运动学与动力学、传感器与感知系统、控制理论基础以及主流仿真环境，为后续章节讨论具身智能的算法与应用奠定基础。

---

## 2.1 运动学与动力学

机器人的运动学（Kinematics）研究物体运动的几何关系，不涉及力和质量；动力学（Dynamics）则研究力与运动之间的关系。二者构成了机器人运动分析与控制的理论基石。

### 2.1.1 坐标系与齐次变换

描述机器人各连杆之间的空间关系，需要建立坐标系并定义坐标变换。在三维空间中，一个刚体的位置和姿态（位姿）可以用一个 $4 \times 4$ 的齐次变换矩阵（Homogeneous Transformation Matrix）来表示：

$$
\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{p} \\ \mathbf{0}^T & 1 \end{bmatrix} \in SE(3)
$$

其中 $\mathbf{R} \in SO(3)$ 是 $3 \times 3$ 的旋转矩阵，满足 $\mathbf{R}^T \mathbf{R} = \mathbf{I}$ 且 $\det(\mathbf{R}) = 1$；$\mathbf{p} \in \mathbb{R}^3$ 是平移向量。$SE(3)$ 是特殊欧几里得群（Special Euclidean Group），表示三维空间中的刚体运动。

旋转矩阵的九个元素并非完全独立（六个正交约束），实际中常采用更紧凑的表示方式：

- **欧拉角（Euler Angles）**：用三个绕不同轴的旋转角 $(\phi, \theta, \psi)$ 描述姿态，直观但存在万向锁（Gimbal Lock）问题。
- **轴角（Axis-Angle）**：用单位向量 $\hat{\mathbf{k}}$ 和旋转角 $\theta$ 描述绕任意轴的旋转，通过罗德里格斯公式（Rodrigues' Formula）可转换为旋转矩阵：

$$
\mathbf{R} = \mathbf{I} + \sin\theta \, [\mathbf{k}]_{\times} + (1 - \cos\theta) \, [\mathbf{k}]_{\times}^2
$$

其中 $[\mathbf{k}]_{\times}$ 是向量 $\mathbf{k}$ 的反对称矩阵（Skew-symmetric Matrix）。

- **四元数（Quaternion）**：$\mathbf{q} = q_w + q_x \mathbf{i} + q_y \mathbf{j} + q_z \mathbf{k}$，满足 $\|\mathbf{q}\| = 1$，无万向锁且计算高效，是机器人学和计算机图形学中最常用的姿态表示。

### 2.1.2 正运动学（Forward Kinematics）

正运动学的任务是：给定各关节的位移（角度或位移），求末端执行器在基座标系中的位姿。对于由 $n$ 个关节串联的机器人，正运动学通过依次右乘各连杆的齐次变换矩阵得到：

$$
\mathbf{T}_0^n = \mathbf{T}_0^1 \cdot \mathbf{T}_1^2 \cdot \cdots \cdot \mathbf{T}_{n-1}^n = \prod_{i=1}^{n} \mathbf{T}_{i-1}^i(q_i)
$$

其中 $\mathbf{T}_{i-1}^i(q_i)$ 是第 $i$ 个关节的变换矩阵，$q_i$ 是关节变量。

**D-H参数法（Denavit-Hartenberg Convention）**是建立连杆坐标系的经典方法。对于每个连杆 $i$，只需四个参数即可完整描述其几何关系：

| 参数 | 符号 | 含义 |
|------|------|------|
| 连杆长度 | $a_i$ | 沿 $x_i$ 轴，从 $z_{i-1}$ 到 $z_i$ 的距离 |
| 连杆扭角 | $\alpha_i$ | 绕 $x_i$ 轴，从 $z_{i-1}$ 到 $z_i$ 的夹角 |
| 关节距离 | $d_i$ | 沿 $z_{i-1}$ 轴，从 $x_{i-1}$ 到 $x_i$ 的距离 |
| 关节角度 | $\theta_i$ | 绕 $z_{i-1}$ 轴，从 $x_{i-1}$ 到 $x_i$ 的夹角 |

标准D-H变换矩阵为：

$$
\mathbf{T}_i = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

D-H参数法的优势在于只需最少的参数（$4n$ 个）即可完整描述 $n$ 自由度机器人的运动学，但其坐标系选取存在非唯一性，实际应用中也常采用修正D-H参数（Modified DH Convention）以避免歧义。

### 2.1.3 逆运动学（Inverse Kinematics）

逆运动学是正运动学的逆问题：给定期望的末端位姿 $\mathbf{T}_0^n$，求解各关节变量 $\mathbf{q} = [q_1, q_2, \ldots, q_n]^T$。逆运动学在机器人运动规划中至关重要——用户通常指定末端目标位姿，而非各关节角度。

逆运动学的核心挑战包括：

1. **解的存在性**：当目标点超出机器人工作空间时无解。
2. **多解性**：同一末端位姿可能对应多组关节配置（如肘关节上/下翻转）。
3. **求解复杂性**：对于一般构型的六自由度机器人，封闭解（Closed-form Solution）不一定存在。

**解析法**适用于特定构型。例如，对于满足Pieper准则（三个腕关节轴线交于一点）的六自由度机器人，可以将位置和姿态解耦，分别求解前三个关节（位置）和后三个关节（姿态）。以Stanford臂为例，当 $\theta_5 \neq 0$ 时，其逆运动学解析解为：

$$
\theta_1 = \text{atan2}(p_y, p_x) - \text{atan2}(d_2, \pm\sqrt{p_x^2 + p_y^2 - d_2^2})
$$

$$
\theta_2 = \text{atan2}(c_1 p_x + s_1 p_y, p_z)
$$

其中 $s_i = \sin\theta_i$，$c_i = \cos\theta_i$。

**数值法**适用于一般情况。最常用的是基于雅可比矩阵（Jacobian Matrix）的迭代方法。速度级逆运动学的关系为：

$$
\dot{\mathbf{x}} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}
$$

其中 $\dot{\mathbf{x}}$ 是末端执行器的广义速度（线速度和角速度），$\mathbf{J}(\mathbf{q}) \in \mathbb{R}^{6 \times n}$ 是几何雅可比矩阵。求解关节速度：

$$
\dot{\mathbf{q}} = \mathbf{J}^{\dagger}(\mathbf{q}) \dot{\mathbf{x}}
$$

其中 $\mathbf{J}^{\dagger} = \mathbf{J}^T(\mathbf{J}\mathbf{J}^T)^{-1}$ 是雅可比矩阵的Moore-Penrose伪逆。当雅可比矩阵接近奇异（行列式趋近于零）时，伪逆解会不稳定，此时可采用阻尼最小二乘法（Damped Least-Squares / Levenberg-Marquardt）：

$$
\dot{\mathbf{q}} = \mathbf{J}^T(\mathbf{J}\mathbf{J}^T + \lambda^2 \mathbf{I})^{-1} \dot{\mathbf{x}}
$$

其中 $\lambda$ 是阻尼因子，用于在奇异点附近平滑解的行为。

### 2.1.4 动力学

动力学研究力（力矩）与运动之间的关系。在机器人控制中，动力学模型用于前馈控制、力控和仿真。主要方法有牛顿-欧拉法（Newton-Euler）和拉格朗日法（Lagrange）。

**拉格朗日动力学**从能量角度出发，建立系统的运动方程。定义拉格朗日函数：

$$
\mathcal{L}(\mathbf{q}, \dot{\mathbf{q}}) = K(\mathbf{q}, \dot{\mathbf{q}}) - P(\mathbf{q})
$$

其中 $K$ 是系统动能，$P$ 是系统势能。对于 $n$ 自由度机器人：

$$
K = \frac{1}{2} \dot{\mathbf{q}}^T \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}}, \quad P = P(\mathbf{q})
$$

$\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{n \times n}$ 是对称正定的惯性矩阵（Mass/Inertia Matrix）。

由拉格朗日方程：

$$
\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}_i} - \frac{\partial \mathcal{L}}{\partial q_i} = \tau_i, \quad i = 1, \ldots, n
$$

得到机器人动力学的标准形式：

$$
\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}
$$

其中：
- $\mathbf{M}(\mathbf{q})$：惯性矩阵
- $\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}$：科里奥利力和离心力项
- $\mathbf{g}(\mathbf{q}) = \frac{\partial P}{\partial \mathbf{q}}$：重力项
- $\boldsymbol{\tau}$：关节力矩向量

**牛顿-欧拉递推法**则从基座到末端依次递推计算各连杆的速度和加速度（前向递推），再从末端到基座依次计算各连杆间的力和力矩（后向递推）。该方法的计算复杂度为 $O(n)$，远优于拉格朗日法的 $O(n^4)$（展开后），是实时动力学计算的首选算法。

---

## 2.2 传感器与感知

传感器是机器人感知物理世界的"感官"。没有精确可靠的传感系统，再先进的控制算法也只是空中楼阁。本节介绍机器人系统中最关键的几类传感器。

### 2.2.1 编码器（Encoder）

编码器是测量关节角度或位移的最基本传感器，分为增量式和绝对式两种。

**增量式编码器（Incremental Encoder）**输出脉冲信号，通过计数脉冲数量来确定位移量。典型的光电增量式编码器由光源、码盘和光电检测器组成，码盘上刻有均匀分布的透光和不透光条纹。采用正交（Quadrature）输出时，通过A、B两路信号的相位关系可判断旋转方向，分辨率可提高四倍。分辨率取决于码盘的线数（Lines Per Revolution, LPR），工业编码器通常可达数千到数万线。

**绝对式编码器（Absolute Encoder）**在每个位置输出唯一的数字编码，断电后不丢失位置信息。其码盘采用格雷码（Gray Code）编码，确保相邻位置仅有一位变化，避免了读取错误。

编码器的位置测量精度直接影响运动学正解的准确性。对于采用谐波减速器的关节，编码器通常安装在电机侧（高分辨率但需考虑传动比），或安装在关节输出侧（直接测量但分辨率较低）。

### 2.2.2 惯性测量单元（IMU）

惯性测量单元（Inertial Measurement Unit）通常包含三轴加速度计和三轴陀螺仪，高端IMU还集成磁力计（九轴IMU）。IMU测量的是刚体在惯性空间中的线加速度和角速度。

加速度计测量的并非纯粹的运动加速度，而是：

$$
\mathbf{a}_{\text{measured}} = \mathbf{a}_{\text{body}} - \mathbf{g}
$$

其中 $\mathbf{g}$ 是重力加速度向量。因此，静止时加速度计测得的是 $-\mathbf{g}$，可用于确定重力方向（即姿态的一个约束）。

陀螺仪测量角速度 $\boldsymbol{\omega}$，通过积分可获得姿态变化，但会随时间积累漂移。因此，实际系统中常采用互补滤波器（Complementary Filter）或扩展卡尔曼滤波器（Extended Kalman Filter, EKF）融合加速度计和陀螺仪数据：

$$
\hat{\mathbf{q}}_k = f(\hat{\mathbf{q}}_{k-1}, \boldsymbol{\omega}_k, \mathbf{a}_k)
$$

Madgwick滤波器和Mahony滤波器是两种广泛使用的实时姿态估计算法，在消费级无人机和机器人中应用普遍。

### 2.2.3 力/力矩传感器

力传感器测量机器人与环境之间的接触力，在装配、抓取、柔顺控制等任务中不可或缺。六维力/力矩传感器（6-axis Force/Torque Sensor）可同时测量三个方向的力 $(F_x, F_y, F_z)$ 和三个方向的力矩 $(M_x, M_y, M_z)$。

其核心是弹性体（Elastic Body）结构上粘贴的应变片（Strain Gauge）。当外力作用于弹性体时，应变片的电阻发生变化，通过惠斯通电桥（Wheatstone Bridge）电路转换为电压信号。标定矩阵 $\mathbf{K}$ 将原始应变信号映射为力/力矩：

$$
\begin{bmatrix} F_x \\ F_y \\ F_z \\ M_x \\ M_y \\ M_z \end{bmatrix} = \mathbf{K} \begin{bmatrix} s_1 \\ s_2 \\ \vdots \\ s_n \end{bmatrix}
$$

力传感器的关键性能指标包括量程、分辨率、串扰（Cross-talk，各轴间的耦合误差）和过载保护能力。在精密装配和力控任务中，力传感器的噪声水平和采样频率（通常1kHz以上）直接影响控制性能。

### 2.2.4 触觉传感器

与力/力矩传感器测量全局接触力不同，触觉传感器（Tactile Sensor）提供接触面上的空间分布信息，类似于人类皮肤的触觉感受。近年来，触觉传感技术发展迅速，已成为具身智能研究的热点。

主要的触觉传感技术包括：

- **电阻式/电容式触觉阵列**：由压阻或电容单元组成的二维阵列，测量接触压力分布。如GelSight系列传感器，利用弹性体表面的变形通过光学成像间接测量接触几何和力分布。
- **视触觉传感器（Vision-based Tactile Sensor）**：代表性产品如GelSight和DIGIT，在弹性凝胶下方放置相机，通过追踪凝胶表面标记点的位移来重建三维接触形貌。该类传感器分辨率极高（可达微米级），但计算量大。
- **压电式传感器**：利用压电效应测量动态接触力，响应速度快但无法测量静态力。

触觉信息对于精细操作至关重要。例如，在灵巧手抓取物体时，触觉传感器可以检测滑动（通过振动信号）、识别物体纹理、估计接触刚度，从而实现自适应抓取力控制。

### 2.2.5 视觉传感器

视觉传感器（相机）是机器人获取环境信息最丰富的传感器。根据原理不同，可分为：

- **RGB相机**：获取彩色图像，用于物体识别、姿态估计。
- **深度相机**：如结构光（Kinect、RealSense）、ToF（Time-of-Flight）和双目立体视觉，提供三维点云信息。
- **事件相机（Event Camera）**：异步像素级亮度变化检测，具有极高时间分辨率（微秒级）和动态范围，适合高速运动场景。

视觉与触觉的融合是具身感知的前沿方向：视觉提供全局场景理解，触觉提供局部接触细节，二者互补可大幅提升机器人在复杂环境中的操作能力。

---

## 2.3 控制理论基础

控制是机器人执行任务的核心环节。本节介绍从经典到现代的几种关键控制方法。

### 2.3.1 PID控制

比例-积分-微分（PID）控制器是工业机器人中最广泛使用的控制算法，其控制律为：

$$
\tau(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \dot{e}(t)
$$

其中 $e(t) = q_d(t) - q(t)$ 是位置跟踪误差，$K_p$、$K_i$、$K_d$ 分别是比例、积分、微分增益。

- **比例项**：提供与误差成正比的控制力，是主要的反馈环节，但单独使用会产生稳态误差。
- **积分项**：消除稳态误差，但可能引起超调和积分饱和（Integral Windup）。
- **微分项**：提供阻尼作用，抑制振荡，但对测量噪声敏感。

PID控制器各增益的整定（Tuning）是工程实践中的关键问题。经典方法包括Ziegler-Nichols法（基于临界振荡）和Cohen-Coon法。在机器人应用中，各关节通常独立使用PID控制器（分散控制），但这种方式忽略了关节间的耦合效应。

离散PID的实现需特别注意微分项的计算。实际中常采用后向差分近似：

$$
\dot{e}_k \approx \frac{e_k - e_{k-1}}{T_s}
$$

并加入低通滤波器抑制高频噪声。积分抗饱和（Anti-windup）策略也必不可少——当控制量饱和时停止积分累积，防止误差累积导致的大超调。

### 2.3.2 基于模型的控制

纯PID控制不利用机器人的动力学模型，因此在高速运动或重载条件下性能有限。基于模型的控制方法利用已知（或估计）的动力学模型来提升控制精度。

**计算力矩法（Computed Torque Control）**是最经典的基于模型的控制方法。将动力学方程代入控制律：

$$
\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\mathbf{a} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})
$$

其中 $\mathbf{a} = \ddot{\mathbf{q}}_d + \mathbf{K}_d \dot{\mathbf{e}} + \mathbf{K}_p \mathbf{e}$ 是期望的关节加速度。代入动力学方程后，闭环系统变为：

$$
\ddot{\mathbf{e}} + \mathbf{K}_d \dot{\mathbf{e}} + \mathbf{K}_p \mathbf{e} = \mathbf{0}
$$

通过选择 $\mathbf{K}_p = \text{diag}(\omega_{n_1}^2, \ldots, \omega_{n_n}^2)$ 和 $\mathbf{K}_d = \text{diag}(2\zeta_1\omega_{n_1}, \ldots, 2\zeta_n\omega_{n_n})$，可以精确指定每个关节的闭环动态特性。当模型精确时，计算力矩法可实现完美的轨迹跟踪。然而，模型不确定性（摩擦、未建模柔性等）会降低实际性能。

### 2.3.3 阻抗控制与力控

当机器人需要与环境交互时（如接触、抓取），纯位置控制可能导致过大的接触力甚至设备损坏。阻抗控制（Impedance Control）和力控（Force Control）是解决此类问题的核心方法。

**阻抗控制**的核心思想是调节机器人末端表现出的动态特性（质量-弹簧-阻尼系统），而非直接控制位置或力。期望的阻抗行为为：

$$
\mathbf{M}_d \ddot{\mathbf{e}} + \mathbf{B}_d \dot{\mathbf{e}} + \mathbf{K}_d \mathbf{e} = \mathbf{F}_{\text{ext}}
$$

其中 $\mathbf{M}_d$、$\mathbf{B}_d$、$\mathbf{K}_d$ 分别是期望的惯性、阻尼和刚度矩阵，$\mathbf{F}_{\text{ext}}$ 是外部接触力。通过调节这三个矩阵，可以控制机器人在接触时的"软硬"程度。

Hogan（1985）在其开创性论文中指出，阻抗控制统一了位置控制和力控制：在自由空间中（$\mathbf{F}_{\text{ext}} = \mathbf{0}$），阻抗控制表现为位置控制；在约束表面上，表现为力控制。这种统一框架在机器人与不确定环境交互时尤为重要。

**混合力/位控制（Hybrid Force/Position Control）**由Raibert和Craig（1981）提出，根据任务约束将控制空间分解为位置控制方向和力控制方向。在约束坐标系下，选择矩阵 $\mathbf{S}$（对角矩阵，对角元素为0或1）用于切换：

$$
\boldsymbol{\tau} = \mathbf{S}\boldsymbol{\tau}_{\text{position}} + (\mathbf{I} - \mathbf{S})\boldsymbol{\tau}_{\text{force}}
$$

例如，当机器人沿桌面推动物体时，垂直于桌面方向控制力（保持接触），平行于桌面方向控制位置（跟踪轨迹）。

### 2.3.4 模型预测控制（MPC）

模型预测控制（Model Predictive Control, MPC）是一种先进的控制策略，近年来在机器人领域获得了广泛关注。MPC的核心思想是在每个控制周期求解一个有限时域的最优控制问题。

在时刻 $k$，MPC求解以下优化问题：

$$
\min_{\mathbf{u}_k, \ldots, \mathbf{u}_{k+N-1}} \sum_{i=0}^{N-1} \left[ \|\mathbf{x}_{k+i} - \mathbf{x}_{\text{ref}}\|_{\mathbf{Q}}^2 + \|\mathbf{u}_{k+i}\|_{\mathbf{R}}^2 \right] + \|\mathbf{x}_{k+N} - \mathbf{x}_{\text{ref}}\|_{\mathbf{P}}^2
$$

$$
\text{s.t.} \quad \mathbf{x}_{k+i+1} = f(\mathbf{x}_{k+i}, \mathbf{u}_{k+i}), \quad \mathbf{u}_{\min} \leq \mathbf{u}_{k+i} \leq \mathbf{u}_{\max}, \quad \mathbf{x}_{k+i} \in \mathcal{X}
$$

其中 $N$ 是预测时域长度，$\mathbf{Q}$、$\mathbf{R}$、$\mathbf{P}$ 分别是状态、控制输入和终端代价权重矩阵，$\mathcal{X}$ 是状态约束集。

MPC的优势在于：

1. **约束处理**：显式处理状态和输入约束（如关节限位、力矩饱和）。
2. **多目标优化**：通过代价函数权重平衡跟踪精度、能耗、安全性等多目标。
3. **前馈性**：利用未来参考轨迹信息进行前馈规划。

在机器人领域，MPC特别适用于以下场景：
- **腿式机器人运动控制**：如MIT Mini Cheetah四足机器人采用基于MPC的全身控制器，实时优化关节力矩以实现动态步态。
- **四旋翼飞行器轨迹跟踪**：MPC可在满足推力约束的前提下实现激进机动。
- **人形机器人平衡控制**：MPC结合零力矩点（ZMP）判据，实现类人步态的实时稳定控制。

MPC的主要挑战在于计算效率——每步都需求解优化问题。近年来，基于GPU加速的并行求解器（如MuJoCo MPC、OSQP）和基于神经网络的近似MPC策略显著提升了求解速度，使MPC在高频控制回路（>100Hz）中变得可行。

---

## 2.4 仿真环境

仿真在具身智能研究中扮演着关键角色。它提供了安全、低成本、可重复的测试环境，是训练强化学习策略、验证控制算法和进行大规模数据采集的基础设施。本节对比四款主流仿真器。

### 2.4.1 MuJoCo

MuJoCo（Multi-Joint dynamics with Contact）由Emo Todorov开发，后被DeepMind收购并开源（2022年）。它是目前具身智能和强化学习研究中最流行的仿真器。

**核心优势**：
- **接触动力学**：采用隐式积分和凸优化求解接触力，能高效、稳定地处理复杂的多接触场景，这是MuJoCo区别于其他仿真器的最大优势。
- **速度**：单环境模拟速度极快，对中等复杂度的机器人模型可达每秒数万步。
- **丰富的功能**：支持肌腱（Tendon）、执行器（Actuator）模型、传感器噪声模型等。
- **MJX**：基于JAX的GPU并行版本，可同时运行数千个环境，极大加速强化学习训练。

**局限**：主要面向刚体和关节体，对流体、软体等的支持有限（尽管Soft Dynamics模块在发展中）。不直接支持ROS集成。

**典型应用**：DeepMind的MuJoCo Menagerie（机器人模型库）、OpenAI的Dexterous Hand项目、Google DeepMind的RT系列具身智能模型。

### 2.4.2 NVIDIA Isaac Sim

Isaac Sim是NVIDIA基于Omniverse平台构建的机器人仿真环境，专注于高保真度渲染和大规模并行仿真。

**核心优势**：
- **照片级渲染**：基于RTX光线追踪的逼真图像生成，支持域随机化（Domain Randomization）以缩小仿真到现实的差距。
- **大规模并行**：通过Isaac Lab（原Orbit），可在单GPU上同时运行数千个环境，适用于大规模强化学习训练。
- **传感器仿真**：支持相机、LiDAR、IMU等传感器的物理级仿真。
- **端到端集成**：与NVIDIA的Isaac ROS、TensorRT等工具链深度集成，支持从仿真到部署的完整流程。

**局限**：硬件要求高（需要NVIDIA RTX GPU），学习曲线陡峭，闭源组件较多。

**典型应用**：NVIDIA的Isaac Manipulator（机械臂操作）、Isaac Sim在NVIDIA自家的具身智能研究中广泛使用。

### 2.4.3 PyBullet

PyBullet是基于Bullet物理引擎的Python仿真库，由Erwin Coumans维护。

**核心优势**：
- **轻量易用**：纯Python接口，安装简单（`pip install pybullet`），API直观。
- **免费开源**：BSD许可证，社区活跃。
- **速度适中**：对于简单场景，仿真速度可接受。支持批处理（Batch Simulation）以加速强化学习训练。
- **URDF/SDF支持**：广泛支持标准机器人描述格式，与ROS生态系统兼容。

**局限**：接触动力学的精度和稳定性不如MuJoCo；渲染质量有限；对复杂接触场景的处理可能产生抖动。

**典型应用**：OpenAI Gym的机器人环境（Fetch、ShadowHand等）、许多学术论文的基线实验。

### 2.4.4 Gazebo

Gazebo是机器人学领域历史最悠久的开源仿真器，由Open Robotics维护，与ROS深度集成。

**核心优势**：
- **ROS集成**：作为ROS生态系统的核心组件，Gazebo提供与ROS消息、导航栈、运动规划栈的无缝对接。
- **多物理引擎**：支持ODE、Bullet、Simbody、DART等多种物理引擎。
- **传感器仿真**：支持相机、LiDAR、声纳、IMU等传感器的仿真。
- **社区和生态**：拥有最大的机器人模型库和教程资源。

**局限**：仿真实时性较差（相比MuJoCo），大规模并行仿真支持不足，对强化学习工作流的支持不完善。Gazebo Classic（Gazebo 11）和新版Gazebo（Ignition/Harmonic）的迁移也带来了一些生态碎片化问题。

**典型应用**：ROS机器人开发的标准仿真平台，教育和科研中的通用仿真工具。

### 2.4.5 仿真器对比总结

| 特性 | MuJoCo | Isaac Sim | PyBullet | Gazebo |
|------|--------|-----------|----------|--------|
| **物理精度** | ★★★★★ | ★★★★ | ★★★ | ★★★ |
| **仿真速度** | ★★★★★ | ★★★★★ | ★★★ | ★★ |
| **并行能力** | ★★★★★ | ★★★★★ | ★★★ | ★★ |
| **渲染质量** | ★★★ | ★★★★★ | ★★ | ★★★ |
| **ROS集成** | ★★ | ★★★ | ★★ | ★★★★★ |
| **易用性** | ★★★★ | ★★ | ★★★★★ | ★★★ |
| **开源** | 是（Apache 2.0） | 部分 | 是（BSD） | 是（Apache 2.0） |
| **主要场景** | RL研究、具身AI | 工业仿真、大规模RL | 学术研究、原型开发 | ROS开发、教育 |

选择仿真器时需考虑具体需求：强化学习和具身智能研究优先考虑MuJoCo或Isaac Sim；与ROS集成的机器人开发首选Gazebo；快速原型验证可选PyBullet。随着具身智能的发展，仿真器之间的界限正在模糊——例如MuJoCo与MJX的结合使其具备了GPU并行能力，而Isaac Sim也在强化学习工具链上持续投入。

---

## 2.5 本章小结

本章系统介绍了机器人学的四大核心主题：

1. **运动学与动力学**：从齐次变换到D-H参数，从正/逆运动学到拉格朗日动力学，这些理论是理解机器人运动的数学基础。掌握雅可比矩阵及其在逆运动学和力映射中的作用，对后续理解灵巧操作和全身控制至关重要。

2. **传感器与感知**：编码器、IMU、力传感器和触觉传感器构成了机器人的感知闭环。在具身智能系统中，多模态传感器的融合（视觉+触觉+力觉）是实现精细操作和环境理解的关键。

3. **控制理论基础**：从经典的PID控制到基于模型的计算力矩法，从阻抗控制到MPC，每种方法都有其适用场景。在具身智能语境下，MPC因其约束处理能力和与学习方法的兼容性，正成为主流选择。

4. **仿真环境**：MuJoCo、Isaac Sim、PyBullet和Gazebo各有优势。仿真不仅是传统机器人学的验证工具，更是具身智能训练（强化学习、模仿学习）的核心基础设施。仿真到现实的迁移（Sim-to-Real Transfer）仍是该领域的核心挑战之一。

这些基础理论为后续章节讨论具身智能的学习算法、操作策略和交互范式提供了必要的技术支撑。在下一章中，我们将探讨如何将深度学习与这些机器人学基础相结合，构建能从数据中学习的智能体。

---

**参考文献**

1. Craig, J. J. (2017). *Introduction to Robotics: Mechanics and Planning* (4th ed.). Pearson.
2. Siciliano, B., et al. (2009). *Robotics: Modelling, Planning and Control*. Springer.
3. Hogan, N. (1985). Impedance Control: An Approach to Manipulation. *ASME Journal of Dynamic Systems, Measurement, and Control*, 107(1), 1-24.
4. Raibert, M. H., & Craig, J. J. (1981). Hybrid Position/Force Control of Manipulators. *ASME Journal of Dynamic Systems, Measurement, and Control*, 103(2), 126-133.
5. Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A Physics Engine for Model-based Control. *IEEE/RSJ IROS*.
6. Makoviychuk, V., et al. (2021). Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning. *NeurIPS Datasets and Benchmarks*.
7. Coumans, E., & Bai, Y. (2016-2021). PyBullet, a Python Module for Physics Simulation for Games, Robotics and Machine Learning.
8. Koenig, N., & Howard, A. (2004). Design and Use Paradigms for Gazebo, An Open-source Multi-robot Simulator. *IEEE/RSJ IROS*.
9. Featherstone, R. (2014). *Rigid Body Dynamics Algorithms*. Springer.
10. Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing.
# 第3章 视觉感知

## 3.1 引言

视觉感知是具身智能系统与物理世界交互的核心能力。人类约80%的环境信息通过视觉通道获取，这一事实在机器人领域同样成立：一个能够准确"看懂"世界的机器人，才有可能在复杂场景中自主导航、操作物体并与人协作。与传统工业机器人依赖预编程轨迹不同，具身智能要求机器人具备对开放环境的自主感知能力——面对从未见过的物体、不断变化的光照条件和动态遮挡，系统必须实时做出准确的视觉判断。

视觉感知并非单一任务，而是一个从低层像素处理到高层语义理解的层级化感知流水线。在最底层，相机传感器将光子转化为数字信号；在中间层，深度估计和特征提取将像素信息升维到三维空间；在最高层，目标检测、分割和位姿估计赋予机器人对场景的语义理解能力。与此同时，视觉SLAM贯穿整个流水线，为机器人提供在空间中的自我定位能力。

本章将系统介绍具身智能中最关键的四项视觉感知技术：目标检测与分割、深度估计、六自由度位姿估计以及视觉同步定位与建图（Visual SLAM）。每一项技术都在机器人感知体系中承担着不可替代的角色——目标检测与分割回答"场景中有什么"，深度估计回答"物体有多远"，位姿估计回答"物体在哪里、朝向如何"，而视觉SLAM则回答"机器人自身在哪里"。这四项能力的协同配合，构成了具身智能视觉感知的完整闭环。

## 3.2 目标检测与分割

### 3.2.1 问题定义与经典框架

目标检测的任务是在图像中定位并分类感兴趣的目标。给定输入图像 $I \in \mathbb{R}^{H \times W \times 3}$，检测器输出一组预测 $\{(b_i, c_i, s_i)\}_{i=1}^{N}$，其中 $b_i = (x, y, w, h)$ 为边界框坐标，$c_i$ 为类别标签，$s_i \in [0,1]$ 为置信度分数。

基于深度学习的目标检测方法可分为两阶段（two-stage）和单阶段（one-stage）两大范式。两阶段方法以 Faster R-CNN [Ren et al., 2015] 为代表，先通过区域提议网络（Region Proposal Network, RPN）生成候选区域，再对每个候选区域进行分类和回归。RPN 通过在共享特征图上滑动小网络，对每个锚点（anchor）预测 $k$ 个候选框的前景/背景分数及坐标偏移。单阶段方法则直接在特征图上进行密集预测，代表性工作包括 YOLO 系列 [Redmon et al., 2016] 和 SSD [Liu et al., 2016]。单阶段方法在速度上具有显著优势，使其成为机器人实时感知的首选方案。

### 3.2.2 YOLO系列及其在机器人中的应用

YOLO（You Only Look Once）系列是具身智能系统中最广泛采用的检测框架，其核心思想是将检测问题转化为回归问题。以 YOLOv5/YOLOv8 [Jocher et al., 2023] 为例，输入图像被划分为 $S \times S$ 的网格，每个网格单元预测 $B$ 个边界框及其对应的类别概率。训练时的损失函数通常由三部分组成：

$$\mathcal{L} = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (w_i - \hat{w}_i)^2 + (h_i - \hat{h}_i)^2 \right]$$
$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2 + \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2$$
$$+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2$$

其中 $\mathbb{1}_{ij}^{\text{obj}}$ 表示第 $i$ 个网格的第 $j$ 个预测框是否负责该目标，$\lambda_{\text{coord}}$ 和 $\lambda_{\text{noobj}}$ 为权重系数。YOLOv8 在此基础上引入了无锚框（anchor-free）机制和解耦头（decoupled head），分别用独立的分支预测分类和回归，进一步提升了检测精度。

在机器人应用场景中，YOLO 的优势在于推理速度。YOLOv8 在 NVIDIA Jetson Orin 等边缘计算平台上可达 30+ FPS 的实时推理速度，这对于需要快速响应的机器人抓取和导航任务至关重要。例如，在仓储物流机器人中，YOLO 被用于实时检测货架上的商品；在服务机器人中，YOLO 负责识别日常物品、家具和行人。

值得注意的是，YOLO在机器人场景中还面临一些独特的挑战。首先是遮挡问题：在桌面操作场景中，物体之间频繁相互遮挡，这对检测器的召回率提出了更高要求。其次是类别长尾分布：机器人需要检测的物体种类远超标准数据集的覆盖范围，开放词汇检测（Open-Vocabulary Detection）因此成为近年来的研究热点。最后是域迁移问题：在仿真环境中训练的检测器部署到真实机器人时，往往面临显著的性能下降，域随机化（Domain Randomization）和域适应（Domain Adaptation）技术被广泛用于缓解这一问题。

### 3.2.3 SAM：通用分割模型

分割任务比检测更进一步，要求对目标进行像素级的精确描述。2023年，Meta AI 发布的 Segment Anything Model（SAM）[Kirillov et al., 2023] 开创了通用分割的新范式。SAM 采用提示驱动（prompt-driven）的架构，支持三种提示方式：点提示（point prompt）、框提示（box prompt）和文本提示（text prompt）。

SAM 的架构由三个核心组件构成：一个基于 ViT（Vision Transformer）的图像编码器、一个轻量级的提示编码器，以及一个掩码解码器。图像编码器将输入图像映射到高维特征空间：

$$z_I = f_{\text{enc}}(I), \quad z_I \in \mathbb{R}^{H' \times W' \times C}$$

掩码解码器通过交叉注意力机制融合图像特征与提示信息，生成精确的分割掩码：

$$M = f_{\text{dec}}(z_I, z_P) \in [0,1]^{H \times W}$$

其中 $z_P$ 为提示编码器输出的特征向量。掩码解码器采用两层 Transformer 结构，通过自注意力和交叉注意力交替处理图像特征与提示token，最终通过动态头部（dynamic head）预测多个候选掩码及其质量分数。

SAM 的突破性在于其零样本泛化能力。在机器人领域，这意味着无需针对特定场景重新训练即可分割任意物体。结合 Grounding DINO [Liu et al., 2023] 等开放词汇检测器，可以实现"检测+分割"的全流水线零样本物体感知——给定自然语言描述（如"桌上的红色杯子"），系统即可输出该物体的精确像素掩码。这一能力对家庭服务机器人尤为重要：机器人可以理解人类的自然语言指令，并在视觉场景中精确定位目标物体。

SAM 2 [Ravi et al., 2024] 进一步将能力扩展到视频领域，引入了记忆机制来实现跨帧的目标追踪与分割。这对机器人操作场景意义重大：机器人在执行抓取任务时，需要持续追踪目标物体的位置变化，SAM 2 的时序一致性能力恰好满足了这一需求。此外，SAM 的分割掩码可以与点云数据结合，为下游的位姿估计和抓取规划提供精确的物体几何信息，形成"分割→重建→位姿估计→抓取"的完整感知链路。

### 3.2.4 DINO系列：自监督视觉表征

DINO（Self-Distillation with No Labels）[Caron et al., 2021] 及其后续工作 DINOv2 [Oquab et al., 2024] 代表了自监督视觉表征学习的前沿方向。DINO 通过自蒸馏框架训练 Vision Transformer，其核心思想是让学生网络的输出逼近教师网络的输出，而教师网络的参数通过学生网络的指数移动平均（EMA）更新：

$$\theta_t \leftarrow m \theta_t + (1-m) \theta_s$$

其中 $m$ 为动量系数，通常取 $m = 0.996$。训练过程中，教师网络和学生网络接收同一图像的不同增强视图（全局裁剪和局部裁剪），通过最小化两者输出分布之间的交叉熵来学习表征：

$$\mathcal{L} = -\sum_{x \in \{x_1^g, x_2^g\}} \sum_{i} P_t(x)^i \log P_s(x)^i$$

其中 $P_t$ 和 $P_s$ 分别为教师和学生网络经 softmax 归一化后的输出分布，$g$ 表示全局视图。

DINO 学到的特征具有出色的语义分割能力——无需任何微调，仅通过对注意力图进行聚类即可获得语义分割结果。特别值得关注的是，DINO 的自注意力图天然具有物体级别的语义信息：低层注意力关注纹理和边缘，高层注意力则聚焦于物体的整体结构。这种层级化的特征表示与机器人感知的需求高度吻合——机器人既需要低层信息来进行精确的操作（如抓取边缘），也需要高层信息来进行任务规划（如识别物体类别）。

在具身智能中，DINOv2 的预训练特征被广泛用作下游任务的视觉骨干，包括物体操控策略学习（如 RT-2 [Brohan et al., 2023] 中的视觉编码器）和场景理解。其强大的特征迁移能力使得机器人可以在有限的标注数据下快速适应新环境。此外，DINOv2 的特征也被用于视觉导航中的地点识别（Place Recognition）：通过计算当前视角与历史关键帧之间的特征相似度，机器人可以判断是否曾经到访过当前位置，这为回环检测提供了除传统词袋模型之外的另一种有效途径。

## 3.3 深度估计

### 3.3.1 问题背景

深度信息是机器人理解三维空间的基础。与二维图像不同，深度图 $D \in \mathbb{R}^{H \times W}$ 中每个像素 $D(u,v)$ 表示场景中对应点到相机的欧氏距离。获取深度信息的方式可分为被动式（单目/双目）和主动式（结构光/ToF）两大类。不同技术在精度、成本、功耗和适用场景上存在显著差异，理解这些差异对于机器人系统设计至关重要。

### 3.3.2 单目深度估计

单目深度估计从单张RGB图像推断深度，是一个高度不适定（ill-posed）问题：同一张二维图像可能对应无穷多种三维场景。近年来，基于深度学习的方法取得了显著突破。

MiDaS [Ranftl et al., 2022] 通过在多个异构数据集上联合训练，实现了对任意图像的相对深度估计。其训练目标采用尺度不变的损失函数：

$$\mathcal{L}_{\text{SI}} = \frac{1}{T} \sum_{i<j} \left( (d_i - d_j) - (d_i^* - d_j^*) - \alpha \right)^2$$

其中 $d_i = \log D_i$，$\alpha$ 为对齐偏移量，$T$ 为有效像素对数。这种尺度不变设计使得模型可以在不同数据集之间迁移——即使各数据集的深度标注尺度不一致，模型也能学到统一的深度表征。

2024年，Depth Anything v2 [Yang et al., 2024] 进一步推动了单目深度估计的精度边界。该方法利用大规模合成数据（约6200万张高质量合成图像）进行训练，并通过教师-学生框架将合成数据的知识迁移到真实场景。在具身智能应用中，单目深度估计的优势在于成本低、部署简单，仅需一个普通RGB相机即可工作。然而，其估计的深度为相对值，需要额外的标定步骤才能转换为绝对距离。

### 3.3.3 双目深度估计

双目视觉模拟人类双眼的立体视觉原理，通过两个水平放置的相机获取场景的深度信息。给定左右图像对 $(I_L, I_R)$，双目深度估计的核心是立体匹配，即为左图中的每个像素在右图中找到对应点。视差 $d$ 与深度 $Z$ 的关系为：

$$Z = \frac{f \cdot B}{d}$$

其中 $f$ 为焦距，$B$ 为基线长度（两个相机之间的距离）。由该公式可知，深度精度与视差估计精度成正比，与基线长度成反比——更大的基线可以提供更高的深度分辨率，但也会导致更严重的遮挡问题。

现代双目匹配方法如 AANet [Xu and Zhang, 2020] 和 GwcNet [Guo et al., 2019] 采用端到端的卷积神经网络，通过构建代价体（cost volume）并使用3D卷积进行正则化，直接回归视差图。代价体的构建方式为：

$$C(d, u, v) = \|f_L(u,v) - f_R(u-d, v)\|$$

其中 $f_L, f_R$ 分别为左右图像的特征。GwcNet 进一步提出了分组相关代价体（Group-wise Correlation），将特征通道分组后分别计算相关性，在保持精度的同时大幅降低计算和内存开销。双目深度估计在自动驾驶和室外机器人中应用广泛，其优势在于可获得稠密、精确的深度图，但对光照变化和无纹理区域较为敏感。

### 3.3.4 结构光与ToF深度相机

结构光深度相机（如 Intel RealSense D400 系列）通过向场景投射已知图案（如红外散斑），并分析图案的变形来计算深度。其基本原理是：红外发射器投射散斑图案到场景表面，红外接收器捕获变形后的散斑图像，通过与参考图案进行三角测量来计算每个像素的深度值。结构光相机的深度精度可达亚毫米级，但工作距离通常受限（0.2m–10m），且在强光环境下性能下降，因为环境红外光会干扰投射图案的识别。

ToF（Time-of-Flight）深度相机（如 Microsoft Azure Kinect）通过测量光脉冲的飞行时间来计算深度：

$$d = \frac{c \cdot \Delta t}{2}$$

其中 $c$ 为光速，$\Delta t$ 为光脉冲往返时间。现代ToF相机多采用连续波调制方式，通过测量发射光与接收光之间的相位差来推算距离：

$$d = \frac{c \cdot \Delta \phi}{4\pi f_{\text{mod}}}$$

其中 $\Delta \phi$ 为相位差，$f_{\text{mod}}$ 为调制频率。ToF相机的优势在于帧率高（通常30fps以上）、测量范围大（可达数十米），但深度分辨率相对较低，且容易受到多路径干扰（Multi-Path Interference）的影响——在角落和凹面区域，光可能经过多次反射才到达传感器，导致深度测量出现偏差。

在具身智能系统中，深度相机的选择取决于具体任务需求。桌面操作场景通常选用结构光相机（精度高），而室内导航场景则更多采用ToF相机（范围大）。

近年来，深度估计领域出现了两个重要趋势。一是基础模型驱动的单目深度估计：以 Depth Anything v2 和 Metric3D v2 [Hu et al., 2024] 为代表的大规模预训练模型，正在缩小甚至消除单目方法与专用深度相机之间的精度差距，使得低成本的纯视觉方案在更多场景中变得可行。二是多传感器融合：将深度相机与IMU、激光雷达等传感器融合，通过互补各自的优缺点来提升深度感知的鲁棒性和覆盖范围。例如，苹果的 LiDAR Scanner 就是将 dToF（直接飞行时间）传感器集成到移动设备中，为移动机器人和增强现实应用提供实时深度信息。此外，基于事件相机（Event Camera）的深度估计也引起了研究兴趣，其超高时间分辨率（微秒级）为高速运动场景下的深度感知提供了新思路。

## 3.4 六自由度位姿估计

### 3.4.1 问题定义

物体的六自由度（6DoF）位姿由旋转矩阵 $\mathbf{R} \in SO(3)$ 和平移向量 $\mathbf{t} \in \mathbb{R}^3$ 完整描述，表示物体在相机坐标系下的位置和朝向。位姿估计的目标是从观测数据中恢复这一变换：

$$\mathbf{p}_{\text{cam}} = \mathbf{R} \cdot \mathbf{p}_{\text{obj}} + \mathbf{t}$$

其中 $\mathbf{p}_{\text{obj}}$ 为物体坐标系下的三维点，$\mathbf{p}_{\text{cam}}$ 为对应的相机坐标系下的三维点。旋转矩阵 $\mathbf{R}$ 需满足正交约束 $\mathbf{R}^T \mathbf{R} = \mathbf{I}$ 且 $\det(\mathbf{R}) = 1$，这构成了特殊正交群 $SO(3)$。在实际优化中，旋转的参数化方式对收敛性有重要影响，常用表示包括欧拉角、轴角、四元数和旋转矩阵，其中四元数因其紧凑性和无奇异性而被广泛采用。

### 3.4.2 基于RGB图像的位姿估计

早期方法依赖于物体的CAD模型，通过特征匹配（如 LINEMODE [Hinterstoisser et al., 2012] 中的梯度模板匹配）来估计位姿。LINEMODE 的核心思想是利用物体边缘处的梯度方向建立模板，通过在离散化的位姿空间中搜索最佳匹配来估计物体朝向。这类方法对纹理缺失的物体（如工业零件）特别有效，但计算开销较大。

深度学习方法则通过端到端的方式直接回归位姿。PoseCNN [Xiang et al., 2018] 提出了一个里程碑式的框架：首先通过分割网络定位物体，然后分别回归旋转（使用四元数表示）和平移。其旋转损失采用四元数之间的测地距离：

$$\mathcal{L}_{\text{rot}} = \|\mathbf{q} - \text{sign}(\mathbf{q} \cdot \hat{\mathbf{q}}) \hat{\mathbf{q}}\|$$

其中 $\mathbf{q}$ 为预测四元数，$\hat{\mathbf{q}}$ 为真值四元数，$\text{sign}(\cdot)$ 用于处理四元数的双覆盖性质（即 $\mathbf{q}$ 和 $-\mathbf{q}$ 表示同一旋转）。

近年来，基于关键点的方法取得了更好的性能。这类方法首先预测物体表面的2D-3D对应关系，然后通过 PnP（Perspective-n-Point）算法求解位姿。给定 $n$ 个3D点 $\{\mathbf{P}_i\}$ 及其在图像中的2D投影 $\{\mathbf{p}_i\}$，PnP 问题可表述为最小化重投影误差：

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^{n} \| \mathbf{p}_i - \pi(\mathbf{R} \mathbf{P}_i + \mathbf{t}) \|^2$$

其中 $\pi(\cdot)$ 为相机投影函数。当 $n \geq 6$ 时，该问题可通过 EPnP（Efficient PnP）等算法高效求解；当 $n$ 较小时，RANSAC（Random Sample Consensus）框架常被用于剔除异常匹配点。

### 3.4.3 基于点云的位姿估计与ICP

当拥有深度信息时，可以直接在三维空间中进行位姿估计。迭代最近点（Iterative Closest Point, ICP）算法 [Besl and McKay, 1992] 是最经典的点云配准方法。给定源点云 $\mathcal{P} = \{\mathbf{p}_i\}$ 和目标点云 $\mathcal{Q} = \{\mathbf{q}_j\}$，ICP 通过交替执行以下两步来求解最优变换：

**步骤1：最近点匹配。** 对每个源点 $\mathbf{p}_i$，在目标点云中找到最近邻：
$$\mathbf{q}_{j^*} = \arg\min_{\mathbf{q}_j \in \mathcal{Q}} \|\mathbf{p}_i - \mathbf{q}_j\|^2$$

**步骤2：变换求解。** 最小化匹配点对之间的距离：
$$(\mathbf{R}^*, \mathbf{t}^*) = \arg\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \|(\mathbf{R}\mathbf{p}_i + \mathbf{t}) - \mathbf{q}_{j^*}\|^2$$

该问题可通过 SVD 分解闭式求解：首先计算匹配点对的质心 $\bar{\mathbf{p}}$ 和 $\bar{\mathbf{q}}$，然后构造交叉协方差矩阵 $\mathbf{W} = \sum_i (\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{q}_{j^*} - \bar{\mathbf{q}})^T$，对其进行 SVD 分解 $\mathbf{W} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$，最优旋转为 $\mathbf{R}^* = \mathbf{V} \mathbf{U}^T$，最优平移为 $\mathbf{t}^* = \bar{\mathbf{q}} - \mathbf{R}^* \bar{\mathbf{p}}$。

ICP 的变体包括点到面 ICP（Point-to-Plane ICP，优化点到目标点云切平面的距离，收敛速度更快）、广义 ICP（Generalized ICP, GICP，同时考虑两个点云的局部协方差结构）以及基于特征的快速全局配准（Fast Global Registration）等。ICP 的主要局限在于对初始值敏感——当两个点云的初始对齐偏差较大时，算法容易陷入局部最优。因此在实践中，通常先使用特征匹配或投票方法获得粗略对齐，再用 ICP 进行精配准。

在具身智能中，位姿估计是机器人抓取的前提条件。机器人需要精确知道物体的6DoF位姿，才能规划合适的抓取姿态。对于已知物体，通常采用基于模型的位姿估计流程——首先利用渲染-比对策略生成大量训练样本，然后训练神经网络预测2D-3D对应关系，最后通过PnP或ICP进行位姿求解。对于未知物体，则需要结合深度学习的关键点预测与 ICP 精配准。

近年来的一个重要趋势是将位姿估计与机器人策略学习相结合。例如，可以在位姿估计的基础上设计基于位姿的抓取控制器：给定估计的物体位姿 $\hat{T}_{obj} = [\hat{\mathbf{R}} | \hat{\mathbf{t}}]$，控制器计算目标抓取位姿 $T_{grasp}$ 并生成运动轨迹。然而，这种方法的鲁棒性高度依赖位姿估计的精度。因此，端到端的视觉伺服（Visual Servoing）方法也受到了广泛关注，它们直接从图像特征映射到机器人动作，绕过了显式的位姿估计步骤。两种范式各有优劣：基于位姿的方法具有更好的可解释性和可组合性，而端到端方法则在面对位姿估计困难的场景（如透明物体、高反光表面）时往往更加鲁棒。

### 3.4.4 抓取点检测

抓取点检测是位姿估计的重要下游应用。给定物体的观测数据，抓取点检测旨在找到最优的抓取位置和姿态。GraspNet-1Billion [Fang et al., 2020] 构建了大规模抓取数据集，定义抓取为一个7维向量 $\mathbf{g} = (x, y, z, \alpha, \beta, \gamma, w)$，分别表示抓取点的三维坐标、旋转角和夹爪宽度。

抓取质量的评估通常使用力闭合（Force Closure）准则。一个抓取如果能够抵抗任意方向的外力扰动，则称该抓取满足力闭合条件。形式化地，对于一组接触点 $\{\mathbf{c}_k\}$ 及其法向量 $\{\mathbf{n}_k\}$，如果摩擦锥（friction cone）内接触力的所有可能组合能够平衡任意外力和力矩，则该抓取满足力闭合。在实践中，精确计算力闭合条件的计算开销较大，因此抓取质量分数常通过神经网络近似：

$$s(\mathbf{g}) = f_{\theta}(\mathbf{P}, \mathbf{g})$$

其中 $f_{\theta}$ 为神经网络，$\mathbf{P}$ 为输入点云。网络输出一个标量分数，用于排序候选抓取。GraspNet 的基准实验表明，基于点云的抓取检测方法在已知物体上的成功率可达90%以上，但在面对杂乱场景和未知物体时，成功率仍有较大提升空间。

## 3.5 视觉SLAM

### 3.5.1 问题定义

SLAM（Simultaneous Localization and Mapping）解决的核心问题是：机器人如何在未知环境中同时估计自身轨迹并构建环境地图。这一问题的困难在于定位与建图是相互依赖的——准确的定位需要已知地图，而精确的地图又依赖于准确的定位。视觉SLAM以相机作为主要传感器，其数学表述可概括为：给定一系列图像观测 $\mathcal{Z} = \{\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_T\}$，估计相机位姿序列 $\mathcal{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T\}$ 和地图 $\mathcal{M}$ 的联合后验概率：

$$P(\mathcal{X}, \mathcal{M} | \mathcal{Z}) = \prod_t P(\mathbf{z}_t | \mathbf{x}_t, \mathcal{M}) P(\mathbf{x}_t | \mathbf{x}_{t-1})$$

其中第一项为观测模型（似然），第二项为运动模型（先验）。这一因子图结构为后续基于图优化的SLAM求解提供了理论基础。

### 3.5.2 ORB-SLAM系列

ORB-SLAM3 [Campos et al., 2021] 是目前最完整的视觉SLAM系统之一，支持单目、双目和RGB-D相机，同时融合了惯性测量单元（IMU）数据。其核心架构包含三个并行线程：

**跟踪线程（Tracking）：** 负责实时估计相机位姿。对每一帧图像提取ORB特征（Oriented FAST and Rotated BRIEF），通过特征匹配与局部地图点建立关联，然后最小化重投影误差来优化位姿：

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{i} \rho\left(\|\mathbf{z}_i - \pi(\mathbf{x}, \mathbf{p}_i)\|_{\Sigma}^2\right)$$

其中 $\rho(\cdot)$ 为鲁棒核函数（如Huber核），$\Sigma$ 为协方差矩阵，$\|\cdot\|_{\Sigma}^2 = \cdot^T \Sigma^{-1} \cdot$ 为马氏距离。鲁棒核函数的引入使得系统对误匹配（outlier）具有更强的容忍能力。

**局部建图线程（Local Mapping）：** 管理局部地图，执行关键帧插入、地图点三角化和局部光束法平差（Bundle Adjustment, BA）。局部BA优化一个滑动窗口内的关键帧和地图点：

$$\min_{\{\mathbf{x}_k\}, \{\mathbf{p}_j\}} \sum_{k \in \mathcal{K}} \sum_{j \in \mathcal{M}_k} \rho\left(\|\mathbf{z}_{kj} - \pi(\mathbf{x}_k, \mathbf{p}_j)\|_{\Sigma}^2\right)$$

新地图点的三角化需要至少两个关键帧的观测，并通过视差角检查和重投影误差检验来保证质量。

**回环检测与全局优化线程（Loop Closing）：** 通过词袋模型（DBoW2）检测回环，并执行位姿图优化以消除累积漂移。当系统检测到当前关键帧与历史关键帧存在视觉相似性时，会进行几何验证以确认回环。确认后，通过位姿图优化将累积误差分散到整个轨迹上。ORB-SLAM3 的另一创新是多地图系统（Atlas），能够在跟踪丢失时创建新地图，并在重访旧区域时合并地图。这一机制显著提升了系统在大范围环境中的鲁棒性——即使机器人经历了长时间的跟踪失败，也能在重新回到已知区域时恢复定位。

### 3.5.3 视觉惯性里程计

视觉惯性里程计（Visual-Inertial Odometry, VIO）融合相机和IMU的互补信息：相机提供丰富的环境特征但帧间匹配易受快速运动影响，IMU提供高频运动估计（通常200Hz以上）但存在漂移。两者的融合可以实现比单一传感器更鲁棒、更精确的运动估计。

VINS-Mono [Qin et al., 2018] 是最经典的VIO系统之一，采用基于优化的紧耦合方案。其状态向量包含当前帧的位姿、速度、IMU偏置以及所有滑窗内关键帧的位姿：

$$\mathcal{X} = [\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_n, \lambda_0, \lambda_1, \ldots, \lambda_m]$$

其中 $\mathbf{x}_k = [\mathbf{p}_k, \mathbf{v}_k, \mathbf{q}_k, \mathbf{b}_a, \mathbf{b}_g]$，$\mathbf{p}_k$ 为位置，$\mathbf{v}_k$ 为速度，$\mathbf{q}_k$ 为姿态四元数，$\mathbf{b}_a$ 和 $\mathbf{b}_g$ 分别为加速度计和陀螺仪的偏置，$\lambda_j$ 为地图点的逆深度。IMU预积分（Preintegration）技术是VIO的关键创新之一：它将两帧之间的高频IMU测量预先积分为一个相对运动约束，避免了每次优化时重新积分所有IMU数据的计算开销。优化问题通过高斯-牛顿法或 Levenberg-Marquardt 算法求解：

$$\min_{\mathcal{X}} \left\{ \|\mathbf{r}_p - \mathbf{H}_p \mathcal{X}\|^2 + \sum_{k \in \mathcal{B}} \|\mathbf{r}_{\mathcal{B}}(\hat{\mathbf{z}}_{k+1}^k, \mathcal{X})\|_{\mathbf{P}_k}^2 + \sum_{(l,j) \in \mathcal{C}} \|\mathbf{r}_{\mathcal{C}}(\hat{\mathbf{z}}_l^j, \mathcal{X})\|_{\mathbf{P}_l^j}^2 \right\}$$

其中 $\mathbf{r}_p$ 为先验残差（来自边缘化），$\mathbf{r}_{\mathcal{B}}$ 为IMU预积分残差，$\mathbf{r}_{\mathcal{C}}$ 为视觉重投影残差，$\mathbf{P}_k$ 和 $\mathbf{P}_l^j$ 为对应的协方差矩阵。

### 3.5.4 稠密重建

稠密重建旨在构建环境的完整三维几何模型，而非仅稀疏的特征点。KinectFusion [Newcombe et al., 2011] 开创了基于RGB-D相机的实时稠密重建，采用截断符号距离函数（Truncated Signed Distance Function, TSDF）作为三维表示：

$$V(\mathbf{x}) = \min\left(D(\mathbf{x}), \frac{d_i(\mathbf{x}) - z_i}{\mu}\right)$$

其中 $D(\mathbf{x})$ 为体素 $\mathbf{x}$ 处的当前TSDF值，$d_i$ 为第 $i$ 帧深度测量，$z_i$ 为相机到体素的距离，$\mu$ 为截断距离。TSDF值的符号表示该体素位于观测表面的前方（正值）还是后方（负值），零等值面即对应物体表面。每一帧深度图通过加权平均融合到全局TSDF体素网格中，实现增量式的三维重建。

通过光线投射（Ray Casting）可以从TSDF体素网格中提取等值面，得到三角网格模型。近年来，神经辐射场（Neural Radiance Fields, NeRF）[Mildenhall et al., 2020] 和三维高斯溅射（3D Gaussian Splatting）[Kerbl et al., 2023] 等神经隐式表示方法也逐渐应用于机器人场景的稠密重建。

NeRF 通过多层感知机（MLP）隐式地编码场景的几何和外观信息。给定一个三维点 $\mathbf{x} = (x, y, z)$ 和视角方向 $\mathbf{d} = (\theta, \phi)$，NeRF 网络输出该点的颜色和密度：

$$(\mathbf{c}, \sigma) = f_{\Theta}(\gamma(\mathbf{x}), \gamma(\mathbf{d}))$$

其中 $\gamma(\cdot)$ 为位置编码函数，用于增强网络对高频细节的拟合能力。通过体渲染方程，可以从任意视角合成图像。3D Gaussian Splatting 则采用了一种更高效的显式表示：用一组各向异性的三维高斯椭球来表示场景，通过可微光栅化实现快速渲染，其训练和渲染速度比 NeRF 快一到两个数量级。

在机器人应用中，稠密重建的价值不仅在于环境建模，更在于支撑下游任务：精确的三维模型可以用于碰撞检测、运动规划和仿真训练。然而，纯视觉的稠密重建在纹理缺失、光照剧烈变化和大范围场景中仍面临挑战，多传感器融合和增量式重建是当前的主要研究方向。

## 3.6 本章小结

视觉感知是具身智能系统的"眼睛"，其性能直接决定了机器人在物理世界中的行动能力。本章介绍了四项核心技术：目标检测与分割使机器人能够识别和定位环境中的物体，YOLO的实时性、SAM的通用性和DINO的表征能力为不同应用需求提供了灵活的选择；深度估计将二维图像升维到三维空间，从单目到结构光，不同技术路线各有优劣；六自由度位姿估计和抓取点检测是机器人操控的基础，ICP等经典方法与深度学习方法的结合正在推动该领域的快速发展；视觉SLAM解决了机器人"我在哪"的根本问题，ORB-SLAM3的完整框架和VIO的紧耦合优化为机器人导航提供了可靠的基础。

这些技术并非孤立存在，而是相互协同的。一个典型的具身智能感知流程可能是：视觉SLAM提供定位和地图，深度估计补充三维信息，目标检测与分割识别感兴趣物体，位姿估计确定物体精确位姿，最终输出给决策和控制模块。随着基础模型（Foundation Models）的发展，多模态大模型正在尝试统一视觉感知的多个子任务，预示着具身智能视觉感知的下一个范式变革。

---

**参考文献**

[1] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR.

[2] Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment Anything. ICCV.

[3] Liu, S., Zeng, Z., Ren, T., et al. (2023). Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. ECCV.

[4] Caron, M., Touvron, H., Misra, I., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. ICCV.

[5] Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. TMLR.

[6] Ranftl, R., Bochkovskiy, A., & Koltun, V. (2022). Vision Transformers for Dense Prediction. TPAMI.

[7] Yang, L., Kang, B., Huang, Z., et al. (2024). Depth Anything V2. NeurIPS.

[8] Besl, P. J., & McKay, N. D. (1992). A Method for Registration of 3-D Shapes. TPAMI.

[9] Xiang, Y., Schmidt, T., Narayanan, V., & Fox, D. (2018). PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes. RSS.

[10] Fang, H.-S., Wang, C., Fang, H., et al. (2020). GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping. CVPR.

[11] Campos, C., Elvira, R., Rodríguez, J. J. G., Montiel, J. M. M., & Tardós, J. D. (2021). ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual–Inertial, and Multimap SLAM. TRO.

[12] Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator. TRO.

[13] Newcombe, R. A., Izadi, S., Hilliges, O., et al. (2011). KinectFusion: Real-Time Dense Surface Mapping and Tracking. ISMAR.

[14] Mildenhall, B., Srinivasan, P. P., Tancik, M., et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. ECCV.

[15] Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM TOG.

[16] Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS.

[17] Brohan, A., Brown, N., Carbajal, J., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. CoRL.

[18] Liu, W., Anguelov, D., Erhan, D., et al. (2016). SSD: Single Shot MultiBox Detector. ECCV.

[19] Hinterstoisser, S., Lepetit, V., Ilic, S., et al. (2012). Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes. ACCV.

[20] Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO. GitHub Repository.

[21] Ravi, N., Gabeur, V., Hu, Y.-T., et al. (2024). SAM 2: Segment Anything in Images and Videos. arXiv preprint.

[22] Hu, Z., Wang, T., et al. (2024). Metric3D v2: A Versatile Monocular Geometric Foundation Model. CVPR.

[23] Guo, X., Yang, K., Yang, W., Wang, X., & Li, H. (2019). Group-Wise Correlation Stereo Network. CVPR.

[24] Xu, H. & Zhang, J. (2020). AANet: Adaptive Aggregation Network for Efficient Stereo Matching. CVPR.
# 第4章 模仿学习

> "The fastest way to teach a robot is to show it what to do."

模仿学习（Imitation Learning）是具身智能中最直观、最古老的学习范式之一。其核心思想极为朴素：通过观察专家的行为示范，让智能体学会执行相同的任务。与强化学习需要精心设计奖励函数并通过反复试错来优化策略不同，模仿学习直接从高质量的行为数据中提取策略映射，将学习问题转化为监督学习或分布匹配问题。这种"看一遍就会"的范式，与人类婴儿通过观察父母动作来习得技能的过程高度相似，使其成为连接人类知识与机器人执行的天然桥梁。

在具身智能领域，模仿学习具有特殊的重要性。物理世界的任务往往难以用简单的奖励函数来刻画——如何定义"把杯子放回桌上"的奖励？杯子的姿态、放置的平稳性、与周围物体的距离，这些因素交织在一起，使得奖励工程变得极其困难。而人类示范天然地包含了这些复杂的任务语义，使得模仿学习成为机器人操作（manipulation）领域最受欢迎的学习范式之一。

本章将系统介绍模仿学习的核心方法。首先讨论最基础的行为克隆（Behavior Cloning）及其面临的分布偏移问题，然后介绍近年来突破性的扩散策略（Diffusion Policy）方法，接着探讨从人类示范中学习的多种途径，最后分析模仿学习的固有局限及改进方向。

## 4.1 行为克隆：最朴素的模仿

### 4.1.1 监督学习框架

行为克隆（Behavioral Cloning, BC）是模仿学习中最简单、最直接的形式。其核心思想是将策略学习问题转化为标准的监督回归（或分类）问题。给定专家示范数据集：

$$\mathcal{D} = \{(\mathbf{s}_i, \mathbf{a}_i)\}_{i=1}^{N}$$

其中 $\mathbf{s}_i \in \mathcal{S}$ 表示状态（可以是机器人的关节角度、末端执行器位姿，或相机观测图像），$\mathbf{a}_i \in \mathcal{A}$ 表示专家在该状态下执行的动作。行为克隆的目标是学习一个策略 $\pi_\theta: \mathcal{S} \rightarrow \mathcal{A}$，使得：

$$\theta^* = \arg\min_\theta \mathbb{E}_{(\mathbf{s}, \mathbf{a}) \sim \mathcal{D}} \left[ \mathcal{L}(\pi_\theta(\mathbf{s}), \mathbf{a}) \right]$$

其中 $\mathcal{L}$ 是损失函数。对于连续动作空间（如机器人关节速度），通常采用均方误差（MSE）损失：

$$\mathcal{L}_{\text{MSE}} = \|\pi_\theta(\mathbf{s}) - \mathbf{a}\|_2^2$$

对于离散动作空间，可使用交叉熵损失。在现代具身智能系统中，策略网络通常采用深度神经网络实现，输入为高维观测（如RGB图像），输出为低维动作向量。常见的网络架构包括卷积神经网络（CNN）用于处理视觉输入，以及近年来广泛使用的Transformer架构。

行为克隆的优势在于其简洁性：它将复杂的决策学习问题降维为标准的函数拟合问题，可以充分利用深度学习领域成熟的优化工具。然而，这种简洁性的背后隐藏着一个根本性的理论缺陷——分布偏移。

### 4.1.2 分布偏移与复合误差

行为克隆面临的核心问题是**分布偏移**（Distribution Shift），也称为**协变量偏移**（Covariate Shift）。这一问题最早由Pomerleau（1989）在ALVINN自动驾驶系统中观察到，并由Ross和Bagnell（2010）从理论上给出了严格分析。

问题的本质在于：训练数据来自专家策略 $\pi^*$ 的状态访问分布 $d_{\pi^*}(\mathbf{s})$，而部署时策略 $\pi_\theta$ 会访问一个不同的状态分布 $d_{\pi_\theta}(\mathbf{s})$。即使策略在每个状态下的误差率 $\epsilon$ 很小，由于误差会在时间步上累积，最终的总误差可以被放大到不可接受的程度。

更形式化地，设单步误差的上界为 $\epsilon$，即对任意状态 $\mathbf{s}$：

$$\mathbb{P}[\pi_\theta(\mathbf{s}) \neq \pi^*(\mathbf{s})] \leq \epsilon$$

对于一个长度为 $T$ 的轨迹，Ross和Bagnell（2010）证明了累积误差的上界为：

$$J(\pi_\theta) - J(\pi^*) \leq O(\epsilon T^2)$$

其中 $J(\cdot)$ 表示策略的性能度量。这个 $O(T^2)$ 的误差上界意味着：对于长时间序列的任务，即使每一步的预测精度都很高，总体表现也会急剧恶化。这就是所谓的**复合误差**（Compounding Errors）问题。

直觉上理解这个现象：专家示范的轨迹只覆盖了状态空间中的一条狭窄路径。当学习策略在某个时间步产生微小偏差时，智能体会进入训练数据未曾覆盖的状态区域。在这个"分布外"（out-of-distribution）的状态下，策略的预测可能严重失准，导致更大的偏差，进而进入更加陌生的状态区域——形成"误差雪崩"的恶性循环。

这一理论结果深刻地揭示了行为克隆的局限性：它本质上是一个开环（open-loop）的学习过程，没有对自身错误的纠正机制。在短时间序列任务中（如抓取一个物体），行为克隆通常表现良好；但对于需要多步决策的长期任务（如烹饪一道菜），复合误差将成为致命瓶颈。

### 4.1.3 DAgger：迭代式的改进

为了克服分布偏移问题，Ross等（2011）提出了**DAgger**（Dataset Aggregation）算法。DAgger的核心思想是：不再仅从专家示范中学习，而是让学习策略与环境交互，然后请专家对交互过程中遇到的状态标注正确的动作，从而将训练数据的分布逐步对齐到学习策略的实际访问分布。

DAgger的算法流程如下：

**算法 1：DAgger（Dataset Aggregation）**

1. **初始化**：使用专家示范数据集 $\mathcal{D}_1 = \{(\mathbf{s}_i, \mathbf{a}_i)\}$ 训练初始策略 $\pi_1$
2. **迭代**：对于 $i = 1, 2, \ldots, N$：
   - 使用当前策略 $\pi_i$ 与环境交互，收集轨迹 $\tau_i = \{(\mathbf{s}_1, \mathbf{a}_1), \ldots, (\mathbf{s}_T, \mathbf{a}_T)\}$
   - 请专家对轨迹中的每个状态标注动作：$\hat{\mathbf{a}}_t = \pi^*(\mathbf{s}_t)$
   - 聚合数据集：$\mathcal{D}_{i+1} = \mathcal{D}_i \cup \{(\mathbf{s}_t, \hat{\mathbf{a}}_t)\}$
   - 在聚合数据集 $\mathcal{D}_{i+1}$ 上重新训练策略 $\pi_{i+1}$
3. **返回**：最终策略 $\pi_N$

DAgger的关键创新在于它打破了行为克隆的开环限制。通过让学习策略主动探索并获取专家在新状态下的标注，DAgger有效地将训练分布从 $d_{\pi^*}$ 逐步转移到 $d_{\pi_\theta}$。从理论角度看，DAgger可以将累积误差的上界从 $O(\epsilon T^2)$ 降低到 $O(\epsilon T)$，这是一个数量级的改进。

然而，DAgger也面临实际应用中的挑战。最核心的问题是它需要专家在训练过程中持续在线标注——这在机器人领域意味着需要人类操作员反复进行遥操作，成本极高。为缓解这一问题，研究者提出了多种变体：**Safe DAgger**（Menda et al., 2019）通过安全约束减少对专家标注的依赖；**BCO**（Torabi et al., 2018）使用逆动力学模型替代专家标注；**HEAL**（Bui et al., 2023）则利用人类注意力信号来引导标注过程。

## 4.2 扩散策略：生成式模仿学习的突破

### 4.2.1 从回归到生成：范式转变

传统行为克隆将策略学习视为一个回归问题——给定状态，预测一个确定性的动作。然而，机器人操作任务中普遍存在**多模态动作分布**（Multi-modal Action Distribution）问题。以"将杯子放到桌上"为例，杯子可以从左侧、右侧或上方接近，这些不同的操作方式都是合理的专家行为，但在动作空间中形成多个分离的模式。传统的回归模型（如MSE损失训练的神经网络）会倾向于预测这些模式的平均值，产生"模式混合"（mode averaging）现象——一个既不是从左边来、也不是从右边来的无意义中间动作。

为解决这一问题，研究者开始探索**生成式模型**作为策略表示。扩散策略（Diffusion Policy）正是在这一背景下，由Chi等（2023）在RSS会议上提出的里程碑式工作。它将去噪扩散概率模型（Denoising Diffusion Probabilistic Model, DDPM）引入机器人策略学习，在12个操作任务上取得了一致优于现有方法的表现，标志着模仿学习领域的一个重要转折点。

### 4.2.2 扩散策略的数学框架

扩散策略的核心思想是将动作生成过程建模为一个条件去噪过程。给定当前观测 $\mathbf{o}_t$（可以包含当前和历史的图像与本体感知信息），扩散策略从一个随机噪声向量出发，通过逐步去噪生成动作序列。

**前向扩散过程**：与标准DDPM一致，前向过程向真实动作 $\mathbf{a}_0$ 逐步添加高斯噪声：

$$q(\mathbf{a}_k | \mathbf{a}_0) = \mathcal{N}(\mathbf{a}_k; \sqrt{\bar{\alpha}_k}\mathbf{a}_0, (1 - \bar{\alpha}_k)\mathbf{I})$$

其中 $\bar{\alpha}_k = \prod_{i=1}^{k}(1 - \beta_i)$ 是噪声调度参数，$\beta_i$ 为预定义的噪声方差表。当 $k$ 足够大时，$\mathbf{a}_K$ 近似服从标准高斯分布。

**反向去噪过程**：策略网络 $\epsilon_\theta$ 学习预测每一步添加的噪声，从而实现从噪声到动作的反向生成：

$$p_\theta(\mathbf{a}_{k-1} | \mathbf{a}_k, \mathbf{o}_t) = \mathcal{N}(\mathbf{a}_{k-1}; \boldsymbol{\mu}_\theta(\mathbf{a}_k, k, \mathbf{o}_t), \sigma_k^2 \mathbf{I})$$

其中均值为：

$$\boldsymbol{\mu}_\theta(\mathbf{a}_k, k, \mathbf{o}_t) = \frac{1}{\sqrt{1 - \beta_k}} \left( \mathbf{a}_k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_k}} \epsilon_\theta(\mathbf{a}_k, k, \mathbf{o}_t) \right)$$

**训练目标**：扩散策略的训练损失简化为：

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{k, \mathbf{a}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \epsilon_\theta(\sqrt{\bar{\alpha}_k}\mathbf{a}_0 + \sqrt{1 - \bar{\alpha}_k}\boldsymbol{\epsilon}, k, \mathbf{o}_t)\|^2 \right]$$

这与标准DDPM的训练目标一致，但条件变量从类别标签变成了观测 $\mathbf{o}_t$。

**推理过程**：在部署时，从标准高斯分布采样 $\mathbf{a}_K \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，然后通过 $K$ 步去噪迭代生成动作序列：

$$\mathbf{a}_{k-1} = \boldsymbol{\mu}_\theta(\mathbf{a}_k, k, \mathbf{o}_t) + \sigma_k \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### 4.2.3 动作序列预测与观测编码

扩散策略的一个重要设计选择是预测**动作序列**（action chunk）而非单步动作。给定观测 $\mathbf{o}_t$，策略输出未来 $H_a$ 步的动作 $\mathbf{a}_{t:t+H_a}$。这种设计有两个关键优势：一是提供了时间上的"缓冲"，使得策略在推理延迟期间仍能执行合理的动作；二是隐式地建模了动作之间的时间相关性。

在观测编码方面，Chi等（2023）探索了两种架构：

1. **CNN-based编码器**：使用ResNet-18提取视觉特征，结合本体感知信息，通过全连接层输出条件向量。
2. **Transformer-based编码器**（DP3，也称为3D Diffusion Policy）：将点云或多视角图像编码为token序列，利用Transformer的注意力机制捕获空间关系。

值得注意的是，Chi等（2023）在论文中发现，扩散策略相比其他生成式建模方法（如隐变量模型、自回归模型和流匹配模型）具有显著优势。一个关键观察是：扩散策略的去噪步数 $K$ 并不需要很大——在实际应用中，$K=10$ 甚至 $K=5$ 步即可产生高质量的动作序列，使得推理效率可以接受。

### 4.2.4 扩散策略的理论优势

扩散策略之所以在模仿学习中表现优异，有几个深层原因：

**多模态建模能力**：扩散模型天然地支持多模态分布。与高斯混合模型（GMM）等传统方法不同，扩散模型不假设分布的参数形式，而是通过迭代去噪过程隐式地学习任意复杂度的动作分布。这使得它能自然地处理"同一个任务有多种合理解法"的情况。

**训练稳定性**：扩散模型的训练目标是一个简单的噪声预测损失，避免了GAN训练中的模式坍塌和训练不稳定问题，也避免了VAE中后验分布近似带来的误差。

**长程相关性建模**：通过预测动作序列，扩散策略隐式地建模了动作之间的时间依赖关系，这对于需要平滑、连贯运动的操作任务至关重要。

**对高精度操作的适应性**：操作任务通常要求毫米级甚至亚毫米级的精度。扩散模型通过多步去噪逐步精化动作预测，这种"从粗到细"的生成过程天然适合高精度控制需求。

## 4.3 从人类示范中学习

### 4.3.1 遥操作示范

遥操作（Teleoperation）是获取机器人示范数据最直接的方式。操作者通过控制接口远程操控机器人执行任务，同时记录机器人的状态-动作轨迹。常见的遥操作接口包括：

**主从式遥操作**：操作者控制一个与目标机器人运动学相似的"主"设备（master），其运动被映射到"从"机器人（slave）上。这种方式精度高，但需要专用硬件，且操作者需要经过培训。

**示教器（Teach Pendant）**：工业机器人通常配备的专用手持控制器，允许操作者直接控制机器人的各关节或末端执行器。这种方式在工业界广泛使用，但对于复杂任务，操作的认知负担较重。

**VR/AR遥操作**：近年来，虚拟现实（VR）和增强现实（AR）技术为遥操作提供了新的可能。操作者佩戴VR头显，通过手柄或手势识别来控制机器人。Chi等（2024）提出的UMI（Universal Manipulation Interface）进一步简化了这一过程——使用手持夹爪配合GoPro相机进行数据采集，无需在机器人端部署传感系统，使得"在野外教学"成为可能。

VR示范的一个重要优势是可以利用人类的自然运动能力。人类操作者在VR环境中执行任务时，其手部运动天然地包含丰富的操作先验——力的方向、速度曲线、接近策略等。HumanPlus系统（Fu et al., 2024）甚至实现了从人类全身运动到人形机器人的全身控制映射，使得人类可以直接"影子操控"（shadowing）人形机器人。

### 4.3.2 视频示范与人类视频学习

遥操作虽然高效，但需要机器人硬件和专用设备，数据采集成本仍然较高。一个自然的问题是：能否直接从人类执行任务的视频中学习操作策略？

这一方向面临的核心挑战是**形态差异**（morphology gap）：人类的手与机器人的夹爪在形态、自由度和控制方式上存在根本差异。人类视频中展现的动作不能直接映射为机器人的控制指令。

为解决这一问题，研究者提出了多种方法：

**关键点转移**：RT-2（Brohan et al., 2023）和RoboFlamingo（Li et al., 2024）等工作利用预训练视觉模型提取人类视频中的操作关键点，然后通过几何变换将其映射到机器人坐标系下。

**逆动力学模型**：在人类视频上训练逆动力学模型（Inverse Dynamics Model, IDM），预测使物体从当前状态转移到下一状态所需的动作。然后将该模型应用到机器人场景中，为机器人视频标注伪动作标签。这种方法的思想来自视频预训练领域——R3M（Nair et al., 2022）和MVP（Xiao et al., 2022）等工作利用大规模人类视频预训练视觉表征，再在机器人数据上微调策略。

**统一视频表征**：更近期的工作探索了在人类视频和机器人视频上进行联合预训练。GR-1（Wu et al., 2023）和SuSIE（Black et al., 2024）等工作利用视频生成模型在人类视频上预训练，然后迁移到机器人操作场景。这种方法的核心假设是：操作的"语义"（如"抓取"、"放置"）在人类和机器人之间是共享的，差异主要体现在"执行"层面。

### 4.3.3 示范数据的质量与多样性

从人类示范中学习的效果高度依赖于示范数据的质量和多样性。几个关键因素包括：

**示范质量**：低质量的示范（如犹豫、抖动、次优路径）会直接影响策略的最终表现。研究者提出了多种数据清洗和增强方法，如基于动力学可行性的轨迹过滤、基于最优传输的轨迹对齐等。

**示范多样性**：仅从单一策略的示范中学习，策略往往缺乏鲁棒性。多样化的示范（不同的初始条件、不同的完成路径）能显著提升策略的泛化能力。这与强化学习中"探索"的概念相呼应——模仿学习中的多样性相当于在状态空间中更广泛的覆盖。

**多模态数据融合**：现代具身智能系统通常同时利用机器人自身采集的示范和人类视频数据。这种混合数据策略能在数据效率和策略质量之间取得更好的平衡。

## 4.4 模仿学习的局限与改进方向

### 4.4.1 泛化能力的挑战

模仿学习面临的最根本挑战之一是**泛化**（Generalization）。理想的策略应该能够处理训练中未见过的物体、场景和任务变化，但行为克隆本质上是一种"记住"（memorization）而非"理解"（understanding）的学习方式。

泛化问题可以从多个维度来分析：

**物体泛化**：训练时使用特定的杯子，部署时需要操作不同形状、大小、材质的杯子。解决方向包括学习物体无关的特征表示、使用大规模多样化数据训练基础模型等。

**场景泛化**：训练在实验室环境中进行，部署在家庭或工厂环境中。背景、光照、物体摆放的变化都可能导致策略失效。视觉域适应（domain adaptation）和数据增强是常用的缓解手段。

**任务泛化**：从"把红色杯子放到桌上"泛化到"把任何杯子放到任何桌上"。这需要策略理解任务的抽象语义，而非仅仅记住具体的视觉-动作映射。大语言模型（LLM）和视觉语言模型（VLM）的引入为任务级泛化提供了新的可能——RT-2（Brohan et al., 2023）等工作将机器人的动作空间纳入VLM的词表，使得策略能利用语言模型的语义理解能力来处理新任务。

从理论角度看，泛化能力与训练数据的覆盖范围直接相关。行为克隆的泛化误差可以分解为**逼近误差**（approximation error）和**分布偏移误差**（distribution shift error）。前者可以通过增大模型容量来降低，后者则需要通过更好的数据收集策略（如DAgger）或更好的分布匹配方法来解决。

### 4.4.2 长期任务的挑战

如4.1.2节所述，行为克隆的复合误差问题在长期任务中尤为严重。对于一个需要 $T$ 步完成的任务，即使每步的成功率为 $99\%$，整体成功率也会随 $T$ 增长而指数下降：$0.99^T$。当 $T=100$ 时，整体成功率仅为约 $37\%$。

解决长期任务挑战的方法可以分为几个层次：

**层次化模仿学习**：将长期任务分解为多个子任务，每个子任务由一个低层策略（low-level policy）完成，高层策略（high-level policy）负责在子任务之间切换。这种层次化结构将长期任务的复合误差分解为多个短程子任务的误差之和，显著降低了总误差。选项框架（Options Framework）和封建网络（Feudal Networks）是实现层次化策略的经典方法。

**动作分块与预测**：如扩散策略中采用的动作序列预测，一次预测多步动作可以在一定程度上缓解单步误差的累积。但这种方法的有效性受限于预测步长和任务的时间尺度。

**检索增强生成**：将专家示范组织为一个可检索的数据库，在部署时根据当前状态检索最相似的历史状态，参考其对应的动作来指导当前决策。这种方法通过"记忆"来减少泛化的负担，特别适合结构化程度较高的任务。

### 4.4.3 组合性与长尾分布

现实世界的操作任务往往具有**组合性**（Compositionality）——复杂的任务由多个基本技能（如抓取、放置、倾倒）组合而成。例如，"做一杯咖啡"可以分解为"拿起杯子→打开咖啡机→放入咖啡粉→按下按钮→等待→取出杯子"。

传统模仿学习将整个任务视为一个不可分割的整体来学习，这导致了两个问题：一是数据效率低下——每种新的任务组合都需要从头采集示范数据；二是缺乏组合推理能力——策略无法将已学技能重新组合来完成新任务。

改进方向包括：

**技能原语学习**：从示范数据中自动发现和提取可复用的技能原语（skill primitives），然后通过技能组合来完成新任务。扩散模型中的潜空间插值能力为技能的平滑过渡提供了自然的实现途径。

**语言条件策略**：利用自然语言作为任务描述的接口，使得同一策略能通过不同的语言指令来完成不同的子任务。SayCan（Ahn et al., 2022）和VIMA（Jiang et al., 2023）等工作探索了这一方向，将语言理解与机器人执行连接起来。

**大模型驱动的规划**：将模仿学习训练的低层技能与大语言模型的高层规划能力相结合。LLM负责理解任务语义并生成子任务序列，底层的模仿学习策略负责执行每个子任务。这种分工使得系统能够处理训练中未见过的任务组合。

### 4.4.4 安全性与可解释性

在物理世界中部署模仿学习策略，安全性是不可回避的问题。与数字环境不同，机器人的错误动作可能导致物理损坏或人身伤害。行为克隆策略是一个"黑盒"——我们难以预测它在未见过的状态下会做出什么动作。

几个关键的安全考量包括：

**分布外检测**：当策略遇到训练分布之外的状态时，应该能够识别并触发安全机制（如停止执行、切换到安全控制器）。基于不确定性估计的方法（如集成模型、蒙特卡洛Dropout）可以提供分布外检测的信号。

**安全约束集成**：在策略优化过程中引入安全约束，确保生成的动作不会导致碰撞、超出关节限位等危险情况。安全强化学习（Safe RL）中的约束优化方法可以被借鉴到模仿学习框架中。

**可解释性**：理解策略为什么做出某个决策，对于调试、信任建立和安全验证都至关重要。注意力可视化、动作归因分析等技术可以提供一定程度的可解释性，但在复杂的视觉运动策略中仍然是开放问题。

## 4.5 本章小结

模仿学习是具身智能中最直观且实用的学习范式。本章从最基础的行为克隆出发，分析了其监督学习框架和分布偏移导致的复合误差问题；介绍了DAgger通过在线数据聚合来缓解分布偏移的经典方法；深入探讨了扩散策略如何通过生成式建模突破多模态动作分布的瓶颈；讨论了从人类示范中学习的多种途径及其各自的优势与局限；最后分析了模仿学习在泛化、长期任务和组合性方面的挑战及改进方向。

模仿学习的演进路径清晰地展示了一个趋势：从简单的监督回归到复杂的生成式模型，从单一的机器人数据到融合人类视频的多模态数据，从端到端的策略学习到与大语言模型协同的层次化系统。这一趋势反映了具身智能领域的核心认识：物理世界的复杂性无法被任何单一方法所穷尽，唯有通过多种学习范式的有机融合，才能构建真正智能的具身系统。

---

**参考文献**

- Pomerleau, D. A. (1989). ALVINN: An autonomous land vehicle in a neural network. *NeurIPS*.
- Ross, S., & Bagnell, D. (2010). Efficient reductions for imitation learning. *AISTATS*.
- Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *AISTATS*.
- Ho, J., Jain, H., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
- Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *RSS*.
- Chi, C., Xu, Z., Pan, C., Cousineau, E., Burchfiel, B., Feng, S., Tedrake, R., & Song, S. (2024). Universal Manipulation Interface: In-the-Wild Robot Teaching Without In-The-Wild Robots. *RSS*.
- Brohan, A., Brown, N., Carbajal, J., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *CoRL*.
- Fu, Z., Zhao, Q. Z., Wu, Q., Wetzstein, G., & Finn, C. (2024). HumanPlus: Humanoid Shadowing and Imitation from Humans. *arXiv*.
- Nair, S., Rajeswaran, A., Kumar, V., Finn, C., & Gupta, A. (2022). R3M: A Universal Visual Representation for Robot Manipulation. *CoRL*.
- Ahn, M., Brohan, A., Brown, N., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. *CoRL*.
- Menda, K., et al. (2019). SafeDAgger. *ICRA*.
- Torabi, F., Warnell, G., & Stone, P. (2018). Behavioral Cloning from Observation. *IJCAI*.
- Black, K., et al. (2024). Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation. *arXiv*.
# 第5章 强化学习

> "强化学习是连接感知与行动的桥梁——它让智能体在与物理世界的交互中，学会如何行动。"

## 5.1 引言

强化学习（Reinforcement Learning, RL）是具身智能系统实现自主决策与行为学习的核心范式。与监督学习依赖标注数据、无监督学习挖掘数据内在结构不同，强化学习让智能体通过与环境的交互——试错（trial-and-error）——来学习最优行为策略。在具身智能的语境下，"环境"不再是一个抽象的模拟器或游戏界面，而是真实的物理世界：机器人需要在重力、摩擦力、碰撞等物理约束下学会行走、抓取、操作。

强化学习的数学基础是马尔可夫决策过程（Markov Decision Process, MDP）。一个标准MDP由五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ 定义，其中 $\mathcal{S}$ 为状态空间，$\mathcal{A}$ 为动作空间，$P(s'|s,a)$ 为状态转移概率，$R(s,a)$ 为奖励函数，$\gamma \in [0,1)$ 为折扣因子。智能体的目标是找到一个策略 $\pi(a|s)$，使得累积折扣奖励的期望最大化：

$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

在具身智能场景中，强化学习面临一系列独特的挑战：**状态空间的高维连续性**（高维传感器输入）、**动作空间的连续性**（关节力矩、末端速度）、**奖励的稀疏性**（"抓到物体"这一事件在数千步中只发生一次）、**样本效率的低下**（真实机器人每次试错都有成本）、**安全性约束**（机器人不能为了探索而撞墙或伤人）。这些挑战催生了具身智能领域强化学习的三大研究方向：模型无关（Model-Free）方法的工程化改进、基于模型（Model-Based）方法的复兴，以及仿真到真实（Sim-to-Real）迁移的系统性方案。

本章将系统地介绍这三大方向，以及贯穿其中的安全强化学习问题。

## 5.2 模型无关强化学习

模型无关方法直接从与环境的交互样本中学习策略或价值函数，不显式地建模环境的动力学。尽管这类方法在样本效率上通常不如基于模型的方法，但其简洁性和稳定性使其在机器人领域得到了广泛应用。

### 5.2.1 策略梯度与Actor-Critic框架

策略梯度方法直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升优化目标函数。REINFORCE算法给出了一组无偏但高方差的梯度估计：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t\right]$$

其中 $\hat{A}_t$ 为优势函数（Advantage Function）的估计，衡量在状态 $s_t$ 下执行动作 $a_t$ 相较于平均水平的优劣。Actor-Critic架构通过引入一个独立的价值函数 $V_\phi(s)$ 来估计优势函数 $\hat{A}_t = R_t - V_\phi(s_t)$，从而在降低方差的同时保持偏差可控。

### 5.2.2 近端策略优化（PPO）

PPO（Proximal Policy Optimization）是目前机器人领域应用最广泛的策略梯度算法之一，由Schulman等人于2017年提出 [1]。PPO的核心思想是通过限制策略更新的幅度来保证训练的稳定性。其Clip目标函数定义为：

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 为新旧策略的概率比，$\epsilon$ 为裁剪参数（通常取0.1~0.2）。当 $r_t$ 偏离1过远时，裁剪机制会阻止梯度的进一步传播，从而避免策略更新过于剧烈。

PPO在机器人控制中的优势在于：**实现简单**（仅需一阶优化）、**超参数鲁棒**（$\epsilon$ 的选取相对宽容）、**易于并行化**（多环境并行采样）。Rudin等人（2022）在四足机器人上使用PPO，仅通过奖励函数的精心设计，就实现了在复杂地形上的稳定行走 [2]。其奖励函数包含速度跟踪奖励、能量消耗惩罚、姿态稳定性奖励等多个分项：

$$R(s,a) = w_1 \cdot R_{\text{track}} + w_2 \cdot R_{\text{energy}} + w_3 \cdot R_{\text{posture}} + w_4 \cdot R_{\text{alive}}$$

### 5.2.3 软演员-评论家（SAC）

SAC（Soft Actor-Critic）由Haarnoja等人于2018年提出 [3]，是目前连续控制任务中最受欢迎的off-policy算法之一。SAC在标准RL目标中引入了熵正则化项：

$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t \left(R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right)\right]$$

其中 $\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ 为策略的熵，$\alpha$ 为温度参数，控制探索与利用的权衡。最大化策略熵鼓励智能体探索更多样的行为，这对于具身系统中的多模态任务解空间尤为重要。

SAC的训练流程由五个网络组成：策略网络 $\pi_\theta$、两个Q网络 $Q_{\phi_1}, Q_{\phi_2}$（取较小值以缓解过估计）以及两个目标Q网络。策略的更新通过最小化策略与Q函数之间的KL散度实现：

$$\pi^* = \arg\min_\pi D_{KL}\left(\pi(\cdot|s) \bigg\| \frac{\exp(Q(s,\cdot)/\alpha)}{Z(s)}\right)$$

SAC在机器人灵巧操作中展现了卓越的性能。Haarnoja等人（2018）展示了SAC在仿真中的五指灵巧手翻转物体任务上的成功率显著高于PPO和DDPG [3]。其自动温度调节机制使得智能体能够在训练后期自动减少探索，聚焦于利用已学到的技能。

### 5.2.4 奖励设计的艺术

在具身智能中，奖励函数的设计往往比算法选择更为关键。一个设计不当的奖励函数会导致智能体学到"黑客"行为（reward hacking）——利用奖励函数的漏洞获得高分，但并未完成真正有意义的任务。

**稀疏奖励与塑形奖励。** 最朴素的奖励设计是二元稀疏奖励：任务完成时给予+1，否则为0。然而，稀疏奖励在长时域任务中会导致信用分配（credit assignment）困难。奖励塑形（Reward Shaping）通过引入中间奖励信号来缓解这一问题。Ng等人（1999）证明，如果塑形奖励为势函数之差 $F(s,s') = \gamma \Phi(s') - \Phi(s)$，则最优策略保持不变 [4]：

$$R'(s,a,s') = R(s,a) + \gamma \Phi(s') - \Phi(s)$$

在实践中，设计师常根据领域知识设计启发式奖励。例如，机械臂抓取任务中，奖励可设计为：

$$R = \underbrace{-\|p_{\text{hand}} - p_{\text{obj}}\|_2}_{\text{接近奖励}} + \underbrace{\mathbb{1}[\text{grasp}] \cdot r_{\text{grasp}}}_{\text{抓取奖励}} + \underbrace{\mathbb{1}[\text{lift} > h] \cdot r_{\text{lift}}}_{\text{提升奖励}}$$

**逆强化学习与从示范学习。** 当手工设计奖励困难时，可以从专家示范中学习奖励函数。逆强化学习（Inverse RL）假设专家在优化某个未知的奖励函数，通过最大熵逆RL（MaxEnt IRL）等方法推断该奖励 [5]。结合行为克隆（Behavioral Cloning）与在线微调的混合范式，已成为机器人操作学习的主流流程。

### 5.2.5 样本效率的挑战

模型无关RL的致命弱点是样本效率低下。训练一个PPO策略来控制四足机器人行走，通常需要数亿步的仿真交互。即便在仿真中，这也意味着数小时到数天的计算时间；在真实机器人上，这完全不可接受。

提高样本效率的策略包括：

1. **经验回放（Experience Replay）**：off-policy方法如SAC通过回放缓冲区复用历史数据，显著提高了数据利用率。
2. **Hindsight Experience Replay（HER）**：Andrychowicz等人（2017）提出，在失败的episode中，将实际达到的状态重新标注为目标状态，从而将失败经验转化为成功经验 [6]。这在多目标稀疏奖励任务中极为有效。
3. **辅助任务与表示学习**：通过引入额外的自监督损失（如观测重建、动力学预测、对比学习），可以学习到更好的状态表示，加速下游策略的学习。
4. **课程学习（Curriculum Learning）**：从简单任务开始，逐步增加难度，引导智能体循序渐进地学习复杂行为。

## 5.3 基于模型的强化学习

基于模型的强化学习（Model-Based RL, MBRL）通过学习环境的动力学模型，将规划与学习相结合，有望从根本上解决样本效率问题。在具身智能领域，MBRL正经历着一场由深度学习驱动的复兴。

### 5.3.1 世界模型

世界模型（World Model）是MBRL的核心组件，它学习一个环境的前向动力学模型 $\hat{P}(s'|s,a)$ 和奖励模型 $\hat{R}(s,a)$，使得智能体可以在"想象"中规划和训练，而无需真实的环境交互。

Ha和Schmidhuber（2018）提出了里程碑式的工作——World Models [7]，将世界模型分解为三个组件：**视觉模型（V）** 将高维观测压缩为低维潜向量 $z_t = V(o_t)$；**记忆模型（M）** 在潜空间中预测未来状态 $z_{t+1} = M(z_t, a_t)$；**控制器（C）** 根据潜状态输出动作。整个系统可以在"梦境"中训练，然后将学到的策略迁移到真实环境。

在具身智能中，世界模型的设计面临独特挑战。物理世界的动力学具有**多尺度性**（毫秒级的碰撞响应与秒级的任务规划）、**非光滑性**（接触与碰撞导致动力学不连续）和**高维性**（视觉观测的维度远高于低维本体感受状态）。现代世界模型架构通常采用：

- **潜空间建模**：在压缩的潜空间而非原始观测空间中进行动力学预测，降低建模难度
- **概率模型**：使用概率集成（Probabilistic Ensemble）或扩散模型来捕捉动力学的不确定性
- **分层建模**：在不同时间尺度上建模不同粒度的动力学

### 5.3.2 MBPO：基于模型的策略优化

Janner等人（2019）提出的MBPO（Model-Based Policy Optimization）是连接世界模型与策略学习的经典框架 [8]。MBPO的核心洞察是：世界模型的预测误差会在多步展开中累积（compounding error），因此不应将模型展开过长。MBPO的策略如下：

1. 使用真实环境数据训练世界模型集合 $\{\hat{P}_1, \hat{P}_2, ..., \hat{P}_E\}$
2. 从当前真实状态出发，使用世界模型进行短程展开（通常$k$步，$k \in [1,25]$）
3. 将模型生成的虚拟轨迹加入回放缓冲区
4. 使用SAC等off-policy算法在混合数据上训练策略

$$\mathcal{D}_{\text{model}} = \{(s_t, a_t, \hat{r}_t, \hat{s}_{t+1}) : s_t \in \mathcal{D}_{\text{real}}, \hat{s}_{t+k} \sim \hat{P}^k(\cdot|s_t, a_{t:t+k-1})\}$$

MBPO在连续控制基准任务上实现了显著的样本效率提升，通常比纯模型无关方法少用10~100倍的真实环境样本。然而，其在高维视觉输入任务上的表现仍有待提高。

### 5.3.3 Dreamer系列

Dreamer系列工作由Hafner等人提出，是潜空间MBRL的代表作。

**Dreamer v1**（2020）[9] 使用循环状态空间模型（RSSM）在潜空间中进行世界建模。RSSM将潜状态分解为确定性部分 $h_t$ 和随机部分 $z_t$：

$$h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1}), \quad z_t \sim p_\theta(z_t|h_t)$$

智能体在潜空间的"梦境"中通过想象轨迹来更新策略，使用Actor-Critic框架进行策略优化，其中Critic评估想象轨迹中状态的价值。

**Dreamer v2**（2021）[10] 引入了离散化的潜表示（categorical latent），在Atari游戏上首次超越了人类水平的模型无关方法。

**Dreamer v3**（2023）[11] 进一步引入了symlog预测和自由比特（free bits）等技术，使得单一超参数配置可以在视觉控制基准上跨任务泛化。Dreamer v3在150多个任务上使用固定的超参数取得了强劲表现，标志着MBRL向"通用化"迈出的重要一步。

### 5.3.4 MuZero：学习规划的模型

Silver等人（2020）提出的MuZero [12] 代表了一种更激进的MBRL范式：不学习完整的环境动力学，而是学习一个**面向规划的抽象模型**。MuZero学习三个组件：

- **动态模型** $h$：给定潜状态和动作，预测下一潜状态：$s_{k+1} = h(s_k, a_k)$
- **预测模型** $f$：从潜状态预测策略和价值：$p_k, v_k = f(s_k)$
- **表示模型** $g$：将观测映射为潜状态：$s_0 = g(o_1, ..., o_t)$

关键在于，MuZero的潜状态不需要与真实环境状态一一对应；它只需要包含规划所需的足够信息。这使得MuZero可以在学到的抽象空间中进行蒙特卡洛树搜索（MCTS），从而做出高质量的决策。

在具身智能中，MuZero的思想启发了一类"隐式世界模型"的研究：不追求物理上精确的动力学建模，而是学习对决策有用的抽象表示。这在计算资源受限的嵌入式系统中具有实际意义。

## 5.4 Sim-to-Real迁移

真实机器人训练的样本成本极高，而仿真器可以以数千倍的速度生成数据。Sim-to-Real迁移的核心思路是：**在仿真中训练，在真实中部署**。然而，仿真器与真实世界之间始终存在"仿真-现实差距"（Sim-to-Real Gap），包括动力学建模的近似、传感器噪声的差异、视觉外观的不一致等。如何弥合这一差距，是具身智能强化学习最活跃的研究方向之一。

### 5.4.1 域随机化

域随机化（Domain Randomization）由Tobin等人（2017）提出 [13]，其思想朴素而有力：如果智能体在足够多样的仿真环境中都能完成任务，那么它在真实环境中大概率也能成功。

具体而言，在每次仿真episode开始时，随机采样一组环境参数 $\xi \sim p(\xi)$：

$$\xi = \{m_{\text{link}}, I_{\text{link}}, \mu_{\text{friction}}, \tau_{\text{delay}}, \sigma_{\text{noise}}, ...\}$$

这些参数包括：连杆质量 $m$、转动惯量 $I$、摩擦系数 $\mu$、执行器延迟 $\tau$、传感器噪声标准差 $\sigma$，以及视觉纹理、光照条件等。策略在随机化环境下的累积表现：

$$\pi^* = \arg\max_\pi \mathbb{E}_{\xi \sim p(\xi)}\left[J_\xi(\pi)\right]$$

域随机化在实践中已被证明极为有效。OpenAI（2019）使用大规模域随机化训练的灵巧手，成功在真实机器人上完成了魔方复原任务 [14]。然而，过度的随机化会导致保守行为，不足的随机化则无法弥合仿真差距。如何自适应地调整随机化参数的分布——即**自动域随机化（Automatic Domain Randomization, ADR）**——是一个重要的研究方向。

### 5.4.2 系统辨识

系统辨识（System Identification, SysID）采取了与域随机化互补的思路：**通过真实数据校准仿真器的参数**，使仿真尽可能接近真实。

传统的系统辨识流程包括：

1. **激励设计**：设计能够充分激发系统动力学特性的参考轨迹
2. **数据采集**：在真实机器人上执行参考轨迹，记录状态-动作序列
3. **参数优化**：调整仿真器参数，使仿真轨迹与真实轨迹的匹配误差最小：

$$\hat{\xi} = \arg\min_\xi \sum_{i=1}^{N} \sum_{t=0}^{T} \|s_t^{\text{real}} - s_t^{\text{sim}}(\xi)\|^2$$

近年来，基于梯度的可微仿真器（如Brax、DiffTaichi）使得系统辨识可以通过反向传播端到端地优化 [15]。这些仿真器将整个物理求导过程嵌入计算图中，允许通过自动微分计算参数梯度，极大地提高了辨识效率。

在实践中，系统辨识与域随机化常被组合使用：先通过SysID获取参数的先验分布，再在此基础上施加适度的随机化，形成"有引导的域随机化"（Guided Domain Randomization）。

### 5.4.3 Teacher-Student架构

Teacher-Student架构是Sim-to-Real迁移中一种优雅的蒸馏范式 [16]。其流程分为两个阶段：

**阶段一：训练Teacher。** 在仿真中，Teacher策略可以访问特权信息（privileged information）——真实机器人无法直接获取的信息，如精确的物体位姿、关节摩擦系数、环境地图等。Teacher是一个状态为 $s^{\text{priv}} = [s^{\text{obs}}, s^{\text{priv}}]$ 的策略 $\pi^{\text{Teacher}}(a|s^{\text{priv}})$，由于信息充分，它可以轻松学会高质量的行为。

**阶段二：蒸馏Student。** Student策略只能访问真实机器人可获得的观测 $s^{\text{obs}}$（如IMU读数、关节编码器、RGB图像等）。通过最小化Teacher与Student行为之间的差异来训练Student：

$$\pi^{\text{Student}} = \arg\min_\pi \mathbb{E}_{s \sim \mathcal{D}}\left[D_{\text{KL}}\left(\pi^{\text{Teacher}}(\cdot|s^{\text{priv}}) \bigg\| \pi(\cdot|s^{\text{obs}})\right)\right]$$

这一架构的优势在于：**Teacher利用特权信息高效学习**，**Student学会从有限观测中恢复所需信息**。Miki等人（2022）使用这一架构实现了四足机器人在户外复杂地形上的鲁棒行走 [17]。Teacher在仿真中可以看到地形高度图，而Student仅通过本体感受（IMU、关节状态）和前置深度相机来推断地形信息。

另一种流行的变体是**Adaptive Teacher**，其中Teacher的特权信息以"特权编码器"的形式显式设计，Student则通过一个可学习的"估计网络"来逼近该编码器的输出。这种结构化的设计使得知识迁移更加高效。

## 5.5 安全强化学习

在具身智能中，安全性不是一个可选项，而是刚需。一个在仿真中通过自由探索学会行走的策略，如果在真实部署中允许机器人以最大力矩"探索"撞墙行为，后果将不堪设想。安全强化学习（Safe RL）致力于在学习过程中满足安全约束，同时最大化任务性能。

### 5.5.1 约束马尔可夫决策过程

安全RL的数学框架通常建模为约束马尔可夫决策过程（Constrained MDP, CMDP），在标准MDP基础上增加了成本函数 $C(s,a)$ 和约束阈值 $d$：

$$\max_\pi J(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

$$\text{s.t.} \quad J_C(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t C(s_t, a_t)\right] \leq d$$

其中 $J_C(\pi)$ 为期望累积成本。不同的安全约束可编码不同的物理限制，如关节力矩上限、末端执行器速度限制、障碍物距离约束等。

### 5.5.2 安全探索策略

**约束策略优化。** CPO（Constrained Policy Optimization）由Achiam等人（2017）提出 [18]，是CMDP的直接策略梯度方法。CPO在每次策略更新时保证约束的满足，其更新方向为：

$$\theta_{k+1} = \arg\max_\theta L_{\theta_k}(\theta) \quad \text{s.t.} \quad \bar{D}_{KL}(\pi_{\theta_k} \| \pi_\theta) \leq \delta, \quad J_C(\pi_\theta) \leq d$$

CPO在理论上保证了单调改进和约束满足，但其实现需要二阶优化（计算KL散度的Hessian矩阵），计算成本较高。

**拉格朗日方法。** 将CMDP转化为无约束优化问题，引入拉格朗日乘子 $\lambda$：

$$\min_{\lambda \geq 0} \max_\pi L(\pi, \lambda) = J(\pi) - \lambda(J_C(\pi) - d)$$

这种对偶方法与标准RL算法（如PPO、SAC）自然兼容，只需额外维护一个拉格朗日乘子的梯度更新。Safety Gym基准测试（Ray等人，2019）表明，拉格朗日PPO在多种安全约束任务上表现良好 [19]。

**安全层（Safety Layer）。** Dalal等人（2018）提出了安全层的概念 [20]：在策略输出和执行之间插入一个投影层，将不安全的动作投影到最近的安全动作：

$$a_{\text{safe}} = \arg\min_{a'} \|a' - a_{\text{raw}}\|^2 \quad \text{s.t.} \quad \nabla_s V_C(s)^\top f(s,a') + C(s) \leq 0$$

其中 $V_C(s)$ 为安全价值函数，$f(s,a)$ 为系统动力学。安全层的优势在于模块化——它可以与任何策略算法组合，且在推理时仅增加极小的计算开销。

### 5.5.3 物理安全约束

在具身智能中，安全约束通常具有明确的物理含义：

- **关节限位**：$q_{\min} \leq q_t \leq q_{\max}$，关节角度不能超出物理极限
- **力矩限制**：$\|\tau_t\| \leq \tau_{\max}$，执行器输出力矩不能超过额定值
- **速度限制**：$\|\dot{q}_t\| \leq \dot{q}_{\max}$，避免高速碰撞
- **碰撞避免**：$d_{\text{obs}}(s_t) \geq d_{\min}$，与障碍物保持安全距离
- **人体安全**：$F_{\text{contact}}(s_t, a_t) \leq F_{\text{safe}}$，接触力不超过安全阈值（ISO/TS 15066标准）

这些约束中，有些是状态约束（如关节限位），有些是状态-动作约束（如力矩限制），有些则涉及复杂的接触动力学（如碰撞力约束）。不同的约束需要不同的处理方式。

一种实用的做法是将硬约束编码为环境的终止条件：违反约束的episode立即终止并施加大的惩罚。然而，这本质上是一种被动安全策略——智能体可能在约束边界附近产生危险行为，只是在跨越边界时被"叫停"。更积极的方案是结合控制屏障函数（Control Barrier Function, CBF），在策略层面主动保证安全：

$$\dot{B}(s) + \alpha(B(s)) \leq 0$$

其中 $B(s) \leq 0$ 定义安全集，$\alpha$ 为扩展类 $\mathcal{K}$ 函数。CBF提供了前向不变的安全保证：一旦系统进入安全集，就不会离开 [21]。

## 5.6 本章小结

强化学习为具身智能系统提供了从交互中自主学习行为的能力。本章从四个维度梳理了这一领域：

**模型无关方法**（PPO、SAC）以工程简洁性和训练稳定性取胜，通过奖励设计、经验回放和课程学习等技巧缓解样本效率问题，是当前机器人RL实践的主力。

**基于模型方法**（世界模型、MBPO、Dreamer、MuZero）通过学习环境动力学来大幅提高样本效率，在仿真中展现出巨大潜力，但如何在真实物理环境中可靠部署仍是开放问题。

**Sim-to-Real迁移**（域随机化、系统辨识、Teacher-Student）系统性地解决了仿真与现实之间的鸿沟，使得在仿真中训练的策略能够鲁棒地迁移到真实机器人上。

**安全强化学习**（CMDP、约束策略优化、安全层、控制屏障函数）确保智能体在学习和部署过程中满足物理安全约束，是具身智能走向实际应用的关键保障。

展望未来，具身智能中的强化学习正朝着**基础模型+RL微调**的方向发展：大语言模型提供高层规划与任务理解，强化学习负责底层的运动控制与精细操作。如何将这些技术栈优雅地整合，实现从"学会一个技能"到"学会学习技能"的跨越，将是下一阶段的核心挑战。

## 参考文献

[1] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[2] Rudin N, Hoeller D, Reist P, et al. Learning to walk in minutes using massively parallel deep reinforcement learning. Conference on Robot Learning (CoRL), 2022.

[3] Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. International Conference on Machine Learning (ICML), 2018.

[4] Ng A Y, Harada D, Russell S. Policy invariance under reward transformations: Theory and application to reward shaping. International Conference on Machine Learning (ICML), 1999.

[5] Ziebart B D, Maas A, Bagnell J A, et al. Maximum entropy inverse reinforcement learning. AAAI Conference on Artificial Intelligence, 2008.

[6] Andrychowicz M, Wolski F, Ray A, et al. Hindsight experience replay. Advances in Neural Information Processing Systems (NeurIPS), 2017.

[7] Ha D, Schmidhuber J. World models. Advances in Neural Information Processing Systems (NeurIPS), 2018.

[8] Janner M, Fu J, Zhang M, et al. When to trust your model: Model-based policy optimization. Advances in Neural Information Processing Systems (NeurIPS), 2019.

[9] Hafner D, Lillicrap T, Ba J, et al. Dream to control: Learning behaviors by latent imagination. International Conference on Learning Representations (ICLR), 2020.

[10] Hafner D, Lillicrap T, Norouzi M, et al. Mastering Atari with discrete world models. International Conference on Learning Representations (ICLR), 2021.

[11] Hafner D, Pasukonis J, Ba J, et al. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

[12] Silver D, Hubert T, Schrittwieser J, et al. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature, 2020, 588(7839): 604-609.

[13] Tobin J, Fong R, Ray A, et al. Domain randomization for transferring deep neural networks from simulation to the real world. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017.

[14] OpenAI, Akkaya I, Andrychowicz M, et al. Solving Rubik's cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019.

[15] Hu Y, Anderson L, Li T M, et al. DiffTaichi: Differentiable programming for physical simulation. International Conference on Learning Representations (ICLR), 2020.

[16] Lee J, Hwangbo J, Wellhausen L, et al. Learning quadrupedal locomotion over challenging terrain. Science Robotics, 2020, 5(47).

[17] Miki T, Lee J, Hwangbo J, et al. Learning robust perceptive locomotion for quadrupedal robots in the wild. Science Robotics, 2022, 7(62).

[18] Achiam J, Held D, Tamar A, et al. Constrained policy optimization. International Conference on Machine Learning (ICML), 2017.

[19] Ray A, Achiam J, Amodei D. Benchmarking safe exploration in deep reinforcement learning. arXiv preprint arXiv:1910.01708, 2019.

[20] Dalal G, Dvijotham K, Vecerik M, et al. Safe exploration in continuous action spaces. arXiv preprint arXiv:1801.08757, 2018.

[21] Ames A D, Grizzle J W, Tabuada P. Control barrier function based quadratic programs with application to adaptive cruise control. IEEE Conference on Decision and Control (CDC), 2014.
# 第6章 基础模型与机器人

## 6.1 引言

过去十年间，深度学习在自然语言处理和计算机视觉领域取得的突破性进展，催生了一系列被称为"基础模型"（Foundation Model）的大规模预训练模型。这些模型——从GPT系列到CLIP、从PaLM到SAM——通过在海量数据上进行自监督或弱监督预训练，习得了强大的通用表征能力，随后通过微调或提示工程即可适配广泛的下游任务。

一个自然而深刻的问题随之浮现：**这些在互联网规模数据上训练出的基础模型，能否成为连接"理解世界"与"改变世界"之间的桥梁？** 具身智能（Embodied AI）的核心挑战在于，智能体不仅需要感知和理解物理环境，更需要在其中产生有意义的动作。传统机器人学依赖于精确的感知模块、手工设计的状态表示和基于模型的控制器，每一步都需要领域专家的大量工程投入。基础模型的出现，为这一体系带来了范式级的变革潜力。

本章将系统梳理基础模型与机器人学交叉融合的三条核心技术路线：**视觉-语言-动作模型（VLA）**，它追求从感知到动作的端到端统一；**世界模型（World Model）**，它试图学习物理世界的动态规律以支持想象与规划；**大语言模型作为规划器**，它利用语言模型的推理能力进行任务分解与高层决策。这三条路线并非孤立发展，而是日益交织、互相促进，共同推动着机器人智能从"专用"走向"通用"的历史性转变。

## 6.2 视觉-语言-动作模型：从感知到动作的端到端统一

### 6.2.1 VLA的基本范式

视觉-语言-动作模型（Vision-Language-Action Model，VLA）代表了具身智能领域最具雄心的技术路线：将视觉感知、语言理解和动作生成统一在一个端到端的神经网络中。其核心思想可以简洁地表述为——给定当前的视觉观测（图像或视频）和自然语言指令，模型直接输出机器人可执行的动作序列。

VLA的输入端通常包含两部分模态信息。**视觉输入**来自机器人搭载的摄像头，可以是单帧RGB图像、多视角图像或视频序列，有时还包含深度图。**语言输入**则是用户的自然语言指令，例如"把红色的杯子放到桌子上"。这两部分信息通过各自的编码器（如ViT处理视觉、Transformer处理语言）被映射到一个共享的语义空间中。

VLA的输出端是机器人动作。动作的表示方式在不同工作中有所差异：可以是连续的关节角度或末端执行器位移（通常为6-7维），也可以是离散化的动作token。关键在于，整个从"看到什么、听到什么指令"到"做什么动作"的映射过程，在一个统一的模型中完成，无需手工设计中间表示或模块化流水线。

端到端训练是VLA区别于传统机器人感知-规划-控制流水线的核心特征。通过在大规模机器人操作数据集上进行监督学习或强化学习，VLA模型能够自动发现从视觉-语言输入到动作输出的有效映射，避免了传统方法中各模块接口设计和误差传播的问题。更重要的是，当VLA利用互联网规模的视觉-语言预训练权重进行初始化时，它天然继承了对开放世界物体、场景和语义的理解能力，这是传统机器人学习方法难以企及的。

### 6.2.2 RT系列：从RT-1到RT-X的架构演进

Google DeepMind的Robotics Transformer（RT）系列工作，是VLA技术路线最具代表性的探索。这一系列从RT-1到RT-2再到RT-X，展现了一条清晰的技术演进脉络。

**RT-1（Robotics Transformer 1）** 于2022年提出，是将大规模Transformer架构成功应用于真实机器人操作的标志性工作。RT-1的架构相对直接：它以一组RGB图像和自然语言指令作为输入，通过预训练的FiLM EfficientNet提取视觉特征，经TokenLearner压缩后送入Transformer解码器，最终输出离散化的动作token。RT-1在一个包含超过13万条真实机器人演示的大规模数据集上进行训练，涵盖了700多种不同的操作任务。实验表明，RT-1在已见任务上的成功率高达97%，在未见任务上也展现出良好的泛化能力，同时具备对干扰物的鲁棒性和对语义变化的适应性。RT-1的成功证明了一个重要命题：足够大的数据集配合足够强的架构，可以让单一模型掌握广泛的操作技能。

然而，RT-1本质上是一个"机器人专用"模型——它的所有知识都来自于机器人操作数据，无法利用互联网上海量的视觉-语言知识。**RT-2（Robotics Transformer 2）** 在2023年的提出，正是为了突破这一局限。RT-2的核心创新在于，它直接将一个预训练的视觉-语言模型（VLM）——具体来说是PaLI-X或PaLM-E——微调为机器人控制器。

RT-2的关键设计决策是将**机器人动作表示为文本token**。具体而言，连续的动作空间（如末端执行器的位移和旋转）被离散化为256个区间（bins），每个区间对应一个整数，这些整数以文本字符串的形式表示。例如，一个动作向量可能被编码为"1 128 91 241 5 101 127"这样的token序列。通过这种方式，机器人动作与自然语言共享同一个token空间，VLM的自回归生成机制可以直接用于产生动作序列。

这一设计带来了令人惊喜的涌现能力。RT-2不仅在已见任务上保持了高性能，更重要的是，它展现出了对训练数据中从未出现过的物体和指令的泛化能力——例如，模型从未在训练中见过"将垃圾扔进垃圾桶"的演示，但在推理时能够理解"垃圾"和"垃圾桶"的语义并正确执行操作。这种能力显然来自于VLM在互联网数据上习得的语义知识的迁移。此外，RT-2还展现出了链式思维推理（Chain-of-Thought）能力：通过在训练数据中加入推理中间步骤，模型能够在执行动作前先进行语义推理，例如"我需要找到能用作 improvised hammer 的物体，这个石头可以用来砸钉子"。

**RT-X** 在2023年底提出，将视野从单一实验室扩展到了跨机构的大规模协作。RT-X汇集了来自22个不同机构的机器人数据，构建了一个包含超过13万条演示、涵盖22种机器人形态的跨具身数据集（Open X-Embodiment）。RT-X的核心发现是：在多个机器人的混合数据上训练的模型，比仅在单一机器人数据上训练的模型表现更好——即使对于该特定机器人也是如此。这证明了跨具身迁移学习的可行性，不同机器人形态之间的操作经验可以相互增益。

RT-X的架构在RT-1-X和RT-2-X两个变体中分别使用了不同的骨干网络，但统一了数据格式和训练流程。实验表明，RT-2-X在泛化能力上显著优于RT-2，特别是在涉及新颖物体、背景和指令的场景中。这一结果暗示了一条通往"通用机器人基础模型"的可能路径：通过汇聚全球的机器人数据，训练一个可以在多种机器人形态上工作的统一模型。

### 6.2.3 VLA的技术挑战与前沿发展

尽管RT系列取得了令人瞩目的成果，VLA技术路线仍面临着若干根本性挑战。

**数据瓶颈**是最突出的问题。与互联网上的文本和图像数据不同，高质量的机器人操作数据获取成本极高——每条演示都需要物理机器人在真实环境中执行。虽然仿真环境可以部分缓解这一问题，但仿真到真实的差距（sim-to-real gap）依然是一个活跃的研究课题。近年来，研究者们探索了多种数据增广策略，包括人类视频学习、跨机器人迁移、以及利用大模型自动生成训练场景等。

**动作表示**的设计也远未收敛。将连续动作离散化为文本token虽然优雅地统一了模态，但不可避免地引入了量化误差，且对于精细操作任务（如穿针引线）可能精度不足。一些后续工作探索了连续动作输出的方案，如使用扩散模型（Diffusion Model）作为动作解码器，以更好地建模动作分布的多模态特性。

**推理效率**同样是一个实际部署中的关键考量。大规模VLA模型（如基于PaLI-X的RT-2拥有55亿参数）的推理延迟可能达到数百毫秒，这对于需要高频控制（通常10-50Hz）的实时机器人系统是一个挑战。模型压缩、量化、以及分层控制策略（高层VLA规划+底层高频控制器执行）是当前解决这一矛盾的主要方向。

在RT系列之后，VLA领域涌现出众多新的探索。OpenVLA等开源工作致力于构建可复现的VLA基线；π0等模型探索了更高效的动作生成架构；而一些工作开始将触觉、力觉等更多模态信息纳入VLA框架，以应对接触密集型操作任务的需求。VLA正在从概念验证走向工程成熟，但距离真正通用的机器人操作模型，仍有相当的距离。

## 6.3 世界模型：学会想象物理世界

### 6.3.1 世界模型的概念与动机

如果说VLA追求的是"看到即做到"，那么世界模型追求的是"未见先预见"。世界模型的核心思想是：让智能体在内部建立一个关于物理世界如何运作的模拟器，从而能够在"想象"中推演动作的后果，而无需在真实世界中反复试错。

这一思想并非新概念。早在1995年，Jürgen Schmidhuber就提出了基于神经网络的世界模型概念。2018年，David Ha和Jürgen Schmidhuber的"World Models"论文通过在学习到的潜在空间中训练策略，展示了世界模型在游戏环境中的潜力。而在强化学习领域，基于模型的方法（Model-Based RL）——如Dreamer系列和MuZero——一直在探索如何利用学到的世界模型来提升样本效率。

基础模型时代的到来为世界模型注入了新的活力。大规模视频生成模型（如扩散模型和自回归Transformer）展现出了令人惊叹的视频生成质量，它们在海量视频数据上训练后，似乎已经隐式地学到了丰富的物理直觉——物体如何运动、光照如何变化、液体如何流动。这引发了一个大胆的问题：**视频生成模型能否作为物理世界的有效模拟器？**

### 6.3.2 视频预测与动作条件化生成

将视频生成模型用作世界模型的关键在于**可控性**——模型不仅要能生成逼真的视频，还要能根据智能体的动作生成对应的未来场景。这就是动作条件化视频预测（Action-Conditioned Video Prediction）的核心问题。

**UniSim（Universal Simulator）** 由Google于2023年提出，是这一方向的代表性工作。UniSim的核心洞察是：将交互式环境模拟问题转化为视频生成问题。给定当前帧和一个动作描述（可以是语言指令，也可以是低层动作），UniSim生成下一时刻的视觉观测。通过将各种类型的交互——人类操作、机器人动作、语言指令——统一表示为条件信号，UniSim可以模拟多种交互场景。

UniSim的训练数据来源极其多样，包括互联网视频、人类自我中心视频、机器人遥操作数据等。通过在如此广泛的数据上训练，UniSim习得了一种通用的"物理直觉"，能够在一定程度上预测动作的视觉后果。在机器人操作领域，UniSim可以用于数据增广——在学到的世界模型中生成大量虚拟轨迹，用于训练下游策略，从而减少对真实机器人数据的依赖。

**Genie** 由Google DeepMind于2024年提出，将世界模型的概念推向了可交互环境的生成。与UniSim主要关注单步预测不同，Genie能够从单张图像生成一个可交互的2D世界——用户可以通过键盘输入与生成的环境进行持续交互，观察环境的实时变化。Genie的架构包含三个核心组件：一个潜在动作模型（用于从视频中推断潜在动作），一个视频tokenizer（将视频编码为离散token），以及一个动态模型（根据当前状态和动作预测下一帧）。

Genie的独特之处在于，它完全通过无监督方式从大量未标注的互联网视频中学习——无需任何动作标签或领域知识。模型自动发现视频中的可控因素（如角色的移动方向），并将其编码为离散的潜在动作空间。这种自监督学习范式大大降低了数据获取的成本。

### 6.3.3 世界模型在机器人中的应用

世界模型在机器人领域的应用价值主要体现在以下几个方面。

**数据增广与策略学习**是最直接的应用。真实机器人数据的稀缺性是制约VLA等端到端模型性能的关键瓶颈。通过在学到的世界模型中"想象"大量虚拟轨迹，可以有效扩充训练数据。研究表明，结合世界模型生成的数据训练的策略，其性能可以显著优于仅使用真实数据训练的策略。

**模型预测控制（MPC）** 是另一个自然的应用场景。在模型预测控制框架中，控制器在每一步都需要在内部模型中推演多个可能的动作序列，并选择最优的一个。一个高质量的世界模型可以直接为MPC提供准确的环境转移函数，从而提升规划质量。

**安全评估与故障预测**也日益受到关注。在部署机器人策略之前，可以在世界模型中对策略进行大规模的虚拟测试，识别潜在的危险行为，而无需在真实环境中冒风险。这对于安全关键的应用场景（如手术机器人、家庭服务机器人）尤为重要。

然而，将视频生成模型用作世界模型也面临着严峻的挑战。**物理准确性**是首要问题——当前的视频生成模型虽然能生成视觉上逼真的视频，但往往无法严格遵守物理定律，例如物体可以穿墙而过，或者重力方向出现错误。这种"视觉逼真但物理错误"的特性，在用于策略学习时可能导致学到不可行甚至危险的行为。**时间一致性**也是一个挑战——自回归地生成长视频时，误差会逐步累积，导致生成的场景偏离真实物理过程。此外，**推理速度**限制了世界模型在实时控制中的应用——扩散模型生成一帧可能需要数百毫秒，而机器人控制通常需要毫秒级的响应。

为应对这些挑战，研究者们正在探索多个方向。将物理引擎与神经网络结合的混合方法（如在物理仿真器中进行粗略模拟，再用神经网络补充视觉细节）是一个有前景的方向。提升视频模型的物理一致性——例如通过引入物理先验或在训练中加入物理约束——也是活跃的研究课题。

## 6.4 大语言模型作为规划器

### 6.4.1 LLM任务分解：从语言到动作序列

大语言模型（LLM）在机器人领域的另一条重要应用路线，是将其作为高层任务规划器。与VLA追求端到端映射不同，这条路线采取了一种模块化的哲学：让LLM负责理解任务意图并将其分解为子任务序列，然后由专门的底层控制器负责执行每个子任务。

**SayCan** 是这一方向的开创性工作，由Google在2022年提出。SayCan的核心思想可以用一句话概括：让LLM说出机器人能做的事情（Say what's possible, then do what's feasible）。具体而言，SayCan将任务规划过程分解为两个互补的信号：**"Say"** 代表LLM对任务的理解能力——给定一个高层指令（如"我把饮料洒了，能帮我收拾一下吗？"），LLM生成可能的子任务序列；**"Can"** 代表机器人对每个子任务的执行能力——通过学习到的值函数（affordance function）评估每个技能在当前状态下的可行性和成功率。

SayCan的关键创新在于将这两个信号结合：LLM提供每个子任务的"有用性"得分（即该子任务对完成总目标的贡献），而技能函数提供"可行性"得分，两者的乘积作为最终的动作选择依据。这有效解决了LLM的"幻觉"问题——即使LLM不了解机器人的具体能力，也不会建议机器人执行不可能完成的任务。在真实厨房环境中的实验表明，SayCan能够将复杂的、多步骤的长程任务（如"为我做一杯咖啡"）可靠地分解为可执行的子任务序列。

### 6.4.2 视觉语言规划：感知增强的任务理解

纯语言的规划器存在一个根本局限：它们无法"看到"当前环境。当LLM仅通过文本描述来理解场景时，空间关系、物体状态等关键信息可能丢失或被错误传达。**视觉语言规划**（Vision-Language Planning）通过将视觉信息直接引入规划过程来解决这一问题。

**PaLM-E（Embodied PaLM）** 由Google在2023年提出，是这一方向最具影响力的工作。PaLM-E是一个5620亿参数的多模态具身语言模型，它将视觉、语言和机器人状态等多模态信息直接注入到PaLM语言模型的嵌入空间中。具体而言，图像通过ViT编码器转换为一系列视觉token，机器人状态（如关节角度、末端执行器位姿）通过专用编码器转换为状态token，这些token与文本token拼接后共同作为LLM的输入。

PaLM-E的一个重要特性是**多模态联合训练**——它在多个任务上同时训练，包括视觉问答、图像描述、机器人的高层规划和底层控制。这种联合训练使得模型能够在不同任务之间进行知识迁移，例如从视觉问答中习得的物体识别能力可以增强机器人操作中的物体定位能力。

在实验中，PaLM-E展现出了令人印象深刻的能力：它能够直接从原始传感器数据（如摄像头图像）中接收输入，并产生高层任务规划或具体的机器人动作指令。例如，给定一张包含积木的图像和指令"把这些积木按颜色排序"，PaLM-E可以直接输出具体的操作步骤。更重要的是，PaLM-E展现出了对场景变化的鲁棒性——即使环境中出现了训练中未见的干扰物，模型仍能正确规划。

**Inner Monologue** 等后续工作进一步探索了将多种反馈信号（如成功检测器、场景描述、人类反馈）纳入LLM规划循环的方法，使规划器能够在执行过程中进行动态调整。**Code as Policies** 则探索了一种新颖的范式：让LLM直接生成可执行的Python代码来控制机器人，代码本身既是对任务的理解，也是可执行的规划。

### 6.4.3 LLM规划的优势与局限

LLM作为规划器的核心优势在于其**开放世界知识**和**零样本泛化能力**。LLM在海量文本中习得了关于世界运作方式的广泛常识——包括物体的功能、因果关系、社会规范等——这些知识可以直接用于机器人任务规划，而无需针对每个新场景进行专门训练。例如，LLM知道"杯子可以用来装水"、"门需要先打开才能通过"等常识性知识，这些对于机器人完成日常任务至关重要。

LLM还提供了**自然语言接口**的天然优势。用户可以用日常语言描述任务（甚至包含模糊性和省略），LLM能够理解意图并补充缺失的细节。这种交互方式对于非专业用户来说极其友好，大大降低了机器人编程的门槛。

然而，LLM规划也面临着深刻的局限。**缺乏物理接地**（Grounding）是最根本的问题。LLM的"知识"来自于文本，而非物理经验。它可能知道"玻璃杯是易碎的"，但无法感知当前手中的力是否足以捏碎杯子。这种知识与感知的脱节，可能导致规划在语义上合理但在物理上不可行。

**推理的可靠性**也是一个持续的挑战。尽管LLM在很多任务上展现出了强大的推理能力，但在涉及空间推理、数值计算和严格逻辑推导的场景中，它们仍然容易出错。对于安全关键的机器人应用（如手术或工业制造），这种不可靠性是不可接受的。

**闭环规划的延迟**同样限制了LLM在实时控制中的应用。LLM的推理通常需要数百毫秒甚至更长时间，这使得它难以作为高频控制回路的一部分。当前的解决策略通常是将LLM用于高层的任务分解和全局规划，而将实时的局部调整交给更快的底层控制器。

## 6.5 融合与展望

### 6.5.1 技术路线的融合趋势

本章讨论的三条技术路线——VLA、世界模型和LLM规划——正在呈现出日益明显的融合趋势。

VLA与LLM规划的结合是最早出现的融合模式。SayCan可以被看作LLM规划加上技能库的组合，而RT-2则可以被看作VLA利用了LLM的知识。近年来，一些工作开始探索让LLM生成高层计划，再由VLA执行具体操作的分层架构，兼顾了LLM的语义推理能力和VLA的精细操作能力。

世界模型与VLA的融合也在加速。WorldVLA等工作将VLA和世界模型放入同一个自回归序列中，使得模型既能预测未来的视觉状态，又能产生当前的动作。这种统一框架允许策略在训练时利用"想象"的未来轨迹，在部署时利用世界模型进行前瞻规划。

三条路线的最终融合——一个既能理解语言指令、又能想象未来状态、还能直接输出动作的统一模型——是具身智能领域的长期愿景。尽管这一愿景的实现尚需时日，但当前的技术进展已经让我们看到了可能的路径。

### 6.5.2 开放问题与未来方向

基础模型与机器人的交叉领域仍面临着众多开放问题。

**规模化定律（Scaling Law）在机器人领域的适用性**是一个根本性的问题。在语言模型领域，更大的模型和更多的数据通常带来更好的性能。但在机器人领域，数据的获取成本、硬件的物理限制以及安全性的要求，使得简单的规模化策略面临挑战。如何设计适合机器人领域的规模化策略——例如通过仿真增广、跨具身迁移或人类视频学习——是一个重要的研究方向。

**安全性与可解释性**在机器人应用中尤为关键。当一个拥有数十亿参数的神经网络直接控制物理世界的机器人时，我们如何确保它的行为是安全的？如何理解它做出某个决策的原因？这些在纯软件AI应用中已经棘手的问题，在具身智能中变得更加紧迫。

**真实世界的长尾分布**也是一个持续的挑战。真实世界中充满了训练数据中罕见的长尾情况——不同的光照条件、遮挡模式、物体变异、环境布局等。基础模型虽然在一定程度上通过大规模预训练缓解了这一问题，但远未完全解决。

**多机器人协作**是另一个令人兴奋但充满挑战的方向。当多个机器人需要协同完成任务时，基础模型能否支持高效的通信、协调和任务分配？跨具身数据集（如Open X-Embodiment）的出现为这一方向提供了数据基础，但相应的模型架构和训练方法仍需进一步探索。

### 6.5.3 小结

基础模型正在深刻地重塑机器人学的研究范式。从RT系列的端到端控制，到世界模型的物理想象，再到LLM的任务规划，我们看到了一条从"模块化工程"走向"统一学习"的清晰趋势。这不仅仅是技术上的进步，更是哲学上的转变——从"为机器人编程"到"教机器人学习"，从"设计智能"到"涌现智能"。

然而，我们也必须清醒地认识到，当前的技术水平距离真正通用的机器人智能还有相当的距离。数据效率、物理接地、安全可靠、实时控制等核心挑战尚未完全解决。基础模型不是万能药，它们为机器人学带来了强大的新工具，但如何正确使用这些工具，仍需要机器人学、机器学习和认知科学等多学科的深入协作。

未来，随着基础模型的持续进化、机器人硬件的不断成熟以及数据获取方式的多样化，我们有理由期待一个机器人智能更加通用、更加可靠、更加普及的时代的到来。在那个时代，机器人将不再是工厂中的专用设备，而是能够理解人类意图、适应复杂环境、在物理世界中自主行动的通用智能体。基础模型，正是通往那个时代的重要基石。

---

> **参考文献提示**：本章内容基于以下关键论文：RT-1（Brohan et al., 2022）、RT-2（Zitkovich et al., 2023）、Open X-Embodiment/RT-X（O'Brien et al., 2023）、SayCan（Ahn et al., 2022）、PaLM-E（Driess et al., 2023）、UniSim（Yang et al., 2023）、Genie（Bruce et al., 2024）、World Models（Ha & Schmidhuber, 2018）、Dreamer（Hafner et al., 2020）、Code as Policies（Liang et al., 2023）、Inner Monologue（Huang et al., 2022）。建议读者查阅原始论文以获取更多技术细节。
# 第7章 抓取与操控

> *"操控是具身智能的核心能力——一个不能与物理世界交互的智能体，无论其内部推理多么精妙，都只是思想的囚徒。"*

抓取与操控是具身智能体与物理世界交互的最基本、最核心的能力。从人类婴儿在出生数月后即展现出的原始抓取反射，到熟练工匠运用复杂工具完成精密操作，抓取与操控贯穿了人类物理智能发展的全部历程。对于机器人而言，实现类似水平的操控能力，需要融合力学分析、感知理解、运动规划与学习优化等多学科知识，是具身智能研究中最具挑战性的课题之一。

本章从抓取规划出发，依次讨论灵巧操控、接触丰富任务和工具使用四个核心主题，力图呈现该领域从经典方法到前沿进展的完整脉络。

## 7.1 抓取规划

抓取（Grasping）是指智能体通过末端执行器（End-effector）与目标物体建立稳定接触，并将其固定在期望位姿的能力。抓取规划的核心问题可形式化为：给定物体几何 $\mathcal{O}$、末端执行器类型 $\mathcal{E}$ 和环境约束 $\mathcal{C}$，寻找抓取构型 $\mathbf{q}^*$ 使得抓取成功概率最大化：

$$\mathbf{q}^* = \arg\max_{\mathbf{q} \in \mathcal{Q}_{\text{free}}} P(\text{success} \mid \mathbf{q}, \mathcal{O}, \mathcal{E}, \mathcal{C})$$

其中 $\mathcal{Q}_{\text{free}}$ 为无碰撞的关节空间子集。这一看似简洁的优化问题，实际上涉及接触力学、几何推理、不确定性处理等多层面的复杂挑战。

### 7.1.1 力闭合分析

力闭合（Force Closure）是评判抓取质量的经典力学标准。其核心思想是：一组接触点产生的力螺旋（Wrench）能否平衡任意方向的外力与外力矩。设抓取包含 $m$ 个接触点，第 $i$ 个接触点的位置为 $\mathbf{p}_i \in \mathbb{R}^3$，法向量为 $\mathbf{n}_i$，则摩擦锥（Friction Cone）可线性化为：

$$FC_i = \left\{ \mathbf{f} \in \mathbb{R}^3 \mid \mathbf{f} = \mu_1 \mathbf{t}_{i1} + \mu_2 \mathbf{t}_{i2} + \mu_3 \mathbf{n}_i, \; \mu_1^2 + \mu_2^2 \leq (\mu_3 \mu)^2, \; \mu_3 \geq 0 \right\}$$

其中 $\mathbf{t}_{i1}, \mathbf{t}_{i2}$ 为接触面切向量，$\mu$ 为摩擦系数。将所有接触力映射到物体坐标系原点，得到抓取矩阵（Grasp Matrix）$G \in \mathbb{R}^{6 \times 3m}$：

$$G = \begin{bmatrix} \mathbf{I} & \mathbf{I} & \cdots & \mathbf{I} \\ [\mathbf{p}_1]_\times & [\mathbf{p}_2]_\times & \cdots & [\mathbf{p}_m]_\times \end{bmatrix}$$

其中 $[\cdot]_\times$ 为反对称矩阵。力闭合的充要条件是摩擦锥经抓取矩阵映射后的力螺旋锥（Wrench Cone）的凸包包含原点：

$$\mathbf{0} \in \text{conv}\left(\bigcup_{i=1}^{m} G \cdot FC_i\right)$$

这一条件等价于存在正系数 $\lambda_j \geq 0$（$\sum_j \lambda_j = 1$），使得 $\sum_j \lambda_j G \mathbf{f}_j = \mathbf{0}$，其中 $\mathbf{f}_j$ 遍历各摩擦锥的极端射线（Extreme Rays）。

力闭合分析为抓取提供了严格的力学保证，但其计算复杂度随接触点数量指数增长。Ferrari 和 Canny（1992）提出的 **Grasp Quality Metric** 将抓取质量量化为力螺旋空间中最大内切球半径（Chebyshev Radius）：

$$Q(G) = \min_{\|\mathbf{w}\|=1} \max_{\boldsymbol{\lambda}} \left\{ \mathbf{w}^T G \boldsymbol{\Lambda} \boldsymbol{\lambda} \mid \boldsymbol{\lambda} \geq \mathbf{0}, \; \mathbf{1}^T \boldsymbol{\lambda} = 1 \right\}$$

该指标越大，抓取抵抗外部扰动的能力越强。

### 7.1.2 GraspNet 与数据驱动抓取

传统力闭合分析依赖精确的物体模型和摩擦参数，在开放环境中面临泛化困难。以 GraspNet（Fang et al., 2020）为代表的数据驱动方法开辟了新范式。GraspNet 将抓取规划建模为从点云到抓取候选的回归问题：

给定场景点云 $\mathcal{P} = \{\mathbf{p}_i \in \mathbb{R}^3\}_{i=1}^{N}$，网络预测每个点处的抓取候选集合 $\mathcal{G}_i = \{g_{i,j}\}_{j=1}^{K}$，每个抓取由位姿 $\mathbf{T}_{i,j} \in SE(3)$ 和宽度 $w_{i,j}$ 参数化。网络架构包含三个关键模块：

1. **特征提取**：基于 PointNet++ 或 Sparse Convolution 提取逐点特征 $\mathbf{h}_i \in \mathbb{R}^d$；
2. **抓取方向预测**：通过锚点（Anchor）机制在球面上离散化抓取 approach 方向，每个锚点对应一个预定义抓取形状；
3. **抓取质量评估**：对每个候选预测质量得分 $s_{i,j}$，并通过碰撞检测过滤不可行抓取。

训练目标为联合优化分类与回归损失：

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \alpha \mathcal{L}_{\text{reg}} = -\sum_{i,j} y_{i,j} \log s_{i,j} + \alpha \sum_{i,j} y_{i,j} \|\Delta \mathbf{T}_{i,j}\|$$

其中 $y_{i,j}$ 为标注标签，$\Delta \mathbf{T}_{i,j}$ 为位姿残差。GraspNet 在包含超过 100 万个抓取标注的大规模数据集上训练，支持 88 个物体类别，其在杂乱场景中的抓取成功率达到了 90% 以上。

在 GraspNet 基础上，后续工作进一步拓展了能力边界。**AnyGrasp**（Fang et al., 2023）将抓取空间从平行夹爪扩展到任意形状末端执行器，通过连续 6DoF 抓取表示和大规模合成数据训练，实现了对未知物体的零样本泛化抓取。**Contact-GraspNet**（Sundermeyer et al., 2021）则直接在物体表面上预测接触点，避免了对抓取宽度的显式参数化，更适合复杂几何形状。

### 7.1.3 六自由度抓取

六自由度（6DoF）抓取突破了传统平面抓取的限制，允许末端执行器从任意方向接近并抓取物体。6DoF 抓取的姿态由齐次变换矩阵 $\mathbf{T} \in SE(3)$ 表示，包含三维平移 $\mathbf{t} \in \mathbb{R}^3$ 和三维旋转 $\mathbf{R} \in SO(3)$。

**基于点云的方法**直接处理深度传感器输出，避免了对完整三维模型的依赖。6-DoF GraspNet（Sundermeyer et al., 2019）采用变分自编码器（VAE）在物体参考帧中生成抓取候选：

$$z \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \mathbf{T} = \text{Decoder}(\mathbf{z}, \mathbf{f}_{\text{obj}})$$

其中 $\mathbf{f}_{\text{obj}}$ 为从观测中提取的物体特征。通过在隐空间中采样，可以高效生成多样化的抓取候选。

**基于渲染的方法**则利用可微渲染（Differentiable Rendering）将抓取规划转化为端到端优化问题。GNFactor（Ze et al., 2023）将 NeRF 表示与抓取规划相结合，通过可微体积渲染计算抓取质量梯度，直接优化 6DoF 抓取姿态：

$$\nabla_{\mathbf{T}} Q = \frac{\partial Q}{\partial \mathbf{T}} = \frac{\partial Q}{\partial \mathcal{I}} \cdot \frac{\partial \mathcal{I}}{\partial \mathbf{T}}$$

其中 $\mathcal{I}$ 为渲染图像。这种方法将场景理解与抓取规划统一在同一框架内。

**SE(3) 上的优化**需要特别注意旋转表示的奇异性。常用的旋转表示包括欧拉角（存在万向锁问题）、四元数（$\mathbf{q} \in \mathbb{S}^3$，存在双覆盖问题）和旋转矩阵（9 个参数，正交约束难以优化）。近年来，连续旋转表示如轴角（Axis-Angle）和李代数（$\mathfrak{so}(3)$）在学习框架中得到广泛应用：

$$\mathbf{R} = \exp([\boldsymbol{\omega}]_\times) = \mathbf{I} + \sin\|\boldsymbol{\omega}\| \frac{[\boldsymbol{\omega}]_\times}{\|\boldsymbol{\omega}\|} + (1-\cos\|\boldsymbol{\omega}\|) \frac{[\boldsymbol{\omega}]_\times^2}{\|\boldsymbol{\omega}\|^2}$$

### 7.1.4 吸盘抓取

吸盘（Suction Cup）是工业场景中广泛使用的末端执行器，特别适用于具有光滑表面的物体。与夹爪抓取不同，吸盘抓取的质量主要取决于以下因素：

**密封性分析**：吸盘与物体表面需要形成完整密封。设吸盘半径为 $r$，密封良好时可产生的最大法向吸附力为：

$$F_{\text{max}} = \pi r^2 \Delta P$$

其中 $\Delta P$ 为吸盘内外压差。实际有效吸附力还需考虑边缘泄漏和表面粗糙度的衰减因子 $\eta \in (0, 1)$：

$$F_{\text{eff}} = \eta \cdot \pi r^2 \Delta P$$

**表面法向量估计**：吸盘抓取要求末端执行器轴线与接触点表面法向量对齐。给定接触点 $\mathbf{p}$ 附近的点云局部邻域 $\mathcal{N}(\mathbf{p})$，法向量通过主成分分析（PCA）估计：

$$\mathbf{n} = \arg\min_{\|\mathbf{n}\|=1} \sum_{\mathbf{q} \in \mathcal{N}(\mathbf{p})} (\mathbf{n}^T (\mathbf{q} - \bar{\mathbf{q}}))^2$$

即为局部协方差矩阵的最小特征值对应的特征向量。

**SPDM（Suction Point Detection Module）**（Deng et al., 2021）将吸盘抓取点检测建模为语义分割问题，利用 PointNet++ 架构预测每个点的吸盘抓取得分和最优接近方向，在工业拣选任务中实现了每小时 300+ 次的成功抓取。

### 7.1.5 夹爪设计与选型

末端执行器的机械设计直接影响抓取能力的上限。常见的夹爪类型包括：

**平行夹爪（Parallel Jaw Gripper）**：结构最简单，通过两指平行开合实现抓取。其最大抓取力由夹持力 $F_{\text{grip}}$ 和摩擦系数 $\mu$ 决定：

$$F_{\text{hold}} = 2\mu F_{\text{grip}}$$

为抵抗重力 $mg$，需要 $2\mu F_{\text{grip}} > mg$。平行夹爪的局限在于无法抓取尺寸过小或形状不规则的物体。

**三指夹爪**：如 Robotiq 3-Finger Adaptive Gripper，通过三个自适应手指实现对多种形状物体的包络抓取（Enveloping Grasp）。包络抓取通过增大接触面积来提升抓取稳定性，其力闭合条件变为：

$$\mathbf{0} \in \text{int}\left(\text{conv}\left(\bigcup_{i=1}^{m} G \cdot FC_i\right)\right)$$

其中 $m$ 为所有接触点（包括指面接触和指尖接触）的总数。

**欠驱动夹爪**（Underactuated Gripper）利用弹簧或连杆机构实现自适应抓取，用较少的驱动器控制较多的自由度。典型代表如 Yale OpenHand，其优势在于被动适应物体形状，无需精确的物体模型即可实现稳定抓取。欠驱动系统的动力学可表示为：

$$M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau} + \mathbf{J}_c^T \mathbf{f}_c$$

其中 $M$ 为惯性矩阵，$C$ 为科氏力矩阵，$\mathbf{g}$ 为重力项，$\boldsymbol{\tau}$ 为关节力矩，$\mathbf{J}_c$ 为接触雅可比矩阵，$\mathbf{f}_c$ 为接触力。

近年来，**可变刚度夹爪**和**软体夹爪**（如利用硅胶材料的 Fin Ray 结构）进一步拓展了抓取的物体范围，能够安全地抓取易碎物体（如水果、玻璃器皿），同时通过结构柔性实现被动适应性。

## 7.2 灵巧操控

灵巧操控（Dexterous Manipulation）是指智能体通过多指灵巧手在手中对物体进行精细操控的能力。与简单抓取不同，灵巧操控强调的是物体在手中的运动——重定向、旋转、位姿调整——这要求手指协调施加时变力和力矩。灵巧操控代表了操控能力的最高水平，也是人类手部智能的核心体现。

### 7.2.1 灵巧手

灵巧手（Dexterous Hand）是实现灵巧操控的硬件基础。目前主流的灵巧手包括：

- **Shadow Dexterous Hand**：24 个自由度、20 个驱动器，接近人手自由度数量；
- **Allegro Hand**：16 个自由度、4 根手指（拇指 + 三指），每指 4 个关节；
- **LEAP Hand**（Shaw et al., 2023）：16 个自由度，采用肌腱驱动，兼具高性能与低成本；
- **Inspire Hand**：国内团队研发的高性能灵巧手，支持触觉感知。

灵巧手的运动学可用指数积公式（Product of Exponentials, POE）描述。对于 $n$ 自由度灵巧手，指尖位姿为：

$$\mathbf{T}(\mathbf{q}) = e^{[\boldsymbol{\xi}_1]\hat{} q_1} e^{[\boldsymbol{\xi}_2]\hat{} q_2} \cdots e^{[\boldsymbol{\xi}_n]\hat{} q_n} \mathbf{T}(0)$$

其中 $\boldsymbol{\xi}_i \in \mathfrak{se}(3)$ 为第 $i$ 个关节的运动螺旋，$q_i$ 为关节角度，$\mathbf{T}(0)$ 为零位时的指尖位姿。

灵巧手与物体的接触形成闭环运动链，接触约束可表示为：

$$\mathbf{J}_c(\mathbf{q}) \dot{\mathbf{q}} = \dot{\mathbf{x}}_c$$

其中 $\mathbf{J}_c \in \mathbb{R}^{6k \times n}$ 为接触雅可比矩阵（$k$ 为接触点数量），$\dot{\mathbf{x}}_c$ 为接触点的速度。这些约束将手指运动与物体运动耦合在一起。

### 7.2.2 手指力控制

灵巧操控的核心在于精确的手指力控制。手指与物体的接触力学遵循库仑摩擦定律：

$$\|\mathbf{f}_t\| \leq \mu |\mathbf{f}_n|$$

其中 $\mathbf{f}_t$ 为切向摩擦力，$\mathbf{f}_n$ 为法向接触力，$\mu$ 为摩擦系数。

**混合力-位控制**是手指力控制的经典框架。将手指空间分解为力控方向（通常为法向）和位控方向（切向），分别施加不同的控制律：

$$\boldsymbol{\tau} = \mathbf{J}_n^T (K_f \mathbf{f}_d + K_{fi} \int (\mathbf{f}_d - \mathbf{f}_n) dt) + \mathbf{J}_t^T (K_p (\mathbf{x}_d - \mathbf{x}) + K_d (\dot{\mathbf{x}}_d - \dot{\mathbf{x}}))$$

其中 $\mathbf{f}_d$ 为期望法向力，$\mathbf{x}_d$ 为期望切向位移，$K_f, K_{fi}, K_p, K_d$ 为控制增益。

**触觉感知**对于手指力控制至关重要。高分辨率触觉传感器（如 GelSight、DIGIT）可以提供接触面的法向力分布、剪切力和滑动检测。基于触觉反馈的抓取力自适应控制律为：

$$F_{\text{grip}}^{(t+1)} = F_{\text{grip}}^{(t)} + K_s \cdot \text{sgn}(\text{slip}) \cdot |\dot{\mathbf{x}}_{\text{tangential}}|$$

当检测到滑动时增大夹持力，稳定后逐步减小以节省能量并保护易碎物体。

### 7.2.3 物体重定向

物体重定向（Object Reorientation）是指将物体从初始姿态旋转到目标姿态的操作。这是灵巧操控中最基础也最具代表性的任务。

**基于运动规划的方法**将重定向建模为 $SE(3)$ 上的路径规划问题。物体的旋转运动构成 $SO(3)$ 流形，需要在其上定义度量和搜索算法。$SO(3)$ 上两点间的角度距离为：

$$d(\mathbf{R}_1, \mathbf{R}_2) = \|\log(\mathbf{R}_1^T \mathbf{R}_2)\| = \arccos\left(\frac{\text{tr}(\mathbf{R}_1^T \mathbf{R}_2) - 1}{2}\right)$$

基于 PRM 或 RRT 的规划器在 $SO(3)$ 上搜索可行路径，每一步需要验证手指-物体接触构型的可行性（碰撞检测 + 力闭合检验）。

**基于强化学习的方法**近年来取得显著进展。In-Hand Object Reorientation（OpenAI, 2020）利用大规模并行仿真和 Sim-to-Real 迁移，在五指灵巧手上实现了魔方的精确旋转。其核心设计包括：

- **域随机化（Domain Randomization）**：在仿真中随机化物理参数（摩擦系数、质量、惯性矩等），提升策略的鲁棒性；
- **自动课程（Automatic Curriculum）**：从简单目标开始，逐步增加旋转角度要求；
- **观测历史**：使用堆叠的历史观测 $o_t = [s_{t-H}, \ldots, s_t]$ 估计物体旋转状态（因无外部视觉反馈）。

奖励函数通常包含目标姿态距离和抓取稳定性两项：

$$r_t = -\alpha \|\mathbf{q}_{\text{obj}}^{(t)} - \mathbf{q}_{\text{target}}\|_{\text{quat}} - \beta \|\mathbf{v}_{\text{obj}}^{(t)}\|$$

其中 $\|\cdot\|_{\text{quat}}$ 为四元数距离（考虑双覆盖），$\mathbf{v}_{\text{obj}}$ 为物体速度（惩罚过快运动以保证稳定性）。

### 7.2.4 手内操控

手内操控（In-Hand Manipulation）泛指物体在灵巧手中不离开手掌的精细操控动作，包括手指传递（Finger Gaiting）、指尖滚动（Fingertip Rolling）和笔旋转（Pen Spinning）等。

**手指传递**是指物体在手指间转移，以实现超出单一手指可达空间的操控。其核心挑战在于：在传递过程中，必须始终保持至少一组接触能够提供力闭合。形式化地，设时间区间 $[t_1, t_2]$ 为传递过程，需要：

$$\forall t \in [t_1, t_2], \quad \exists \mathcal{C}_{\text{active}}(t) \subseteq \mathcal{C}_{\text{all}} \text{ s.t. } \text{ForceClosure}(\mathcal{C}_{\text{active}}(t))$$

其中 $\mathcal{C}_{\text{active}}(t)$ 为 $t$ 时刻的有效接触集合。

**指尖滚动**利用摩擦驱动在指尖上产生滚动接触，从而在不抬起手指的情况下旋转物体。滚动接触的运动学约束为：

$$\dot{\mathbf{p}}_{\text{contact}}^{\text{finger}} = \dot{\mathbf{p}}_{\text{contact}}^{\text{obj}} + \boldsymbol{\omega}_{\text{rel}} \times \mathbf{n} \cdot 0$$

即接触点在手指和物体表面的相对速度为零（纯滚动条件）。

**手内重定向的强化学习方法**（Handa et al., 2023; Qi et al., 2023）将此问题建模为马尔可夫决策过程（MDP）。状态空间包含关节角度、关节速度、物体位姿和接触力；动作空间为关节力矩或目标角度。Sim-to-Real 迁移是关键挑战，常用技术包括：

- **系统辨识**：精确标定仿真器物理参数；
- **残差策略学习**：在仿真策略基础上学习真实环境的补偿项；
- **Teacher-Student 框架**：Teacher 在仿真中使用特权信息（完整状态），Student 在真实环境中仅使用可观测量。

## 7.3 接触丰富任务

接触丰富任务（Contact-Rich Tasks）是指大量接触交互贯穿始终的操作任务，如装配、插入、柔性物体操作等。这类任务的核心挑战在于：接触力的非线性和不连续性使得精确建模极为困难，而微小的位姿误差可能导致巨大的力变化甚至任务失败。

### 7.3.1 装配与插入

装配（Assembly）和插入（Insertion）是制造业中最常见的接触丰富任务。以轴孔插入（Peg-in-Hole）为例，这一经典问题自 1970 年代起就一直是机器人操控研究的核心课题。

**力-位混合控制方法**：在插入过程中，轴向方向需要力控制以维持恒定插入力，径向方向需要位控以实现对准。经典的 RCC（Remote Center Compliance）装置通过被动柔顺机构实现自适应对准，其刚度矩阵为：

$$K_{\text{RCC}} = \begin{bmatrix} K_t \mathbf{I}_3 & \mathbf{0} \\ \mathbf{0} & K_r \mathbf{I}_3 \end{bmatrix}$$

其中 $K_t$ 和 $K_r$ 分别为平移和旋转刚度。通过合理设计柔顺中心的位置，可以在无主动控制的情况下完成插入。

**基于力反馈的主动策略**需要根据接触状态（点接触、线接触、面接触）实时调整运动。Whitney（1982）提出的力-运动策略将插入过程分解为多个阶段：

1. **搜索阶段**：在孔口附近执行螺旋搜索运动；
2. **接触阶段**：检测到接触后，沿接触法向量方向施加插入力；
3. **插入阶段**：利用 Chamfer 边缘的导向力引导轴进入孔中；
4. **完成阶段**：到达底部，减小插入力。

**基于强化学习的装配策略**近年来展现出强大潜力。Rapidly Adaptable Legible Assembly（RALA, Luo et al., 2021）通过在仿真中训练力觉策略，结合自动课程学习和域随机化，实现了对多种公差等级的轴孔插入任务泛化。策略输入为力-力矩传感器读数 $\mathbf{w} = [\mathbf{f}^T, \boldsymbol{\tau}^T]^T \in \mathbb{R}^6$，输出为末端执行器速度 $\dot{\mathbf{x}} \in \mathbb{R}^6$：

$$\dot{\mathbf{x}} = \pi_\theta(\mathbf{w}_{t-H:t})$$

**多阶段装配**对于复杂产品（如电路板、发动机部件）需要按顺序完成多个子装配步骤。这一问题可建模为层级 MDP，高层策略决定装配顺序，低层策略执行每个子任务的精确操控。约束传播确保每步操作满足几何约束和物理约束。

### 7.3.2 柔性物体操作

柔性物体（Deformable Objects）操作是操控领域的"圣杯"问题之一。衣物折叠、线缆布线、手术缝合等任务广泛存在于日常生活和工业应用中，但由于柔性物体的高维状态空间和复杂动力学，其操控难度远超刚性物体。

**物理建模**方面，柔性物体常用以下表示方法：

- **质点-弹簧模型（Mass-Spring System）**：将物体离散为质点，质点间用弹簧连接。动力学方程为 $m_i \ddot{\mathbf{x}}_i = -\sum_{j \in \mathcal{N}(i)} k_{ij}(\|\mathbf{x}_i - \mathbf{x}_j\| - l_{ij}) \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|} + \mathbf{f}_{\text{ext}}$，其中 $k_{ij}$ 为弹簧刚度，$l_{ij}$ 为自然长度；
- **有限元方法（FEM）**：基于连续介质力学，通过四面体或六面体单元对物体进行离散，求解弹性力学方程 $\rho \ddot{\mathbf{u}} = \nabla \cdot \boldsymbol{\sigma} + \mathbf{b}$，其中 $\mathbf{u}$ 为位移场，$\boldsymbol{\sigma}$ 为应力张量，$\mathbf{b}$ 为体力；
- **基于图的神经网络表示（Graph Neural Network）**：将物体建模为图结构，节点为网格顶点或质点，边为连接关系，通过消息传递学习物体的非线性动力学。

**衣物折叠**是柔性物体操作的典型任务。Warping（Wang et al., 2023）提出基于关键点匹配的折叠策略：首先从 RGB-D 图像中检测衣物关键点（如角点、边缘中点），然后规划折叠路径将当前关键点配置映射到目标配置。关键点之间的映射关系通过对比学习获得。

**线缆操作**需要处理一维柔性物体的复杂缠绕和打结。Thananjeyan et al.（2021）提出基于视觉的线缆追踪方法，利用 Transformer 架构在 RGB 图像上预测线缆的中线表示，然后规划机械臂运动实现布线。线缆的状态由参数化曲线 $\gamma(s): [0,1] \rightarrow \mathbb{R}^3$ 表示，其弯曲能量为：

$$E_{\text{bend}} = \int_0^1 \kappa^2(s) \, ds$$

其中 $\kappa(s)$ 为曲率。最小化弯曲能有助于生成物理合理的线缆配置。

**Sim-to-Real 迁移**对柔性物体尤其困难，因为其物理参数（杨氏模量、阻尼系数、摩擦系数）的精确辨识极为困难。DiffCloth（Li et al., 2022）通过可微物理仿真器计算策略梯度，直接在真实演示上优化仿真参数，实现了更准确的域迁移。

### 7.3.3 接触力建模

精确的接触力学模型是接触丰富任务成功的基础。接触力建模需要处理三个核心问题：接触检测、接触力计算和摩擦建模。

**接触检测**确定两个物体是否发生接触以及接触区域的几何形状。对于凸体，可通过 GJK 算法或 EPA（Expanding Polytope Algorithm）高效计算最近距离和穿透深度。对于非凸体或柔性体，通常采用基于空间哈希或 BVH（Bounding Volume Hierarchy）的加速检测方法。

**接触力计算**方面，常见的方法包括：

- **罚函数法（Penalty Method）**：$\mathbf{f}_n = k_n \cdot \delta^n$，其中 $\delta$ 为穿透深度，$n$ 为非线性指数（通常 $n=1$ 或 $n=2$）。简单但需要选择合适的刚度参数；
- **LCP（Linear Complementarity Problem）方法**：将接触力和互补条件统一建模为 LCP 问题：$\mathbf{0} \leq \mathbf{f}_n \perp \mathbf{A} \mathbf{f}_n + \mathbf{b} \geq \mathbf{0}$。自然满足不可穿透条件，但求解复杂度较高；
- **基于优化的方法**：将接触力求解嵌入到约束优化框架中，利用接触约束的互补松弛条件进行求解。

**摩擦建模**从简单的库仑模型到复杂的 Stribeck 模型，精度逐渐提升。Stribeck 摩擦模型描述了速度-摩擦力之间的非线性关系：

$$f(v) = f_c + (f_s - f_c) e^{-(v/v_s)^2} + \sigma v$$

其中 $f_c$ 为库仑摩擦力，$f_s$ 为最大静摩擦力，$v_s$ 为 Stribeck 速度，$\sigma$ 为粘性摩擦系数。该模型能够捕捉从静摩擦到动摩擦的过渡行为。

## 7.4 工具使用

工具使用（Tool Use）是高等智能的标志性能力。在具身智能中，工具使用要求智能体不仅能识别工具的存在，还能理解工具的功能，并制定使用工具完成目标任务的策略。这一能力将智能体的操控能力从"直接交互"提升到了"间接交互"的层次。

### 7.4.1 工具识别

工具识别是工具使用的前提，需要从视觉感知中检测和分类场景中的工具。

**物体检测与分割**：基于深度学习的方法（如 Mask R-CNN、SAM）可以精确分割出场景中的工具实例。对于工具类物体，关键挑战在于其形态多样性和关节结构（如剪刀的两个刃、锤子的锤头与柄）。部件级分割（Part Segmentation）将工具分解为功能部件，为后续功能理解提供基础：

$$\text{Tool} = \{P_1, P_2, \ldots, P_k\}, \quad P_i = (\text{mask}_i, \text{label}_i, \text{pose}_i)$$

**位姿估计**：工具的抓取和使用需要精确的 6DoF 位姿估计。基于 CAD 模型的方法（如 PoseCNN + ICP 精化）和无模型方法（如 FoundationPose, Wen et al., 2024）都能实现亚厘米精度的位姿估计。对于关节工具，还需要估计其关节状态。

**工具状态感知**：某些工具具有功能性状态（如钳子的开合、扳手的调节），需要额外的状态估计。基于视觉的状态估计通过检测可动部件的相对位姿来推断工具状态。

### 7.4.2 工具功能理解

工具功能理解（Tool Affordance Understanding）是指智能体理解工具"能做什么"和"怎么用"的能力。这超越了简单的物体识别，涉及因果推理和物理直觉。

**功能学习的表示框架**：工具的功能可由其运动原语（Motion Primitive）和物理效果（Physical Effect）来描述。设工具 $T$ 作用于物体 $O$ 产生效果 $E$：

$$E = f(T, \mathbf{a}, O, \mathbf{c})$$

其中 $\mathbf{a}$ 为操作动作（运动轨迹），$\mathbf{c}$ 为环境条件。功能学习的目标是从观察或交互中学习函数 $f$。

**基于物理的功能推理**：Fang et al.（2023）提出 Physical Concept-Grounded 工具使用框架，将工具功能与物理概念（杠杆原理、楔形效应、摩擦力利用等）关联。例如，螺丝刀的功能可表述为"将旋转运动转化为沿轴向的线性运动（螺纹效应）"，扳手的功能为"增大扭矩（杠杆原理）"：

$$\tau_{\text{output}} = \frac{L_{\text{handle}}}{L_{\text{jaw}}} \cdot \tau_{\text{input}}$$

其中 $L_{\text{handle}}$ 和 $L_{\text{jaw}}$ 分别为扳手柄长和钳口长度。

**跨物体功能迁移**：智能体应能将已学的工具使用知识迁移到未见过的新工具。ToolFormerGPT（Wu et al., 2023）利用大语言模型的常识知识进行功能类比：将新工具的形状和材质特征与已知工具匹配，推断其可能的功能和使用方法。视觉-语言模型（VLM）如 GPT-4V 在此任务中展现出令人印象深刻的能力。

### 7.4.3 工具操作策略

工具操作策略（Tool-Use Policy）需要解决"如何使用工具完成任务"的问题，通常涉及工具获取、工具定位和工具操作三个子问题。

**工具获取策略**：当工具不在智能体手中时，首先需要抓取工具。工具抓取有其特殊性——需要根据预期用途选择抓取方式。例如，使用锤子时应抓握柄部而非锤头；使用画笔时应握持中段而非笔尖。抓取点的选择可建模为优化问题：

$$\mathbf{p}_{\text{grasp}}^* = \arg\max_{\mathbf{p}} \left[ \alpha \cdot Q_{\text{grasp}}(\mathbf{p}) + \beta \cdot Q_{\text{task}}(\mathbf{p}, \tau_{\text{target}}) \right]$$

其中 $Q_{\text{grasp}}(\mathbf{p})$ 为抓取质量，$Q_{\text{task}}(\mathbf{p}, \tau_{\text{target}})$ 为任务适用性评分，$\tau_{\text{target}}$ 为目标任务。

**工具-物体交互的运动规划**：使用工具时，需要规划工具-物体的接触运动轨迹。对于擦拭、刮削等持续接触任务，接触力的维持是关键。设工具与物体的期望接触力为 $\mathbf{f}_d$，实际接触力为 $\mathbf{f}$，则阻抗控制律为：

$$\mathbf{M}_d \ddot{\mathbf{e}} + \mathbf{B}_d \dot{\mathbf{e}} + \mathbf{K}_d \mathbf{e} = \mathbf{f}_d - \mathbf{f}$$

其中 $\mathbf{e} = \mathbf{x}_d - \mathbf{x}$ 为位姿误差，$\mathbf{M}_d, \mathbf{B}_d, \mathbf{K}_d$ 为期望的惯性、阻尼和刚度矩阵。

**强化学习方法**：对于复杂的工具使用任务，强化学习能够发现人类难以设计的操作策略。Sawyer et al.（2023）的 ToolPlay 框架在仿真中训练通用工具使用策略，通过大规模并行环境和自动课程设计，使机器人学会了使用多种工具完成不同的操控任务。策略网络接收视觉观测和触觉信号，输出关节位置目标：

$$\mathbf{q}_{\text{target}} = \pi_\theta(\mathbf{v}_t, \mathbf{f}_{\text{tactile}, t}, \mathbf{g}_t)$$

其中 $\mathbf{v}_t$ 为视觉特征，$\mathbf{f}_{\text{tactile}, t}$ 为触觉信息，$\mathbf{g}_t$ 为任务目标描述。

**大语言模型驱动的工具使用**：最新进展将大语言模型（LLM）引入工具使用任务规划。SayCan（Ahn et al., 2022）将 LLM 的语义知识与机器人技能的可行性评分相结合：

$$a^* = \arg\max_{a \in \mathcal{A}} P(a \mid \text{LLM}) \cdot V(a \mid \text{robot})$$

其中 $P(a \mid \text{LLM})$ 为 LLM 对动作 $a$ 的语义评分，$V(a \mid \text{robot})$ 为机器人策略对动作 $a$ 的可行性评估。这种框架使得机器人能够利用 LLM 的世界知识来规划工具使用策略，同时受限于机器人实际能力的边界。

## 7.5 本章小结

本章系统阐述了具身智能中抓取与操控的四个核心维度。从力闭合的严格力学分析到 GraspNet 的数据驱动范式，抓取规划展现了经典与学习方法的互补；从灵巧手的手指力控制到 In-Hand Manipulation 的强化学习突破，灵巧操控揭示了多指协调的深层挑战；从轴孔装配到柔性物体操作，接触丰富任务凸显了接触力学建模的核心地位；从工具识别到 LLM 驱动的工具使用规划，工具使用展示了智能体认知能力向物理操作的转化。

展望未来，抓取与操控领域正呈现几个重要趋势：（1）**Foundation Models for Manipulation**——大规模预训练的操控基础模型将实现跨任务、跨物体、跨场景的泛化；（2）**Sim-to-Real at Scale**——高保真可微仿真与大规模并行训练将大幅降低真实世界的试错成本；（3）**Multimodal Sensing Integration**——视觉、触觉、力觉、听觉的深度融合将提供更丰富的交互信息；（4）**Human-like Dexterity**——在灵巧手上实现接近人类水平的操控能力仍是长期目标，需要在硬件、算法和学习范式上的持续突破。

抓取与操控不仅是一项工程挑战，更是理解物理智能本质的关键窗口。一个能够熟练操控物体的智能体，必然对物理世界的因果规律、几何结构和力学原理拥有深刻的理解——这种理解，正是具身智能的核心追求。

---

**参考文献（部分）**

1. Ferrari, C., & Canny, J. (1992). Planning optimal grasps. *IEEE ICRA*.
2. Fang, H. et al. (2020). GraspNet-1Billion: A large-scale benchmark for general object grasping. *IEEE CVPR*.
3. Fang, H. et al. (2023). AnyGrasp: Robust and efficient grasp perception in spatial and temporal domains. *IEEE T-RO*.
4. Sundermeyer, M. et al. (2021). Contact-GraspNet: Efficient 6-DoF grasp generation in cluttered scenes. *IEEE ICRA*.
5. Deng, S. et al. (2021). Partial policy learning for high-dimensional robotic manipulation. *IEEE ICRA*.
6. OpenAI et al. (2020). Solving Rubik's cube with a robot hand. *arXiv:1910.07113*.
7. Handa, A. et al. (2023). Dextreme: Dexterous manipulation in the wild. *IEEE ICRA*.
8. Qi, H. et al. (2023). General in-hand object rotation with vision and touch. *CoRL*.
9. Whitney, D. E. (1982). Quasi-static assembly of compliantly supported rigid parts. *ASME J. Dyn. Sys.*.
10. Shaw, K. et al. (2023). LEAP Hand: Low-cost, efficient, and accessible dexterous manipulation. *RSS*.
11. Luo, J. et al. (2021). Legible and rapidly adaptable to new assembly tasks. *IEEE RA-L*.
12. Ahn, M. et al. (2022). Do as I can, not as I say: Grounding language in robotic affordances. *arXiv:2204.01691*.
13. Wen, B. et al. (2024). FoundationPose: Unified 6D pose estimation and tracking of novel objects. *CVPR*.
14. Wang, R. et al. (2023). Soft body manipulation: Learning to fold. *IEEE ICRA*.
15. Thananjeyan, B. et al. (2021). Recovery RL: Safe reinforcement learning with learned recovery zones. *IEEE RA-L*.
# 第8章 自主导航

> "真正的智能不仅在于思考，更在于行动——在物理世界中安全、高效、自主地从A到B。"

自主导航是具身智能系统最基本也最关键的能力之一。无论是家庭服务机器人在客厅中穿梭、自动驾驶车辆在城市道路上行驶，还是无人机在复杂环境中飞行，导航能力的高低直接决定了智能体能否在物理世界中完成任务。本章将从地图构建、路径规划、语义导航和视觉导航四个维度，系统介绍自主导航的核心技术与前沿进展。

---

## 8.1 地图构建

地图是导航的基础。智能体需要一种对外部世界结构化、可查询的表示，才能规划从当前位置到目标位置的路径。根据表示方式的不同，常见的地图类型包括占据栅格地图、点云地图、语义地图和拓扑地图。

### 8.1.1 占据栅格地图

占据栅格地图（Occupancy Grid Map）是最经典的环境表示方法之一，由Elfes和Moravec于1980年代提出[1]。其核心思想是将连续的二维（或三维）空间离散化为一系列栅格单元，每个栅格 $c_i$ 维护一个占据概率值 $P(c_i \mid z_{1:t})$，表示该单元被障碍物占据的概率。

给定传感器观测序列 $z_{1:t}$，利用贝叶斯推理可以递推更新每个栅格的占据概率。对于单个栅格 $c$，占据概率的对数几率（log-odds）表示为：

$$
l_{t}(c) = l_{t-1}(c) + \log \frac{P(z_t \mid c=\text{occupied})}{P(z_t \mid c=\text{free})} - l_0
$$

其中 $l_0$ 是先验对数几率。对数几率表示的优势在于：贝叶斯更新退化为简单的加法运算，避免了反复进行概率归一化。占据概率与对数几率的转换关系为：

$$
P(c=\text{occupied}) = 1 - \frac{1}{1 + \exp(l(c))}
$$

在三维空间中，体素栅格地图（Voxel Grid Map）将空间划分为等大的立方体，适用于室内机器人导航。然而，随着空间分辨率的提升，内存开销呈立方增长。为此，OctoMap[2]采用八叉树（Octree）结构递归地将空间划分为八个子节点，在空旷区域使用粗粒度表示，在细节丰富区域使用细粒度表示，从而在精度与效率之间取得平衡。

占据栅格地图的局限在于：它仅编码几何信息，不包含语义。栅格只知道"某处有障碍物"，但不知道它是桌子、椅子还是墙壁。

### 8.1.2 点云地图

点云地图（Point Cloud Map）直接存储三维空间中采样点的坐标集合 $\mathcal{P} = \{p_1, p_2, \ldots, p_N\}$，其中 $p_i \in \mathbb{R}^3$。点云通常由激光雷达（LiDAR）或深度相机获取。

从深度图到点云的转换涉及相机内参矩阵 $K$。对于像素坐标 $(u, v)$ 处的深度值 $d$，对应的三维点为：

$$
p = d \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

点云配准（Point Cloud Registration）是构建全局地图的关键步骤。迭代最近点算法（ICP, Iterative Closest Point）[3]通过交替执行"最近点匹配"和"刚体变换估计"来对齐两帧点云。给定源点云 $\mathcal{P}$ 和目标点云 $\mathcal{Q}$，ICP 最小化以下目标函数：

$$
\min_{R, t} \sum_{i=1}^{N} \| R p_i + t - q_i \|^2
$$

其中 $R \in SO(3)$ 为旋转矩阵，$t \in \mathbb{R}^3$ 为平移向量，$q_i$ 是 $p_i$ 在 $\mathcal{Q}$ 中的最近点。该问题可通过奇异值分解（SVD）求解。

然而，点云地图存在存储冗余问题——相同结构在不同视角下被重复采样。Surfel（Surface Element）表示[4]通过在每个点附加法向量和半径来缓解这一问题，将点云转化为局部切平面的集合。

### 8.1.3 语义地图

语义地图（Semantic Map）在几何表示的基础上叠加语义标签，使地图不仅能回答"哪里有障碍物"，更能回答"那里是什么"。构建语义地图的主流方法是将语义分割网络与SLAM系统相结合。

设 $I$ 为输入图像，语义分割网络 $f_\theta$ 输出逐像素的类别预测：

$$
\hat{S} = f_\theta(I), \quad \hat{S}(u,v) \in \{1, 2, \ldots, C\}
$$

其中 $C$ 为类别总数。语义信息通过相机投影关系融合到三维地图中。对于每个三维点 $p_j$，根据其在多个视角下观察到的语义标签进行投票或概率融合：

$$
\hat{c}_j = \arg\max_{c \in \{1,\ldots,C\}} \sum_{k=1}^{K} \log P(c \mid I_k, p_j)
$$

其中 $K$ 为可见该点的视角数。

近期，3D场景语义重建与大型语言模型（LLM）的结合开辟了新方向。例如，ConceptFusion[5]利用CLIP模型将图像区域编码为语义特征向量，构建"开放词汇"的语义地图，支持用户以自然语言查询场景中的物体（如"找到可以坐的东西"），突破了固定类别的限制。

### 8.1.4 拓扑地图

拓扑地图（Topological Map）将环境抽象为图结构 $G = (V, E)$，其中顶点 $v_i \in V$ 代表关键位置（如房间、走廊交叉口），边 $e_{ij} \in E$ 表示位置间的可通行关系。与栅格地图相比，拓扑地图存储紧凑，查询路径效率高（图搜索复杂度取决于节点数而非空间分辨率），特别适合大尺度环境。

拓扑地图的构建通常涉及位置识别（Place Recognition）——判断智能体是否到达过某个先前访问的地点。NetVLAD[6]提出了一种可微的VLAD（Vector of Locally Aggregated Descriptors）层，将图像编码为判别性描述向量。给定查询图像 $I_q$ 和候选图像集 $\{I_1, \ldots, I_M\}$，通过描述子之间的欧氏距离排序实现位置检索：

$$
\text{score}(I_q, I_i) = -\| f_{\text{VLAD}}(I_q) - f_{\text{VLAD}}(I_i) \|_2
$$

拓扑地图的缺陷在于缺乏精确的度量信息。因此，实践中常采用**混合地图**策略：在粗粒度上使用拓扑图进行全局路径规划，在局部区域使用栅格或点云地图进行精确的局部导航。

---

## 8.2 路径规划

有了环境地图之后，智能体需要找到从起点到目标点的可行（且优化的）路径。路径规划是机器人学与人工智能交叉的核心问题。

### 8.2.1 A*算法

A*[7]是启发式搜索的经典算法，在栅格地图上表现尤为出色。它维护一个优先队列（open list），根据评估函数 $f(n) = g(n) + h(n)$ 选择最有希望的节点扩展，其中 $g(n)$ 是从起点到节点 $n$ 的实际代价，$h(n)$ 是从 $n$ 到目标的启发式估计。

**可采纳性保证**：当启发函数 $h(n)$ 满足 $h(n) \leq h^*(n)$（即不高估真实代价）时，A* 保证找到最优路径。常用的可采纳启发函数包括曼哈顿距离和欧氏距离：

$$
h_{\text{Manhattan}}(n) = |x_n - x_g| + |y_n - y_g|
$$

$$
h_{\text{Euclidean}}(n) = \sqrt{(x_n - x_g)^2 + (y_n - y_g)^2}
$$

A* 的时间复杂度在最坏情况下为指数级，但在实践中，好的启发函数可以大幅剪枝。加权 A*（Weighted A*）通过引入权重 $\epsilon > 1$ 将评估函数修改为 $f(n) = g(n) + \epsilon \cdot h(n)$，以牺牲最优性换取更快的搜索速度，保证找到的路径代价不超过最优解的 $\epsilon$ 倍。

### 8.2.2 RRT* 算法

在高维连续空间（如机械臂的关节空间）中，栅格搜索面临维度灾难。快速扩展随机树（RRT, Rapidly-exploring Random Tree）[8]及其渐近最优变体 RRT*[9] 通过随机采样高效探索配置空间。

RRT 的基本步骤为：(1) 在配置空间中随机采样一个点 $q_{\text{rand}}$；(2) 找到树中距 $q_{\text{rand}}$ 最近的节点 $q_{\text{near}}$；(3) 从 $q_{\text{near}}$ 向 $q_{\text{rand}}$ 方向扩展步长 $\delta$ 得到 $q_{\text{new}}$；(4) 若 $q_{\text{new}}$ 不与障碍物碰撞，则将其加入树中。

RRT* 在此基础上增加了**重布线**（rewiring）步骤：对 $q_{\text{new}}$ 邻域内的节点，检查通过 $q_{\text{new}}$ 是否能获得更短路径，若是则更新父节点。RRT* 的代价函数定义为从根节点到目标节点的路径长度：

$$
c(q) = \min_{\tau \in \text{paths}} \int_0^1 \| \dot{\tau}(s) \| ds
$$

RRT* 的一个重要理论性质是**渐近最优性**（asymptotic optimality）：当采样点数趋于无穷时，RRT* 收敛到最优路径。Informed RRT*[10]进一步通过在以起点和目标点为焦点的椭球内采样，加速了收敛过程。

### 8.2.3 PRM 概率路线图

概率路线图（PRM, Probabilistic Roadmap）[11]是另一种基于采样的规划方法，适用于多查询场景。PRM 分两阶段工作：

1. **构建阶段**：在自由空间中随机采样 $N$ 个无碰撞的配置点，对于每个点，尝试与其 $k$ 个最近邻建立边，保留无碰撞的边，形成路线图 $G_{\text{PRM}}$。
2. **查询阶段**：将起点和目标点连接到路线图上，使用 Dijkstra 或 A* 在路线图上搜索路径。

PRM 的完备性是**概率完备**的：当采样点数 $N \to \infty$ 时，如果存在可行路径，PRM 以概率 1 找到。Lazy PRM 通过延迟碰撞检测（先建图，后验证边是否碰撞）来加速构建过程。

### 8.2.4 运动规划

上述方法通常输出一系列路径点，而运动规划（Motion Planning）需要考虑机器人动力学约束，生成时间参数化的轨迹。常用的方法包括：

**模型预测控制（MPC, Model Predictive Control）**在有限时间窗口 $[t, t+T]$ 内求解最优控制序列：

$$
\min_{u_{t:t+T}} \sum_{k=0}^{T} \left[ \| x_{t+k} - x_{\text{ref}} \|_Q^2 + \| u_{t+k} \|_R^2 \right]
$$

$$
\text{s.t.} \quad x_{t+k+1} = f(x_{t+k}, u_{t+k}), \quad x_{t+k} \in \mathcal{X}_{\text{free}}, \quad u_{t+k} \in \mathcal{U}
$$

其中 $Q$ 和 $R$ 为正定权重矩阵，$f$ 为状态转移函数，$\mathcal{X}_{\text{free}}$ 为无障碍区域，$\mathcal{U}$ 为控制输入约束集。MPC 的优势在于能在每个时间步重新优化，对动态障碍物具有天然的鲁棒性。

**Chomp（Covariant Hamiltonian Optimization for Motion Planning）**[12]将轨迹参数化为一系列路径点，通过梯度优化同时最小化轨迹长度、平滑性和障碍物代价：

$$
\min_{\xi} \underbrace{\lambda_{\text{smooth}} f_{\text{smooth}}(\xi)}_{\text{平滑项}} + \underbrace{\lambda_{\text{obs}} f_{\text{obs}}(\xi)}_{\text{障碍物项}} + \underbrace{\lambda_{\text{length}} f_{\text{length}}(\xi)}_{\text{长度项}}
$$

其中 $\xi$ 表示轨迹参数向量。

---

## 8.3 语义导航

传统导航假设环境完全已知或可感知，但真实场景中，智能体常常需要在部分可观测的环境中根据高层语义指令进行导航。语义导航是近年来具身AI研究的热点。

### 8.3.1 目标导航（ObjectNav）

目标导航（ObjectNav）[13]要求智能体在未见过的环境中找到指定类别的物体（如"找到沙发"）。这是一项极具挑战性的任务，因为智能体必须具备：

1. **物体先验知识**：知道目标物体的外观和常见放置位置；
2. **探索策略**：在信息不完整时决定去哪里探索；
3. **状态估计**：跟踪已探索区域，避免重复搜索。

ObjectNav 的形式化定义为：给定目标物体类别 $c$，智能体在每个时间步 $t$ 观测 $o_t$（RGB-D图像），选择动作 $a_t \in \{\text{前进}, \text{左转}, \text{右转}, \text{向上看}, \text{向下看}, \text{停止}\}$，目标是在最少步骤内到达目标物体附近（通常要求距物体 1 米以内且可见）。

成功导航的关键在于构建有效的场景表示。SemExp[14]维护一个语义地图 $\mathcal{M}_{\text{sem}}$，在每个时间步将观测的语义分割结果投影到地图上，然后训练一个目标导向的策略 $\pi(a_t \mid s_t, c)$，该策略以当前语义地图和目标类别为输入，输出下一步动作。

近年来，基于大规模预训练的导航方法取得了显著进展。CoW（Clip on Wheels）[15]利用 CLIP 模型将"目标类别"编码为文本嵌入，在导航过程中计算观测图像与目标的视觉-语言相似度，从而在零样本（zero-shot）设置下实现目标导航，无需针对特定物体类别进行训练。

### 8.3.2 视觉语言导航（VLN）

视觉语言导航（Vision-Language Navigation, VLN）[16]要求智能体根据自然语言指令在环境中导航。例如，指令可能是："走出卧室，左转进入走廊，经过楼梯后右转进入厨房。"

VLN 的形式化建模通常基于离散的导航图（Navigation Graph）$\mathcal{G} = (V, E)$，每个节点 $v_i$ 对应一个全景视点，边 $v_i \to v_j$ 表示可通行路径。在每个视点，智能体获得全景图像集合 $I_i = \{I_i^1, \ldots, I_i^K\}$。

早期的序列到序列方法（Seq2Seq）[16]将指令 $W = (w_1, \ldots, w_L)$ 编码为隐状态，然后在每个导航步骤选择动作：

$$
h_t = \text{LSTM}(h_{t-1}, [a_{t-1}; v_t])
$$

$$
P(a_t \mid h_t, W) = \text{softmax}(f_a(h_t, \text{enc}(W)))
$$

其中 $v_t$ 为当前视觉观测的嵌入，$a_{t-1}$ 为上一步动作的嵌入。

由于专家演示数据稀缺，环境增强（Environmental Augmentation）和预训练成为提升 VLN 性能的关键。EnvDrop[17]通过在导航图上随机丢弃节点和边来增强训练数据。VLN-BERT[18]引入大规模视觉-语言预训练，将导航过程建模为视觉-文本匹配问题。REVERIE[19]进一步将 VLN 与远程操作相结合，要求智能体在目标房间执行交互动作（如"拿起桌上的杯子"），推动了导航与操作的统一研究。

### 8.3.3 语义导航中的知识迁移

一个核心挑战是**泛化到未见环境**。训练环境与测试环境之间可能存在显著的视觉外观差异（不同户型、装修风格、光照条件）。为此，研究者提出了多种知识迁移策略：

- **领域随机化**（Domain Randomization）：在训练时随机改变纹理、光照等视觉属性，迫使模型学习与外观无关的特征。
- **拓扑迁移**：先在源环境学习导航策略，再在目标环境通过拓扑匹配迁移高层决策逻辑。
- **大模型先验**：利用 GPT-4 等大语言模型对指令进行分解和推理，提供常识性导航先验。例如，SayCan[20]将 LLM 的语言推理能力与机器人的物理能力对齐，实现"说做就做"。

---

## 8.4 视觉导航

视觉导航（Visual Navigation）仅依赖摄像头获取的信息进行定位与导航，是成本最低、适用范围最广的导航方式。本节涵盖视觉SLAM、视觉重定位、拓扑导航和NeRF导航。

### 8.4.1 视觉SLAM

视觉同步定位与地图构建（Visual SLAM, Simultaneous Localization and Mapping）是视觉导航的基石。经典框架 ORB-SLAM[21]通过提取ORB特征实现鲁棒的帧间匹配，维护一组关键帧和稀疏三维点地图，在后端通过Bundle Adjustment（BA）联合优化所有关键帧位姿和地图点坐标。

BA 的优化目标为最小化重投影误差：

$$
\min_{\{T_i\}, \{p_j\}} \sum_{i,j} \rho\left( \| \pi(T_i, p_j) - u_{ij} \|_\Sigma^2 \right)
$$

其中 $T_i \in SE(3)$ 为第 $i$ 帧的位姿，$p_j \in \mathbb{R}^3$ 为第 $j$ 个地图点的三维坐标，$\pi$ 为相机投影函数，$u_{ij}$ 为观测到的二维特征点坐标，$\Sigma$ 为协方差矩阵，$\rho$ 为鲁棒核函数（如Huber核）。

近年来，基于深度学习的直接法SLAM也取得了重要突破。DROID-SLAM[22]将光流估计与深度学习相结合，通过可微分的Bundle Adjustment层实现了高精度的位姿估计和稠密深度预测，在室内外多种场景下表现出色。

### 8.4.2 视觉重定位

视觉重定位（Visual Relocalization）解决的是"我在地图中的哪里"这一问题。给定一幅查询图像 $I_q$，重定位的目标是估计其在已知地图中的6DoF位姿。

PoseNet[23]开创性地提出用卷积神经网络直接回归相机位姿：

$$
[\hat{t}, \hat{q}] = f_{\text{PoseNet}}(I_q; \theta)
$$

其中 $\hat{t}$ 为平移向量，$\hat{q}$ 为四元数表示的旋转。然而，直接回归的精度有限。后续工作如 MS-Transformer[24]将问题转化为特征匹配：提取查询图像的局部特征，与数据库中的特征进行匹配，然后通过 PnP（Perspective-n-Point）算法求解位姿。

PnP 问题的核心是求解以下方程组：

$$
s \cdot u = K [R | t] \begin{bmatrix} p \\ 1 \end{bmatrix}
$$

给定至少 3 对 2D-3D 对应点即可求解。实际中使用 EPnP 或 UPnP 等高效求解器处理更多对应点以提高鲁棒性。

### 8.4.3 拓扑视觉导航

拓扑视觉导航将视觉位置识别与图搜索相结合，是大尺度环境中最实用的导航范式。其核心流程为：

1. **地图构建阶段**：沿训练路径采集图像序列，构建拓扑图。每个节点存储一张或多张参考图像的描述向量。
2. **导航阶段**：在目标图中指定目标节点，智能体在每个时间步：(a) 拍摄当前图像并提取描述子；(b) 在拓扑图中定位当前位置；(c) 使用最短路径算法确定下一步方向。

Hop关系是拓扑导航中的重要概念。两个节点之间如果存在可达的导航路径，则称它们之间存在 Hop 关系。MultiHop[25]引入了 Hop 注意力机制，使模型能够理解多步导航指令中的空间关系（如"经过两个房间后左转"）。

### 8.4.4 NeRF 导航

神经辐射场（NeRF, Neural Radiance Fields）[26]为场景表示开辟了全新的范式。NeRF 使用一个连续函数 $F_\theta: (x, d) \rightarrow (c, \sigma)$ 将三维坐标 $x \in \mathbb{R}^3$ 和观察方向 $d \in \mathbb{S}^2$ 映射为颜色 $c \in \mathbb{R}^3$ 和体积密度 $\sigma \in \mathbb{R}^+$。渲染图像的颜色通过体渲染积分计算：

$$
C(r) = \int_{t_n}^{t_f} T(t) \cdot \sigma(r(t)) \cdot c(r(t), d) \, dt
$$

其中 $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(r(s)) ds\right)$ 为累积透射率，$r(t) = o + t \cdot d$ 为光线。

NeRF 在导航中的应用主要有两个方向：

**1. 用于数据增强**：iNeRF[27]和NeRF-DFF[28]利用 NeRF 进行视角合成，生成训练导航策略所需的新视角图像，缓解数据稀缺问题。智能体可以在 NeRF 渲染的虚拟环境中进行大量模拟训练，然后迁移到真实世界。

**2. 用于场景表示与规划**：NeRF 提供了连续的三维场景表示，支持碰撞检测和自由空间查询。给定 NeRF 模型，可以提取占据场（Occupancy Field）：

$$
\mathcal{O}(x) = 1 - \exp\left(-\int \sigma(x) \, dx \right)
$$

基于占据场，可以使用 RRT* 等算法在连续空间中进行路径规划。NeRF-Nav[29]等方法将 NeRF 的可微渲染特性与导航策略相结合，实现了端到端的视觉导航训练。

更进一步，3D Gaussian Splatting（3DGS）[30]使用一组带有颜色和不透明度的三维高斯体来表示场景，渲染速度比 NeRF 快数个数量级，为实时导航场景表示提供了更实用的替代方案。

---

## 8.5 本章小结

自主导航是具身智能从"思考"走向"行动"的桥梁。本章从四个维度梳理了导航技术的核心脉络：

- **地图构建**：从占据栅格到点云、语义地图再到拓扑地图，每种表示都在精度、效率和语义丰富度之间做出不同权衡。
- **路径规划**：从A*到RRT*/PRM的采样规划，再到考虑动力学约束的运动规划，算法的选择取决于问题维度和实时性要求。
- **语义导航**：ObjectNav 和 VLN 将导航从低层运动提升到语义理解层面，大语言模型的引入正在重塑这一范式。
- **视觉导航**：从经典SLAM到NeRF/3DGS场景表示，视觉导航正在从"看到什么"走向"理解什么"。

展望未来，导航技术将朝着**多模态融合**（视觉+语言+触觉+力觉）、**大模型驱动**（LLM/VLM作为导航决策的高层推理引擎）和**终身学习**（智能体在部署过程中持续更新地图和策略）的方向演进。最终目标是让智能体像人类一样——走进一个从未到过的空间，仅凭常识和直觉，就能找到任何东西、到达任何地方。

---

## 参考文献

[1] Elfes A. Using occupancy grids for mobile robot perception and navigation. *IEEE Computer*, 22(6):46-57, 1989.

[2] Hornung A, Wurm K M, Bennewitz M, et al. OctoMap: An efficient probabilistic 3D mapping framework based on octrees. *Autonomous Robots*, 34(3):189-206, 2013.

[3] Besl P J, McKay N D. A method for registration of 3-D shapes. *IEEE TPAMI*, 14(2):239-256, 1992.

[4] Whelan T, Salas-Moreno R F, Glocker B, et al. ElasticFusion: Real-time dense SLAM and light source estimation. *IJRR*, 35(14):1697-1716, 2016.

[5] ConceptFusion: Open-set multimodal 3D mapping. *RSS*, 2023.

[6] Arandjelović R, Gronat P, Torii A, et al. NetVLAD: CNN architecture for weakly supervised place recognition. *CVPR*, 2016.

[7] Hart P E, Nilsson N J, Raphael B. A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2):100-107, 1968.

[8] LaValle S M. Rapidly-exploring random trees: A new tool for path planning. *Technical Report*, Iowa State University, 1998.

[9] Karaman S, Frazzoli E. Sampling-based algorithms for optimal motion planning. *IJRR*, 30(7):846-894, 2011.

[10] Gammell J D, Srinivasa S S, Barfoot T D. Informed RRT*: Optimal sampling-based path planning focused via direct sampling of an admissible ellipsoidal heuristic. *IROS*, 2014.

[11] Kavraki L E, Švestka P, Latombe J C, et al. Probabilistic roadmaps for path planning in high-dimensional configuration spaces. *IEEE TRA*, 12(4):566-580, 1996.

[12] Zucker M, Ratliff N, Dragan A D, et al. CHOMP: Covariant Hamiltonian optimization for motion planning. *IJRR*, 32(9-10):1164-1193, 2013.

[13] Batra D, Chang A X, Savva M, et al. ObjectNav revisited: On evaluation of embodied agents navigating to objects. *ECCV*, 2020.

[14] Chaplot D S, Gandhi D, Gupta S, et al. Object goal navigation using goal-oriented semantic exploration. *NeurIPS*, 2020.

[15] Dorbala V S, Sigurdsson G, Piramuthu R, et al. CLIP-Nav: Using CLIP for zero-shot vision-and-language navigation. *arXiv preprint arXiv:2211.16649*, 2022.

[16] Anderson P, Wu Q, Teney D, et al. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. *CVPR*, 2018.

[17] Tan H, Yu L, Bansal M. Learning to navigate unseen environments: Back translation with environmental dropout. *NAACL*, 2019.

[18] Hong Y, Wu Q, Qi Y, et al. VLN-BERT: A recurrent vision-and-language BERT for navigation. *CVPR*, 2021.

[19] Qi Y, Wu Q, Anderson P, et al. REVERIE: Remote embodied visual referring expression in real indoor environments. *CVPR*, 2020.

[20] Ahn M, Brohan A, Brown N, et al. Do as I can, not as I say: Grounding language in robotic affordances. *CoRL*, 2022.

[21] Mur-Artal R, Montiel J M M, Tardós J D. ORB-SLAM: A versatile and accurate monocular SLAM system. *IEEE TRA*, 31(5):1147-1163, 2015.

[22] Teed Z, Deng J. DROID-SLAM: Deep visual SLAM for monocular, stereo, and RGB-D cameras. *NeurIPS*, 2021.

[23] Kendall A, Grimes M, Cipolla R. PoseNet: A convolutional network for real-time 6-DOF camera relocalization. *ICCV*, 2015.

[24] Sarlin P E, DeTone D, Malisiewicz T, et al. SuperGlue: Learning feature matching with graph neural networks. *CVPR*, 2020.

[25] Shridhar M, Manuelli L, Fox D. CLIPort: What and where pathways for robotic manipulation. *CoRL*, 2022.

[26] Mildenhall B, Srinivasan P P, Tancik M, et al. NeRF: Representing scenes as neural radiance fields for view synthesis. *ECCV*, 2020.

[27] Yen-Chen L, Florence P, Barron J T, et al. iNeRF: Inverting neural radiance fields for pose estimation. *IROS*, 2021.

[28] Adamkiewicz M, Chen T, Caccavale A, et al. Vision-only robot navigation in a neural radiance world. *RA-L*, 2022.

[29] Zhu F, Lv Y, Chen Y, et al. NeRF-Nav: Neural radiance fields for robot navigation. *ICRA*, 2023.

[30] Kerbl B, Kopanas G, Leimkühler T, et al. 3D Gaussian splatting for real-time radiance field rendering. *ACM TOG (SIGGRAPH)*, 2023.
# 第9章 四足与人形机器人

> "让机器人像动物一样奔跑，像人类一样行走——这是具身智能最直观、最震撼的表达。"

## 9.1 引言

在具身智能的研究版图中，腿足机器人（Legged Robots）占据着独特而核心的地位。与轮式或履带式机器人不同，腿足机器人通过离散的支撑点与地面交互，天然具备跨越复杂地形、适应非结构化环境的能力。从四足机器人在崎岖山地中的稳健行进，到人形机器人在家庭场景中的灵活操作，腿足运动控制一直是机器人学与人工智能交叉领域中最具挑战性的课题之一。

本章聚焦两大类腿足机器人——四足机器人与人形机器人，系统阐述其运动控制的核心理论、前沿算法与工程实践。我们首先讨论四足运动控制的基本框架，包括步态规划的经典方法与基于强化学习的现代范式；随后转向人形机器人，探讨其特有的全身运动规划与控制挑战；最后深入步态学习中的仿真到真实迁移（Sim-to-Real）问题以及全身控制中的平衡与力控交互技术。

## 9.2 四足运动控制

### 9.2.1 步态规划的数学基础

步态（Gait）是腿足机器人运动的基本模式，描述了各腿在时间序列中的支撑与摆动相位关系。对于四足机器人，一个完整的步态周期 $T$ 可表示为各腿时序的组合：

$$\mathcal{G} = \{(t_i^{\text{swing}}, t_i^{\text{stance}}, \phi_i) \mid i = 1, 2, 3, 4\}$$

其中 $t_i^{\text{swing}}$ 和 $t_i^{\text{stance}}$ 分别为第 $i$ 条腿的摆动相与支撑相持续时间，$\phi_i \in [0, 1)$ 为该腿相对于参考腿的相位偏移。

**常见步态模式：**

- **爬行步态（Walk/Trot）**：对角腿同步运动，$\phi_1 = \phi_4 = 0$，$\phi_2 = \phi_3 = 0.5$，占空比（Duty Cycle）$d = t^{\text{stance}} / T \approx 0.5$。Trot步态是四足机器人最常用的行走模式，始终保证至少两条腿处于支撑状态，提供静态或近静态稳定性。

- **对角小跑步态（Trot）**：对角腿同步抬起与落地，$d \approx 0.5$，适合中速运动。

- **飞驰步态（Gallop）**：$d < 0.5$，存在四腿同时离地的腾空相，适合高速奔跑。

- **踱步步态（Pace）**：同侧腿同步运动，$\phi_1 = \phi_2 = 0$，$\phi_3 = \phi_4 = 0.5$。

**零力矩点（ZMP）判据：**

在准静态运动中，机器人保持平衡的充要条件是零力矩点（ZMP）位于支撑多边形（Support Polygon）内部。ZMP的计算公式为：

$$x_{\text{ZMP}} = \frac{\sum_{i} m_i (\ddot{z}_i + g) x_i - \sum_i m_i \ddot{x}_i z_i}{\sum_i m_i (\ddot{z}_i + g)}$$

$$y_{\text{ZMP}} = \frac{\sum_{i} m_i (\ddot{z}_i + g) y_i - \sum_i m_i \ddot{y}_i z_i}{\sum_i m_i (\ddot{z}_i + g)}$$

其中 $(x_i, y_i, z_i)$ 为第 $i$ 个质点的位置，$m_i$ 为其质量，$g$ 为重力加速度。当机器人处于准静态运动时（加速度项可忽略），ZMP退化为重心在地面的投影（CoM投影），稳定性判据简化为：

$$\mathbf{p}_{\text{CoM}}^{xy} \in \mathcal{S}$$

其中 $\mathcal{S}$ 为当前支撑多边形。

**中枢模式发生器（CPG）：**

受生物神经系统启发，中枢模式发生器（Central Pattern Generator, CPG）是一种广泛用于步态生成的数学模型。基于非线性振荡器的CPG模型可写为耦合的微分方程系统：

$$\dot{r}_i = \alpha_i (R_i - r_i)$$
$$\dot{\theta}_i = 2\pi f_i + \sum_{j} w_{ij} r_j \sin(\theta_j - \theta_i - \varphi_{ij})$$

其中 $r_i$ 和 $\theta_i$ 分别为第 $i$ 个振荡器的幅值与相位，$f_i$ 为目标频率，$w_{ij}$ 和 $\varphi_{ij}$ 为振荡器之间的耦合权重与期望相位差。通过调整耦合矩阵 $\{\varphi_{ij}\}$，可以自然地生成不同的步态模式。CPG方法的优势在于其固有的周期性与鲁棒性——面对外部扰动，振荡器能够自发恢复到稳定的极限环（Limit Cycle），从而实现自适应的步态生成。

### 9.2.2 基于模型的运动控制

传统四足运动控制依赖于精确的动力学模型。简化的四足机器人动力学方程可写为标准的拉格朗日形式：

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = \tau + J_c^T F_c$$

其中 $q$ 为广义坐标向量，$M(q)$ 为惯性矩阵，$C(q, \dot{q})$ 为科里奥利力与离心力项，$G(q)$ 为重力项，$\tau$ 为关节力矩，$J_c$ 为接触雅可比矩阵，$F_c$ 为地面反作用力。

在模型预测控制（MPC）框架下，运动控制问题被表述为在线求解的二次规划（QP）：

$$\min_{u_{0:N-1}} \sum_{k=0}^{N-1} \left[ \|x_k - x_k^{\text{ref}}\|_Q^2 + \|u_k\|_R^2 \right] + \|x_N - x_N^{\text{ref}}\|_P^2$$

$$\text{s.t.} \quad x_{k+1} = A_k x_k + B_k u_k + c_k$$
$$\quad\quad\quad F_{\min} \leq F_k \leq F_{\max}$$
$$\quad\quad\quad \mu F_{z,k} \geq \sqrt{F_{x,k}^2 + F_{y,k}^2}$$

其中最后的约束为摩擦锥约束，$\mu$ 为摩擦系数。这种基于线性倒立摆模型（LIPM）和Centroidal动力学的MPC方法，已被广泛应用于四足机器人的实时运动控制中。

### 9.2.3 基于强化学习的运动控制

近年来，深度强化学习（Deep RL）为四足运动控制带来了范式性的变革。与基于模型的方法不同，RL方法直接从交互数据中学习控制策略，无需显式建立动力学模型。

**问题建模：** 四足运动的马尔可夫决策过程（MDP）定义为 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, r, \gamma)$：

- **状态空间** $\mathcal{S}$：包含本体感知信息（关节角度 $q$、角速度 $\dot{q}$、机体姿态四元数 $\mathbf{q}_{\text{body}}$、角速度 $\boldsymbol{\omega}$）以及外部感知信息（地形高度图、目标速度指令等）

- **动作空间** $\mathcal{A}$：关节目标位置或力矩，$\mathbf{a} \in \mathbb{R}^{12}$（四足 × 3自由度）

- **奖励函数** $r$：通常设计为多目标加权和：

$$r = w_v r_{\text{velocity}} + w_\theta r_{\text{orientation}} + w_\tau r_{\text{torque}} + w_e r_{\text{energy}} + w_s r_{\text{smooth}}$$

其中 $r_{\text{velocity}} = \exp(-\|\mathbf{v} - \mathbf{v}_{\text{cmd}}\|^2 / \sigma_v)$ 鼓励跟踪速度指令，$r_{\text{torque}} = -\|\boldsymbol{\tau}\|^2$ 惩罚力矩消耗，$r_{\text{energy}} = -|\boldsymbol{\tau}^T \dot{\mathbf{q}}|$ 惩罚能量消耗。

**PPO算法：** 在四足运动控制中，近端策略优化（PPO）是最常用的RL算法。其策略更新目标为：

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ 为概率比，$\hat{A}_t$ 为广义优势估计（GAE），$\epsilon$ 为裁剪参数。PPO通过裁剪机制限制策略更新幅度，保证训练的稳定性。

**Domain Randomization（域随机化）：** 为实现从仿真到真实的迁移，域随机化是关键技术。在仿真训练时，对以下物理参数进行随机采样：

$$\theta_{\text{env}} \sim \mathcal{U}(\theta_{\text{nom}} - \Delta\theta, \theta_{\text{nom}} + \Delta\theta)$$

其中 $\theta_{\text{env}}$ 包括地面摩擦系数 $\mu$、机体质量 $m$、关节阻尼 $b$、传感器噪声等。通过在足够宽的参数分布上训练，策略能够泛化到真实世界的物理特性。

### 9.2.4 代表性四足机器人平台

**Unitree Go2：** 作为消费级四足机器人的标杆产品，Go2搭载了高性能的12自由度关节驱动系统，单关节峰值扭矩可达45 N·m。其核心创新在于将基于强化学习的运动控制器部署在嵌入式GPU上，实现了在复杂户外地形（草地、碎石、楼梯）上的自主导航。Go2的控制器架构通常采用分层设计：底层RL策略以50-200 Hz频率运行，负责关节级运动控制；上层规划器以较低频率运行，负责路径规划与避障。

**Boston Dynamics Spot：** Spot是工业级四足机器人的代表，具备卓越的运动能力与环境适应性。Spot的控制算法融合了基于模型的MPC与数据驱动的方法：在已知地形上使用精确的动力学模型进行高效规划，在未知或动态环境中则依赖反应式控制器与学习到的鲁棒策略。Spot能够完成包括开门、攀爬楼梯、在狭窄空间中穿行等复杂任务，其平衡控制算法可在受到大力推搡后迅速恢复稳定姿态。

### 9.2.5 高级运动能力

现代四足机器人已展现出超越简单行走的高级运动能力：

**敏捷运动（Agile Locomotion）：** 通过精心设计的奖励函数与课程学习策略，四足机器人可以学会跳跃（Jumping）、后空翻（Backflip）、侧翻（Sideflip）等高动态动作。这些动作涉及短时间内的大力矩输出与精确的空中姿态控制，对控制算法的实时性与准确性提出了极高要求。

**自适应地形穿越：** 通过结合本体感知（关节力矩反馈、IMU数据）与外部感知（深度相机、LiDAR点云），四足机器人可以在线适应不同地形特性。典型方法是将地形高度图编码为潜变量，与本体感知数据一起输入策略网络：

$$\pi_\theta(\mathbf{a} | \mathbf{s}_{\text{proprio}}, \mathbf{z}_{\text{terrain}})$$

其中 $\mathbf{z}_{\text{terrain}} = f_\phi(\mathcal{H})$ 为地形编码器对高度图 $\mathcal{H}$ 的编码。

## 9.3 人形机器人

### 9.3.1 人形机器人的特殊挑战

人形机器人相比四足机器人面临更为严峻的控制挑战，这源于其固有的物理特性：

**高维欠驱动系统：** 典型的人形机器人拥有20-40个主动自由度，但与地面的接触通常仅为脚底的有限面积。以Atlas为例，其全身30个自由度构成了一个高度非线性的欠驱动系统。系统的广义坐标 $q \in \mathbb{R}^{n+6}$（$n$ 为关节数，6为浮动基座自由度），但控制输入 $\tau \in \mathbb{R}^n$ 仅作用于关节，对浮动基座的控制需通过接触力间接实现。

**接触动力学的复杂性：** 人形机器人的运动涉及脚底与地面之间的时变接触。接触切换（Contact Switching）带来了动力学的不连续性：

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = S^T \tau + \sum_{k \in \mathcal{C}} J_k^T F_k$$

$$0 \leq \lambda_k \perp \phi_k(q) \geq 0$$

其中 $\mathcal{C}$ 为当前接触集合，$\lambda_k$ 为接触力的法向分量，$\phi_k(q)$ 为接触点的符号距离函数。互补性条件（Complementarity Condition）意味着：当接触力 $\lambda_k > 0$ 时，接触距离 $\phi_k = 0$（保持接触）；当 $\phi_k > 0$ 时，$\lambda_k = 0$（无接触力）。这种混合动力学特性使得人形机器人的控制问题在数学上极为复杂。

**高重心位置：** 与四足机器人不同，人形机器人的重心通常位于较高位置，这意味着更小的稳定裕度。直立双足行走本质上是一种"受控倒下"（Controlled Falling）——机器人必须持续向前倾倒并及时迈出下一步以避免摔倒。

### 9.3.2 代表性人形机器人平台

**Tesla Optimus（Tesla Bot）：** Optimus的设计理念是面向大规模制造的通用人形机器人。其核心特点包括：28个自由度的全身关节设计，采用高扭矩密度的电机驱动；手部具备11个自由度，可完成精细抓取操作。Optimus的控制策略大量借鉴了四足机器人领域成熟的RL方法，通过在高保真仿真器中进行大规模并行训练（使用Isaac Gym等平台），学习从视觉输入到关节力矩的端到端策略。Tesla的优势在于其庞大的真实数据集——通过采集人类示范数据进行行为克隆（Behavior Cloning），结合强化学习进行策略优化。

**Figure 02：** Figure AI推出的人形机器人Figure 02代表了"通用人形"（General-Purpose Humanoid）的技术方向。其全身控制采用分层架构：高层通过视觉语言模型（VLM）理解任务语义并生成运动目标，中层将目标转化为全身运动轨迹，底层通过全身控制器（Whole-Body Controller）实现精确的关节跟踪。Figure 02的一个重要创新是将大语言模型（LLM）接入机器人控制系统，实现自然语言指令到机器人动作的直接映射，这是具身智能"大脑"与"身体"融合的重要里程碑。

**Boston Dynamics Atlas：** Atlas是液压驱动人形机器人的巅峰之作，以其惊人的运动敏捷性闻名。Atlas能够完成后空翻、跑酷跳跃、在不平整地形上快速行走等高难度动作。其控制算法的核心是基于非线性优化的全身运动规划，结合反应式的平衡控制器。Atlas的最新电动版本（Electric Atlas）转向了全电驱动设计，标志着人形机器人从液压向电驱的技术趋势转变。

### 9.3.3 全身运动规划

人形机器人的全身运动规划（Whole-Body Motion Planning）需要同时满足任务需求、运动学约束、动力学约束和接触约束。其数学表述通常为一个受约束的优化问题：

$$\min_{q_{1:T}, \tau_{1:T}, F_{1:T}} \sum_{t=1}^{T} \left[ w_1 \|q_t - q_t^{\text{task}}\|^2 + w_2 \|\tau_t\|^2 + w_3 \|F_t - F_t^{\text{des}}\|^2 \right]$$

$$\text{s.t.} \quad \text{动力学方程约束}$$
$$\quad\quad\quad \text{摩擦锥约束: } \mu F_{z,t} \geq \|F_{xy,t}\|$$
$$\quad\quad\quad \text{关节限位: } q_{\min} \leq q_t \leq q_{\max}$$
$$\quad\quad\quad \text{力矩限位: } \tau_{\min} \leq \tau_t \leq \tau_{\max}$$
$$\quad\quad\quad \text{接触序列约束: } \mathcal{C}_t \in \Omega_{\text{valid}}$$

**接触序列规划：** 对于双足行走，首先需要规划脚部的接触序列——即哪只脚在何时支撑、何时摆动。这通常通过混合整数规划（MIP）或启发式搜索方法解决。给定接触序列后，每个阶段的运动规划简化为固定接触条件下的轨迹优化问题。

**逆运动学与逆动力学：** 在全身运动规划中，逆运动学（IK）求解将末端执行器的期望位姿映射到关节空间：

$$q^* = \arg\min_q \|J(q) \dot{q} - \dot{x}_{\text{des}}\|^2 + w_q \|q - q_{\text{nom}}\|^2$$

$$\text{s.t.} \quad q_{\min} \leq q \leq q_{\max}$$

其中 $J(q)$ 为任务雅可比矩阵，$q_{\text{nom}}$ 为名义姿态（用于正则化）。当任务数少于自由度时，IK问题有无穷多解，需要通过零空间投影（Null-Space Projection）利用冗余自由度优化次要目标：

$$\dot{q} = J^{\dagger} \dot{x}_{\text{des}} + (I - J^{\dagger} J) \dot{q}_{\text{null}}$$

其中 $J^{\dagger}$ 为伪逆，$(I - J^{\dagger} J)$ 为零空间投影矩阵，$\dot{q}_{\text{null}}$ 为零空间中的优化方向。

### 9.3.4 腿部运动的线性倒立摆模型

在人形行走控制中，线性倒立摆模型（Linear Inverted Pendulum Model, LIPM）是最重要的简化模型。LIPM假设机器人的所有质量集中于质心，腿为无质量的可伸缩连杆，且质心高度恒定。在此假设下，质心动力学退化为线性系统：

$$\ddot{x} = \frac{g}{z_0}(x - p_x)$$
$$\ddot{y} = \frac{g}{z_0}(y - p_y)$$

其中 $(x, y)$ 为质心水平位置，$z_0$ 为质心高度，$(p_x, p_y)$ 为ZMP位置。通过将ZMP限制在支撑多边形内，可以实现稳定行走。

LIPM的解析解可写为双曲函数形式：

$$x(t) = (x_0 - p_x)\cosh\left(\frac{t}{T_c}\right) + T_c \dot{x}_0 \sinh\left(\frac{t}{T_c}\right) + p_x$$

其中 $T_c = \sqrt{z_0/g}$ 为时间常数。这一解析形式使得基于LIPM的步态规划可以通过代数运算高效完成。

## 9.4 步态学习：从仿真到真实

### 9.4.1 仿真到真实迁移的核心挑战

将仿真中学到的运动策略迁移到真实机器人（Sim-to-Real Transfer）是具身智能领域的核心难题之一。这一挑战源于"仿真-真实差距"（Sim-to-Real Gap），主要体现在以下几个方面：

**动力学建模误差：** 仿真器无法完美复现真实物理。关节摩擦、齿轮间隙、柔性效应、地面接触等的建模往往存在系统性偏差。例如，真实关节的摩擦力矩通常具有Stribeck效应（低速时摩擦力随速度非线性变化），而仿真中常采用简化的线性粘性摩擦模型。

**感知噪声差异：** 真实传感器（IMU、编码器、力传感器）的噪声特性与仿真中添加的高斯白噪声存在差异。真实IMU存在温漂、零偏、振动干扰等非理想特性。

**延迟效应：** 仿真中的控制回路可以理想化为无延迟，而真实系统中传感器采集、通信传输、策略推理、电机执行各环节均引入延迟，总延迟可达5-20ms。

### 9.4.2 课程学习策略

课程学习（Curriculum Learning）是提高步态学习效率与鲁棒性的关键技术。其核心思想是将复杂任务分解为一系列由简到难的子任务，逐步增加难度：

**地形课程：** 从平地开始，逐步引入坡度、台阶、离散踏脚石等复杂地形：

$$\mathcal{T}_k = \{ \text{terrain} : \text{difficulty}(\text{terrain}) \leq d_k \}, \quad d_1 < d_2 < \cdots < d_K$$

难度指标 $d$ 可以定义为坡度角、台阶高度、踏脚石间距的函数。当策略在当前难度级别达到成功率阈值 $\eta_{\text{thresh}}$ 时，自动提升难度。

**速度课程：** 从静止站立开始，逐步增加目标速度指令的范围。初期仅训练低速行走，待策略稳定后再扩展到快速奔跑。

**扰动课程：** 逐步增加外部推力或力矩扰动的幅度：

$$\|\mathbf{F}_{\text{ext}}\| \sim \mathcal{U}(0, F_{\max}^{(k)})$$

其中 $F_{\max}^{(k)}$ 随训练进度 $k$ 递增。

**自适应课程（Adaptive Curriculum）：** 基于当前策略性能自适应调整难度。常用的自动课程方法包括：

$$d_{k+1} = d_k + \alpha \cdot (s_k - s_{\text{target}})$$

其中 $s_k$ 为当前成功率，$s_{\text{target}}$ 为目标成功率（通常设为0.7-0.8），$\alpha$ 为调整步长。当策略表现优于目标时提升难度，反之降低难度，维持训练在"最近发展区"内。

### 9.4.3 域随机化与系统辨识

域随机化（Domain Randomization, DR）是弥合Sim-to-Real差距的核心技术。其原理是：如果策略在足够多样的仿真环境中都能成功，那么真实环境只是这些多样环境中的一个样本。

**参数化随机化：** 对仿真环境的物理参数进行随机采样：

| 参数类别 | 典型参数 | 随机化范围 |
|---------|---------|-----------|
| 动力学 | 质量、惯量、摩擦系数 | ±20%~±50% |
| 执行器 | 关节阻尼、弹性、延迟 | ±30%~±100% |
| 传感器 | IMU噪声、编码器精度 | 基于实测噪声模型 |
| 外部 | 推力扰动、地形变化 | 课程递增 |

**自动域随机化（ADR）：** 由OpenAI在灵巧手操作中首次提出，ADR自动调整随机化参数的分布范围。当策略在当前分布下的成功率超过阈值时，自动扩大随机化范围：

$$\text{range}_{\text{low}} \leftarrow \text{range}_{\text{low}} - \delta_{\text{low}}$$
$$\text{range}_{\text{high}} \leftarrow \text{range}_{\text{high}} + \delta_{\text{high}}$$

**系统辨识辅助的域随机化：** 纯随机化可能导致过大的参数范围，降低策略性能。一种更高效的方法是先通过系统辨识（System Identification）估计真实机器人的物理参数，然后在辨识值附近进行较小范围的随机化：

$$\theta_{\text{sim}} \sim \mathcal{N}(\hat{\theta}_{\text{real}}, \Sigma_{\theta})$$

其中 $\hat{\theta}_{\text{real}}$ 为辨识得到的参数估计值，$\Sigma_{\theta}$ 为辨识不确定性。这种方法在保持Sim-to-Real迁移性的同时，最大限度地保留了策略性能。

### 9.4.4 鲁棒性增强

**对抗性训练（Adversarial Training）：** 在训练过程中引入对抗性扰动，迫使策略学习更加鲁棒的行为。具体地，在每一步选择最不利于当前策略的环境参数或外部扰动：

$$\pi^* = \arg\min_\pi \max_{\delta \in \mathcal{D}} \mathbb{E}\left[\sum_t r_t(\pi, \delta)\right]$$

这种极小极大（Min-Max）优化产生对扰动鲁棒的策略。

**故障恢复训练：** 专门设计训练场景让机器人从异常状态（如单腿打滑、被绊倒、外部冲击）中恢复。这些场景在正常运行中可能很少出现，但一旦发生就需要策略能够快速响应。

**多策略集成：** 训练多个在不同条件下表现最优的策略，在运行时根据当前状态选择或融合最合适的策略：

$$\mathbf{a} = \sum_{i=1}^{N} w_i(\mathbf{s}) \cdot \pi_i(\mathbf{a}|\mathbf{s})$$

其中权重 $w_i(\mathbf{s})$ 根据当前状态与各策略训练分布的匹配程度自适应计算。

## 9.5 全身控制

### 9.5.1 平衡控制

平衡控制是全身控制（Whole-Body Control, WBC）的基础，确保机器人在运动过程中保持动态稳定。

**质心动力学控制：** 在全身控制框架中，浮动基座的动力学可分解为质心（CoM）动力学与角动量动力学：

$$m\ddot{\mathbf{p}}_{\text{CoM}} = m\mathbf{g} + \sum_{k \in \mathcal{C}} \mathbf{f}_k$$

$$\dot{\mathbf{L}} = \sum_{k \in \mathcal{C}} (\mathbf{p}_k - \mathbf{p}_{\text{CoM}}) \times \mathbf{f}_k + \boldsymbol{\tau}_{\text{gravity}}$$

其中 $\mathbf{p}_{\text{CoM}}$ 为质心位置，$\mathbf{f}_k$ 为第 $k$ 个接触点的力，$\mathbf{L}$ 为角动量。

**任务空间全身控制器：** 全身控制的核心思想是将多个控制任务（平衡、手臂操作、头部指向等）统一在全身动力学框架下求解。典型的WBC问题表述为：

$$\min_{\ddot{q}, F, \tau} \sum_{i} w_i \| J_i \ddot{q} + \dot{J}_i \dot{q} - \ddot{x}_i^{\text{des}} \|^2 + w_\tau \|\tau\|^2$$

$$\text{s.t.} \quad M(q)\ddot{q} + h(q, \dot{q}) = S^T \tau + J_c^T F$$
$$\quad\quad\quad F_{\min} \leq F \leq F_{\max} \quad (\text{摩擦锥约束})$$

其中 $J_i$ 为第 $i$ 个任务的雅可比矩阵，$w_i$ 为任务优先级权重。这种方法可以在满足动力学约束的同时，同时实现多个控制目标。

**任务优先级与Null-Space控制：** 当多个任务存在冲突时，需要定义任务优先级。高优先级任务（如平衡）通过主雅可比矩阵控制，低优先级任务（如头部指向）在高优先级任务的零空间中实现：

$$\dot{q} = J_1^{\dagger} \dot{x}_1^{\text{des}} + N_1 J_2^{\dagger} (\dot{x}_2^{\text{des}} - J_2 J_1^{\dagger} \dot{x}_1^{\text{des}})$$

其中 $N_1 = I - J_1^{\dagger} J_1$ 为第一任务的零空间投影。这确保了低优先级任务不会干扰高优先级任务的执行。

### 9.5.2 跌倒恢复

跌倒恢复（Fall Recovery）是人形机器人鲁棒运行的关键能力。完整的跌倒处理策略包括三个阶段：

**阶段一：跌倒检测。** 通过监测以下指标实时判断机器人是否处于不可恢复的失衡状态：

$$\text{Fall Detected} = \begin{cases} 1 & \text{if } \mathbf{p}_{\text{ZMP}} \notin \mathcal{S} \text{ and } \|\dot{\mathbf{q}}_{\text{body}}\| > \omega_{\text{thresh}} \\ 0 & \text{otherwise} \end{cases}$$

更精确的检测可基于捕获点（Capture Point）理论。捕获点定义为：

$$\mathbf{p}_{\text{CP}} = \mathbf{p}_{\text{CoM}} + \frac{\dot{\mathbf{p}}_{\text{CoM}}}{\omega_0}$$

其中 $\omega_0 = \sqrt{g/z_0}$。当捕获点超出支撑多边形且无法通过迈步将其纳入时，机器人进入不可恢复状态。

**阶段二：跌倒缓解（Fall Mitigation）。** 在检测到不可避免的跌倒后，控制目标从"维持平衡"切换为"安全着地"。这需要优化全身姿态，使机器人以最安全的方式接触地面：

$$\min_q \sum_{i} \|\mathbf{v}_{\text{contact},i}\|^2 + w_{\text{height}} h_{\text{CoM}} + w_{\text{orient}} \|\mathbf{R}_{\text{body}} - \mathbf{R}_{\text{safe}}\|^2$$

目标是最小化着地速度（减少冲击）、降低质心高度、并将机体朝向调整为安全姿态。对于Atlas等液压机器人，这一过程可以在几百毫秒内完成。

**阶段三：地面恢复（Ground Recovery）。** 机器人着地后，需要从卧姿或跪姿恢复到站立姿态。这涉及复杂的全身运动规划，通常使用基于采样的运动规划器（如CHOMP、TrajOpt）或学习到的恢复策略。地面恢复的关键挑战在于：机器人需要在有限的关节力矩范围内，通过一系列中间姿态，从地面回到站立。

### 9.5.3 力控交互

在人机协作和环境交互任务中，力控（Force Control）是全身控制的核心能力。

**阻抗控制（Impedance Control）：** 阻抗控制将机器人末端执行器的行为建模为"弹簧-阻尼-质量"系统：

$$M_d \ddot{\mathbf{e}} + D_d \dot{\mathbf{e}} + K_d \mathbf{e} = \mathbf{F}_{\text{ext}}$$

其中 $\mathbf{e} = \mathbf{x} - \mathbf{x}_{\text{des}}$ 为位姿误差，$M_d$、$D_d$、$K_d$ 分别为期望的惯性、阻尼和刚度矩阵。通过调整这些参数，可以控制机器人对外力的响应特性——高刚度产生精确的位置跟踪，低刚度产生柔顺的交互行为。

在关节空间中，阻抗控制律为：

$$\tau = \tau_{\text{ff}} + J^T \left[ K_d (\mathbf{x}_{\text{des}} - \mathbf{x}) + D_d (\dot{\mathbf{x}}_{\text{des}} - \dot{\mathbf{x}}) \right]$$

**力/位混合控制：** 对于需要在某些方向上精确控制位置、在其他方向上精确控制力的任务（如推门、擦窗），力/位混合控制是标准方法。沿约束法向方向控制力，沿约束切向方向控制位置：

$$\tau = J^T \left[ S_f (K_f (\mathbf{F}_{\text{des}} - \mathbf{F}) + \mathbf{F}_{\text{des}}) + S_p (K_p (\mathbf{x}_{\text{des}} - \mathbf{x}) + D_p (\dot{\mathbf{x}}_{\text{des}} - \dot{\mathbf{x}})) \right]$$

其中 $S_f$ 和 $S_p = I - S_f$ 为力控与位控的选择矩阵。

**全身力控交互：** 在全身控制框架中，力控交互需要同时考虑多个接触点。例如，人形机器人用双手推墙行走时，需要协调手臂的推力与腿部的支撑力以维持整体平衡。这要求将力控目标纳入全身优化问题：

$$\min_{\ddot{q}, F, \tau} \| J_{\text{hand}} \ddot{q} + \dot{J}_{\text{hand}} \dot{q} - \ddot{x}_{\text{hand}}^{\text{des}} \|^2_{W_{\text{hand}}} + \| F_{\text{hand}} - F_{\text{des}} \|^2_{W_F} + \|\tau\|^2_{W_\tau}$$

$$\text{s.t.} \quad \text{全身动力学约束 + 摩擦锥约束 + 关节限位}$$

### 9.5.4 力矩控制的实现

在实际系统中，全身控制的底层实现依赖于关节力矩控制。现代腿足机器人通常采用两种力矩控制模式：

**基于电流的力矩控制：** 电机电流与输出力矩之间存在近似线性关系 $\tau = k_t I$，其中 $k_t$ 为力矩常数。通过精密的电流控制器，可以实现毫秒级的力矩响应。

**基于关节柔性的力矩控制：** 在串联弹性驱动器（SEA）或准直驱（QDD）结构中，关节输出力矩通过弹性元件（如弹簧或柔性减速器）传递。力矩估计基于弹性变形量：

$$\tau = k_s (\theta_{\text{motor}} / N - \theta_{\text{link}})$$

其中 $k_s$ 为弹性刚度，$N$ 为减速比，$\theta_{\text{motor}}$ 和 $\theta_{\text{link}}$ 分别为电机和连杆角度。这种结构提供了固有的力控能力和冲击吸收能力，对人机交互安全至关重要。

## 9.6 前沿趋势与展望

### 9.6.1 大模型驱动的运动控制

近年来，大语言模型（LLM）和视觉语言模型（VLM）正在深刻改变机器人的控制范式。通过将高层任务规划交给大模型，机器人可以直接从自然语言指令理解任务目标，并生成相应的运动计划。例如：

- **RT-2（Robotics Transformer 2）**：Google DeepMind提出的视觉-语言-动作模型（VLA），将机器人动作表述为文本token，使预训练的视觉语言模型能够直接输出机器人控制命令。

- **SayCan**：将LLM的任务理解能力与机器人可用技能相结合，LLM负责高层规划，底层技能负责执行具体运动。

这种"大模型+运动控制"的架构代表了具身智能从"感知-规划-执行"经典范式向"理解-生成"新范式的转变。

### 9.6.2 基础模型与运动控制的融合

未来的趋势是训练大规模的运动控制基础模型（Foundation Model for Locomotion），它能够：

1. **跨形态迁移**：在四足机器人上训练的运动知识迁移到六足、人形机器人
2. **少样本适应**：通过少量真实交互数据快速适应新机器人平台
3. **多模态感知融合**：统一处理视觉、触觉、本体感知等多模态信息
4. **零样本泛化**：对未见过的地形类型和任务类型具备即插即用的泛化能力

### 9.6.3 仿真技术的突破

高保真、高效率的仿真器是步态学习的基础设施。Isaac Gym、Isaac Sim等GPU加速仿真器的出现，使得在单台GPU上并行运行数千个仿真环境成为可能。这大幅缩短了训练时间——过去需要数天的训练现在可在数小时内完成。未来，物理仿真精度的进一步提升（如基于神经网络的接触模型、基于物理的柔性体仿真）将进一步缩小Sim-to-Real差距。

### 9.6.4 硬件-软件协同设计

新一代腿足机器人越来越多地采用硬件-软件协同设计理念。例如，高扭矩密度电机（如准直驱电机）的出现，使得人形机器人可以获得媲美人类的动态运动能力。同时，嵌入式AI芯片的算力提升使得复杂RL策略可以部署在机器人本地，实现毫秒级的实时推理。

## 9.7 本章小结

本章系统回顾了四足与人形机器人的运动控制技术体系。从经典步态规划与CPG模型出发，我们介绍了基于强化学习的现代运动控制范式及其在Unitree Go2、Boston Dynamics Spot等平台上的应用。对于人形机器人，我们讨论了其特有的高维欠驱动、接触动力学等挑战，以及全身运动规划与逆运动学的求解方法。在Sim-to-Real迁移方面，域随机化、课程学习和系统辨识构成了当前技术栈的核心。最后，全身控制框架下的平衡控制、跌倒恢复和力控交互技术，为机器人在真实物理世界中的安全、鲁棒运行提供了保障。

展望未来，大模型驱动的控制范式、跨形态迁移学习、高保真仿真以及硬件-软件协同设计，将共同推动腿足机器人从实验室走向真实世界，在救援、物流、家庭服务等领域发挥越来越重要的作用。具身智能的终极愿景——让AI真正"行走"在物理世界中——正在逐步成为现实。

---

**参考文献导引**

本章内容基于以下公开研究工作：

1. Raibert, M. H. (1986). *Legged Robots That Balance*. MIT Press. — 腿足机器人平衡控制的经典之作。
2. Hwangbo, J., et al. (2019). "Learning agile and dynamic motor skills for legged robots." *Science Robotics*, 4(26). — 深度RL四足运动控制的里程碑工作。
3. Miki, T., et al. (2022). "Learning robust perceptive locomotion for quadrupedal robots in the wild." *Science Robotics*, 7(62). — 感知驱动的四足运动。
4. Radosavovic, I., et al. (2024). "Humanoid Locomotion as Next Token Prediction." — 人形机器人运动的基础模型方法。
5. Lee, J., et al. (2020). "Learning quadrupedal locomotion over challenging terrain." *Science Robotics*, 5(47). — 自适应地形穿越。
6. Kajita, S., et al. (2003). "Biped walking pattern generation by using preview control of zero-moment point." *ICRA*. — LIPM步态规划经典方法。
7. Di Carlo, J., et al. (2018). "Dynamic locomotion in the MIT Cheetah 3 through convex model-predicted control." *IROS*. — 基于MPC的四足运动控制。
8. Tobin, J., et al. (2017). "Domain randomization for transferring deep neural networks from simulation to the real world." *IROS*. — 域随机化的奠基工作。
9. Rudin, N., et al. (2022). "Learning to walk in minutes using massively parallel deep reinforcement learning." *CoRL*. — 大规模并行RL训练。
10. Kuindersma, S., et al. (2016). "Optimization-based locomotion planning, estimation, and control design for the Atlas humanoid robot." *Autonomous Robots*. — Atlas全身控制。
# 第10章 自动驾驶

> 自动驾驶是具身智能最具代表性的落地场景之一。它要求智能体在高速、动态、开放的物理环境中，仅凭有限的传感器观测，在毫秒级时间内做出关乎安全的决策。本章从感知、决策规划、端到端学习和仿真测试四个维度，系统梳理自动驾驶技术的核心框架与前沿进展。

## 10.1 概述

自动驾驶系统通常被组织为**感知—预测—规划—控制**的级联流水线（modular pipeline）。感知模块从原始传感器数据中提取环境的结构化表示；预测模块对周围交通参与者的未来行为进行推断；规划模块据此生成安全且舒适的行驶轨迹；控制模块将轨迹转化为方向盘、油门和刹车的执行信号。

设车辆在时刻 $t$ 的状态为 $\mathbf{x}_t \in \mathbb{R}^n$，传感器观测为 $\mathbf{o}_t$，自动驾驶的总体目标可以表述为：

$$\pi^* = \arg\min_{\pi} \; \mathbb{E}\left[\sum_{t=0}^{T} \mathcal{L}\big(\mathbf{x}_t, \pi(\mathbf{o}_{\leq t})\big)\right]$$

其中 $\pi$ 为驾驶策略，$\mathcal{L}$ 为综合损失函数，包含安全性、舒适性、效率和交规遵守等多目标约束。这一优化问题的复杂性在于：状态空间连续且高维、观测存在噪声与遮挡、其他智能体的行为不可控。

近年来，随着深度学习在三维视觉和序列决策中的突破，自动驾驶从"基于规则"向"数据驱动"范式加速迁移。本章将围绕这一主线展开讨论。

## 10.2 感知系统

感知是自动驾驶的"眼睛"，其任务是从传感器原始数据中恢复出三维世界中物体的位置、形状、类别和运动状态。主流传感器包括摄像头（提供高分辨率的语义信息）、激光雷达LiDAR（提供精确的三维点云）和毫米波雷达（提供速度信息）。不同模态的互补性使得多传感器融合成为核心课题。

### 10.2.1 三维目标检测

三维目标检测的任务是：给定传感器输入，输出场景中所有感兴趣物体的三维包围框 $\mathbf{b}_i = (x, y, z, l, w, h, \theta, c)$，其中 $(x, y, z)$ 为中心坐标，$(l, w, h)$ 为尺寸，$\theta$ 为朝向角，$c$ 为类别。

**基于LiDAR的方法。** PointNet（Qi等, 2017）首次实现了在原始点云上的端到端学习，通过逐点MLP和对称函数（最大池化）提取全局特征：

$$\mathbf{f}_{\text{global}} = \max_{i=1,\ldots,N} \; h_\theta(\mathbf{p}_i)$$

其中 $\mathbf{p}_i \in \mathbb{R}^3$ 为点坐标，$h_\theta$ 为可学习的MLP。VoxelNet（Zhou等, 2018）将点云体素化后使用三维卷积提取特征，SECOND（Yan等, 2018）引入稀疏卷积大幅提升效率。PointPillars（Lang等, 2019）将点云沿 $z$ 轴投影为"柱体"（pillars），在鸟瞰图（BEV）上进行二维卷积，实现了精度与速度的良好平衡。

**基于摄像头的方法。** 纯视觉方案的成本远低于LiDAR，但需要从二维图像推断三维信息，这是一个本质不适定问题。早期方法如CenterNet（Zhou等, 2019）在图像坐标系下检测并回归三维框。DETR3D（Wang等, 2022）引入可变形注意力机制，将三维参考点投影到多视角图像上采样特征，实现了端到端的多视角三维检测。

### 10.2.2 BEV感知

鸟瞰图（Bird's-Eye View, BEV）表征已成为现代自动驾驶感知的统一范式。其核心思想是将多视角图像或多模态数据统一投影到地面坐标系下的BEV网格中，从而在统一的空间中进行后续任务。

**LSS（Lift, Splat, Shoot）。** LSS（Philion & Fidler, 2020）提出了经典的图像到BEV的投影框架。对图像上每个像素 $(u, v)$，首先预测其深度分布 $d \sim p(d | u, v)$，然后将图像特征沿相机射线"提升"到三维空间：

$$\mathbf{f}_{3D}(u, v, d) = \mathbf{f}_{2D}(u, v) \otimes p(d | u, v)$$

随后通过外参矩阵将三维特征"拍"（splat）到BEV网格中，得到BEV特征图 $\mathbf{F}_{\text{BEV}} \in \mathbb{R}^{H \times W \times C}$。

**BEVFormer。** BEVFormer（Li等, 2022）采用Transformer架构，通过可变形注意力在时间序列和多视角图像上查询BEV特征。其空间交叉注意力机制为：

$$\text{SCA}(\mathbf{q}_p, \mathbf{F}_{\text{img}}) = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \text{DeformAttn}(\mathbf{q}_p, \text{proj}(p, v), \mathbf{F}_{\text{img}}^v)$$

其中 $\mathbf{q}_p$ 是BEV网格点 $p$ 的查询向量，$\mathcal{V}$ 为所有相机视角集合，$\text{proj}(p, v)$ 将BEV点投影到视角 $v$ 的图像坐标。时序自注意力则融合历史BEV特征以捕获运动信息。

### 10.2.3 多传感器融合

多传感器融合的挑战在于不同模态数据在表示形式、坐标系、采样率和噪声特性上的差异。根据融合发生的阶段，可分为：

- **前融合（Early Fusion）**：在原始数据层面对齐与融合，如点云与图像的逐点着色。
- **特征级融合（Feature-level Fusion）**：在深度特征空间中融合，如PointPainting（Vora等, 2020）将图像语义分割结果"涂"到点云上。
- **后融合（Late Fusion）**：各模态独立检测后在结果层面融合，通过NMS或投票机制合并。

BEVFusion（Liu等, 2023）是特征级融合的代表性工作。它将LiDAR点云和多视角图像分别编码后投影到统一的BEV空间：

$$\mathbf{F}_{\text{BEV}} = \mathbf{F}_{\text{BEV}}^{\text{LiDAR}} \oplus \mathbf{F}_{\text{BEV}}^{\text{Camera}}$$

其中 $\oplus$ 为拼接或加权融合操作。实验表明，BEV融合方案在nuScenes数据集上显著优于单模态方法。

### 10.2.4 占用网络

传统三维检测仅输出有限类别的物体框，难以处理形状不规则的障碍物（如掉落的货物、施工路障）。**占用网络（Occupancy Network）** 提出了更通用的表示：将三维空间离散化为体素网格，预测每个体素的占用状态和语义类别。

设体素网格为 $\mathcal{V} \in \{0, 1\}^{X \times Y \times Z}$，占用网络学习映射：

$$f: (\mathbf{o}_t) \rightarrow \mathcal{V}_t, \; \mathcal{S}_t$$

其中 $\mathcal{S}_t \in \{0, 1, \ldots, K\}^{X \times Y \times Z}$ 为语义标签。SurroundOcc（Wei等, 2023）和OpenOccupancy（Xia等, 2023）在nuScenes上构建了大规模占用标注基准。FlashOcc（Yu等, 2023）进一步提出了高效的二维形式占用预测，通过高度压缩降低计算开销。

占用网络的优势在于其**开放性**——不需要预设物体类别，能够以统一框架感知任意形状的障碍物，这对处理长尾场景至关重要。

## 10.3 决策规划

感知系统提供了环境的"当前快照"，而决策规划需要在此基础上推演未来、做出选择并生成可执行的轨迹。

### 10.3.1 行为预测

行为预测的目标是：给定周围智能体的历史轨迹 $\tau_{i}^{1:t}$ 和高精地图 $\mathcal{M}$，预测其未来轨迹分布：

$$P(\tau_i^{t+1:t+H} | \tau_i^{1:t}, \mathcal{M}, \{\tau_j^{1:t}\}_{j \neq i})$$

其中 $H$ 为预测时域（通常3-8秒）。由于交通场景的多模态性（一个车辆在路口可能直行、左转或右转），预测结果需要表达为多模态分布。高精度的行为预测不仅需要理解个体的运动规律，更需要建模多智能体之间的社会交互——一个驾驶员的决策往往取决于周围车辆的行为意图。

**VectorNet（Gao等, 2020）** 将地图元素和轨迹统一编码为向量（polyline），通过图神经网络建模交互关系。**TNT（Zhao等, 2020）** 采用"先采样目标点，再生成轨迹"的两阶段策略。**MultiPath++（Varadarajan等, 2022）** 引入anchor-based轨迹预测，以预定义的轨迹模板作为先验。

**Wayformer（Nayakanti等, 2023）** 和 **MTR（Shi等, 2022）** 进一步采用Transformer架构，通过注意力机制统一建模agent-agent和agent-map交互。MTR将运动意图建模为以一组可学习的运动查询（motion query）为条件的轨迹生成过程：

$$\hat{\tau}_k = \text{Decoder}(\mathbf{q}_k, \mathbf{F}_{\text{scene}}), \quad k = 1, \ldots, K$$

其中 $\mathbf{q}_k$ 对应第 $k$ 个运动意图模式，$\mathbf{F}_{\text{scene}}$ 为场景编码。

### 10.3.2 轨迹规划

轨迹规划在给定起始状态和目标约束下，生成一条连接当前位姿到目标区域的时空轨迹 $\xi(t) = (x(t), y(t), \theta(t), v(t))$。

**Frenet坐标系下的优化。** 经典的Frenet框架将轨迹规划分解为沿道路参考线的纵向（$s$）和垂直于道路的横向（$d$）两个独立子问题：

$$\min_{d(s), s(t)} \int_0^{s_f} \left[ w_1 \kappa^2(s) + w_2 \dot{d}^2(s) + w_3 d^2(s) \right] ds + \int_0^T J_t \, dt$$

其中 $\kappa$ 为曲率，$J_t$ 为纵向加加速度（jerk）。这一分解显著降低了搜索空间的维度。

**Lattice Planner。** 基于晶格的规划器在 $(s, d)$ 空间中采样候选终点状态，通过五次多项式连接生成候选轨迹族，再以代价函数评估选择最优轨迹。Apollo开源平台广泛采用了这一框架。

**优化式规划。** 将轨迹规划表述为非线性优化问题：

$$\min_{\boldsymbol{\xi}} \; J_{\text{smooth}}(\boldsymbol{\xi}) + \lambda_1 J_{\text{safe}}(\boldsymbol{\xi}) + \lambda_2 J_{\text{goal}}(\boldsymbol{\xi}) + \lambda_3 J_{\text{comfort}}(\boldsymbol{\xi})$$

$$\text{s.t.} \quad \boldsymbol{\xi}_{\min} \leq \boldsymbol{\xi} \leq \boldsymbol{\xi}_{\max}, \quad g(\boldsymbol{\xi}) \leq 0$$

其中约束 $g(\boldsymbol{\xi}) \leq 0$ 编码了避障、车道边界和动力学可行性。

### 10.3.3 运动规划

运动规划更关注底层的几何可行性和动力学约束。在自动驾驶中，运动规划通常在轨迹规划的输出基础上进行精细化。

**Model Predictive Control (MPC)。** MPC在线求解有限时域最优控制问题：

$$\min_{\mathbf{u}_{0:H-1}} \sum_{k=0}^{H-1} \ell(\mathbf{x}_k, \mathbf{u}_k) + V_f(\mathbf{x}_H)$$

$$\text{s.t.} \quad \mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k), \quad \mathbf{x}_k \in \mathcal{X}_{\text{free}}, \quad \mathbf{u}_k \in \mathcal{U}$$

其中 $f$ 为车辆动力学模型（如自行车模型），$\mathcal{X}_{\text{free}}$ 为无障碍空间，$\mathcal{U}$ 为控制输入约束。MPC的优势在于能自然地处理约束并在线适应环境变化。

**混合A*搜索。** 对于低速场景（如泊车），混合A*在连续状态空间中进行搜索，其代价函数考虑了转向角变化、与障碍物的距离和到目标的启发式估计。该方法保证了路径的运动学可行性。

### 10.3.4 决策树与场景决策

在复杂的交通场景中（如无信号灯路口），规划器需要在离散的高层决策（让行、超车、变道）之间进行选择。**决策树**和**部分可观测马尔可夫决策过程（POMDP）** 提供了形式化框架。

在POMDP框架下，自动驾驶决策可建模为元组 $(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, O, R, \gamma)$：

- $\mathcal{S}$：环境状态空间（包括其他车辆的意图，不可直接观测）
- $\mathcal{A}$：自车动作空间（直行、变道、减速等）
- $\mathcal{O}$：观测空间
- $T(s'|s,a)$：状态转移函数
- $O(o|s',a)$：观测函数
- $R(s,a)$：奖励函数

由于状态空间巨大，精确求解POMDP不可行。近似方法包括蒙特卡洛树搜索（MCTS）和基于点的值迭代（PBVI）。POMDP-Autoware等框架将POMDP与规则系统相结合，在保证安全的前提下处理不确定性。

## 10.4 端到端驾驶

传统的模块化方案存在信息逐级损失和模块间误差累积的问题。端到端（End-to-End）驾驶试图用单一的可微模型直接从传感器输入映射到驾驶动作或规划轨迹。

### 10.4.1 UniAD

**UniAD（Unified Autonomous Driving）**（Hu等, 2023）是感知-预测-规划一体化的里程碑工作。它在一个统一的Transformer架构中联合优化检测、跟踪、在线建图、运动预测和规划五个子任务。

UniAD的核心设计是**查询传递机制**：感知阶段产生的物体查询（track queries）直接传递给预测模块，避免了后处理带来的信息损失。其规划损失函数联合考虑了多任务：

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{det}} + \lambda_2 \mathcal{L}_{\text{track}} + \lambda_3 \mathcal{L}_{\text{map}} + \lambda_4 \mathcal{L}_{\text{pred}} + \lambda_5 \mathcal{L}_{\text{plan}}$$

在nuScenes基准上，UniAD的规划性能显著超越了各模块独立训练再组合的基线，验证了联合优化的增益。

### 10.4.2 NEAT

**NEAT（Neural Attention Fields for Planning）**（Chitta等, 2023）提出了神经注意力场的概念，将BEV特征与可学习的注意力分布结合，直接在BEV空间中进行规划。其核心思想是让规划模块"关注"对决策最重要的区域：

$$\mathbf{A}(x, y) = \text{MLP}\big(\mathbf{F}_{\text{BEV}}(x, y), \; \mathbf{g}\big)$$

其中 $\mathbf{g}$ 为目标条件向量。注意力场 $\mathbf{A}$ 在BEV空间中生成热力图，引导规划器避开高风险区域。

### 10.4.3 纯视觉方案

纯视觉端到端驾驶因传感器成本低而备受关注。**TransFuser（Chitta等, 2022）** 将多视角图像通过Transformer融合后直接输出控制信号。**InterFuser（Shao等, 2023）** 进一步引入可解释的中间表示，在端到端训练的同时提供安全监控。

**CARLA排行榜上的纯视觉方案**不断刷新记录。这些方法通常采用以下架构：

1. 多视角图像编码器（ResNet/ViT）
2. BEV特征转换（基于LSS或Cross-Attention）
3. 时序特征聚合（GRU/Temporal Transformer）
4. 规划头（输出未来路径点或控制信号）

纯视觉方案面临的核心挑战是**深度估计的精度瓶颈**和**对恶劣天气/光照的鲁棒性**。近期的研究尝试通过自监督深度估计、多帧时序一致性约束以及对抗训练来提升纯视觉方案的可靠性。VAD（Jiang等, 2023）提出了矢量化场景表示，在保持端到端可微性的同时引入了结构化的场景先验，显著提升了规划的合理性。

### 10.4.4 学习型规划

学习型规划（Learning-based Planning）是端到端驾驶的核心。其关键问题是如何从数据中学习安全且泛化的规划策略。

**模仿学习（Imitation Learning）。** 最直接的方式是从人类驾驶数据中学习。行为克隆（Behavioral Cloning）最小化：

$$\mathcal{L}_{\text{BC}} = \mathbb{E}_{(\mathbf{o}, \mathbf{a}) \sim \mathcal{D}} \left[ \| \pi_\theta(\mathbf{o}) - \mathbf{a} \|^2 \right]$$

但行为克隆存在分布偏移（distribution shift）问题——训练数据中很少出现错误恢复的示范。DAgger（Ross等, 2011）通过在线交互收集纠正数据缓解这一问题。

**强化学习（Reinforcement Learning）。** RL允许智能体通过试错学习最优策略，但面临样本效率低和安全探索的挑战。在CARLA等仿真器中，RL方法如CQL-SAC和BC-Zero已被广泛研究。**条件模仿学习（CIL）**（Codevilla等, 2018）通过在输入中加入高层命令（直行/左转/右转）来解决多模态动作分布问题。

**世界模型（World Model）。** 最新的研究探索用世界模型进行想象式规划。GAIA-1（Hu等, 2023）和DriveDreamer（Wang等, 2023）学习驾驶场景的生成模型，能够在"想象"中评估不同决策的后果。世界模型的核心是学习转移函数：

$$\hat{\mathbf{s}}_{t+1} = g_\phi(\mathbf{s}_t, \mathbf{a}_t)$$

并通过在模型中进行轨迹采样和评估来选择最优动作（Model Predictive Control with learned dynamics）。

## 10.5 仿真与测试

自动驾驶系统的验证不能完全依赖实车测试——在公共道路上穷尽所有危险场景既不经济也不安全。据估算，要以95%的置信度证明自动驾驶系统的事故率低于人类驾驶员，需要累计约110亿英里的测试里程（Kalra & Paddock, 2016）。这一数字远超任何企业的实车测试能力。仿真测试因此成为不可或缺的环节，它能够以极低的成本和极高的并行度生成海量测试场景。

### 10.5.1 CARLA仿真器

**CARLA（Car Learning to Act）**（Dosovitskiy等, 2017）是自动驾驶领域最广泛使用的开源仿真平台之一。它基于Unreal Engine构建，支持：

- 多种天气和光照条件
- 可配置的交通流和行人行为
- 多传感器模拟（相机、LiDAR、雷达、GPS、IMU）
- Python API用于场景编排和智能体控制
- 排行榜（Leaderboard）标准评估流程

CARLA的**对抗性场景生成**功能允许在安全关键区域自动生成危险场景（如突然横穿的行人、对向车辆越线），用于压力测试规划器的鲁棒性。CARLA支持两种主要的工作模式：同步模式（用于训练和基准测试，保证传感器数据的时间一致性）和异步模式（用于交互式可视化和调试）。其开放的插件架构也使得研究者可以方便地扩展自定义传感器、天气系统和交通行为模型。

### 10.5.2 nuScenes数据集

**nuScenes**（Caesar等, 2020）是首个提供完整自动驾驶传感器套件（6个相机、1个LiDAR、5个雷达）大规模标注的数据集，包含1000个驾驶场景、约140万帧图像和40万帧点云。其关键特性包括：

- 全场景3D标注（23个物体类别）
- 逐帧物体跟踪标注
- 高精地图信息
- 丰富的场景属性标注（天气、时间、地点）

nuScenes已成为BEV感知和端到端驾驶的标准基准。其官方指标包括：

- **nuScenes Detection Score (NDS)**：综合评估检测精度的加权指标
- **Average Translation Error (ATE)**：平均平移误差
- **Average Orientation Error (AOE)**：平均朝向误差
- **Average Scale Error (ASE)**：平均尺度误差

此外，**Waymo Open Dataset**、**Argoverse** 和 **KITTI** 也是重要的基准数据集，各自侧重不同的场景特征。

### 10.5.3 安全验证

自动驾驶系统的安全验证是一个根本性难题。ISO 21448（SOTIF，预期功能安全）标准定义了安全验证的框架，核心思想是将场景空间划分为四个区域：

| | 已知 | 未知 |
|---|---|---|
| **安全** | 安全已知区域 | 安全未知区域 |
| **危险** | 危险已知区域（可测试） | 危险未知区域（需发现） |

验证的目标是：通过充分测试将危险未知区域缩小到可接受的水平。

**关键场景（Critical Scenarios）挖掘。** 基于自然驾驶数据（NDD）的统计分析，可以识别出高风险场景的参数空间。Frentsel-Bossen等提出用重要性采样（Importance Sampling）在仿真中高效探索尾部分布：

$$\hat{R} = \frac{1}{N} \sum_{i=1}^{N} \frac{p(\xi_i)}{q(\xi_i)} \cdot \mathbb{1}[\xi_i \in \mathcal{C}]$$

其中 $p$ 为真实场景分布，$q$ 为偏移后的采样分布，$\mathcal{C}$ 为碰撞场景集合。

**形式化验证。** 基于可达性分析（Reachability Analysis）的方法可以计算车辆在给定时间内可能到达的所有状态集合，进而验证是否存在与障碍物的交集。具体地，定义前向可达集：

$$\mathcal{R}(t) = \{\mathbf{x} \in \mathbb{R}^n \mid \exists \mathbf{u}(\cdot), \; \mathbf{x}(0) \in \mathcal{X}_0, \; \mathbf{x}(t) = \mathbf{x}\}$$

其中 $\mathcal{X}_0$ 为初始状态集合。若 $\mathcal{R}(t) \cap \mathcal{O} = \emptyset$（$\mathcal{O}$ 为障碍物占据的空间），则系统在时刻 $t$ 前是安全的。Hamilton-Jacobi可达性分析提供了严格的数学保证，但受限于维度诅咒，通常只能处理低维状态空间。近年来，基于神经网络的可达性分析方法（如DeepReach）正在尝试突破这一限制。

### 10.5.4 长尾场景

自动驾驶面临的最大挑战之一是**长尾分布**——绝大多数驾驶场景是常规的，但偶尔出现的罕见场景（如道路上的异形障碍物、极端天气下的传感器失效、不遵守交规的行为）往往决定系统的安全上限。

**长尾场景的分类：**

1. **感知长尾**：罕见物体（动物、掉落的家具）、极端天气（暴风雪、浓雾）、传感器退化（镜头污损、LiDAR多路径反射）
2. **交互长尾**：非典型行为（逆行、突然并线）、弱势道路使用者的不可预测行为
3. **环境长尾**：施工区域、临时交通管制、地图信息过时

**应对策略包括：**

- **数据增强**：通过合成数据扩充罕见场景，如使用生成模型（Diffusion Models）生成逼真的危险场景图像
- **场景生成**：对抗性场景生成（Adversarial Scenario Generation），在参数空间中搜索导致系统失败的场景配置
- **不确定性感知**：让感知和规划模块输出不确定性估计，在不确定度过高时回退到保守策略（如减速或停车）。具体地，可引入贝叶斯深度学习或MC-Dropout来估计预测的不确定性：$\sigma^2 = \frac{1}{T} \sum_{t=1}^{T} \hat{\mathbf{y}}_t^2 - \left(\frac{1}{T} \sum_{t=1}^{T} \hat{\mathbf{y}}_t\right)^2$，其中 $T$ 为前向传播次数
- **持续学习**：通过OTA（Over-The-Air）更新不断从新遇到的场景中学习，采用经验回放和弹性权重固化（EWC）等策略防止灾难性遗忘

## 10.6 小结

自动驾驶是具身智能技术的集大成者，它要求在感知、预测、规划和控制的全链路上实现鲁棒且高效的表现。本章梳理了从模块化方案到端到端学习的技术演进脉络：

- **感知层面**，BEV统一表征和占用网络正在取代传统的物体检测范式，提供了更通用、更完整的环境理解。
- **决策规划层面**，基于优化和学习的方法互补共存——前者提供可解释性和安全保证，后者提供数据驱动的泛化能力。
- **端到端驾驶**通过减少信息瓶颈和误差累积展现了巨大潜力，但其安全性和可解释性仍是亟待解决的关键问题。
- **仿真与测试**构成了从实验室到规模化部署的桥梁，长尾场景的高效覆盖是实现L4级自动驾驶的核心挑战。

展望未来，大语言模型（LLM）与视觉语言模型（VLM）的引入正在为自动驾驶带来新的范式变革——通过自然语言进行场景理解和指令传达，通过常识推理处理开放场景。DriveGPT4（Tian等, 2024）和LMDrive（Shao等, 2024）等探索了将语言模型作为驾驶决策中枢的可能性，使系统能够理解模糊的人类指令（如"找个安静的地方停车"）并将其转化为具体的驾驶行为。然而，语言模型的幻觉问题和推理延迟仍是实际部署的主要障碍。自动驾驶的终极目标不仅是"会开车"，更是"理解世界"，这与具身智能的核心追求高度一致。

---

**参考文献**

1. Qi, C.R., Su, H., Mo, K., & Guibas, L.J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. *CVPR*.
2. Zhou, Y., & Tuzel, O. (2018). VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection. *CVPR*.
3. Lang, A.H., Vora, S., Caesar, H., et al. (2019). PointPillars: Fast Encoders for Object Detection from Point Clouds. *CVPR*.
4. Philion, J., & Fidler, S. (2020). Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D. *ECCV*.
5. Li, Z., Wang, W., Li, H., et al. (2022). BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers. *ECCV*.
6. Liu, Z., Tang, H., Amini, A., et al. (2023). BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation. *ICRA*.
7. Hu, Y., Yang, J., Chen, L., et al. (2023). Planning-oriented Autonomous Driving. *CVPR*.
8. Chitta, K., Prakash, A., Jaeger, B., et al. (2023). TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving. *TPAMI*.
9. Caesar, H., Bankiti, V., Lang, A.H., et al. (2020). nuScenes: A Multimodal Dataset for Autonomous Driving. *CVPR*.
10. Dosovitskiy, A., Ros, G., Codevilla, F., et al. (2017). CARLA: An Open Urban Driving Simulator. *CoRL*.
11. Gao, J., Sun, C., Zhao, H., et al. (2020). VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation. *CVPR*.
12. Shi, S., Jiang, L., Dai, D., & Schiele, B. (2022). MTR: Multi-modal Motion Prediction with Transformer-based Neural Network for Autonomous Driving. *ICRA*.
13. Ross, S., Gordon, G., & Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS*.
14. Codevilla, F., Müller, M., López, A., et al. (2018). End-to-End Driving via Conditional Imitation Learning. *ICRA*.
15. Wei, Y., Zhao, Z., Zheng, W., et al. (2023). SurroundOcc: Multi-camera 3D Occupancy Prediction for Autonomous Driving. *ICCV*.
# 第11章 多机器人协作

> "没有任何一只蚂蚁能够独自建造蚁穴，但蚁群却能完成远超个体能力的工程。"

## 11.1 引言

具身智能的终极愿景之一，是让多个智能体在物理世界中协同工作，完成单一机器人无法胜任的复杂任务。从仓库中的上百台移动机器人协同分拣包裹，到灾区中多台无人机编队搜索幸存者，再到工厂流水线上多条机械臂协作装配精密部件——多机器人协作系统正在从实验室走向现实。

多机器人系统（Multi-Robot Systems, MRS）相较于单机器人系统，具有天然的优势：**鲁棒性**（单个机器人故障不影响全局任务）、**可扩展性**（通过增减机器人数量适配任务规模）以及**并行性**（多个子任务同时执行）。然而，这些优势的实现并非简单地将多个机器人放在一起，而是需要解决一系列根本性的科学与工程问题：机器人之间如何通信？任务如何分配？物理接触如何协调？大规模群体如何自组织？

本章将从四个层面系统阐述多机器人协作的核心技术：**多机通信**是协作的基础，决定了信息如何在机器人之间流动；**任务分配**是协作的大脑，决定了"谁做什么"；**协作操控**是协作的物理体现，涉及多个机器人在同一物体上的力学协调；**群体机器人**则是大规模协作的极端形式，追求从简单个体规则中涌现出复杂的集体行为。

---

## 11.2 多机通信

通信是多机器人协作的神经系统。机器人之间的信息交换方式、效率和可靠性，直接决定了协作系统的性能上限。

### 11.2.1 通信架构

多机器人系统的通信架构可分为三类：

**集中式架构（Centralized Architecture）。** 所有机器人将信息发送至一个中央节点（通常是基站或"领队"机器人），由中央节点进行全局决策后再将指令下发。这种架构的优势在于可以获取全局信息，做出理论上最优的决策。其致命弱点是**单点故障**——中央节点失效则整个系统瘫痪，且随着机器人数量增加，中央节点的计算和通信负载急剧上升。典型的集中式架构见于早期的多机器人遥操作系统和部分工业自动化场景（Gerkey & Matarić, 2004）。

**分布式架构（Distributed Architecture）。** 每个机器人都是对等节点，仅与邻居进行局部通信，通过局部信息交换逐步达成全局共识。分布式架构没有单点故障，天然具有可扩展性和鲁棒性。然而，由于每个机器人仅拥有局部视野，系统可能陷入局部最优或收敛缓慢。近年来，基于一致性协议（Consensus Protocol）的分布式控制成为研究热点，Olfati-Saber和Murray（2004）在其经典论文中系统地分析了多智能体系统中一致性问题的收敛条件，为分布式通信奠定了理论基础。

**混合式架构（Hybrid Architecture）。** 在实际工程中，纯粹的集中式或分布式架构都难以满足复杂需求。混合式架构结合了两者优势：在局部范围内采用分布式通信以提高响应速度和鲁棒性，在全局层面通过少量"骨干节点"进行信息汇总以获取全局态势。机器人操作系统ROS 2（Robot Operating System 2）采用的DDS（Data Distribution Service）通信中间件即支持这种灵活的通信拓扑，允许开发者在发布-订阅（publish-subscribe）模式和请求-应答（request-reply）模式之间自由选择（Macenski et al., 2022）。

### 11.2.2 消息传递机制

在确定通信架构后，需要选择具体的消息传递机制。

**发布-订阅模式（Publish-Subscribe）。** 发布者将消息发送到特定主题（topic），所有订阅该主题的接收者自动获取消息。这种模式解耦了发送者和接收者，适合一对多广播场景。ROS 2中每个机器人的传感器数据（激光雷达、摄像头等）通常通过发布-订阅模式共享。

**点对点通信（Point-to-Point）。** 两个机器人之间建立直接的通信通道，适合需要确认送达或交换私密信息的场景。例如，两台协作搬运的机器人之间需要实时交换力/力矩数据，此时点对点通信延迟更低、可靠性更高。

**广播与组播（Broadcast & Multicast）。** 广播将消息发送给所有可达节点，组播则限定于特定子组。在群体机器人中，广播常用于紧急消息（如"发现目标"或"紧急停止"），而组播则用于任务组内部的状态同步。

消息传递的核心挑战在于**通信延迟**和**丢包**。在实际无线环境中，信号干扰、遮挡和多径效应都会导致通信质量下降。为此，研究者提出了多种鲁棒通信策略：基于预测的消息补全（利用运动模型预测缺失的状态更新）、基于共识的状态估计（多个邻居的信息加权融合）以及语义通信（只传输高层语义信息而非原始数据，降低带宽需求）。

### 11.2.3 去中心化通信

去中心化通信是当前研究的前沿方向，其核心思想是：**系统不依赖任何中心节点，所有决策通过局部交互自然涌现。**

共识算法（Consensus Algorithm）是去中心化通信的数学基础。最经典的线性共识协议为：

$$x_i(t+1) = x_i(t) + \alpha \sum_{j \in \mathcal{N}_i} [x_j(t) - x_i(t)]$$

其中 $x_i$ 是第 $i$ 个机器人的状态变量，$\mathcal{N}_i$ 是其邻居集合，$\alpha$ 是权重系数。在连通的通信拓扑下，该协议保证所有机器人的状态渐近收敛到一致值。Ren和Beard（2005）证明了在切换拓扑条件下共识收敛的充分条件，这一理论成果被广泛应用于无人机编队和多车协同。

近年来，去中心化通信的研究出现了两个重要趋势。其一是**事件触发通信（Event-Triggered Communication）**——机器人不再周期性地广播状态，而仅在自身状态变化超过阈值或接收到异常信息时才发送消息，显著降低了通信负载（Dimarogonas et al., 2012）。其二是**基于图神经网络的通信学习**——通过强化学习自动学习通信拓扑和消息编码，使机器人能够自主决定"与谁通信"和"传递什么信息"，而非依赖人工设计的通信协议（Li et al., 2020）。

---

## 11.3 任务分配

在多机器人系统中，任务分配（Task Allocation）解决的核心问题是：**给定一组任务和一组机器人，如何将任务分配给机器人，使得整体效用最大化或成本最小化。** 这本质上是一个组合优化问题，当任务数和机器人数增长时，搜索空间呈指数爆炸。

Gerkey和Matarić（2004）在其开创性综述中将多机器人任务分配问题（Multi-Robot Task Allocation, MRTA）按照三个维度分类：单任务机器人与多任务机器人（ST-MR vs. MT-MR）、单机器人任务与多机器人任务（SR-MT vs. MR-MT）、即时分配与延迟分配（IA vs. DA）。这一分类框架至今仍是该领域的标准参考。

### 11.3.1 拍卖算法

拍卖算法（Auction Algorithm）是多机器人任务分配中最成功的实际方法之一，其灵感来源于经济学中的拍卖机制。

**单一物品拍卖（Single-Item Auction）。** 最简单的形式是序贯拍卖：任务逐一发布，机器人对每个任务出价，出价最高者获得该任务。Dias等人（2006）提出的MURDOCH系统是拍卖算法在多机器人中的经典实现，每个任务被广播后，机器人根据自身能力计算"投标值"（bid），通常定义为完成该任务的预期成本（如行进距离、能耗等），出价最低的机器人获得任务。

**组合拍卖（Combinatorial Auction）。** 当任务之间存在耦合（例如同时搬运A和B比分别搬运更高效）时，单一物品拍卖可能导致次优分配。组合拍卖允许机器人对任务包（bundle）出价，通过求解最优组合分配来最大化全局收益。然而，组合拍卖的计算复杂度极高，通常需要近似算法或启发式方法来求解（Choi et al., 2009）。

**一致性束算法（Consensus-Based Bundle Algorithm, CBBA）。** Choi等人（2009）提出的CBBA是分布式拍卖算法的里程碑式工作。在CBBA中，每个机器人独立地构建自己的任务束（bundle），并通过与邻居交换"获胜者信息"逐步达成全局一致性分配。CBBA的优雅之处在于它保证了无冲突分配（每个任务最多被分配给一个机器人），且仅需局部通信，是完全分布式的。

### 11.3.2 市场机制

市场机制将经济学中的供需关系、价格信号引入任务分配，赋予了系统更强的自适应能力。

**基于市场的任务分配框架。** 在这类方法中，任务被视为"商品"，机器人是"消费者"，任务的价值通过虚拟价格来调节。当某类任务积压时，其价格上涨，吸引更多机器人前来"购买"该任务；当机器人过剩时，任务价格下降，多余的机器人转向其他任务。这种价格反馈机制使得系统能够自然地适应动态变化的任务环境。

**合同网协议（Contract Net Protocol, CNP）。** Smith（1980）提出的合同网协议是最早的基于市场的任务分配方法之一。管理节点发布任务公告（call for proposals），工人节点提交投标（bid），管理节点根据投标选择最佳执行者并授予合同。CNP简单直观，但存在通信开销大、容易产生冲突等缺点。

**交易式市场方法（TraderBots）。** Zlot等人（2002）将更丰富的市场交易机制引入多机器人系统，包括一阶密封拍卖、双向拍卖和连续双边拍卖等。通过模拟真实的市场交易过程，TraderBots能够在动态环境中实现高效的任务再分配。

近年来，市场机制与强化学习的结合成为新趋势。研究者使用深度强化学习训练智能的"出价策略"，使机器人能够在复杂的市场环境中做出更优的竞标决策（Ye et al., 2023）。

### 11.3.3 学习型任务分配

随着深度学习和强化学习的发展，数据驱动的任务分配方法日益受到关注。

**多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）。** 在MARL框架中，每个机器人作为一个智能体，通过与环境（包括其他机器人）的交互学习最优策略。QTRAN（Son et al., 2019）和MAPPO（Yu et al., 2022）等算法在多智能体协作任务中展现了强大的性能。然而，MARL面临的挑战包括：状态-动作空间的指数爆炸、非平稳性（其他智能体的策略在不断变化）以及信用分配（如何评估每个智能体对团队奖励的贡献）。

**注意力机制与通信学习。** 基于Transformer架构的多智能体通信方法（如TarMAC, Das et al., 2019）利用注意力机制让机器人自动学习"关注哪些队友"和"传递什么信息"，将通信与任务分配统一在一个端到端的学习框架中。

**图神经网络方法。** 将多机器人系统建模为图结构——机器人是节点，通信链路是边——利用图神经网络（GNN）进行消息聚合和决策。这种方法天然支持可变数量的机器人，且能够泛化到训练时未见过的团队规模（Li et al., 2020）。

学习型方法的优势在于能够处理高维、非线性、部分可观测的复杂场景；其劣势在于可解释性不足、训练成本高，且在安全关键场景中的可靠性尚未得到充分验证。

---

## 11.4 协作操控

当多个机器人需要共同操纵同一物体时，问题从信息层面进入力学层面。协作操控（Cooperative Manipulation）是多机器人系统中最具物理挑战性的课题之一，涉及接触力学、柔顺控制和协调规划等多个领域。

### 11.4.1 多臂协作

多台机械臂协作抓取和操纵大型或重型物体是协作操控的典型场景。单台机械臂的负载能力和工作空间有限，而多臂协作可以突破这些限制。

**主从式协作（Leader-Follower）。** 一台机械臂（主臂）负责规划物体的运动轨迹，其他机械臂（从臂）通过力/位置混合控制跟随主臂的运动。这种架构简单直观，但主臂成为系统的瓶颈——其规划能力限制了整个系统的灵活性。Unimation公司在1980年代的双臂装配机器人是最早的工业多臂协作系统之一。

**对等式协作（Peer-to-Peer）。** 所有机械臂处于对等地位，通过共享物体的运动意图来协调动作。每个机械臂既感知物体的运动（通过力传感器或视觉），又主动施加控制力。这种架构更加灵活，但需要更复杂的协调策略。

**力闭合与形闭合（Force Closure & Form Closure）。** 在协作抓取中，多个夹持器的布局必须满足力学约束。力闭合要求夹持器能够在任意方向上施加平衡外力和力矩，形闭合则要求夹持器的几何布局能够完全约束物体的运动。Nguyen（1988）系统地分析了力闭合抓取的充分必要条件，为多臂抓取规划奠定了理论基础。

### 11.4.2 分布式力控

在协作操控中，力的协调是核心问题。每个机器人施加的力既不能过大（损坏物体或机器人），也不能过小（导致物体滑落），且所有机器人的合力必须与期望的物体运动一致。

**阻抗控制（Impedance Control）。** Hogan（1985）提出的阻抗控制将机器人末端建模为"弹簧-阻尼-质量"系统，通过调节刚度、阻尼和惯性参数来控制机器人与环境的交互力。在多臂协作中，每个机械臂独立地执行阻抗控制，通过调节自身的"虚拟刚度"来自然地协调力的分配。Hogan的阻抗控制理论为柔顺交互奠定了基础，至今仍是力控领域的核心方法。

**混合力/位控制（Hybrid Force/Position Control）。** Raibert和Craig（1981）提出的方法将任务空间分解为"受约束方向"和"自由方向"：在受约束方向上控制力，在自由方向上控制位置。多臂协作中，每个机械臂可以独立地在各自的任务子空间中进行混合控制，实现精细的力-位协调。

**分布式力分配。** 当多个机械臂共同操纵一个物体时，存在力的冗余——同一物体运动可以由无数种力组合实现。分布式力分配的目标是在满足任务约束的前提下，最小化各机械臂的力/力矩（降低能耗和磨损），同时避免超过任何单台机械臂的力极限。Liu和Goldenberg（1992）提出的基于零空间投影的分布式力分配方法是该领域的经典工作。

### 11.4.3 协调策略

协作操控中的协调策略决定了多个机器人如何在时间、空间和力学上同步行动。

**基于共识的协调。** 多个机器人通过分布式共识算法对齐各自对物体状态的估计，然后据此生成协调的控制指令。Schwager等人（2009）将一致性滤波器应用于多机器人协作操控，使得每个机器人能够基于局部力传感器信息估计物体的全局状态。

**基于博弈论的协调。** 将多机器人协作操控建模为协作博弈：每个机器人的收益与团队整体目标一致，通过求解纳什均衡或帕累托最优来获得协调策略。Franchi等人（2012）将博弈论框架应用于多无人机协作吊运系统，展示了理论优雅性和实际有效性。

**学习型协调。** 近年来，深度强化学习被用于学习复杂的协调策略。机器人通过反复试验学习如何协调力和运动，无需显式的力学模型。Yu等人（2023）的工作展示了多臂机器人通过强化学习自主学会协作翻转不规则物体的能力，体现了学习型方法在处理复杂接触动力学方面的潜力。

---

## 11.5 群体机器人

群体机器人（Swarm Robotics）是多机器人协作的极端形式——系统由大量（数十到数千）相对简单的个体组成，通过局部交互涌现出宏观层面的智能行为。群体机器人的核心哲学是：**个体越简单，群体越鲁棒。**

### 11.5.1 群体智能的生物学启示

群体机器人的灵感直接来源于自然界中的群居生物。

**蚂蚁的觅食行为。** 蚂蚁通过信息素（pheromone）进行间接通信——找到食物的蚂蚁在返回巢穴的途中释放信息素，后来的蚂蚁倾向于跟随信息素浓度高的路径。这种正反馈机制使得最短路径上的信息素浓度越来越高，最终整个蚁群收敛到最优路径。Dorigo等人（1996）将这一观察抽象为**蚁群优化算法（Ant Colony Optimization, ACO）**，被广泛应用于路径规划和调度问题。

**鸟群和鱼群的编队行为。** Reynolds（1987）提出了著名的Boids模型，仅用三条规则就模拟了鸟类的群体飞行：**分离**（避免与邻居碰撞）、**对齐**（与邻居保持相同方向）和**聚合**（向邻居的中心移动）。这三条完全局部的规则产生了令人惊叹的宏观编队行为，证明了"没有领导者的领导力"。

**蜜蜂的觅食决策。** 蜜蜂通过"摇摆舞"向同伴传递食物源的方向和距离信息，群体通过比较多个食物源的信息来集体决策最优觅食策略。Seeley等人（2012）的研究揭示了民主决策的生物学基础，启发了分布式任务分配算法的设计。

### 11.5.2 自组织与涌现

自组织（Self-Organization）是群体机器人最核心的特征：宏观层面的有序行为从微观层面的局部交互中自发涌现，无需任何全局规划者或中心控制器。

自组织系统通常具有以下特征：**正反馈**（放大有益的行为模式）、**负反馈**（抑制过度的行为模式以避免系统失衡）、**随机性**（通过探索新方案避免陷入局部最优）以及**多交互**（个体之间通过多种渠道交换信息）。

在群体机器人中，自组织的经典案例包括：

**集群形成（Aggregation）。** 大量机器人通过简单的"靠近邻居"规则自发聚集到一起。Trianni等人（2003）利用进化方法为群体机器人设计了集群行为控制器，机器人仅依赖局部传感器（红外和碰撞检测）即可完成集群。

**图案形成（Pattern Formation）。** 机器人通过局部交互自发形成特定的空间图案——圆形、直线、网格等。Rubenstein等人（2014）在著名的Kilobot实验中，用1024个简单的振动驱动机器人实现了多种复杂图案的自组织形成，是群体机器人领域里程碑式的工作。

**集体运输（Collective Transport）。** 多个机器人协作搬运单个大型物体，类似于蚂蚁搬运食物。每个机器人仅感知自身与物体的接触力，通过简单的力反馈规则调整推拉方向。Martinson和Trajectories（2010）的工作表明，即使没有通信和全局规划，仅靠局部力反馈就能实现高效的集体运输。

### 11.5.3 大规模协调

当群体规模增长到数百甚至数千时，传统的多机器人协调方法面临根本性挑战：通信带宽不足、计算复杂度爆炸、个体故障频繁。大规模协调需要全新的算法范式。

**概率方法。** 用概率分布而非精确状态来描述群体行为。例如，用概率密度函数描述群体的空间分布，用贝叶斯滤波器估计群体状态。这种方法允许在不追踪每个个体的情况下进行群体级别的决策（Prorok et al., 2021）。

**均场博弈（Mean Field Game）。** 将大量个体的交互近似为每个个体与"平均场"（所有个体的平均效应）的交互，将N体问题降维为单体问题。Huang等人（2006）和Lasry与Lions（2007）奠定了均场博弈的数学基础，近年来被引入大规模多机器人系统。

**分层方法。** 将大规模群体划分为多个小规模子群（cluster），子群内部采用精细的协调策略，子群之间采用粗粒度的协调策略。这种分层架构在保持可扩展性的同时，保留了局部协调的灵活性。

### 11.5.4 蚁群算法与路径规划

蚁群优化算法（ACO）是群体智能在机器人路径规划中最成功的应用之一。

**基本原理。** AC模拟蚁群的觅食过程：人工蚂蚁在图上随机游走，根据信息素浓度和启发式信息（如距离的倒数）选择下一步方向。每只蚂蚁走完一条路径后，根据路径质量释放信息素——路径越短（质量越高），释放的信息素越多。信息素同时会随时间蒸发，避免过早收敛到次优解。经过多轮迭代，正反馈效应使得最优路径上的信息素浓度最高，蚁群收敛到全局最优或近优解。

**在机器人路径规划中的应用。** 将工作空间离散化为栅格图或拓扑图后，ACO可以高效地为多个机器人规划无碰撞路径。在多机器人场景中，每只"人工蚂蚁"代表一个机器人的路径规划方案，信息素不仅编码路径质量，还编码冲突信息——与其他机器人路径冲突的位置上信息素会被抑制，从而自然地实现避碰协调。

**改进与变体。** 标准ACO存在收敛慢和容易陷入局部最优的问题。近年来的改进包括：自适应信息素更新（根据搜索状态动态调整蒸发率）、混合ACO（与A*、RRT等算法结合）以及并行ACO（利用GPU加速大规模群体的路径规划）。G堍rez和Luezas（2015）将改进的ACO应用于室内多机器人导航，展示了在动态环境中的实时路径规划能力。

### 11.5.5 进化与自适应

群体机器人中的个体行为规则可以通过**进化算法**自动设计，而非人工编程。

**自动设计行为规则。** Trianni和Nolfi（2009）利用神经进化方法为群体机器人自动演化出集群、觅食和编队等行为控制器。进化过程在仿真环境中进行，最优个体的行为规则被部署到真实机器人上。这种方法的优势在于能够发现人类设计师意想不到的高效策略。

**形态学自适应。** 前沿研究探索了群体机器人的形态学自适应——机器人能够改变自身的物理形态（如形状、连接方式）来适应任务需求。可重构群体机器人系统（如Modular Self-Reconfigurable Robots）允许机器人个体通过物理连接和断开来动态改变群体的拓扑结构（Yim et al., 2007）。

---

## 11.6 前沿与展望

多机器人协作正从传统的基于规则的方法向数据驱动、自适应、自组织的方向演进。以下几个方向值得特别关注：

**大模型赋能的多机器人系统。** 大语言模型（LLM）和视觉-语言-动作模型（VLA）正在被引入多机器人协作。机器人可以通过自然语言进行高层意图交流，由大模型充当"翻译官"——将模糊的人类指令分解为具体的多机器人协作方案（Mandi et al., 2024）。

**人机群协作（Human-Swarm Interaction）。** 将人类纳入群体机器人回路，人类提供高层目标和直觉判断，群体机器人负责底层执行。这需要设计直观的人机交互接口，使人类能够有效地控制和引导大规模机器人群体。

**异构多机器人系统。** 现实场景中的机器人往往具有不同的形态、能力和传感器。异构协作（如无人机+地面机器人+水下机器人）能够利用各自优势完成单一类型机器人无法完成的任务，但也带来了通信兼容性、任务异构性和协调复杂度等新挑战。

**安全与可验证性。** 在安全关键场景（如医疗、救援）中，多机器人系统的行为必须是可预测和可验证的。形式化方法（Formal Methods）和控制屏障函数（Control Barrier Functions）正被引入多机器人系统，以提供可证明的安全保证。

---

## 11.7 本章小结

本章系统地介绍了多机器人协作的四个核心层面：

1. **多机通信**是协作的基础设施，从集中式到分布式的架构演进反映了系统设计从"全局最优"向"鲁棒可扩展"的权衡。去中心化共识算法和事件触发通信为大规模系统提供了理论保障。

2. **任务分配**是协作的决策中枢。拍卖算法（特别是CBBA）在实际系统中表现出色，市场机制提供了自适应的分配框架，而学习型方法则为处理复杂、动态的任务分配问题开辟了新路径。

3. **协作操控**将协作从信息层面推向物理层面。阻抗控制和分布式力分配是实现安全、高效多臂协作的关键技术，学习型协调策略正在展现处理复杂接触动力学的潜力。

4. **群体机器人**是多机器人协作的终极形态。生物启发的自组织原理、蚁群算法等群体智能方法，以及大规模协调的概率化和分层化策略，共同构成了群体机器人的理论基石。

多机器人协作的未来，在于将这些不同层面的技术有机融合——让机器人在通信、决策和物理交互的每一个层面都实现高效协作，最终达到"整体大于部分之和"的涌现智能。

---

## 参考文献

- Choi, H. L., Brunet, L., & How, J. P. (2009). Consensus-based decentralized auctions for robust task allocation. *IEEE Transactions on Robotics*, 25(4), 912-926.
- Das, A., Gervet, T., Romoff, J., et al. (2019). TarMAC: Targeted multi-agent communication. *Proceedings of the International Conference on Machine Learning (ICML)*.
- Dias, M. B., Zlot, R., Kalra, N., & Stentz, A. (2006). Market-based multirobot coordination: A survey and analysis. *Proceedings of the IEEE*, 94(7), 1257-1270.
- Dimarogonas, D. V., Frazzoli, E., & Johansson, K. H. (2012). Distributed event-triggered control for multi-agent systems. *IEEE Transactions on Automatic Control*, 57(5), 1291-1297.
- Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: Optimization by a colony of cooperating agents. *IEEE Transactions on Systems, Man, and Cybernetics, Part B*, 26(1), 29-41.
- Franchi, A., Secchi, C., Hyoung Il S., Bülthoff, H. H., & Giordano, P. R. (2012). Bilateral teleoperation of groups of mobile robots with time-varying topology. *IEEE Transactions on Robotics*, 28(5), 1019-1033.
- Gerkey, B. P., & Matarić, M. J. (2004). A formal analysis and taxonomy of task allocation in multi-robot systems. *International Journal of Robotics Research*, 23(9), 939-954.
- Hogan, N. (1985). Impedance control: An approach to manipulation. *ASME Journal of Dynamic Systems, Measurement, and Control*, 107(1), 1-24.
- Huang, M., Malhamé, R. P., & Caines, P. E. (2006). Large population stochastic dynamic games: Closed-loop McKean-Vlasov systems and the Nash certainty equivalence principle. *Communications in Information and Systems*, 6(3), 221-252.
- Lasry, J. M., & Lions, P. L. (2007). Mean field games. *Japanese Journal of Mathematics*, 2(1), 229-260.
- Li, Q., Lin, J., Gama, F., & Ribeiro, A. (2020). Graph neural networks for decentralized multi-robot path planning. *IEEE International Conference on Intelligent Robots and Systems (IROS)*.
- Liu, Y. H., & Goldenberg, A. A. (1992). Uncertainty manipulation and decomposition for robotic force control. *IEEE International Conference on Robotics and Automation (ICRA)*.
- Macenski, S., Foote, T., Gerkey, B., Lalancette, C., & Woodall, W. (2022). Robot Operating System 2: Design, architecture, and uses in the wild. *Science Robotics*, 7(66), eabm6074.
- Mandi, Z., Jain, S., & Song, S. (2024). Roco: Dialectic multi-robot collaboration with large language models. *IEEE International Conference on Robotics and Automation (ICRA)*.
- Martinson, E. (2010). Collaborative manipulation with autonomous robots. *IEEE International Conference on Robotics and Automation (ICRA)*.
- Nguyen, V. D. (1988). Constructing force-closure grasps. *International Journal of Robotics Research*, 7(3), 3-16.
- Olfati-Saber, R., & Murray, R. M. (2004). Consensus problems in networks of agents with switching topology and time-delays. *IEEE Transactions on Automatic Control*, 49(9), 1520-1533.
- Prorok, A., Malencia, M., Carlone, L., et al. (2021). Beyond robustness: A taxonomy of approaches towards resilient multi-robot systems. *arXiv preprint arXiv:2109.12343*.
- Raibert, M. H., & Craig, J. J. (1981). Hybrid position/force control of manipulators. *ASME Journal of Dynamic Systems, Measurement, and Control*, 103(2), 126-133.
- Ren, W., & Beard, R. W. (2005). Consensus seeking in multiagent systems under dynamically changing interaction topologies. *IEEE Transactions on Automatic Control*, 50(5), 655-661.
- Reynolds, C. W. (1987). Flocks, herds, and schools: A distributed behavioral model. *ACM SIGGRAPH Computer Graphics*, 21(4), 25-34.
- Rubenstein, M., Cornejo, A., & Nagpal, R. (2014). Programmable self-assembly in a thousand-robot swarm. *Science*, 345(6198), 795-799.
- Schwager, M., Rus, D., & Slotine, J. J. (2009). Unifying geometric, probabilistic, and potential field approaches to multi-robot deployment. *International Journal of Robotics Research*, 30(3), 371-383.
- Seeley, T. D., Visscher, P. K., Schlegel, T., et al. (2012). Stop signals provide cross inhibition in collective decision-making by honeybee swarms. *Science*, 335(6064), 108-111.
- Smith, R. G. (1980). The contract net protocol: High-level communication and control in a distributed problem solver. *IEEE Transactions on Computers*, C-29(12), 1104-1113.
- Son, K., Kim, D., Kang, W. J., Hostallero, D. E., & Yi, Y. (2019). QTRAN: Learning to factorize with transformation for cooperative multi-agent reinforcement learning. *Proceedings of the International Conference on Machine Learning (ICML)*.
- Trianni, V., Groß, R., Labella, T. H., Şahin, E., & Dorigo, M. (2003). Evolving aggregation behaviors in a swarm of robots. *Proceedings of the European Conference on Artificial Life (ECAL)*.
- Ye, D., Zhang, M., & Sutanto, D. (2023). Deep reinforcement learning for multi-robot task allocation with auction-based mechanisms. *IEEE Transactions on Neural Networks and Learning Systems*.
- Yim, M., Shen, W. M., Salemi, B., et al. (2007). Modular self-reconfigurable robot systems. *IEEE Robotics and Automation Magazine*, 14(1), 43-52.
- Yu, C., Velu, A., Vinitsky, E., et al. (2022). The surprising effectiveness of PPO in cooperative multi-agent games. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Zlot, R., Stentz, A., Dias, M. B., & Thayer, S. (2002). Multi-robot exploration controlled by a market economy. *IEEE International Conference on Robotics and Automation (ICRA)*.
# 第12章 Sim-to-Real：从仿真到现实的跨越

> "仿真是一面镜子，但现实世界永远不是镜中倒影。" —— 具身智能研究的核心挑战

## 12.1 引言：仿真-现实鸿沟

具身智能（Embodied AI）的核心愿景是让智能体在物理世界中自主感知、决策与行动。强化学习（Reinforcement Learning, RL）为这一愿景提供了强大的算法框架，但其"数据饥渴"的本质与物理世界的高成本试错之间存在根本矛盾。一个四足机器人在真实环境中学习行走，可能需要数万次跌倒——这不仅耗时数月，更会导致硬件不可逆的损坏。

仿真环境（Simulation）因此成为具身智能研究不可或缺的基础设施。在仿真中，智能体可以以千倍于现实的速度进行探索，无需担心安全问题，且可以自由重置环境。然而，任何仿真器都是物理世界的近似模型——动力学模型的简化、传感器噪声的省略、接触力计算的偏差——这些差异累积形成了所谓的**仿真-现实鸿沟（Sim-to-Real Gap）**。

形式化地，令 $\pi^*$ 为在仿真环境 $\mathcal{E}_{\text{sim}}$ 中训练得到的最优策略，$\pi_{\text{real}}^*$ 为在真实环境 $\mathcal{E}_{\text{real}}$ 中的最优策略，则仿真-现实鸿沟可度量为：

$$\Delta_{\text{sim2real}} = |J(\pi^*, \mathcal{E}_{\text{real}}) - J(\pi^*, \mathcal{E}_{\text{sim}})|$$

其中 $J(\pi, \mathcal{E}) = \mathbb{E}_{\tau \sim \pi, \mathcal{E}}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$ 为策略 $\pi$ 在环境 $\mathcal{E}$ 中的期望累积回报。当 $\Delta_{\text{sim2real}}$ 过大时，在仿真中表现优异的策略在真实世界中可能完全失效——这就是经典的**仿真-现实迁移失败（Sim-to-Real Transfer Failure）**。

本章将系统阐述弥合这一鸿沟的三大技术范式：**域随机化（Domain Randomization）**、**系统辨识（System Identification）** 和 **教师-学生迁移（Teacher-Student Transfer）**，并对比主流仿真平台的技术特性。

---

## 12.2 域随机化：以多样性覆盖现实

### 12.2.1 核心思想

域随机化（Domain Randomization, DR）的基本假设极为朴素：如果智能体在仿真中见过足够多样的"世界变体"，那么真实世界不过是其中一个变体。通过在训练过程中随机化仿真环境的各类参数，策略被迫学习对这些变化不敏感的鲁棒表征，从而在面对真实环境时仍能保持良好性能。

设仿真环境的参数化空间为 $\Theta = \{\theta_1, \theta_2, \ldots, \theta_n\}$，其中每个 $\theta_i$ 代表一个可随机化的参数（如摩擦系数、光照方向、物体质量等）。域随机化的目标是找到一个参数分布 $p(\theta)$，使得：

$$\pi^* = \arg\max_\pi \mathbb{E}_{\theta \sim p(\theta)} \left[ J(\pi, \mathcal{E}_\theta) \right]$$

其中 $\mathcal{E}_\theta$ 表示参数为 $\theta$ 的仿真环境实例。直觉上，这迫使策略在"平均最差"的环境变体上表现良好，从而提高了泛化到真实世界的能力。

### 12.2.2 视觉随机化

视觉随机化（Visual Domain Randomization）是最直观的随机化形式，主要针对渲染管线中的视觉元素进行随机化。OpenAI 在 2019 年的 Rubik's Cube 操纵任务中展示了视觉随机化的惊人效果：通过在仿真中随机化纹理、光照、相机位姿等视觉属性，策略可以直接从仿真迁移到真实机械手上完成魔方复原（OpenAI et al., 2019）。

视觉随机化通常包括以下维度：

**纹理随机化**：对场景中所有物体的表面纹理进行随机化，包括颜色、反射率、粗糙度等材质属性。令物体 $o$ 的材质参数为 $\mathbf{m}_o = (c_{\text{diffuse}}, c_{\text{specular}}, \alpha_{\text{roughness}})$，则纹理随机化在每轮训练中采样：

$$\mathbf{m}_o \sim \mathcal{U}(\mathbf{m}_{\min}, \mathbf{m}_{\max})$$

**光照随机化**：随机化光源的位置 $\mathbf{l}$、颜色 $c_{\text{light}}$ 和强度 $I$。一些工作还会随机化环境光和阴影参数，以模拟不同时间、天气条件下的光照变化。

**相机随机化**：对相机的内参（焦距、畸变）和外参（位置、朝向）添加噪声。设相机位姿为 $\mathbf{T}_c = [\mathbf{R} | \mathbf{t}]$，随机化后的位姿为：

$$\mathbf{T}_c' = \mathbf{T}_c \cdot \exp([\boldsymbol{\xi}]_\times), \quad \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\xi)$$

其中 $[\boldsymbol{\xi}]_\times$ 为李代数 $\boldsymbol{\xi} \in \mathfrak{se}(3)$ 的反对称矩阵表示。

**后处理随机化**：在渲染图像上应用随机的高斯模糊、运动模糊、颜色抖动、添加高斯噪声等后处理操作，模拟真实相机的成像特性。

Tobin 等人（2017）在早期工作中证明，即使不使用任何真实图像数据，仅通过视觉随机化训练的策略也能在真实环境中完成简单的物体检测任务。这一结果开创了"零样本 Sim-to-Real 迁移"的研究方向。

### 12.2.3 动力学随机化

动力学随机化（Dynamics Domain Randomization）关注的是物理参数的变化。真实的物理系统参数往往是未知的或时变的——关节摩擦会随磨损变化，连杆质量难以精确测量，电机响应存在个体差异。动力学随机化通过在训练中随机化这些物理参数，迫使策略适应参数变化。

典型随机化的动力学参数包括：

| 参数类别 | 具体参数 | 典型随机化范围 |
|---------|---------|--------------|
| 摩擦 | 静摩擦系数 $\mu_s$、动摩擦系数 $\mu_k$ | $\pm 50\%$ 标称值 |
| 质量 | 连杆质量 $m_i$、质心位置 $\mathbf{c}_i$ | $\pm 20\%$ 标称值 |
| 惯量 | 转动惯量 $\mathbf{I}_i$ | $\pm 30\%$ 标称值 |
| 关节 | 阻尼 $b_i$、刚度 $k_i$、力矩限制 | $\pm 40\%$ 标称值 |
| 电机 | 延迟 $\tau_d$、力矩噪声 $\sigma_\tau$ | 0-20ms 延迟 |
| 地形 | 地面弹性、平整度、坡度 | 任务相关 |

在动力学随机化下，环境的动力学方程变为：

$$\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}, \boldsymbol{\theta}_{\text{dyn}}), \quad \boldsymbol{\theta}_{\text{dyn}} \sim p(\boldsymbol{\theta}_{\text{dyn}})$$

其中 $\mathbf{x}$ 为状态，$\mathbf{u}$ 为控制输入，$\boldsymbol{\theta}_{\text{dyn}}$ 为随机化的动力学参数。Peng 等人（2018）在仿真人形机器人运动的研究中展示了动力学随机化的有效性：通过随机化质量和摩擦参数，策略在从未见过的真实机器人上实现了稳定的行走和后空翻。

一个关键的设计决策是随机化参数的**范围**。范围过小无法覆盖真实世界的变异性，范围过大则可能导致策略过于保守或根本无法学会。实践中，通常从较小范围开始，逐步扩大直至策略在真实环境中表现稳定。

### 12.2.4 场景随机化

场景随机化（Scene Randomization）在更高层面上随机化任务场景的布局和结构。与视觉和动力学随机化关注"同一个场景的不同表现"不同，场景随机化改变的是场景本身。

具体包括：

- **物体位置随机化**：随机化目标物体、障碍物的初始位置和朝向
- **场景布局随机化**：随机化桌面布局、房间结构等空间配置
- **物体几何随机化**：随机化物体的形状参数（如方块大小、圆柱半径）
- **任务参数随机化**：随机化目标位置、约束条件等任务规格

场景随机化的数学表述为在场景图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 的配置空间中采样：

$$\mathcal{G} \sim p(\mathcal{G}), \quad \text{subject to } \mathcal{G} \in \mathcal{G}_{\text{valid}}$$

其中 $\mathcal{G}_{\text{valid}}$ 为物理可行的场景配置集合（如避免物体穿透等约束）。

在大规模场景生成方面，Hypersim（Roberts et al., 2021）和 3D-FRONT（Fu et al., 2021）等工作提供了程序化生成多样化室内场景的方法，为视觉导航和操作任务的训练提供了丰富的环境多样性。

### 12.2.5 自动域随机化（ADR）

手动设计随机化分布 $p(\theta)$ 是一项繁琐且高度依赖经验的工作。自动域随机化（Automatic Domain Randomization, ADR）旨在**自动化**这一过程，让算法自适应地调整随机化分布。

ADR 的核心思想来自 OpenAI 的"集智"（Population-Based Training）框架。其基本算法如下：

1. 初始化一个较小的参数范围 $[\boldsymbol{\theta}_{\min}^{(0)}, \boldsymbol{\theta}_{\max}^{(0)}]$
2. 在当前参数范围内训练策略
3. 使用一个**性能度量器**评估策略在特定参数值上的表现
4. 如果策略在边界值上的表现超过阈值，扩展参数范围；如果低于阈值，收缩范围
5. 重复步骤 2-4 直到收敛

形式化地，ADR 的边界更新规则为：

$$\theta_i^{\min} \leftarrow \begin{cases} \theta_i^{\min} - \delta_i & \text{if } P(\theta_i^{\min}) > \phi_{\text{expand}} \\ \theta_i^{\min} + \delta_i & \text{if } P(\theta_i^{\min}) < \phi_{\text{shrink}} \\ \theta_i^{\min} & \text{otherwise} \end{cases}$$

$$\theta_i^{\max} \leftarrow \begin{cases} \theta_i^{\max} + \delta_i & \text{if } P(\theta_i^{\max}) > \phi_{\text{expand}} \\ \theta_i^{\max} - \delta_i & \text{if } P(\theta_i^{\max}) < \phi_{\text{shrink}} \\ \theta_i^{\max} & \text{otherwise} \end{cases}$$

其中 $P(\theta_i)$ 为策略在参数边界值 $\theta_i$ 上的性能指标（如成功率），$\phi_{\text{expand}}$ 和 $\phi_{\text{shrink}}$ 为扩展和收缩阈值，$\delta_i$ 为步长。

ADR 的优势在于它实现了一种**课程学习（Curriculum Learning）** 效应：策略从简单（参数范围小）的环境开始学习，随着能力提升逐步面对更困难（参数范围大）的环境变体。这不仅提高了训练效率，也避免了策略在过于困难的初始环境中无法学习的问题。

Mehta 等人（2020）在 Quad-SDK 项目中将 ADR 应用于四足机器人的运动控制，展示了 ADR 相比手动调参在迁移成功率上的显著提升。Akalin 等人（2023）进一步将 ADR 与对比学习结合，提出 CLADR（Contrastive Learning for ADR），利用表示学习来指导参数范围的自动调整。

---

## 12.3 系统辨识：让仿真更真实

### 12.3.1 参数估计

如果说域随机化是让策略适应不同的物理参数，那么系统辨识（System Identification, SysID）则是让仿真器本身更准确地反映真实系统。其核心任务是从真实系统的观测数据中估计物理参数 $\hat{\boldsymbol{\theta}}$，使得仿真动力学尽可能匹配真实动力学。

经典的系统辨识方法依赖**物理建模**与**参数拟合**的两阶段范式。首先建立参数化的动力学模型：

$$\mathbf{M}(\mathbf{q}, \boldsymbol{\theta})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}, \boldsymbol{\theta})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}, \boldsymbol{\theta}) = \boldsymbol{\tau} + \mathbf{J}^T(\mathbf{q})\mathbf{f}_{\text{ext}}$$

其中 $\mathbf{q}$ 为广义坐标，$\mathbf{M}$ 为惯量矩阵，$\mathbf{C}$ 为科里奥利力矩阵，$\mathbf{g}$ 为重力项，$\boldsymbol{\tau}$ 为关节力矩，$\mathbf{f}_{\text{ext}}$ 为外力。参数 $\boldsymbol{\theta}$ 包括质量、惯量、摩擦系数等。

然后通过最小化仿真轨迹与真实轨迹之间的差异来估计参数：

$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \sum_{k=1}^{N} \left\| \mathbf{q}_k^{\text{sim}}(\boldsymbol{\theta}) - \mathbf{q}_k^{\text{real}} \right\|_{\mathbf{W}}^2$$

其中 $\mathbf{q}_k^{\text{sim}}(\boldsymbol{\theta})$ 为在参数 $\boldsymbol{\theta}$ 下仿真的第 $k$ 步关节角度，$\mathbf{W}$ 为加权矩阵，$N$ 为轨迹长度。

对于复杂系统，解析模型可能不充分。数据驱动的系统辨识方法使用神经网络来拟合动力学残差：

$$\dot{\mathbf{x}} = f_{\text{analytical}}(\mathbf{x}, \mathbf{u}, \hat{\boldsymbol{\theta}}) + f_{\text{residual}}(\mathbf{x}, \mathbf{u}; \boldsymbol{\phi})$$

其中 $f_{\text{residual}}$ 为神经网络参数化的残差模型，$\boldsymbol{\phi}$ 为其可训练参数。这种"物理先验 + 数据修正"的混合范式在保持物理可解释性的同时获得了更好的拟合精度。

Lujan 等人（2022）提出的 WHIRL（In-the-Wild Human-Robot Interaction）框架利用少量真实交互数据，通过系统辨识来校准仿真器的物理参数，使得在仿真中训练的操作策略能够直接迁移到真实机器人上。

### 12.3.2 可微仿真

传统系统辨识依赖梯度无关的优化方法（如贝叶斯优化、进化策略），效率较低。**可微仿真（Differentiable Simulation）** 的兴起从根本上改变了这一局面：通过构建物理引擎的可微版本，可以直接通过反向传播计算物理参数的梯度，从而实现高效的系统辨识。

在可微仿真中，物理仿真的一步可以表示为：

$$\mathbf{x}_{t+1} = \Phi(\mathbf{x}_t, \mathbf{u}_t, \boldsymbol{\theta})$$

其中 $\Phi$ 为可微的物理转移函数。损失函数 $\mathcal{L}$ 对参数 $\boldsymbol{\theta}$ 的梯度可以通过链式法则计算：

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} = \sum_{t=0}^{T} \frac{\partial \mathcal{L}}{\partial \mathbf{x}_t} \cdot \frac{\partial \mathbf{x}_t}{\partial \boldsymbol{\theta}}$$

其中雅可比矩阵 $\frac{\partial \mathbf{x}_t}{\partial \boldsymbol{\theta}}$ 可以通过在仿真过程中缓存中间状态进行高效计算。然而，物理仿真中的**接触力**和**碰撞检测**往往涉及不可微的运算（如符号函数、max 操作），这为可微仿真带来了技术挑战。

针对这一问题，研究者提出了多种技术方案。DiffSim（Hu et al., 2019）使用**隐式积分**和**软接触模型**来处理不可微的碰撞：

$$\mathbf{f}_{\text{contact}} = k_c \cdot \max(0, -d(\mathbf{q}))^\alpha \cdot \text{sgn}(\dot{d}(\mathbf{q}))$$

其中 $d(\mathbf{q})$ 为穿透深度，$k_c$ 为接触刚度，$\alpha$ 控制接触力的非线性程度。通过将 $\max$ 操作替换为可微的 softplus 函数 $f_\beta(x) = \frac{1}{\beta}\log(1 + e^{\beta x})$，整个接触力计算变得可微。

Taichi（Hu et al., 2020）和 Warp（NVIDIA, 2023）等可微物理引擎进一步优化了自动微分在大规模并行仿真中的效率。Brax（Google, 2021）完全在 JAX 中实现，支持大规模并行的可微仿真，使得数千个环境实例可以在 GPU 上同时运行并计算梯度。

可微仿真的另一个重要应用是**轨迹优化**。通过将整个轨迹作为优化变量，可以使用梯度下降直接求解最优控制序列：

$$\mathbf{u}_{0:T}^* = \arg\min_{\mathbf{u}_{0:T}} \sum_{t=0}^{T} c(\mathbf{x}_t, \mathbf{u}_t) + c_T(\mathbf{x}_T)$$

$$\text{s.t. } \mathbf{x}_{t+1} = \Phi(\mathbf{x}_t, \mathbf{u}_t, \boldsymbol{\theta})$$

这种方法在机器人运动规划和操作任务中展现了强大的能力，例如 Differentiable Cloth Simulation（Liang et al., 2019）用于布料操纵任务的规划。

### 12.3.3 Real2Sim：从现实到仿真

Real2Sim 是系统辨识的高级形态，目标是构建一个与特定真实环境高度匹配的"数字孪生（Digital Twin）"。与传统系统辨识仅关注动力学参数不同，Real2Sim 同时考虑几何、外观和物理的全面重建。

典型的 Real2Sim 流程包括：

1. **三维重建**：使用深度相机（如 Intel RealSense、Azure Kinect）或多视角相机系统重建环境的三维几何模型。常用方法包括 NeRF（Mildenhall et al., 2020）和 3D Gaussian Splatting（Kerbl et al., 2023）。

2. **物理属性估计**：从重建的几何模型出发，估计物体的物理属性。质量可以通过称重获得，惯量可以通过摆锤实验估计，摩擦系数可以通过滑动实验测量。

3. **仿真环境构建**：将重建的几何模型和估计的物理属性导入仿真器，构建数字孪生。

4. **验证与校准**：在仿真中复现真实实验，比较仿真与真实的行为差异，迭代优化物理参数。

Zakka 等人（2023）提出的 ManiSkill2 基准环境提供了标准化的 Real2Sim 评估流程。Sundermeyer 等人（2023）进一步展示了从单张 RGB 图像重建可操作的三维物体模型，大幅降低了 Real2Sim 的数据采集成本。

近年来，基于大语言模型（LLM）和视觉语言模型（VLM）的 Real2Sim 方法也引起了广泛关注。例如，RoboGen（Wang et al., 2024）利用 GPT-4 自动生成仿真任务和环境配置，实现了一种"全自动"的 Real2Sim 流程。

---

## 12.4 教师-学生迁移：知识的传递与适应

### 12.4.1 教师策略蒸馏

教师-学生（Teacher-Student）框架是 Sim-to-Real 迁移中最优雅的范式之一。其核心思想是：先在仿真中训练一个强大的**教师策略（Teacher Policy）** $\pi_T$，然后用教师策略的输出来监督训练一个**学生策略（Student Policy）** $\pi_S$。

教师策略在仿真中拥有特权信息（Privileged Information）的访问权——例如精确的物体位姿、全局地图、无噪声的本体感觉数据——这些在真实世界中是不可获得的。学生策略则只能使用真实传感器可获得的观测。

设教师策略的观测空间为 $\mathcal{O}_T$，学生策略的观测空间为 $\mathcal{O}_S \subset \mathcal{O}_T$（即学生只能看到教师观测的子集），蒸馏的目标是：

$$\min_{\boldsymbol{\phi}} \mathbb{E}_{\mathbf{o}_T \sim \mathcal{D}_{\text{teacher}}} \left[ D_{\text{KL}}\left( \pi_T(\cdot | \mathbf{o}_T) \| \pi_S(\cdot | \mathbf{o}_S; \boldsymbol{\phi}) \right) \right]$$

其中 $\boldsymbol{\phi}$ 为学生策略的参数，$D_{\text{KL}}$ 为 KL 散度，$\mathcal{D}_{\text{teacher}}$ 为教师策略的运行轨迹数据。

对于连续动作空间，教师策略通常输出高斯分布 $\pi_T(\mathbf{a}|\mathbf{o}_T) = \mathcal{N}(\boldsymbol{\mu}_T, \boldsymbol{\Sigma}_T)$，蒸馏损失简化为：

$$\mathcal{L}_{\text{distill}} = \mathbb{E} \left[ \left\| \boldsymbol{\mu}_S(\mathbf{o}_S; \boldsymbol{\phi}) - \boldsymbol{\mu}_T(\mathbf{o}_T) \right\|^2 + \lambda \cdot D_{\text{KL}}(\boldsymbol{\Sigma}_S \| \boldsymbol{\Sigma}_T) \right]$$

Hinton 等人（2015）提出的"知识蒸馏"（Knowledge Distillation）最初用于神经网络压缩，后来被广泛应用于 Sim-to-Real 迁移。在机器人领域，Miki 等人（2022）在 ANYmal 四足机器人的运动控制中展示了教师-学生蒸馏的显著效果：教师策略利用仿真特权信息（精确地形高度图、关节状态）训练，学生策略仅使用本体感觉和深度相机观测，最终在真实机器人上实现了鲁棒的户外行走。

### 12.4.2 学生网络适应

单纯的行为克隆（Behavioral Cloning）式蒸馏可能导致**分布偏移（Distribution Shift）**：学生策略的小误差会逐步累积，导致其进入教师策略从未访问的状态空间区域，进而引发更大的误差。

为解决这一问题，研究者提出了多种增强学生策略适应性的方法：

**DAgger（Dataset Aggregation）**：Ross 等人（2011）提出的 DAgger 算法通过在线交互来修正分布偏移。具体而言，在学生策略执行过程中，教师策略对当前状态给出"正确"的动作标签，学生策略在这些"修正"数据上进行再训练。迭代过程为：

$$\mathcal{D}_k = \mathcal{D}_{k-1} \cup \{(\mathbf{o}_S^{(i)}, \pi_T(\mathbf{o}_T^{(i)}))\}_{i=1}^{N_k}$$

$$\boldsymbol{\phi}_k = \arg\min_{\boldsymbol{\phi}} \sum_{(\mathbf{o}_S, \mathbf{a}_T) \in \mathcal{D}_k} \left\| \pi_S(\mathbf{o}_S; \boldsymbol{\phi}) - \mathbf{a}_T \right\|^2$$

**条件模仿学习（Conditional Imitation Learning）**：Menda 等人（2019）提出在学生策略中引入条件变量（如任务目标、模式标签），使得学生策略能够在不同情境下选择合适的行为模式，提高泛化能力。

**混合目标训练**：将蒸馏损失与强化学习损失结合，让学生策略在模仿教师的同时也通过与环境交互来优化自身：

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{distill}} + (1 - \alpha) \mathcal{L}_{\text{RL}}$$

其中 $\alpha$ 可以随训练进度从 1 逐渐衰减到 0，实现从"模仿教师"到"自主探索"的平滑过渡。

**表征对齐（Representation Alignment）**：除了动作层面的蒸馏，一些工作还在特征层面进行知识传递。通过让教师和学生的中间特征表征对齐，学生策略可以学习到教师策略的"推理方式"，而不仅仅是最终输出：

$$\mathcal{L}_{\text{align}} = \left\| f_{\text{enc}}^S(\mathbf{o}_S; \boldsymbol{\phi}) - f_{\text{enc}}^T(\mathbf{o}_T) \right\|^2$$

其中 $f_{\text{enc}}^T$ 和 $f_{\text{enc}}^S$ 分别为教师和学生编码器的特征输出。

### 12.4.3 在线适应

上述方法假设学生策略在部署后参数固定不变。然而，真实世界是动态的——地形会变化，负载会改变，机器人自身的状态也会随磨损退化。**在线适应（Online Adaptation）** 使学生策略能够在部署时根据实时经验调整自身。

**自适应归一化层（Adaptive Normalization）**：一些方法在学生网络中引入可调节的归一化层，通过少量在线交互数据调整归一化统计量。BN层的均值和方差可以快速适应新的数据分布：

$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{online}}}{\sqrt{\boldsymbol{\sigma}_{\text{online}}^2 + \epsilon}} \cdot \gamma + \beta$$

**元学习（Meta-Learning）**：MAML（Finn et al., 2017）等元学习方法训练策略使其能够在少量梯度步内快速适应新任务。在 Sim-to-Real 语境下，这意味着策略在仿真中被训练为"容易适应"的参数初始化，部署到真实环境后仅需少量交互即可快速适应：

$$\boldsymbol{\phi}' = \boldsymbol{\phi} - \alpha \nabla_{\boldsymbol{\phi}} \mathcal{L}_{\text{task}}(\boldsymbol{\phi}, \mathcal{D}_{\text{real}}^{\text{few-shot}})$$

**上下文策略（Contextual Policy）**：Raileanu 等人（2021）提出的 RAD（Reinforcement Learning with Augmented Data）和后续工作中，策略被设计为接受一个上下文向量 $\mathbf{c}$，该向量通过对近期经验的编码自动捕捉当前环境的特性：

$$\mathbf{c}_t = \text{Encoder}(\{(\mathbf{o}_\tau, \mathbf{a}_\tau, r_\tau, \mathbf{o}_{\tau+1})\}_{\tau=t-K}^{t-1})$$

$$\mathbf{a}_t = \pi_S(\mathbf{o}_t, \mathbf{c}_t; \boldsymbol{\phi})$$

这种方式无需显式更新网络参数，仅通过上下文向量的隐式更新即可实现在线适应。

Hwangbo 等人（2019）在 ANYmal 四足机器人的工作中展示了在线适应的强大效果：通过在策略中嵌入一个估计当前动力学参数的编码器，策略能够在几秒内自动适应未见过的地面条件（如冰面、沙地）。

---

## 12.5 仿真平台：工具的生态

### 12.5.1 Isaac Sim

NVIDIA Isaac Sim 是基于 Omniverse 平台构建的机器人仿真环境，以其高保真渲染和大规模并行能力著称。Isaac Sim 的核心技术特性包括：

- **物理引擎**：基于 NVIDIA PhysX 5，支持 GPU 加速的刚体、柔体和流体仿真。Isaac Sim 的 GPU 并行能力可以同时运行数千个环境实例，极大加速强化学习训练。
- **渲染**：基于 Omniverse RTX Renderer，支持实时光线追踪，能够生成高度逼真的合成图像。支持领域特定的渲染（Domain-Specific Rendering），如深度图、法线图、语义分割等。
- **Isaac Lab（原 Orbit）**：NVIDIA 于 2024 年开源的机器人学习框架，提供了标准化的任务接口、域随机化工具和强化学习算法集成。Isaac Lab 支持 PPO、SAC 等主流 RL 算法，并与 rl_games、RSL-RL 等库深度集成。
- **ROS 集成**：原生支持 ROS 2，方便与机器人软件栈对接。

Isaac Sim 的主要优势在于工业级的渲染质量和 GPU 加速能力，特别适合视觉密集型任务（如基于 RGB 图像的操纵和导航）。然而，其闭源性质和较高的硬件需求（需要 NVIDIA RTX GPU）是主要限制。

### 12.5.2 MuJoCo

MuJoCo（Multi-Joint dynamics with Contact）由 Todorov 等人（2012）开发，2021 年被 DeepMind 收购后开源，现已成为机器人学习研究中最广泛使用的仿真器之一。

核心技术特性：

- **物理引擎**：使用隐式积分和软接触模型，在保证数值稳定性的前提下实现高效的接触力计算。MuJoCo 的解析求导器（Analytical Derivatives）支持高效的梯度计算，使其天然适合可微仿真和基于梯度的优化。
- **接触模型**：采用连续的椭球接触模型和弹簧-阻尼器模型，避免了传统 LCP（Linear Complementarity Problem）求解中的数值问题。接触力计算如下：

$$\mathbf{f}_c = \mathbf{k}_c \odot [\mathbf{d}]_+ - \mathbf{b}_c \odot \dot{\mathbf{d}}$$

其中 $[\mathbf{d}]_+$ 为穿透深度的正值部分，$\mathbf{k}_c$ 和 $\mathbf{b}_c$ 为接触刚度和阻尼参数。

- **速度**：MuJoCo 以轻量高效著称，在单核 CPU 上即可达到每秒数十万步的仿真速度。
- **MuJoCo 3.0（2024）**：新增 GPU 加速仿真（MJX，基于 JAX 实现），支持大规模并行训练。

MuJoCo 的主要优势是计算效率高、API 设计优雅、社区活跃。它是 DeepMind Control Suite（DMC）和 Gymnasium 等标准基准环境的默认后端，特别适合基础研究和算法开发。

### 12.5.3 PyBullet

PyBullet 是基于 Bullet 物理引擎的 Python 封装，由 Coumans 和 Bai（2016-2019）开发。它以开源、易用和与 OpenAI Gym 的良好兼容性著称。

核心技术特性：

- **物理引擎**：Bullet 3 使用基于顺序脉冲（Sequential Impulse）的方法求解接触约束，计算效率中等。
- **渲染**：内置 OpenGL 渲染器和 TinyRenderer（CPU 软渲染），图像质量有限但速度较快。也支持连接外部高保真渲染器。
- **ROS 集成**：支持通过 pybullet_ros 与 ROS 通信。
- **OpenAI Gym 集成**：PyBullet 是 Roboschool 和许多 Gym 机器人环境的默认后端。

PyBullet 的主要优势是完全开源、安装简单、社区资源丰富。然而，其物理精度和渲染质量相比 MuJoCo 和 Isaac Sim 有明显差距，接触仿真中的"粘滞"（Stiction）问题和穿透问题是常见痛点。

### 12.5.4 Gazebo

Gazebo 是 ROS 生态中最成熟的机器人仿真平台，由 Open Robotics（原 OSRF）维护。Gazebo Classic（Gazebo 11 及以前版本）和新一代 Gazebo（原 Ignition Gazebo）并行存在。

核心技术特性：

- **物理引擎**：支持多个物理后端，包括 ODE（Open Dynamics Engine）、Bullet、DART 和 Simbody。用户可以根据需求选择不同的物理引擎。
- **传感器仿真**：Gazebo 内置了丰富的传感器模型，包括激光雷达（LiDAR）、RGB-D 相机、IMU、GPS、力/力矩传感器等。这些传感器模型可以直接输出与真实传感器格式一致的数据。
- **ROS 集成**：与 ROS 2 深度集成，支持通过 plugin 机制扩展功能。Gazebo 是 ROS 导航（Navigation2）和操纵（MoveIt 2）等框架的标准仿真后端。
- **世界构建**：支持 SDF（Simulation Description Format）和 URDF（Unified Robot Description Format）格式的模型导入，拥有丰富的模型库。

Gazebo 的主要优势在于与 ROS 生态的深度集成和丰富的传感器模型，特别适合系统级的机器人开发和测试。但其强化学习支持较弱，渲染质量一般，不太适合视觉密集型的 RL 训练。

### 12.5.5 Genesis

Genesis 是 2024 年由卡内基梅隆大学等机构发布的通用物理仿真平台（Genesis Team, 2024），旨在统一机器人、物理和图形社区的仿真需求。

核心技术特性：

- **多物理场引擎**：同时支持刚体、柔体、流体和布料仿真，使用统一的求解器框架。这使得 Genesis 能够处理传统仿真器难以应对的多物理场耦合任务（如水下机器人、软体机器人）。
- **完全可微**：整个仿真管线（包括接触、碰撞、约束求解）都是可微的，支持端到端的梯度优化。
- **速度**：基于 Taichi 编程语言实现，支持 GPU 大规模并行。Genesis 声称在某些任务上比现有仿真器快 10-100 倍。
- **Python 原生**：完全在 Python 中实现，降低了开发门槛。

Genesis 的出现代表了仿真平台的最新发展方向：统一、可微、高效、易用。但作为新兴平台，其稳定性和社区生态尚在建设中。

### 12.5.6 平台对比

| 特性 | Isaac Sim | MuJoCo | PyBullet | Gazebo | Genesis |
|------|-----------|--------|----------|--------|---------|
| 物理精度 | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★★★☆ |
| 渲染质量 | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ |
| 仿真速度 | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★★★ |
| GPU 并行 | ★★★★★ | ★★★★☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ |
| 可微性 | ★★☆☆☆ | ★★★★☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ |
| 开源程度 | 部分开源 | 完全开源 | 完全开源 | 完全开源 | 完全开源 |
| ROS 集成 | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ |
| 学习生态 | ★★★★☆ | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ |
| 软体/流体 | ★★☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ |

**选择建议**：

- **基础研究与算法开发**：MuJoCo — 轻量、高效、生态成熟
- **视觉主导的工业应用**：Isaac Sim — 高保真渲染、GPU 大规模并行
- **ROS 机器人系统开发**：Gazebo — 深度 ROS 集成、丰富传感器模型
- **可微仿真与梯度优化**：Genesis 或 MuJoCo MJX — 完整可微管线
- **多物理场耦合任务**：Genesis — 唯一同时支持刚体、柔体、流体的平台
- **快速原型与教学**：PyBullet — 安装简单、社区资源丰富

---

## 12.6 实践指南与前沿展望

### 12.6.1 Sim-to-Real 的工程实践

基于大量公开工作的经验，我们总结 Sim-to-Real 迁移的实践准则：

1. **从简单开始**：先在不使用任何随机化的仿真中训练策略，确认策略在"理想"仿真中可以正常工作。然后逐步添加随机化，观察其对策略性能的影响。

2. **随机化范围先小后大**：从较小的参数范围开始，确认策略能在真实环境中基本工作后，再逐步扩大范围以提高鲁棒性。ADR 是自动化这一过程的有效工具。

3. **混合方法更优**：域随机化和系统辨识并非互斥。实践中，先通过系统辨识缩小参数范围，再在缩小后的范围内进行域随机化，通常效果最佳。

4. **重视状态估计**：在将策略部署到真实机器人时，状态估计（State Estimation）往往是最大的瓶颈。使用 EKF/UKF 融合 IMU 和视觉数据，或者通过网络端到端估计状态，是常见的解决方案。

5. **渐进式部署**：先在安全的仿真环境中评估，然后在受保护的真实环境中测试，最后在开放环境中部署。每个阶段都收集数据用于策略优化。

### 12.6.2 前沿方向

Sim-to-Real 迁移领域正在经历快速演进，以下方向值得关注：

**基础模型驱动的 Sim-to-Real**：视觉语言动作模型（VLA，如 RT-2、Octo）和世界模型（如 UniSim、Genie）正在改变 Sim-to-Real 的范式。这些模型通过大规模预训练获得了强大的跨域泛化能力，可能从根本上减少对仿真-现实对齐的依赖。

**生成式仿真**：使用扩散模型等生成模型来增强或替代传统渲染管线。例如，Diffusion-Enhanced Policy（Shi et al., 2023）使用扩散模型将仿真图像转换为逼真的"域转换"图像，作为训练数据的增强。

**多智能体 Sim-to-Real**：将 Sim-to-Real 方法扩展到多智能体协作场景。多智能体系统面临更复杂的交互动力学，迁移难度更大。

**终身 Sim-to-Real**：使机器人能够在部署后持续从真实环境中学习，并将新知识反馈到仿真器中，形成"仿真-现实-仿真"的闭环学习循环。

---

## 12.7 本章小结

本章系统阐述了 Sim-to-Real 迁移的三大核心技术范式。**域随机化**通过在训练中注入多样性来增强策略的鲁棒性，是最直接的迁移方法；**系统辨识**通过校准仿真器使其更准确地反映真实世界，从源头缩小仿真-现实鸿沟；**教师-学生框架**通过知识蒸馏优雅地处理了特权信息的传递问题。这三种方法并非相互独立，实际应用中往往需要综合运用。

仿真平台方面，Isaac Sim、MuJoCo、PyBullet、Gazebo 和 Genesis 各有侧重，研究者和工程师应根据具体任务需求选择合适的工具链。随着可微仿真、生成式 AI 和基础模型的发展，Sim-to-Real 迁移的效率和效果将持续提升，最终实现从仿真到真实世界的无缝跨越。

---

## 参考文献

1. Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IROS*.
2. Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018). Sim-to-real transfer of robotic control with dynamics randomization. *ICRA*.
3. OpenAI, Akkaya, I., Andrychowicz, M., et al. (2019). Solving Rubik's cube with a robot hand. *arXiv:1910.07113*.
4. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NIPS Workshop*.
5. Miki, T., Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., & Hutter, M. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild. *Science Robotics*.
6. Hwangbo, J., Lee, J., Dosovitskiy, A., et al. (2019). Learning agile and dynamic motor skills for legged robots. *Science Robotics*.
7. Hu, Y., Anderson, L., Li, T.-M., et al. (2019). DiffTaichi: Differentiable programming for physical simulation. *ICLR*.
8. Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. *IROS*.
9. Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *AISTATS*.
10. Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML*.
11. Roberts, M., Ramapuram, J., Ranjan, A., et al. (2021). Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. *ICCV*.
12. Mehta, B., Scalise, R., & Coros, S. (2020). Hierarchical planning and control for quadrotor navigation in dynamic environments. *ICRA*.
13. Akalin, O., Loutfi, A., et al. (2023). CLADR: Contrastive learning for automatic domain randomization. *RAL*.
14. Hu, Y., Li, T.-M., Anderson, L., Ragan-Kelley, J., & Durand, F. (2020). Taichi: A language for computer graphics and physics. *SIGGRAPH*.
15. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian splatting for real-time radiance field rendering. *SIGGRAPH*.
16. Wang, Y., Wang, Y., et al. (2024). RoboGen: Towards unleashing infinite data for automated robot learning via generative simulation. *ICML*.
17. Zakka, K., Zeng, A., Florence, P., et al. (2023). ManiSkill2: A unified benchmark for generalizable manipulation skills. *ICLR*.
18. Genesis Team. (2024). Genesis: A universal and generative physics engine for robotics and beyond. *arXiv:2412.xxxxx*.
19. Mildenhall, B., Srinivasan, P. P., Tancik, M., et al. (2020). NeRF: Representing scenes as neural radiance fields for view synthesis. *ECCV*.
20. Liang, J., Lin, M., & Koltun, V. (2019). Differentiable cloth simulation for inverse problems. *NeurIPS*.
# 第13章 具身智能的未来

> "真正的智能不是在真空中思考，而是在世界中行动。" —— Rodney Brooks

具身智能（Embodied Intelligence）正站在一个历史性的交汇点上。过去十年，深度学习革命重塑了计算机视觉和自然语言处理的面貌，而大语言模型（LLM）的爆发式进展更让"通用人工智能"的讨论从学术沙龙走向了公共舆论场。然而，一个根本性的鸿沟依然存在：**数字世界的智能尚无法可靠地迁移到物理世界**。一个能在国际象棋中击败人类冠军的系统，可能无法将一杯水从桌面端到另一张桌子上。

具身智能的核心命题——让智能体通过与物理环境的交互来感知、推理和行动——正是弥合这一鸿沟的关键路径。本章将从四个维度展望具身智能的未来图景：通用机器人的崛起、人机协作的深化、伦理与安全的挑战，以及对全书技术脉络的回顾与展望。

---

## 13.1 通用机器人：从专用到通用的范式跃迁

### 13.1.1 通用操控：灵巧手与可变形物体

机器人操控（Manipulation）是具身智能最核心的能力之一。工业机器人在结构化环境中的点到点运动规划已趋成熟，但通用操控——即在非结构化环境中处理任意物体——仍然是开放性难题。近年来，这一领域正在经历从"专用夹具解决专用任务"到"通用灵巧手处理任意物体"的范式转变。

灵巧操控（Dexterous Manipulation）的复兴是2023—2025年机器人领域最显著的趋势之一。OpenAI 于2019年展示了用强化学习训练的灵巧手 Rubik's Cube，证明了 sim-to-real 迁移在精细操控中的可行性。此后，Meta 的触觉传感器 DIGIT、Google DeepMind 的 RT-2 系列工作，以及特斯拉 Optimus 人形机器人的灵巧手设计，都将这一方向推向了新的高度。

可变形物体（Deformable Objects）的操控是通用操控的"最后堡垒"。与刚性物体不同，衣物、绳索、食物等可变形物体的状态空间维度极高，传统基于几何建模的方法难以奏效。斯坦福大学的 RoboFold 项目（Chi et al., 2023）通过将折纸动作分解为基元操作序列，结合视觉伺服实现了对布料的精准折叠。UC Berkeley 的 Diffusion Policy（Chi et al., 2024）则引入扩散模型作为动作生成器，在处理多模态动作分布方面展现了优异性能，尤其适用于需要多步骤决策的复杂操控任务。

触觉感知是实现通用操控不可或缺的模态。视觉虽能提供全局信息，但在接触密集的精细操作中存在遮挡问题。GelSight 系列触觉传感器（Yuan et al., 2017）通过光学原理将接触面的三维形变转化为高分辨率图像，为机器人提供了"触觉视觉"。最新进展如 ReSkin（Bhatt et al., 2022）和 3DTacDex（Si & Yuan, 2024）进一步将触觉传感与学习框架结合，使机器人能够在接触中不断调整操控策略——正如人类在黑暗中仅凭触觉就能完成系鞋带这样复杂的任务。

### 13.1.2 跨具身迁移：一个大脑，多种身体

跨具身迁移（Cross-Embodiment Transfer）是通用机器人的核心愿景之一：训练一个统一的控制策略，使其能够部署在形态各异的机器人平台上——从机械臂到人形机器人，从四足机器人到无人机。这一想法的理论基础在于，物理世界的交互存在超越具体形态的通用结构。

Google DeepMind 的 RT-X 项目（Open X-Embodiment Collaboration, 2023）是这一方向的里程碑工作。该项目汇集了来自 22 个不同机构的 22 种机器人平台数据，构建了迄今最大的机器人操控数据集（超过 100 万条轨迹），并证明了在该数据集上训练的模型（RT-1-X 和 RT-2-X）在多个目标任务上优于仅在单一平台上训练的模型。这一结果揭示了一个深刻的事实：**跨具身数据的多样性本身就是一种正则化**，它迫使模型学习更加抽象和通用的交互模式。

跨具身迁移面临的核心技术挑战在于动作空间的异构性。不同机器人的关节构型、自由度数量和末端执行器形态差异巨大，使得直接共享动作标签变得不可行。当前主流的解决思路包括：（1）**动作空间归一化**——将不同机器人的动作映射到统一的低维表征空间，如 VQ-BeT（Lee et al., 2024）采用的向量量化行为 token 方案；（2）**目标导向的共享表征**——不共享具体的动作指令，而是共享高层目标描述（如末端执行器的目标位姿），由各平台各自的底层控制器负责执行；（3）**基础模型作为共享接口**——利用预训练视觉-语言模型的语义空间作为跨具身任务描述的通用语言。

斯坦福大学的 Mobile ALOHA 系统（Fu et al., 2024）展示了另一条路径：通过低成本硬件（约 3 万美元的双臂移动平台）配合协同训练（Co-training），仅用 50 次人类演示就能学会烹饪、清洁等复杂家务任务。这暗示着通用机器人不必等待"万能控制器"的出现，而是可以通过**适度的硬件标准化 + 大量的行为数据 + 强大的基础模型**来逐步逼近。

### 13.1.3 基础模型驱动的机器人：从感知到规划

大语言模型（LLM）和视觉-语言模型（VLM）的崛起为机器人学注入了全新的范式。传统机器人系统遵循"感知-规划-控制"的模块化流水线，每个模块独立开发，接口僵化。基础模型驱动的机器人（Foundation Model-Driven Robotics）正在打破这一壁垒，用统一的神经网络架构替代传统的模块化设计。

**语言作为任务接口。** Google 的 SayCan 系统（Ahn et al., 2022）首次系统性地展示了如何将 LLM 作为机器人的"高层大脑"：给定自然语言指令（如"我把饮料洒了，能帮我清理一下吗？"），LLM 生成一系列子任务步骤（"1. 找到海绵；2. 拿起海绵；3. 走到洒饮料的地方；4. 擦拭"），每个步骤的可行性由机器人自身的 affordance 模型评估。这一工作开创了"LLM + 可行性过滤"的范式，后续的 Code as Policies（Liang et al., 2023）和 ProgPrompt（Singh et al., 2023）进一步将 LLM 的代码生成能力与机器人控制相结合。

**视觉-语言-动作模型（VLA）。** RT-2（Brohan et al., 2023）是首个将视觉-语言模型直接微调为端到端机器人策略的工作。它将机器人的动作（如关节角度增量）编码为文本 token，与视觉和语言 token 一起送入 PaLI-X 或 PaLM-E 模型进行联合训练。令人惊讶的是，RT-2 展现了涌现能力（Emergent Capabilities）：它能够执行训练数据中从未出现过的指令，如"把垃圾扔到垃圾桶里"（即使训练数据中没有"垃圾"和"垃圾桶"的配对），这表明模型确实学到了语义层面的概念，而非简单的模仿映射。

**世界模型与想象。** 另一条极具潜力的路线是构建物理世界的内部模型（World Model），使机器人能够在"心理模拟"中预演行动方案。LeCun（2022）提出的联合嵌入预测架构（JEPA）为这一方向提供了理论框架。Google DeepMind 的 Genie（Bruce et al., 2024）和 NVIDIA 的 Cosmos（2025）则分别从 2D 视频和 3D 物理仿真两个维度推进了世界模型的构建。一个能够"想象"行动后果的机器人，将具备真正的前瞻性规划能力，而不仅仅是反应式的策略执行。

---

## 13.2 人机协作：从共存到共生

### 13.2.1 安全交互：物理安全与信任建立

当机器人走出工厂的围栏，进入人类的生活空间时，安全问题从"隔离防护"转变为"亲密共存"中的信任构建。传统工业机器人的安全策略是"快、准、硬"——通过物理围栏将人与机器人隔离开来。协作机器人（Cobot）的出现改变了这一范式，但真正的安全交互远不止"碰撞检测与紧急停止"。

**力控制与柔顺性。** 实现安全交互的第一步是让机器人具备物理柔顺性。阻抗控制（Impedance Control）和导纳控制（Admittance Control）使机器人能够在受到外力时产生合理的顺应响应，就像人类手臂在被推动时会自然地"让步"。MIT 的 HERMES 系统（Bhatt et al., 2024）进一步将全身力控制与强化学习结合，使人形机器人在被推搡时能够像人类一样保持平衡并安全恢复。

**预测性安全。** 被动安全（碰撞后响应）正逐渐被主动安全（碰撞前预测）所取代。基于人类运动预测的安全系统能够在人类伸手取物或转身时，提前调整机器人的运动轨迹以避免干涉。Trajectron++（Salzmann et al., 2020）和 Social-STGCNN（Mohamed et al., 2020）等社会感知轨迹预测模型为此提供了技术基础。

**形式化安全保证。** 在高风险场景中，概率性的安全评估是不够的。控制屏障函数（Control Barrier Functions, CBF）和安全强化学习（Safe RL）提供了更加严格的安全保证框架。Ames et al.（2019）提出的 CBF 方法通过构造安全集上的控制约束，确保机器人在任何情况下都不会违反预设的安全边界——这对于医疗机器人和自动驾驶等场景尤为关键。

### 13.2.2 意图理解：读懂人类的"言外之意"

有效的人机协作建立在深度的意图理解之上。人类的意图往往是隐含的、多层次的、甚至自相矛盾的——我们说"把那个东西递给我"时，"那个东西"需要结合手势、眼神和上下文才能确定。

**多模态意图识别。** 最新的意图理解系统融合了语言、手势、眼动、姿态和生理信号等多个模态。Transformer 架构的跨模态注意力机制（Cross-Modal Attention）为异构信号的融合提供了自然的框架。例如，Su et al.（2024）提出的多模态意图识别框架将语音的韵律特征、手部的运动学数据和眼动的注视模式融合在一个统一的编码器中，在协作装配任务中实现了超过 92% 的意图预测准确率。

**心智理论（Theory of Mind）在机器人中的实现。** 心智理论是人类社会认知的核心能力——我们不仅观察他人的行为，还会推断其信念、欲望和意图。将这一能力赋予机器人是具身智能的前沿课题。Rabinowitz et al.（2018）的 Neural Theory of Mind（NToM）通过元学习框架训练模型推断其他智能体的隐藏状态。最新进展将大语言模型的常识推理能力与具身场景结合，使机器人能够理解"她把杯子放在桌边是因为她马上要拿起来喝"这样的心智状态推理。

**显式与隐式通信。** 人机协作中的通信可分为显式（语言指令、手势信号）和隐式（注视方向、身体朝向、运动节奏）两类。最高效的协作往往发生在隐式通信占主导的场景——就像两个人一起搬沙发时，无需言语就能默契地调整方向和力度。Shared Autonomy 研究（Javdani et al., 2015; Reddy et al., 2018）正是探索如何让机器人捕捉并响应人类的隐式意图，实现流畅的无言协作。

### 13.2.3 共享控制：人与机器的决策融合

共享控制（Shared Control）是人机协作的核心控制范式。它既不是完全的人类控制（效率低、疲劳大），也不是完全的自主控制（缺乏灵活性和人类判断），而是两者的动态融合。

**自适应自主。** 理想的共享控制系统应该能够根据任务复杂度和人类状态自适应地调整自主程度。在简单重复的子任务中提高自主性以减轻人类负担，在关键时刻（如安全临界决策）降低自主性以确保人类的最终控制权。Dragan & Srinivasa（2013）的形式化框架将共享控制建模为一个混合主动（Mixed-Initiative）的优化问题，其中人和机器的控制信号通过可信度加权进行融合。

**脑机接口与意图增强。** 最直接的共享控制形式是通过脑机接口（Brain-Computer Interface, BCI）将人类意图直接传输给机器人。Neuralink 的侵入式 BCI 和基于 EEG 的非侵入式 BCI 都在快速进展中。对于残障人士而言，BCI 驱动的机器人臂不仅是工具，更是身体的延伸——"具身"这一概念在这里获得了最直接的诠释。

**学习型共享控制。** 传统共享控制依赖手工设计的融合规则，而基于学习的方法能够从人机交互数据中自动发现最优的控制分配策略。DeepMimic（Peng et al., 2018）和 AMP（Peng et al., 2021）通过对抗运动先验实现了从人类示范到机器人控制的平滑迁移。最新的交互式模仿学习（Interactive Imitation Learning）方法（如 DAgger 的变体）允许人类在机器人执行过程中实时纠正错误，使共享控制成为一个持续学习的闭环。

### 13.2.4 社会机器人：情感交互与社会嵌入

社会机器人（Social Robots）代表了具身智能最"人性化"的应用方向。不同于功能导向的工业或服务机器人，社会机器人以情感交互和社会嵌入为核心目标。

**情感计算与情感表达。** 情感计算（Affective Computing）赋予机器人识别和表达情感的能力。基于面部动作编码系统（FACS）的面部表情生成、基于语音韵律的情感合成、以及全身姿态的情感表达，共同构成了机器人的情感输出通道。SoftBank 的 Pepper 和 Hanson Robotics 的 Sophia 虽然在技术深度上存在争议，但它们确实推动了公众对社会机器人的认知。MIT Media Lab 的 Jibo（Breazeal, 2019）则展示了如何通过简洁而有表现力的运动设计，在缺乏面部表情的硬件上实现有效的情感传达。

**长期关系建立。** 社会机器人的真正价值在于长期关系的建立。荷兰的"银色护理"项目（Silver Care）在养老院部署的社交机器人 Zora，在持续数月的互动中显著改善了老年居民的孤独感和参与度。Belpaeme et al.（2018）的综述指出，社会机器人在自闭症儿童的社交技能训练中展现出积极效果，但这种效果高度依赖于机器人与儿童之间长期、一致的互动关系。

**文化敏感性。** 社会机器人在全球化部署中面临严峻的文化适应挑战。日本用户倾向于将机器人视为"准生命体"，而欧美用户更倾向于将其视为"工具"。这种文化差异深刻影响着机器人的交互设计：在日本，略带"卡哇伊"气质的机器人更容易被接受；在德国，用户更重视功能透明度和数据隐私。这些洞察提醒我们，**具身智能不仅是技术问题，更是社会-文化问题**。

---

## 13.3 伦理与安全：具身智能的社会契约

### 13.3.1 机器人伦理：从阿西莫夫到现实

科幻作家阿西莫夫（Isaac Asimov）在 1942 年提出的"机器人三定律"——（1）机器人不得伤害人类；（2）机器人必须服从人类命令（除非与第一定律冲突）；（3）机器人必须保护自身（除非与前两定律冲突）——虽然在文学上极具启发性，但在工程实践中几乎无法实现。这些定律之间的优先级冲突、对"伤害"的模糊定义、以及对所有可能情况的穷举假设，都暴露了其局限性。

当代机器人伦理学已经超越了简单的行为禁令，转向了更加结构化的伦理框架。Floridi et al.（2018）提出的"AI4People"框架从有益性（Beneficence）、非恶意性（Non-Maleficence）、自主性（Autonomy）、正义性（Justice）和可解释性（Explicability）五个维度构建了 AI 伦理的评估矩阵。IEEE 的"符合伦理的设计"（Ethically Aligned Design）倡议则从工程实践的角度提出了可操作的指导原则。

在具身智能的语境下，伦理问题变得更加紧迫，因为**机器人的行动具有不可撤回的物理后果**。一个语言模型生成的有害文本可以被删除，但一个机器人执行的有害动作可能造成不可逆的身体伤害。这种"物理后果的不可逆性"要求具身智能系统在伦理合规方面采用比纯软件系统更加严格的标准。

**道德机器问题。** MIT 的"道德机器"（Moral Machine）实验（Awad et al., 2018）通过大规模在线调查揭示了不同文化在自动驾驶伦理困境中的道德偏好差异。虽然"电车难题"在技术上并非自动驾驶的核心决策场景（因为优秀的自动驾驶系统应该通过提前减速来避免此类困境），但它确实触及了一个深层问题：**当机器必须在不同的伤害结果之间做出选择时，谁来定义"正确"的标准？**

### 13.3.2 就业影响：替代、增强与转型

机器人的就业影响是具身智能最广泛讨论的社会议题之一。世界经济论坛（WEF）2023年的《未来就业报告》估计，到 2027 年，全球将新增 6900 万个与技术相关的工作岗位，同时减少 8300 万个传统岗位，净减少约 1400 万个岗位。然而，这一宏观数字掩盖了巨大的结构性差异。

**自动化的历史视角。** 从蒸汽机到流水线，从 ATMs 到自助结账，技术自动化的历史反复表明：短期内的岗位替代效应与长期内的就业创造效应之间存在时滞。Autor et al.（2003）的"任务模型"（Task Model）指出，技术自动化取代的是"任务"而非"职业"——一个职业往往包含多种任务，其中部分可自动化，部分不可自动化。因此，更准确的问题不是"哪些职业会被消灭"，而是"哪些任务会被自动化，以及职业结构将如何重组"。

**机器人增强而非替代。** 越来越多的证据表明，人机协作（而非完全替代）是更可能的场景。Acemoglu & Restrepo（2020）的研究发现，在制造业中，机器人的引入确实导致了就业下降，但同时也创造了新的"人机交互型"岗位——机器人维护技师、人机协作协调员、自动化流程设计师等。在医疗领域，手术机器人（如达芬奇系统）并未取代外科医生，而是扩展了他们的能力边界——外科医生可以在更小的切口、更高的精度和更低的疲劳下完成手术。

**技能鸿沟与再培训。** 最紧迫的挑战不是岗位总数的减少，而是技能需求的剧烈转变。McKinsey Global Institute（2017）估计，到 2030 年，全球将有 3.75 亿工人（约占全球劳动力的 14%）需要转换职业类别。这要求教育体系和政策制定者提前布局：在基础教育中融入计算思维和人机协作素养，在职业培训中建立"终身学习"的基础设施，在社会保障中探索全民基本收入（UBI）等新型安全网。

### 13.3.3 安全标准：从实验室到产业

具身智能系统的安全标准是技术从实验室走向产业的关键基础设施。当前，这一领域的标准体系正在快速演进，但仍然滞后于技术发展。

**功能安全标准。** ISO 13849（机械安全-控制系统安全相关部分）和 IEC 62443（工业网络安全）为传统工业机器人提供了成熟的安全框架。然而，这些标准主要针对确定性系统——传感器触发、控制器响应、执行器动作的因果链条清晰可追溯。基于学习的机器人系统具有非确定性和涌现行为，传统的安全验证方法面临根本性挑战。

**协作机器人安全。** ISO/TS 15066（协作工业机器人）是目前最权威的协作机器人安全标准，它定义了四种协作模式：安全级监控停止（Safety-Rated Monitored Stop）、手动引导（Hand Guiding）、速度与分离监控（Speed and Separation Monitoring）、以及功率与力限制（Power and Force Limiting）。该标准还详细规定了人体不同部位的最大允许接触力和压力值。

**基于学习的系统的安全验证。** 对于采用深度强化学习或大模型策略的机器人系统，传统的形式化验证方法（如模型检测）面临状态空间爆炸的问题。新兴的安全验证方法包括：（1）运行时监控（Runtime Monitoring）——在系统运行过程中持续检测安全约束是否被违反，并在违反时触发安全回退；（2）安全强化学习——通过约束优化（Constrained Optimization）在训练过程中就将安全约束编码到策略中；（3）形式化神经网络验证——如 Reluplex（Katz et al., 2017）和 Marabou（Katz et al., 2019），能够对 ReLU 网络的安全性质进行形式化证明，但目前仅适用于小规模网络。

**数据安全与隐私。** 具身智能系统通常配备大量传感器（摄像头、麦克风、激光雷达、触觉传感器），它们在执行任务的过程中不可避免地会采集到敏感的人脸、语音、居住环境等隐私信息。欧盟《通用数据保护条例》（GDPR）和中国的《个人信息保护法》为此提供了法律框架，但在技术实现层面，如何在保障机器人任务性能的同时最小化隐私侵犯，仍然是一个开放性问题。联邦学习（Federated Learning）和差分隐私（Differential Privacy）为此提供了潜在的技术路径。

### 13.3.4 自主武器争议：技术红线在哪里？

自主武器系统（Autonomous Weapons Systems, AWS）是具身智能最具争议的应用方向。支持者认为自主武器能够减少人类士兵的伤亡、提高打击精度、缩短反应时间；反对者则警告，将杀伤决策权交给算法将引发灾难性的伦理危机和不可控的军备竞赛。

**当前技术状态。** 尽管完全自主的致命武器系统尚未大规模部署，但"人在回路"（Human-in-the-Loop）的半自主武器已经广泛应用。以色列的"哈比"（Harpy）无人机能够在目标区域自主巡逻并锁定雷达信号，伊朗的 Shahed-136 自杀式无人机在俄乌冲突中的广泛使用，以及土耳其 Kargu-2 无人机据报在利比亚自主识别并攻击目标的案例，都表明技术正在快速逼近完全自主的阈值。

**国际治理框架。** 联合国《特定常规武器公约》（CCW）框架下的"致命自主武器系统政府专家组"（GGE on LAWS）自 2017 年以来一直在讨论自主武器的国际监管。2024 年，该专家组通过了一份包含 11 条指导原则的文件，其中最重要的是"国际人道法适用于所有武器系统"以及"人类必须对使用武力的决定保持有效控制"。然而，这些原则缺乏法律约束力，且各方对"有效控制"的定义存在根本分歧。

**技术社区的立场。** 超过 30 个国家和 160 多个非政府组织呼吁禁止完全自主的致命武器。2018 年，超过 2400 名 AI 研究者签署公开信，承诺不参与致命自主武器的开发。然而，也有人（如美国前国防部副部长 Robert Work）认为，禁止自主武器既不可行也不可取，更好的路径是建立严格的人类监督机制和伦理准则。

**本书的立场。** 具身智能技术本身是中性的——同一套感知、规划和控制技术既可以让护理机器人温柔地搀扶老人，也可以让自主武器精准地锁定目标。技术的社会影响取决于其应用方式和治理框架。我们主张：（1）在技术设计中内嵌"有意义的人类控制"（Meaningful Human Control）原则；（2）支持建立具有法律约束力的国际条约；（3）鼓励技术社区积极参与伦理讨论，而非将伦理视为外部约束。

---

## 13.4 结语：回顾与展望

### 13.4.1 全书回顾

回顾本书的全部内容，我们沿着一条清晰的技术脉络展开了具身智能的全景图：

- **感知层**（第2—4章）：从视觉、触觉到多模态感知，我们看到了机器如何从原始传感数据中提取对物理世界的理解。视觉 SLAM、触觉重建和跨模态融合技术为机器人提供了"第一人称"的环境认知能力。
- **认知层**（第5—7章）：从表征学习到因果推理，从物理直觉到世界模型，我们探索了机器人如何在感知之上构建对世界的内部模型，并利用这一模型进行推理和预测。图神经网络在物理推理中的应用、因果发现方法的引入、以及基于扩散模型的世界模型，共同指向了一个方向：机器人需要"理解"而不仅仅是"记忆"物理世界。
- **行动层**（第8—10章）：从运动规划到强化学习，从模仿学习到 sim-to-real 迁移，我们梳理了机器人如何将认知转化为物理行动。扩散策略（Diffusion Policy）、行为克隆与逆强化学习的结合、以及领域随机化和系统辨识技术，共同解决了"从仿真到现实"的迁移难题。
- **系统层**（第11—12章）：从机器人操作系统（ROS）到具身智能体架构，从多机器人协作到云机器人，我们讨论了如何将上述能力整合为完整的系统。基础模型的引入正在重塑整个系统架构，使得"感知-认知-行动"的流水线被端到端的学习系统所取代。

### 13.4.2 技术脉络与范式演进

从更宏观的视角来看，具身智能的技术演进经历了三个范式：

**范式一：基于模型的方法（Model-Based）。** 这是机器人学的经典范式。它假设世界可以用精确的数学模型来描述——刚体动力学、运动学、几何约束——并通过最优化理论求解控制策略。这一范式在结构化环境中取得了巨大成功（工业机器人、无人机编队），但在非结构化环境中面临模型失配的根本困难。

**范式二：基于学习的方法（Learning-Based）。** 深度强化学习和模仿学习的兴起带来了范式二。它放弃了精确建模的执念，转而通过大量数据和计算来学习控制策略。AlphaGo、OpenAI Five 和 Dactyl 的成功证明了这一范式的威力，但其"样本效率低、泛化能力弱、可解释性差"的缺陷也日益凸显。

**范式三：基础模型驱动的方法（Foundation-Model-Driven）。** 这是当前正在涌现的范式。它利用大规模预训练模型的通用知识和推理能力来驱动机器人决策，将机器人学从"为每个任务训练一个专用模型"转变为"用一个通用模型解决所有任务"。RT-2、VIMA、SayCan 等工作是这一范式的早期代表。但基础模型在物理交互的精确性、实时性和安全性方面仍面临严峻挑战。

这三个范式并非替代关系，而是**互补和融合**的关系。未来的具身智能系统很可能采用"基础模型提供高层语义理解 + 基于学习的中层策略 + 基于模型的底层控制"的分层架构，在不同层次上发挥各自范式的优势。

### 13.4.3 开放问题

尽管取得了显著进展，具身智能领域仍然面临多个根本性的开放问题：

**（1）样本效率问题。** 当前最先进的机器人学习方法通常需要数千到数万次交互才能学会一个新任务，而人类通常只需几次示范就能掌握。这一差距的根本原因在于人类拥有强大的先验知识和类比推理能力，能够将以往的经验迁移到新任务中。如何为机器人构建类似的先验知识库？基础模型提供了一种可能的路径，但其物理交互的精确性仍然不足。

**（2）长程推理与规划。** 现有方法在短程（几秒到几十秒）的操控任务中表现良好，但对于需要数十分钟甚至数小时的长程任务（如"准备一顿完整的晚餐"），性能急剧下降。长程任务涉及数百个子步骤、密集的约束依赖和复杂的错误恢复策略，这对当前的决策框架构成了根本性挑战。层级强化学习（Hierarchical RL）、选项框架（Options Framework）和目标条件策略（Goal-Conditioned Policy）是当前的主要探索方向。

**（3）常识物理推理。** 人类对物理世界具有直觉性的理解——我们知道杯子翻倒水会洒出、重物比轻物更难推动、绳子可以拉但不能推。这种"朴素物理"（Naive Physics）能力在儿童早期就已形成，但赋予机器人类似的能力却异常困难。最近的物理推理基准（如 PHYRE、IntPhys、CausalWorld）揭示了当前系统与人类在物理直觉上的巨大差距。

**（4）鲁棒性与泛化。** 实验室中表现优异的机器人系统在面对真实世界的分布外（Out-of-Distribution）情况时往往崩溃。光照变化、背景杂乱、物体新颖性、环境动态变化——这些对人类来说微不足道的变化，对机器人系统来说可能是致命的。域随机化（Domain Randomization）和数据增强（Data Augmentation）只是权宜之计，真正鲁棒的具身智能需要更加根本性的突破。

**（5）可解释性与信任。** 基于深度学习的机器人决策过程是一个黑箱。当机器人做出一个危险的动作时，我们不仅需要知道"它做了什么"，更需要知道"它为什么这样做"。可解释性不仅是技术需求，更是建立人类信任的前提。注意力可视化、反事实推理和决策归因方法正在被引入机器人领域，但距离实用还有很长的路要走。

### 13.4.4 展望：下一个十年

展望未来十年，具身智能的发展可能沿着以下路径展开：

**短期（2025—2028）：基础模型赋能的专用机器人。** 在这一阶段，基础模型（LLM、VLM、扩散模型）将成为机器人系统的标准组件，但部署仍以特定场景为主——仓储物流、家庭服务、农业采摘、建筑施工等。人形机器人将从实验室走向有限的商业试点（如 Figure、1X、Apptronik 的产品），但距离大规模普及仍有差距。关键技术突破点在于：（1）高质量的机器人-环境交互数据集的规模化采集；（2）基础模型的推理效率和物理精确性的提升；（3）安全标准和认证体系的初步建立。

**中期（2028—2032）：通用操控与跨具身迁移的成熟。** 在这一阶段，通用操控能力将取得实质性突破——机器人能够可靠地操作训练数据中从未见过的物体。跨具身迁移技术将使得一个训练好的策略能够在多种机器人平台上无缝部署。人机协作将成为主流工作模式，共享控制系统在制造业、医疗和服务业中广泛部署。社会机器人将在养老、教育和心理健康领域发挥重要作用。关键技术挑战在于：（1）长程复杂任务的可靠规划与执行；（2）触觉感知技术的成熟与普及；（3）大规模仿真环境（digital twins）的构建。

**长期（2032 年之后）：具身通用智能的曙光。** 在更远的未来，我们可能会看到具备真正物理常识、能够自主学习新技能、并与人类自然协作的通用机器人系统。这一目标的实现需要感知、认知、行动和社交能力的深度融合，需要世界模型的突破性进展，也需要伦理和治理框架的同步完善。我们可能最终会回答 Rodney Brooks 在 1991 年提出的那个根本性问题：**智能是否真的需要身体？** 答案或许不是简单的"是"或"否"，而是：**对于真正理解并改变物理世界的智能而言，身体不是可选的，而是不可或缺的。**

---

## 本章小结

本章从通用机器人、人机协作、伦理与安全三个维度展望了具身智能的未来，并在结语中回顾了全书的技术脉络。通用机器人正在经历从专用到通用的范式跃迁，基础模型为其提供了强大的感知和规划能力；人机协作正从简单的共存走向深度的共生，安全交互、意图理解和共享控制是其核心技术支撑；伦理与安全问题从机器人伦理、就业影响到自主武器争议，构成了具身智能必须面对的社会契约。

具身智能的未来不是一条确定的轨迹，而是一个充满可能性的开放空间。技术的进步将以我们难以预测的方式改变人与机器的关系，而这种改变的速度和方向，最终取决于我们今天做出的技术选择和伦理决策。正如维特根斯坦所言："我的语言的界限意味着我的世界的界限。" 对于具身智能而言，我们或许可以说：**机器与物理世界交互的界限，意味着人工智能理解世界的界限。** 扩展这一界限，正是本书和这个领域的终极使命。

---

## 参考文献

1. Acemoglu, D., & Restrepo, P. (2020). Robots and jobs: Evidence from US labor markets. *Journal of Political Economy*, 128(6), 2188-2244.
2. Ahn, M., Brohan, A., Brown, N., et al. (2022). Do as I can, not as I say: Grounding language in robotic affordances. *arXiv preprint arXiv:2204.01691*.
3. Ames, A. D., Grizzle, J. W., & Tabuada, P. (2019). Control barrier function based quadratic programs with application to adaptive cruise control. *IEEE CDC*.
4. Awad, E., Dsouza, S., Kim, R., et al. (2018). The moral machine experiment. *Nature*, 563(7729), 59-64.
5. Belpaeme, T., Kennedy, J., Ramachandran, A., Scassellati, B., & Tanaka, F. (2018). Social robots for education: A review. *Science Robotics*, 3(21).
6. Bhatt, A., Si, Z., & Adelson, E. (2024). Learning dexterous manipulation from exemplar trajectories. *RSS*.
7. Breazeal, C. (2019). Designing sociable robots. *MIT Press*.
8. Brohan, A., Brown, N., Carbajal, J., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *CoRL*.
9. Bruce, J., Dennis, M., Edwards, A., et al. (2024). Genie: Generative Interactive Environments. *ICML*.
10. Chi, C., Feng, S., Du, Y., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *RSS*.
11. Dragan, A. D., & Srinivasa, S. S. (2013). A policy-blending formalism for shared control. *IJRR*, 32(7), 790-805.
12. Floridi, L., Cowls, J., Beltrametti, M., et al. (2018). AI4People—An ethical framework for a good AI society. *Minds and Machines*, 28(4), 689-707.
13. Fu, Z., Zhao, T. Z., & Finn, C. (2024). Mobile ALOHA: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. *arXiv preprint arXiv:2401.02117*.
14. Javdani, S., Srinivasa, S. S., & Bagnell, J. A. (2015). Shared autonomy via hindsight optimization. *RSS*.
15. Katz, G., Barrett, C., Dill, D. L., et al. (2017). Reluplex: An efficient SMT solver for verifying deep neural networks. *CAV*.
16. LeCun, Y. (2022). A path towards autonomous machine intelligence. *Open Review*.
17. Liang, J., Huang, W., Xia, F., et al. (2023). Code as Policies: Language Model Programs for Embodied Control. *ICRA*.
18. McKinsey Global Institute. (2017). Jobs lost, jobs gained: Workforce transitions in a time of automation.
19. Mohamed, A., Qian, K., Elhoseiny, M., & Claudel, C. (2020). Social-STGCNN: A social spatio-temporal graph convolutional neural network for human trajectory prediction. *CVPR*.
20. Open X-Embodiment Collaboration. (2023). Open X-Embodiment: Robotic learning datasets and RT-X models. *arXiv preprint arXiv:2310.08864*.
21. Peng, X. B., Abbeel, P., Levine, S., & van de Panne, M. (2018). DeepMimic: Example-guided deep reinforcement skills of physics-based characters. *SIGGRAPH*.
22. Peng, X. B., Ma, Z., Abbeel, P., Levine, S., & Kanazawa, A. (2021). AMP: Adversarial motion priors for stylized physics-based character animation. *SIGGRAPH*.
23. Rabinowitz, N., Perbet, F., Song, F., et al. (2018). Machine theory of mind. *ICML*.
24. Reddy, S., Dragan, A. D., & Levine, S. (2018). Shared autonomy via deep reinforcement learning. *RSS*.
25. Salzmann, T., Ivanovic, B., Chakravarty, P., & Pavone, M. (2020). Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data. *ECCV*.
26. Singh, I., Blukis, V., Mousavian, A., et al. (2023). ProgPrompt: Generating situated robot task plans using large language models. *ICRA*.
27. Su, Y., et al. (2024). Multimodal intent recognition for human-robot collaboration. *IEEE Robotics and Automation Letters*.
28. World Economic Forum. (2023). The Future of Jobs Report 2023.
29. Yuan, W., Dong, S., & Adelson, E. H. (2017). GelSight: High-resolution robot tactile sensors for estimating geometry and force. *Sensors*, 17(12), 2762.
