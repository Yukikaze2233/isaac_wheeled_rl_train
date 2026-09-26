# V5 SCUT35 部署对接规范：模型结构与使用方法

本文对应本项目V5单策略、SCUT35接口，供仿真／上位机／嵌入式部署端对接。
**模型身份以导出ONNX、配套合同、manifest和SHA回执为准。**同为35D输入，不表示可以互换华南虎模型、旧V3模型或其他35D策略。
本规范覆盖历史12486-update平地候选与V5.4／V5.6沿用的观测／动作接口；文件名保留以兼容已有引用。

配套可执行NumPy参考：[examples/v5_policy_io.py](examples/v5_policy_io.py)。
它与训练侧观测、动作映射和关节输出端控制数学做逐项对照测试，不包含硬件通信驱动。
完整域设计见[V54_FULL_RANGE_TRAINING.md](V54_FULL_RANGE_TRAINING.md)，当前训练方式见[V56_ADAPTIVE_TRAINING.md](V56_ADAPTIVE_TRAINING.md)。
接口兼容不等于能力相同，下面先明确本次交付的具体模型。

## 0. 本次交付：此前的平地移动／旋转模型

- **ONNX**：[models/v5_flat_12486/policy.onnx](../models/v5_flat_12486/policy.onnx)。
- **完整轻量交付包**：[models/v5_flat_12486.zip](../models/v5_flat_12486.zip)。
- 逐项能力及离线运行说明：[交付包README](../models/v5_flat_12486/README.md)。
- 来源：`v5-speed-resume-20260922/train/stage_00_flat/block_016`，累计12486次更新。
- ONNX原始字节已保留：204185字节（约199.4KiB），SHA-256：
  `ae58b862be5547195d8c4b3e71aa9be37b147792ebc903c68f032f341d92be6d`。
- 对应checkpoint SHA：`e6307dd7419c052a96285e58ce6a6624b8832e6c5986aadf568287bfcfc37f81`。

该候选在已保存的27案例固定评测中通过18项，每案例4个episode。平移／旋转评测高度为**0.305m**：

| 行为 | 有全判据通过记录的命令点 |
|---|---|
| 站立 | vx=0、wz=0；高度MAE 1.93mm，最大漂移10.44mm |
| 平移 | vx=+0.5、+2、−2、+3、−3m/s，wz=0 |
| 启停 | 0.5m/s启停轨迹 |
| 弧线 | vx=0.25、wz=±0.3；vx=0.5、wz=±0.6 |
| 双向慢转 | vx=0、wz=±1rad/s |
| 正yaw旋转 | vx=0、wz=4、2π、8、4π rad/s（最高正向2圈/s） |

后退0.5m/s、前后1m/s的速度精度未过阈值；负yaw的−4、−2π、−8、−4π rad/s主要是高度精度未过。
原地0.29–0.32m指定升降轨迹通过，但两个端点独立驻留没有全部达标。
这是一份有明确部分能力证据的历史候选，原始`accepted_stage=null`保留在包内；不是完整阶段验收模型。
不能把V5.4／V5.6的0.21–0.35m、±5m/s、±3圈/s训练目标当作本候选已经学会的范围。
平地行走与原地旋转也不等同于第6节的“旋转平移”复合模式。

### 已确定的硬件控制架构

用户指定WheelLegInfantryRL硬件：2×DM髋＋2×同型号DM膝＋2×M3508轮。
2026-09-26最新确认：**前膝电机、后髋电机，经1:1外部链传动至同轴嵌套输出**。
**六轴最终都接收力矩／电流接口指令，位置与速度闭环由PC计算。DM使用模式A的`control_torque`，不使用模式B的电机内PD。**
策略产生的四路位置目标、两路速度目标只是PC内的中间量。
电机驱动器仍以自身电流／FOC环实现目标力矩；PC并不凭空新增一套未定义的实测力矩反馈PID。

## 1. 模型由哪些部分组成

### 1.1 机械／仿真模型

- 资产目录：`model/纯底盘_v5/urdf/`；`robot.usda`用于Isaac，`robot.xml`用于MuJoCo。
- 19个刚体、18个树关节（16R＋2P）、6个球闭合约束；闭合约束Jacobian已核对秩12。
- 六个内部独立控制坐标：**四个腿部根端主动轴＋两个轮轴**；基座为自由浮动。
- `L_joint2/R_jonit2`是模型中从动的相对膝铰坐标；**硬件仍有膝电机**，其驱动作用通过另一个根部输入和闭链传到膝。不能把“被动膝铰”理解成“没有膝电机”。
- 模型质量约14.642kg，轮半径0.06m，轮距0.4373m；气簧额定行程80mm，使用10MPa研究力曲线。
- 接地高度指令域0.21–0.35m，指base_link原点相对轮支撑面的高度，不是车体底部离地间隙。

结构和名义位置以`manifest.json`、`model_spec.json`为准。当前模型中的电机映射／传动标定仍标记为未确认；
下述量均为**模型关节输出端量**，不是CAN-ID、转子编码器原值或电流指令。

### 1.2 策略网络：部署运行的究竟是什么

```text
Actor:  35 → Linear(256) → ELU → Linear(128) → ELU → Linear(64) → ELU → Linear(6)
Critic: 81 → Linear(256) → ELU → Linear(128) → ELU → Linear(64) → ELU → Linear(1)
```

- Actor是单帧MLP，**确定性网络精确为50,758参数**；无RNN／Transformer，无多帧堆叠。
- “上一动作”是35D输入中的六项，不是隐藏的历史帧队列。
- 训练是普通PPO＋非对称critic：actor和critic是两个独立MLP，不共享隐藏层；critic有62,209参数。
- **部署只运行actor。**Gaussian动作分布和critic用于训练，ONNX导出的是确定性动作均值。
- 没有运行均值／方差归一化；输入按下面固定尺度缩放，整体裁剪到`[-100,100]`。

ONNX接口：输入`obs`为`float32[batch,35]`，输出`actions`为`float32[batch,6]`；通常batch=1。
当前导出使用opset 17、动态batch、不依赖外置权重文件；输出层是线性层，**调用方需要裁剪动作**。

本次交付已读取ONNX实际图和checkpoint张量核对，而非仅引用训练配置：

| 层 | 权重形状[out,in] | bias长度 | 层参数量 | 激活 |
|---|---|---:|---:|---|
| `mlp.0` | [256,35] | 256 | 9216 | ELU，α=1 |
| `mlp.2` | [128,256] | 128 | 32896 | ELU，α=1 |
| `mlp.4` | [64,128] | 64 | 8256 | ELU，α=1 |
| `mlp.6` | [6,64] | 6 | 390 | 无，线性输出 |
| 合计 | 50304个weight | 454个bias | **50758** | 三个隐藏层 |

ONNX包含4个`Gemm`、3个`Elu`节点，没有输出`Tanh`／`Clip`、输入标准化层或隐藏状态接口。
全部参数的FP32数据占203032字节；一次单样本前向有50304次乘加，不含激活、观测组装和低层控制。
训练actor另外保存6个状态无关的可学习Gaussian标准差；它们与critic、优化器均不进入ONNX。

### 1.3 Actor与critic的输入边界

Actor的35D可以按`3+1+3+3+6+6+6+7`核对：速度命令、高度命令、gyro、重力方向、
编码器位置偏差（两轮槽位为0）、编码器速度、上一动作、命令上下文。
部署端只提供这些量；“非对称critic”的额外真值不需要实机提供。

| Critic索引 | 维数 | 训练时的数据 |
|---|---:|---|
| 0–34 | 35 | 未加观测扰动的actor当前帧 |
| 35–37 | 3 | 机体系根部COM线速度 |
| 38 | 1 | 相对支撑面的实际高度 |
| 39–40 | 2 | 两轮接触力模长÷125 |
| 41–58 | 18 | 树关节位置，轮转角清零；按仿真关节序而非P序 |
| 59–76 | 18 | 树关节速度×0.1 |
| 77–78 | 2 | 两轮所处表面摩擦系数 |
| 79–80 | 2 | 两轮下方支撑面高度 |

Critic输出一个状态价值估计用于PPO训练，不输出电机指令。
Actor也不直接预测车体速度／腿长：它根据命令和可部署反馈输出四个主动腿轴的位置偏置与两个轮速目标的归一化表示。

### 1.4 网络与部署控制链的模块边界

1. **观测适配**：同步并标定编码器／IMU，按第3节组装35D；缩放和裁剪在ONNX外执行。
2. **Actor推理**：50Hz，35D→原始6D均值，无内部持续状态。
3. **动作解码**：按P序分腿／轮裁剪、排列，生成四路位置目标和两路轮速目标，保存裁剪动作供下一帧。
4. **PC反馈控制**：按200Hz训练参考使用最新q/dq计算力矩；两次actor推理之间保持目标，持续更新反馈力矩。
5. **硬件适配**：将模型输出端力矩映射到驱动侧单位及通道。

ONNX没有包含机械闭链求解器、气簧前馈或PC低层PD；气簧／连杆动力学通过训练影响策略参数。

## 2. 策略、模型与硬件顺序

本规范按功能定义：**hip＝大腿根部主动输入，knee＝通过连杆调节膝运动的第二主动输入**。
以下表格给出部署方应实现的功能绑定。它不是CAN接线、链轮传动比或编码器符号的实测报告。
安装布局为**前膝、后髋**，覆盖此前写反的记录。策略轴按功能绑定；前电机对应辅助根轴
`LL_joint1/RR_joint1`，后电机对应大腿根轴`L_joint1/R_joint1`。

### 策略顺序P：网络中的电机观测、上一动作和六维输出

| P索引 | 模型主动轴 | 直接驱动的模型构件 | 硬件功能通道 | 名义q₀，rad |
|---:|---|---|---|---:|
| 0 | `L_joint1` | `base_link → L_link1`，左大腿根 | 左髋DM（H0） | 0.42 |
| 1 | `LL_joint1` | `base_link → LL_link1`，左连杆曲柄根 | 左膝DM（H1） | −0.13742282595254576 |
| 2 | `R_joint1` | `base_link → R_link1`，右大腿根 | 右髋DM（H3） | −0.42 |
| 3 | `RR_joint1` | `base_link → RR_link1`，右连杆曲柄根 | 右膝DM（H4） | 0.13741557625658019 |
| 4 | `L_joint3` | 左轮轴 | 左轮M3508（H2） | 0 |
| 5 | `R_joint3` | 右轮轴 | 右轮M3508（H5） | 0 |

### 控制／采集顺序C：`manifest.control_joint_names`

```text
C = [L_joint1, LL_joint1, L_joint3, R_joint1, RR_joint1, R_joint3]
H = [left_hip, left_knee, left_wheel, right_hip, right_knee, right_wheel]
P = C[[0, 1, 3, 4, 2, 5]]
C = P[[0, 1, 4, 2, 3, 5]]
```

落实上述功能绑定后，C与H的**功能排列**一致，因此P的通道索引为`H[[0,1,3,4,2,5]]`。
这不表示原始编码器H数值已经等于模型C数值：仍需零位、方向、传动比及接口参考端转换。
模型能确定大腿根与曲柄根的拓扑；现有导出材料没有给出四台电机的完整链轮接线、内外输出绑定和CAN-ID，不把这些信息标为已验证。

`L_joint2`和拼写为`R_jonit2`的关节是被动膝；不能把旧V4串联腿的动作表套到V5。
名义q₀是模型坐标的参考，不是每次启用时把当前编码器随意清零后的值。

## 3. 35D输入逐项定义（索引从0开始）

| 索引 | 内容 | 来源／处理 |
|---|---|---|
| 0 | vx命令 | m/s，不再缩放 |
| 1 | vy参考命令 | 普通运动填0；旋转平移见第6节 |
| 2 | yaw角速度命令 | rad/s，不再缩放 |
| 3 | 目标高度 | m × 5；0.21–0.35m对应1.05–1.75 |
| 4–6 | 机体系角速度xyz | IMU角速度rad/s × 0.5 |
| 7–9 | 机体系单位重力方向xyz | 姿态融合得到，名义水平朝上约为`[0,0,-1]` |
| 10–13 | 四个主动腿轴角度偏差 | 按P顺序，`atan2(sin(q-q₀), cos(q-q₀))`，rad |
| 14–15 | 轮位置占位 | 必须填0，不输入累计轮角 |
| 16–21 | 六主动轴角速度 | 按P顺序，输出端rad/s × 0.1 |
| 22–27 | 上一策略周期的裁剪动作 | 按P顺序，归一化动作；不是力矩、目标角或目标转速 |
| 28 | normal | 非跳跃请求时1，否则0 |
| 29–31 | 保留位 | 全部为0；不填地形类别、接触phase或recover标志 |
| 32 | jump request | 激活为1，否则0 |
| 33 | 请求跳高 | 请求激活时`apex_delta_m × 5`，否则0 |
| 34 | 请求后计时 | 请求激活时`clip(elapsed_seconds,0,5)`，否则0 |

最终拼接并裁剪到`[-100,100]`，转换为连续的`float32[1,35]`。
gyro的0.5缩放与yaw命令的1.0缩放不同；第3项是**目标**高度，不能替换为实测车体高度。

### 3.1 IMU与坐标系

- 控制坐标：**X前、Y左、Z上**，右手系；正yaw绕+Z。
- 将IMU角速度先通过固定外参转到`base_link`坐标，再乘0.5。
- 若`R_WB`把机体系向量转到世界系，重力方向为`g_B = R_WBᵀ [0,0,-1]`。
- 加速度计供姿态融合使用；原始加速度或`[0,0,-9.81]`不能直接填入7–9。
- Isaac运行状态使用xyzw四元数；MuJoCo自由关节状态使用wxyz。ONNX本身不接收四元数。
- V5的IMU外参应独立标定，不能默认使用旧V4合同里的`source_to_control_rotation`。

### 3.2 编码器与传动映射

部署适配层应得到`q_C`和`dq_C`：各模型主动轴输出端的rad、rad/s。
一般写作`q_C=f(theta_H)`、`dq_C=J_CH dtheta_H`。theta_H必须明确是API反馈的转子角、减速器输出角还是外部链传动输出角。
电机固定在机身、固定轴独立链传动时，通常可简化为`q_C[i]=s_i*theta_H[i]/n_i+b_i`；n_i只包含该API反馈端到模型输出端之间的传动，不能重复计算驱动器已换算的减速比。
用户已确认上述外部链传动比为1:1；若API反馈在电机输出轴端，该段比例的绝对值为1。
API若给转子侧角度，电机内部减速比仍需按接口语义处理；零位、左右正方向及CAN绑定分别确定。
策略需要两根主动轴相对机身的角度，因此不应先把膝驱动输入改成`q_aux-q_hip`再填入actor。
这个差值用于闭链内角推算：在同一装配分支，`inner_knee = F(q_aux-q_hip)`，其中`F`通常非线性。
需要标定每轴的输出归属、零位、正方向、比例／耦合关系，并在动作反向映射中保持一致。

部署配置需明确：H功能通道→驱动口/CAN-ID、输出件、API角度与力矩参考端、符号、传动比、模型零位、力矩单位／电流换算。
四台DM型号相同，不代表镜像两侧的符号和零位相同；本文的q₀也不是DM上电编码器读数。

Actor不需要真实膝角、气簧位移、接触力、根部线速度或地形高度输入。电流反馈可用于驱动层监控，不直接拼入35D。

#### 两条机械运动关系的核验（2026-09-26）

针对V5.9实际训练资产（manifest `dbec8e58…`，本机归档`model/纯底盘_v5_232mm/urdf/`），
用编译后的MuJoCo闭链约束独立求解，并从大腿／小腿空间向量测量内角：

| 条件 | 模型主动输出轴行为 | 核验 |
|---|---|---|
| 大小腿夹角固定，大腿相对机身摆动 | 髋轴和膝驱动轴等增量随动 | 双侧40–110°内角、±90°摆动，70个采样姿态通过 |
| 大腿相对机身不动，改变大小腿夹角 | 髋轴保持，只有该侧膝驱动轴改变 | 双侧内角增加10°，其他五个主动轴增量为数值零 |

等增量最大偏差约0.000616°，与导出根轴约0.56μm的CAD同轴线偏差相符。
例如左侧名义内角67.854971°：两根主动轴同时+10°后为67.854943°；
固定髋轴、只把内角增加10°时，膝驱动轴转约−12.4418°。
**链轮1:1不表示膝驱动轴角与大小腿内角1:1**，后者还经过闭链连杆。

证据：[两条件数值回执](evidence/v5_motor_coupling_two_conditions_20260926.json)。
复现脚本：`scripts/audit_v5_motor_coupling.py`；回归测试：`tests/test_v5_motor_coupling.py`。
这些核验确认模型的运动学关系；实机编码器正方向／零位和驱动通道映射由硬件标定给出。

## 4. 六维动作如何在PC侧变成六轴力矩

网络原始输出记为`a_P`，先裁剪：前四维到`[-3,3]`，后两维到`[-9,9]`，再映射到C顺序。
这与训练端的“软件PD／速度环→effort”控制边界一致。

```text
50Hz ONNX → 裁剪／排列 → PC缓存四腿位置目标＋两轮速度目标
最新模型侧q/dq → PC控制循环 → tau_C → J_CHᵀ → tau_H
→ 实测电机包络／驱动单位适配 → 四DM control_torque＋两M3508力矩适配
```

PD在模型输出端坐标C计算；不能把60/2直接当作未经传动转换的转子坐标增益。

### 腿部：PC内部位置目标与PD

```text
q_des = q₀ + 0.25 × a_leg
q_target = q_current + atan2(sin(q_des - q_current), cos(q_des - q_current))
tau_leg = clip(60 × (q_target - q_current) - 2 × dq_current, -40, +40)
```

角度为rad，速度为rad/s，力矩为关节输出端N·m。0.25是rad/归一化动作；±3对应名义目标偏置±0.75rad。
这不是“六路力矩策略”，也不是虚拟腿长／腿角VMC策略。
零速度命令下也可能有非零腿／轮动作来维持平衡；接口校验应对照参考输出，而不是要求六维输出全为零。

### 轮部：PC内部速度目标与速度环

```text
wheel_speed_target = 10 × a_wheel                  # rad/s
tau_requested = 0.2 × (wheel_speed_target - dq_wheel)
tau_wheel = clip(tau_requested, -tau_limit(dq), +tau_limit(dq))
```

当前研究先验用输出端速度绝对值乘11查询电机侧线性曲线：
`(0 rad/s, 0.348880597 N·m)`到`(999.063012 rad/s, 0 N·m)`，再乘传动比11、效率1，
并限制在3.837686567 N·m以内。超过曲线速度域，模型中的力矩上限为0。
准确实现见`v40/core.py::motor_torque_limit`和配套NumPy函数。

以上PD／速度环均在PC运行，产生`tau_C`。保持模型侧限幅后，再通过已标定传动映射转换到驱动接口侧，执行真实电机包络与电流限幅。
若`q_C=f(theta_H)`，理想虚功关系为`tau_H=J_CHᵀ tau_C`；角度和力矩必须对应同一功率共轭参考端。
DM接口若已经以减速器输出端N·m为单位，就不能再按转子端重复换算一次；M3508力矩到电调电流的常数及协议刻度由实际驱动配置提供。
配套`joint_to_drive_torques(tau_C, J_CH)`要求显式传入标定矩阵；通过瞬时功率一致性测试，不默认J_CH为单位阵。

### 4.1 力矩曲线的三个不同含义

- **当前冻结训练的电机包络**：腿部为固定±40N·m上限，轮部为上述两点线性速度—力矩先验，并非已辨识的六轴非线性电机曲线。
- **实机非线性电机包络**：可以用按速度采样的多点表分段插值表示；四台同型号DM可共享电机侧表，但应分别经过各轴传动映射。需提供真实表、采样侧、单位及制动／驱动象限定义后对齐，不能凭“同型号”生成曲线。
- **已有非线性气簧模型**：`F(s)=280+122.735918491*u²+53.969804409*u³`，`u=s/0.08`，属于内部弹性力。实机有物理气簧，不把这份仿真力再次作为电机力矩补偿叠加；V5.4策略控制没有使用预览工具的气簧前馈。

曲线查表框架支持多点，并不等于当前模型已按实测非线性曲线训练。更换执行器曲线后应形成版本化参数，完成仿真／部署动态一致性验证，再决定是否需要重新训练。

### 4.2 上一动作历史

下一帧保存的是**本周期生成并裁剪、准备发出的归一化policy动作P**。
即使底层存在通信／执行延迟，也不应再把延迟后的动作当作这六维历史，训练中的延迟分支同样保存当前裁剪指令。
控制器重新初始化时清零；普通指令变化不随意清零。

## 5. 时序和复位

- 策略周期20ms，即50Hz；每次只运行一次ONNX，持有目标直到下一策略周期。
- 训练物理／PD更新周期5ms，即200Hz，一次策略更新对应4次低层更新。
- PC力矩循环每次用最新q/dq重新计算PD／速度环力矩，位置／速度目标在策略周期内保持；六路最终输出均为力矩。
- PC控制循环应先按训练200Hz参考对齐；若工程实现采用更高频率，验证等效动态、延迟和增益。驱动器内部FOC电流环与PC循环是不同层次。
- 启用前完成传感器同步与坐标转换，设置已定义的初始指令、普通上下文和零上一动作。
- 每周期检查输入／输出有限值与时间戳；无有效新输出时由部署控制器的既有停机／保持逻辑处理，避免使用未初始化数组。
- 最终鲁棒阶段训练观测与动作各0–20ms延迟及噪声；阶段是否已通过必须读取产物回执。

### 5.1 INIT / IDLE / PREPARE / RL

参考华南虎StateMachine的状态职责与enter/run/exit组织；所有状态统一采用本项目的纯力矩输出约定：

| 状态 | PC侧行为 | 发往驱动器的量 |
|---|---|---|
| INIT | 建立通信、读有效反馈、加载模型与标定，清理历史动作／jump时钟／slew状态 | 零力矩；DM配置为模式A |
| IDLE | 停止RL输出接管，保持零力矩／零电机内PD增益的空闲语义 | 六轴零力矩；不是`action=0`后再执行RL的PD |
| PREPARE | 从测得的四个模型主动轴角度出发，按准备速度插值到本模型q₀；在PC计算准备PD | 四腿准备力矩，轮轴零力矩；DM仍为模式A |
| RL | 50Hz更新目标，PC力矩循环持续使用最新反馈闭环 | 四DM调用`control_torque`，两M3508经力矩到驱动接口的适配输出 |

进入RL时统一初始化上一动作、指令slew和jump上下文；退出RL时由目标状态接管输出，不能继续发送旧RL缓存。
准备目标是表中的V5名义q₀，而不是四腿全0。PREPARE的速度／增益由部署配置明确给出，不复制其他机器的数值。
状态名称／职责参考华南虎，不把其通用多种输出模式或电机内PD分支带入本模式A方案。

## 6. 上层如何发命令

### 普通站立、升降、行走、转向

上层提供`[vx, yaw_rate, height_target]`；vy=0、jump=false，上下文为`[1,0,0,0,0,0,0]`。
速度／yaw命令采用合同中的slew：0.6m/s²、4rad/s²。升降可直接调用训练侧`height_reference`生成相同的平滑过渡与驻留轨迹。
高度、速度和旋转的全范围必须结合模型已验收阶段使用。

组合包络：`max(|vx - b*wz/2|, |vx + b*wz/2|)/r <= 90`，且`|vx*wz| <= 2.06`，其中r=0.06、b=0.4373。
最高5m/s与最高3圈/s分别是任务目标，不能同时叠加为一个可行命令。

### 旋转平移

设固定参考系目标速度为`[Vx,Vy]`，当前车体相对该参考系yaw为ψ：

```text
vx_body_ref =  cos(ψ) × Vx + sin(ψ) × Vy
vy_body_ref = -sin(ψ) × Vx + cos(ψ) × Vy
```

第0项使用vx_body_ref，第1项使用vy_body_ref，第2项保持旋转命令。
坐标旋转产生的参考变化不再额外按普通vx slew处理；训练实现对此有明确旁路。
两轮底盘不具备瞬时横移自由度，这个vy是旋转平移参考，训练按旋转时间尺度评价平均平移。
本项目使用虚拟固定参考系，没有额外云台执行器。

### 主动跳跃

跳跃开始时锁存request并以秒计时，传入相对请求跳高6cm或10cm；它不是绝对车体高度。
普通高度命令仍是基准站立高度。训练先在初始姿态停留约0.8s再发请求；计时从请求发出起算。
上层管理请求结束与控制器复位；policy上下文不接收真实接触phase。
当前保留的recover槽位为0，不能通过把它置1来获得倒地自起能力。

## 7. 需要交付哪些文件

本次历史模型已整理为以下bundle，原始ONNX、metadata、训练合同和资产manifest保持对应SHA：

```text
models/v5_flat_12486/
  policy.onnx
  policy.onnx.json                 # 导出SHA、输入输出、Torch/ORT对照结果
  policy.onnx.contract.json        # 对应的已展开阶段合同
  manifest.json                   # 同一机械资产manifest与名义位置
  own_v40_v2.json                  # control_math_source原文件，用于轮电机先验和SHA
  artifact_selection.json         # 候选／已验收角色、accepted_stage
  agent_config.json               # 来源训练块的实际actor/critic/PPO配置
  evaluation.json                 # 同一checkpoint的27案例历史结果
  evaluation.contract.json        # 上述评测所用合同，不替换训练合同
  provenance.json                 # 来源checkpoint、SHA、通过/失败案例
  verification.json               # 图结构、权重一致性和CPU ORT核对
  io_fixture.json                 # 固定输入、网络输出与控制解码比对值
  v5_policy_io.py                 # 可独立运行的NumPy参考
  infer_example.py                # 本包离线推理与fixture核对
  SHA256SUMS                     # 包内文件完整性清单
  README.md
```

仿真对接再携带完整V5资产：`robot.usda`或`robot.xml`、网格、`model_spec.json`、`fit_10mpa.json`及所需构建元数据。
训练checkpoint `.pt`用于恢复／再导出，不是ONNX推理所必需。

**`own_v40_v2.json`不是本模型的完整部署合同。**这里只复用其执行器数学先验；它的125D观测、100Hz时序、串联腿动作顺序和名义角度都不能套到V5。
范围、35D布局、50Hz时序和V5控制字段以`policy.onnx.contract.json`为准。

已结束训练块中的`policy.onnx`也可能是未通过验收的候选。读取`artifact_selection.json`：
`accepted_stage`说明已验收的能力范围；完整能力发布应对应`full_curriculum_accepted`，不能只挑最新mtime文件。
阶段目录里的原始`contract.json`也可用于校验，只要与导出metadata的contract_sha256一致。
注意，训练目录中的`asset_manifest.json`可能经过JSON重新序列化；即使内容相同也可能字节SHA不同。
本包提供的是导出metadata所引用的原始资产`manifest.json`，不要用重新排版后的副本替换。

## 8. 可运行的离线推理示例

以下只打印推理结果，不连接硬件。先安装NumPy和ONNX Runtime，在训练仓库根目录执行：

```bash
python models/v5_flat_12486/infer_example.py
```

该命令校验真实交付模型和输入输出fixture。若要从传感器值开始复现观测组装，可执行：

```bash
PYTHONPATH=docs/examples python - <<'PY'
from pathlib import Path
import numpy as np
import onnxruntime as ort
from v5_policy_io import CONTROL_ORDER, verify_bundle, build_observation, decode_actions, reference_joint_torques

p = Path("models/v5_flat_12486")
c, m, prior, meta = verify_bundle(
    p / "policy.onnx", p / "policy.onnx.contract.json",
    p / "manifest.json", p / "own_v40_v2.json")
session = ort.InferenceSession(str(p / "policy.onnx"), providers=["CPUExecutionProvider"])
q0 = np.array([m["nominal_joint_pos"][name] for name in CONTROL_ORDER], dtype=np.float32)

# Offline nominal fixture. Replace these with calibrated, synchronized feedback.
q, dq = q0.copy(), np.zeros(6, dtype=np.float32)
previous = np.zeros(6, dtype=np.float32)
obs = build_observation(q, dq, [0, 0, 0], [0, 0, -1],
                        [0, 0, 0.305], q0, previous)
raw = session.run(["actions"], {"obs": obs})[0][0]
legs, wheels, clipped = decode_actions(raw, q, q0, c["v5_control"])
tau_C = reference_joint_torques(q, dq, legs, wheels, c["v5_control"], prior["actuators"]["wheel"])
print("obs:", obs.shape, obs.dtype)
print("leg targets [rad], P[0:4]:", legs)
print("wheel targets [rad/s], P[4:6]:", wheels)
print("PC joint-output torque [N m], C order:", tau_C)
# A calibrated J_CH maps tau_C to drive-side tau_H via J_CH.T @ tau_C.
# Issue torque commands through the hardware adapter, then retain clipped.copy().
PY
```

真实循环中，ONNX与`decode_actions`每20ms更新一次目标；`reference_joint_torques`对应的PC闭环使用每个低层周期的新反馈重新计算，不能只按50Hz计算一次力矩后保持。
硬件适配层执行`tau_H=J_CH.T @ tau_C`、实际驱动包络／单位转换并发送六轴力矩。位置与速度目标不作为DM模式B命令下发。
驱动器内部位置／速度PD不再参与，避免同一PD计算两次。这里没有填写未经标定的J_CH或电流换算常数。

## 9. 对接验收与能力边界

按同一传感器／指令fixture依次比较：35D向量 → ONNX原始6D → 裁剪和排列 → 四腿／两轮目标 → 输出端PD。
配套测试`tests/test_v54_deployment_io.py`已对照训练侧PyTorch实现，覆盖角度wrap、索引、jump上下文和饱和。
ONNX导出另要求与Torch actor在零输入和真实观测上达到`atol=rtol=1e-5`。

本次12486候选交付核验：8个ONNX参数张量与checkpoint逐元素相同；ONNX checker通过；
CPU ONNX Runtime在零输入、名义输入和固定种子接口样本上测试batch=1/8/32/256，
Torch／ORT最大绝对差`5.7220458984375e-6`，通过`atol=rtol=1e-5`。
配套10项NumPy／训练侧接口对照测试通过。这里新增的是文件与接口验证，没有重新执行物理能力评测；能力表来自包内历史评测。

正式模型的技能通过情况必须看固定评测。当前训练目标包含落地恢复，但**未定义或验收任意倒地自起**。
落地后的内部RECOVERY相位不是35D中的recover命令位；不得混用。
实机主动轴映射、编码器／IMU标定、执行器动态与时延仍需由对接方提供实测绑定数据。

华南虎准备／恢复机制的逐文件核对、V5脚本PREPARE与可选倒地恢复训练设计，见[V5_SELF_RIGHTING_DESIGN.md](V5_SELF_RIGHTING_DESIGN.md)。

源码权威入口：`scut_observation.py::build_scut35`、`env.py::get_observations/step`、
`v5_control.py::decode/motor_efforts`、`chassis_export.py::export_actor`，均位于`src/`对应包下。
