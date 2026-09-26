# V5 自起：部署对接与仿真复现规范

核对日期：2026-09-26。面向 `RMCS/rmcs_ws/src/rmcs_rl`、底盘硬件适配及仿真验证的对接人员。

**推荐接入方式：在现有 `RlController` 的 `PREPARE` 内增加恢复子状态机，由同一个控制器统一输出六轴力矩。**
已验证的动作参考是本仓库的 `scripts/inspect_v5_activation.py`；RMCS 当前还没有完整移植该状态机。

本文件给出三种不同性质的信息，并在具体条目中标明：

- **当前实现**：从本机代码核对到的实际行为。
- **仿真参考**：已有物理试验和冻结机械资产支撑的公式、参数、数值。
- **接入要求／建议接口**：部署侧尚需实现或提供的数据，不能理解成已存在的 ROS 接口或硬件标定。

## 导航

1. [交付身份与能力证据](#1-交付身份与能力证据)
2. [现有部署代码与接入位置](#2-现有部署代码与接入位置)
3. [坐标、顺序、单位与标定](#3-坐标顺序单位与标定)
4. [连续角、闭链与机械限位](#4-连续角闭链与机械限位)
5. [控制状态机与切换条件](#5-控制状态机与切换条件)
6. [参考姿态与轨迹计算](#6-参考姿态与轨迹计算)
7. [六轴力矩、气簧与输出仲裁](#7-六轴力矩气簧与输出仲裁)
8. [ONNX 接管与时序](#8-onnx-接管与时序)
9. [部署所需的观测与估计接口](#9-部署所需的观测与估计接口)
10. [配置及建议代码结构](#10-配置及建议代码结构)
11. [逐级对接验收](#11-逐级对接验收)
12. [Isaac Sim 仿真方法及复现](#12-isaac-sim-仿真方法及复现)
13. [日志、回放与问题定位](#13-日志回放与问题定位)
14. [交付清单与源码索引](#14-交付清单与源码索引)

## 1. 交付身份与能力证据

### 1.1 自起动作当前绑定的策略

自起方案为**分阶段反馈脚本恢复，再交给 normal RL 策略平衡**。翻身阶段不是这个 ONNX 学出来的动作。

| 对象 | 身份 |
|---|---|
| 自起试验策略 | `models/v5_flat_12486/policy.onnx`，历史12486-update平地策略 |
| ONNX SHA-256 | `ae58b862be5547195d8c4b3e71aa9be37b147792ebc903c68f032f341d92be6d` |
| 策略接口 | `obs: float32[batch,35]` → `actions: float32[batch,6]` |
| 网络 | 单帧MLP，256/128/64，ELU；无RNN、无在线均值方差归一化 |
| 恢复机械来源 | Git提交 `eb4a3fd12978bda55e7636c9f95addc28cc7c302` |
| 恢复资产manifest SHA | `dbec8e586cf29d9540db7040b133b3ff1170c1c700ff05eda64d94d8561c3dec` |
| 恢复模型spec SHA | `d87023eb9d1bd1c7d55a0c3750a34d1936e5f51b3f47587491d6c6030ee86583` |
| 当前恢复脚本SHA | `c5c4cd31077bb544da13f0acb1186c0280de1054556704064b10dc116a1df991` |

12486策略最初配套的是旧机械资产。实验先认证原ONNX包，再显式允许同ABI的新机械资产评测；
不会把旧策略的原始manifest改写成新资产。部署包也应分别保留**策略来源身份**与**运行机械身份**。

数值参考附件：[v5_self_righting_reference_20260926.json](evidence/v5_self_righting_reference_20260926.json)。
其中包含四主动轴姿态、静态前馈、带前馈偏置的准备目标、45点气簧映射和轴方向。
这些是冻结仿真模型的输出端参考量，**不包含实机CAN、零位、链传动或电流标定**。

### 1.2 与当前V5.9训练候选的关系

2026-09-26打开的普通WASD回放使用另一个候选：

- `models/v59_candidate_6554_20260926/policy.onnx`；累计6554次V5.9更新，当前阶段554次。
- SHA：`7fbedc12bb15e6ee8892185a22efbaac57ec49fb168050f1efe74edf4b76ba05`。
- 它是正常策略回放对象；下面的自起成功率对应**12486 ONNX＋恢复脚本**。

把新的V5.9策略接在同一恢复脚本后面，需要重新验证交接状态分布、上一动作历史、混合接管及完整恢复矩阵。
35D/6D形状相同只是接口兼容条件，不是恢复能力相同的证明。

### 1.3 当前物理试验结果

协议：10种初态，每种5次小扰动；12秒观察窗口、释放后8秒主动恢复预算。

| 初态 | 末尾严格通过 | 曾达到稳定RL |
|---|---:|---:|
| 正立直接使能 | 5/5 | 5/5 |
| 低蹲直接使能 | 5/5 | 5/5 |
| 正立卸力0.2秒 | 4/5 | 5/5 |
| 低蹲卸力0.2秒 | 5/5 | 5/5 |
| 左侧翻 | 5/5 | 5/5 |
| 右侧翻 | 5/5 | 5/5 |
| 前倾45°释放 | 3/5 | 3/5 |
| 后仰45°释放 | 4/5 | 5/5 |
| 前倒90°释放 | 4/5 | 5/5 |
| 后倒90°释放 | 4/5 | 5/5 |
| 合计 | **44/50** | **48/50** |

前后倒扣10次均达到稳定RL，开始混合接管的主动时间为3.195–3.355秒；捕获仍有约60–73°反向摆动。
两次倒扣工况的最终失败是RL阶段角速度短暂超过0.5rad/s，不把“曾站稳”改记成末尾通过。
前／后标签是reset角度，释放后按实际姿态选方向，不代表穷尽所有倒扣构型。

完整证据：`reports/v5_activation_capture_20260925/`及
[v5_capture_recovery_20260925.json](evidence/v5_capture_recovery_20260925.json)。
当前证据只覆盖单一质量／摩擦配置及关闭自碰撞的仿真，不包含整机实物恢复验收。

## 2. 现有部署代码与接入位置

### 2.1 本次核对的RMCS边界

本机仓库：`/home/yukikaze/Documents/workspace/RMCS`，核对时HEAD为
`bd75ba8877044b2f9f9bc5ad26f42606c1218b35`。硬件和bringup配置有工作树修改，不能仅用HEAD描述其全部内容。

| 文件，均相对RMCS根目录 | 当前职责 | 自起接入重点 |
|---|---|---|
| `rmcs_ws/src/rmcs_rl/src/rl_controller.hpp` | 四态、输入输出接口、时钟和缓存 | `PREPARE`内部恢复子状态及独立两套目标缓存 |
| `rmcs_ws/src/rmcs_rl/src/rl_controller.cpp` | 请求处理、普通PREPARE、50/200Hz调度 | 恢复路由、计时、BLEND期间推理、失败退出 |
| `rmcs_ws/src/rmcs_rl/src/observation.cpp` | 编码器映射、IMU外参、35D组装 | 连续角输入、恢复期间固定零速命令、估计状态接口 |
| `rmcs_ws/src/rmcs_rl/src/action.cpp` | 动作解码、模型PD、软限位、驱动映射 | 恢复力矩／RL力矩仲裁、共同圈数和非线性膝约束 |
| `rmcs_ws/src/rmcs_rl/src/onnx_policy.cpp` | ORT推理、名字／形状检查 | 启动时补足模型与合同SHA校验 |
| `rmcs_ws/src/rmcs_core/src/hardware/wheel-leg-infantry-rl.cpp` | CAN、反馈新鲜度、DM模式选择 | 明确保持`joint_control_mode: torque`及反馈时间语义 |
| `rmcs_ws/src/rmcs_core/src/hardware/device/dm_motor.hpp` | MIT量化、符号、反馈解析 | 明确角度回绕行为及实际寄存器定标 |
| `rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-rl.yaml` | 1000Hz执行器图、模型和标定参数 | 保存已验证恢复配置，不能用占位标定启用 |

### 2.2 已有实现与本方案之间的差异

1. 当前RMCS只有`INIT/IDLE/PREPARE/RL`，普通PREPARE向`nominal_`逐轴最短角插值。
   尚无`FOLD/ORBIT/THRUST/SIDE_SWING/CAPTURE/BLEND`。
2. 当前PREPARE判据是关节误差、倾角、角速度、关节速度和0.25秒驻留；没有高度、轮承重、机身接触判据。
3. 当前PREPARE轮环实际输出`0.2*(0-dq_wheel)`，是零轮速制动，不能按“零轮力矩”理解。
4. 当前`read_model_state_()`直接对驱动角做矩阵映射，DM解析器也没有本方案需要的连续圈数管理。
5. 当前膝软限位使用线性`a*q_hip+b*q_aux+bias`。全域闭链约束仍需按第4节补齐。
6. 当前轮力矩只在驱动侧按`max_torque`裁剪，未实现训练模型的速度相关轮力矩包络。
7. 当前推理只在外层`RL`运行；200ms混合接管需要在进入外层RL前就获得有效策略目标。
8. 当前ONNX加载器检查名字和形状，尚不等于已认证模型字节与配套合同。

**接入原则：只保留一个六轴输出拥有者。**恢复模块输出模型侧力矩或内部目标，由`RlController`统一仲裁；
不要让另一个组件同时注册／驱动同名`control_torque`接口。

## 3. 坐标、顺序、单位与标定

### 3.1 三种数组顺序

```text
P = [L_joint1, LL_joint1, R_joint1, RR_joint1, L_joint3, R_joint3]
C = [L_joint1, LL_joint1, L_joint3, R_joint1, RR_joint1, R_joint3]
P_from_C = [0, 1, 3, 4, 2, 5]
C_from_P = [0, 1, 4, 2, 3, 5]
D = [left_hip, right_hip, left_knee_drive, right_knee_drive]
```

- **P**：ONNX观测／动作顺序，也是当前RMCS `kMotorNames`及`q_`、`dq_`、`targets_`的顺序。
- **C**：仿真`manifest.control_joint_names`、`V5Control`力矩及反馈顺序。
- **D**：当前硬件配置中四个`dm_*`参数数组的顺序。D既不是P的前四项，也不是C的前四项。

| 功能 | 模型轴 | P索引 | C索引 | D索引 | RMCS接口前缀 |
|---|---|---:|---:|---:|---|
| 左髋 | `L_joint1` | 0 | 0 | 0 | `/wheel_leg/left_hip_joint` |
| 左膝驱动／辅助根轴 | `LL_joint1` | 1 | 1 | 2 | `/wheel_leg/left_knee_joint` |
| 右髋 | `R_joint1` | 2 | 3 | 1 | `/wheel_leg/right_hip_joint` |
| 右膝驱动／辅助根轴 | `RR_joint1` | 3 | 4 | 3 | `/wheel_leg/right_knee_joint` |
| 左轮 | `L_joint3` | 4 | 2 | — | `/wheel_leg/left_wheel` |
| 右轮 | `R_joint3` | 5 | 5 | — | `/wheel_leg/right_wheel` |

`L_joint2`、`R_jonit2`是闭链中的被动膝铰，不是策略直接控制的膝电机坐标。
2026-09-26用户最新确认：**前膝电机、后髋电机，外部链传动比1:1**，覆盖此前相反记录。
前电机按功能绑定`LL_joint1/RR_joint1`，后电机绑定`L_joint1/R_joint1`；编码器符号、零位及驱动口分别标定。
固定大小腿夹角时两主动输出轴等增量随动；固定大腿相对车体角度时，改变膝内角只有膝驱动轴转动。
V5.9训练资产的两条运动学关系已独立验证，见[数值回执](evidence/v5_motor_coupling_two_conditions_20260926.json)。

### 3.2 单位与IMU

- 位置：模型主动轴输出端rad；速度：rad/s；力矩：同一模型输出端Nm。
- 位置PD增益单位为Nm/rad，阻尼为Nm/(rad/s)。本文的80/2、120/4不是未经传动换算的转子增益。
- 机体系：X前、Y左、Z上，右手系。`R_WB`把机体系向量转到世界系。
- `R_BI=imu_to_base`把IMU向量转到机体系：`omega_B=R_BI*omega_I`。
- `R_WB=R_WI*R_BI^T`，`g_B=R_WB^T*[0,0,-1]`，重力输入是单位方向。
- `tilt=acos(clamp(-g_B.z,-1,1))`；矢状面`pitch=atan2(g_B.x,-g_B.z)`。
- 控制和门禁使用未缩放的rad/s；只有组装ONNX时gyro乘0.5。
- Isaac姿态存储xyzw；Eigen构造函数参数通常为wxyz。按语义转换，不直接重解释数组内存。

近纯侧躺时pitch本身接近奇异，优先使用`|g_B.y|`识别侧平面。

#### 角度零位的物理含义

冻结模型FK核对：两侧髋角为0时，从髋轴到膝轴的向量在base_link中约为`[-0.210,0,0]m`，
即**大腿相对机身向后水平**，不是竖直下垂。左髋正转、右髋负转使大腿向机身下方摆；
理想矢状面下左髋+π/2、右髋−π/2约对应朝机身−Z，实际轴保留约0.264°的CAD倾斜。

辅助根轴（膝电机对应的曲柄）0位是其CAD参考方向；零位曲柄根到销的向量约为
`[0.105736,-0.000191,-0.040982]m`。它不是大小腿内角0°，也不能脱离髋角独自确定真实内角。
被动膝铰raw坐标为0时，实际内角为44.93665895°；策略nominal的被动膝raw为左右±0.4rad，
对应实际内角67.85497076°。

actor的四腿nominal为`[0.42,-0.137422826,-0.42,0.137415576]rad`，约为
`[24.0642,-7.87375,-24.0642,7.87333]°`。观测角偏差在该姿态为0，不代表机械关节坐标为0。
实机必须用可识别构型标定编码器到这些模型坐标的offset，不能把任意上电姿态直接当模型零位。

### 3.3 驱动到模型映射

当前RMCS腿部采用：

```text
q_model_P4  = J_leg * q_api_P4 + offset
dq_model_P4 = J_leg * dq_api_P4
tau_api_P4  = J_leg^T * tau_model_P4
q_model_wheel[i] = wheel_scale[i] * q_api_wheel[i]
tau_api_wheel[i] = wheel_scale[i] * tau_model_wheel[i]
```

这里`q_api`已经经过驱动的MIT解码、`reversed`和`angle_bias`处理。J必须从这个API参考端标定，
外部链传动的绝对比例已确认1:1，不能再凭DM型号重复加入一次9:1减速比或重复镜像反向。
这不表示真实膝内角对膝驱动轴也是1:1；闭链映射单独处理。非恒定传动则需要实际`J(q)`而不是常数矩阵。

必交标定字段：功能轴、物理输出件、总线与ID、反馈参考端、力矩参考端、零位、方向、比例／耦合、
反馈回绕规则、驱动量化范围、真实力矩／电流限幅。校验瞬时功率：
`tau_model^T*dq_model ≈ tau_api^T*dq_api`，损耗模型另行定义。

当前源码路由是：轮在CAN0、发送组帧0x200；髋在CAN1、膝驱动在CAN2，各侧命令ID为1/2。
这只是源码配置，不是已完成的实物接线验收。DM反馈MST_ID与命令ID可以不同。

DM `P_MAX/V_MAX/T_MAX`用于协议量化；`T_MAX=54`这个配置占位值不表示允许输出54Nm。
当前驱动限幅是`min(dm_control_torque_max, dm_mit_torque_max)`，还必须与模型侧40Nm及实际传动后的限制共同满足。
M3508驱动当前配置减速比15.8，训练包络先验使用11；不能把任一数值当成另一参考端已经标定。

## 4. 连续角、闭链与机械限位

### 4.1 反馈解缠发生在传动映射之前

对周期为T的真实驱动反馈，基本累计式为：

```text
delta = raw_now - raw_previous
delta -= T * round(delta / T)
unwrapped_api_angle += delta
q_model = J * unwrapped_api_angle + offset
```

T必须来自实际反馈语义。**MIT的±P_MAX序列化范围不是回绕周期证明**；饱和到端点也不能通过解缠恢复。
仿真反馈使用`atan2(sin(delta),cos(delta))`处理已验证的整圈等效跳变，不能无条件照搬到任意DM反馈格式。
例如驱动端先跳2π，再经过非整数比例J后，不一定仍是模型端的整2π跳变。

接入要求：

- 用新反馈序号／时间戳更新解缠状态，检查`|delta|`与速度上限、采样间隔是否一致。
- 丢帧到跨圈不唯一、编码器清零、设备重启时，不继续沿用旧圈数和旧轨迹。
- 初始圈数可以有共同整圈偏置，但同侧两轴必须位于一致的闭链装配分支。
- `/wheel_leg/calibrate`会给四台DM发设零指令，它不是模型坐标标定，也不能在恢复途中调用。

### 4.2 返回目标时同侧两轴共享圈数

对每侧`hip/aux`，先用髋轴选择共同圈数k：

```text
k = round((q_goal_hip - q_ref_hip) / (2*pi))
delta_hip = q_goal_hip - q_ref_hip - 2*pi*k
delta_aux = q_goal_aux - q_ref_aux - 2*pi*k
```

独立包角会让两轴分走相反方向，命令出一整圈错误的相对膝运动；用两轴平均值选k也可能把髋扫腿方向选反。
例：`q_ref=[-2.44,-3.86]`、`q_goal=[0.30,-0.05]`时，正确增量是`[2.74,3.81]`。
辅助轴增量超过π是有意保留闭链分支，不应单独折回负方向。

Torch的`round`在精确半整数采用ties-to-even；C++ `std::round`不是同一规则。
离线对照应覆盖±π及其两侧，明确一致的tie规则。

`ORBIT/THRUST/SIDE_SWING`使用有方向的连续参考，增量直接取`goal-reference`，不做上述最短路折返。
本限制作用于恢复轨迹，不修改原ONNX动作解码的包角语义。

### 4.3 真膝角和气簧约束

- 大小腿**内角40–110°**；髋允许连续旋转，但同侧辅助根轴必须协调。
- 气簧自身全伸205mm，AB销距为气簧当前长度＋27mm；全伸销距232mm，全压销距152mm，行程80mm。
- 对本冻结模型，`theta0=pi-2.3573`：左内角=`theta0+q_L_joint2`，右内角=`theta0-q_R_jonit2`。
- 上式里的被动膝角须通过闭链几何或实际膝角传感器得到，不能直接用主动辅助轴角度替代。

参考附件提供了`delta=q_aux-q_hip`到内角、slider位移的45点映射。
该装配分支的45点普通最小二乘线性拟合，左右最大内角误差约**2.66°**，
大于当前RMCS `hinge_margin=0.03rad≈1.72°`。因此线性占位映射必须验证全域残差，
建议用闭链FK／单调LUT及其导数实现真实机械余量。

机械保护应同时检查膝角、气簧行程、向限位运动的速度和驱动力矩方向。
当前静态软限位力矩投影不能保证高速轨迹的制动距离；后续逐轴饱和也可能改变投影方向。
模型侧限幅、机械保护、驱动侧限幅应统一校核，记录每一级改写了多少力矩。

仿真判失败容许的0.03rad膝求解误差、1mm气簧误差、3mm闭合残差是数值判据，
**不是实机允许越过机械止挡的余量**。40°端点压缩约72.795mm，已进入气簧90–100%行程的研究外推段。

## 5. 控制状态机与切换条件

### 5.1 外层状态与内部阶段不可混用

现有RMCS外层：`INIT=0, IDLE=1, PREPARE=2, RL=3`。
建议恢复各阶段运行在外层PREPARE，混合完成后再进入外层RL；新增独立恢复阶段输出用于诊断。

**仿真阶段3是PREPARE，RMCS外层状态3是RL，不能直接把两个枚举数值互相发布。**

| 仿真阶段ID | 名称 | 职责 |
|---:|---|---|
| 0 | ZERO | 试验用零主动力矩释放窗口 |
| 1 | FOLD | 收到50°内角参考构型 |
| 2 | PLANT | 中等倾斜工况的支点放置 |
| 3 | PREPARE | 建立站立目标并等待接管条件 |
| 4 | BLEND | 200ms模型侧力矩混合 |
| 5 | RL | normal策略控制 |
| 6 | FAILED | 有界失败退出 |
| 7 | ORBIT | 两根主动轴协调定向回转 |
| 8 | THRUST | 接触后70°内角蹬伸 |
| 9 | SIDE_SWING | 上侧伸长、下侧摆动 |
| 10 | WAIT_GROUND | 近正立释放后等待接触及动量降低 |
| 11 | CAPTURE | 翻正后的支撑、制动和落脚捕获 |

实机收到准备／恢复请求后从真实反馈开始，不执行仿真reset、不设置根位姿。
仿真0.2／1秒释放窗口用于产生测试初态，**不是实机每次请求自起都先主动卸力摔一次**。

### 5.2 初始路由

当前`recovery_auto`在使能瞬间按以下优先级选择：

1. `tilt>=70° && abs(g_y)>0.7`：侧翻路径。
2. `tilt<15° && h>0.20m`：`stand_rl`正立／低蹲准备。
3. 其余`tilt<70°`：`fold_plant_rl`路径。
4. 其余姿态：按pitch正负选正／反向矢状面回转。

ORBIT启动前再次检查pitch并锁存本次方向。状态选择使用实际使能姿态，不使用“前倒／后倒”测试标签。
实机没有可信高度估计时，不能把未知h默认为0.305来完成此分类。

### 5.3 主要转移条件：按当前脚本复现

| 转移 | 当前条件／锁存内容 |
|---|---|
| 近正立进入PREPARE | 记录当前连续q；stand路径高度>0.20m，倾角<15° |
| 释放后的stand进入WAIT_GROUND | 两轮最小垂向载荷<2N；这条判断在释放测试中启用 |
| WAIT_GROUND结束 | 两轮最小垂向载荷>3N或非轮接触峰值>10N；角速度模长<1.5rad/s、腿速绝对值最大<2rad/s，连续40ms；重置参考为当前q |
| FOLD完成 | 参考距目标最大<0.02rad且阶段时间≥0.3s；回转／侧摆路径还要求实测跟踪误差<0.2rad |
| PLANT进入PREPARE | 参考误差<0.15rad，两轮接触力模长均>3N持续60ms |
| PLANT失败 | 2.5秒仍未建立上述支撑 |
| ORBIT进入THRUST | 两轮接触力模长均>3N、倾角<145°、高度>0.12m、回转参考角>0.2rad；锁存回转角和连续pitch |
| ORBIT直接捕获到PREPARE | 倾角<30°、高度>0.22m、两轮接触力模长均>3N；本条件优先于同周期THRUST切换 |
| THRUST进入CAPTURE | 倾角<65°、高度>0.24m、角速度模长<8rad/s；尚不允许直接接RL |
| 侧平面转换 | 已在侧平面启动后，`abs(g_y)<0.65`且侧摆时间>50ms；转回FOLD并按实际pitch选择后续方向 |
| ORBIT失败 | 扫到配置的完整圈数仍未捕获；本配置最多1圈 |
| SIDE_SWING失败 | 往返时间超过`2*(pi/6)+0.4≈1.447s`仍未转换 |
| 全局超时 | 主动恢复超过8秒；部署按恢复开始计时，不把试验释放时间计入主动预算 |
| BLEND／RL失姿 | 倾角>45°退出，记录`lost_upright_during_takeover` |

当前PLANT／ORBIT的承重判据用接触力**模长**，世界对齐与严格接管用**垂向分量**。
移植时不要悄悄互换：侧向碰撞也可能使模长变大。若改成更严格的法向判据，应作为动作版本变更重做回归。

### 5.4 接管门禁与成功门禁

`stand_rl`和回转恢复路径开始BLEND，要求连续100ms满足：

- 倾角<8°；角速度模长<0.75rad/s。
- 高度处于0.27–0.36m；机体系水平速度模长<0.25m/s；机体系垂向速度绝对值<0.15m/s。
- 两轮垂向载荷均>5N；所有非轮构件接触力模长的最大值<5N。

legacy `fold_plant_rl`当前使用另一组较宽门禁：参考就绪、实测关节误差<0.12rad、倾角<20°、
角速度<1.5rad/s、两轮接触力模长>2N、高度>0.27m，持续100ms。
前倾45°仅3/5通过，因此不能把这条路径当成与严格CAPTURE门禁已经等价。

实验“最终成功”要求进入RL后，在**观察窗口末尾连续≥1秒**满足：倾角<10°，高度距0.305m<20mm，
水平速度<0.2m/s，角速度<0.5rad/s，两轮接触力模长>2N，非轮接触峰值<5N，且机械未失败。
建议部署单独输出`handover_ready`、`recovery_completed`和持续稳定时间；外层进入RL不立即等于恢复完成。

### 5.5 失败与取消

当前仿真FAILED下主动六轴力矩为0，气簧仍有力，GUI在选中失败时暂停。
实机没有“暂停物理”：失败应进入既有IDLE／驱动停机路径，并锁存原因、清除RL缓存与恢复计时。
急停、失联、非有限值和机械越界在每个执行器tick优先处理，不等下一次50Hz推理。
重新尝试需要新的有效请求和重新初始化，不能在失败后无限自动扫腿。

## 6. 参考姿态与轨迹计算

### 6.1 参考量全部在模型输出端

下面四维顺序统一为`[L_joint1, LL_joint1, R_joint1, RR_joint1]`，单位rad，表内保留6位小数。
完整精度及SHA见数值参考附件。

| 参考 | 左髋 | 左辅助根轴 | 右髋 | 右辅助根轴 |
|---|---:|---:|---:|---:|
| actor名义q0 | 0.420000 | -0.137423 | -0.420000 | 0.137416 |
| FOLD，内角50° | 0.177818 | 0.043622 | -0.177818 | -0.043629 |
| THRUST，内角70° | 0.434238 | -0.170630 | -0.434238 | 0.170623 |
| 侧翻上侧伸长，内角95° | 0.711288 | -0.417986 | -0.711288 | 0.417978 |
| 0.305m几何平衡姿态 | 0.359514 | -0.106477 | -0.359514 | 0.106470 |
| 0.305m准备目标，Kp120 | 0.326994 | -0.071460 | -0.332353 | 0.073090 |
| 0.305m正立目标，Kp80 | 0.310734 | -0.053952 | -0.318772 | 0.056400 |
| 0.350m捕获支撑目标，Kp120 | 0.486003 | -0.208743 | -0.491306 | 0.210578 |
| 0.400m普通恢复支撑目标，Kp120 | 0.673618 | -0.377482 | -0.678504 | 0.379293 |
| 0.400m正立支撑目标，Kp80 | 0.654732 | -0.357185 | -0.662062 | 0.359905 |

准备目标不等于actor的q0；不能为了起立而改掉ONNX观测／解码的nominal。
带前馈的目标由`q_prepare=q_geometry+tau_static/Kp`得到。0.305m静态前馈P4约为
`[-3.902411, 4.202065, 3.259362, -4.005600] Nm`，左右不完全对称。
更换Kp时必须重算偏置，不能将Kp120的目标原样交给Kp80。

### 6.2 普通参考推进

周期`dt=0.005s`。普通阶段先用第4节共同圈数得到delta，再限制参考增量。

```text
q_ref += clip(delta, -speed*dt, speed*dt)
```

正立路径PREPARE／BLEND使用四轴共享进度：

```text
lambda = min(1, speed*dt / max(max_abs(delta), 1e-6))
q_ref += lambda * delta
```

默认速度：FOLD／PLANT 2rad/s，普通PREPARE快推6rad/s，stand路径1rad/s，
ORBIT／SIDE_SWING 6rad/s，THRUST／CAPTURE 4rad/s。非stand BLEND沿用普通参考的2rad/s。

### 6.3 回转和蹬伸

冻结模型的根轴Y分量P4约为`[-0.9999894,-0.9999894,+0.9999894,+0.9999894]`，记为`s_i`。
实际轴还有很小的X/Z分量；完整向量在附件，不凭左右标签猜符号。

```text
orbit_angle += orbit_speed * dt
q_goal_i = q_orbit_start_i + direction * orbit_angle / s_i
direction = +1 for rollover_positive, -1 for rollover_negative
```

进入THRUST时锁存`plant_orbit=direction*orbit_angle`和`plant_pitch=continuous_pitch`：

```text
offset = plant_orbit - (continuous_pitch - plant_pitch)
q_goal_i = q_thrust70_i + offset / s_i
```

pitch也要连续解缠。仅使用每帧`atan2`主值，会在跨±π时制造约2π的参考跳变。

### 6.4 世界对齐与支撑高度

只有两轮垂向载荷均>2N且姿态在允许域内才做世界对齐；普通PREPARE／BLEND限制65°，CAPTURE限制85°。

```text
extension = clip((tilt - 12deg) / 43deg, 0, 1)
q_base = q_stand + extension * (q_extended - q_stand)
q_goal_i = q_base_i - pitch / s_i
```

CAPTURE使用0.35m的`q_extended`，其余使用0.40m。失去支撑时不继续更新这个世界对齐目标。
当前实现回到对应基础准备目标，并继续经过参考限速；它仍可能产生较大摆动，不能当作已完成最优制动。

### 6.5 侧翻动作

当前成功配置是`side_pattern=lower_only`、`side_direction=+1`、`side_long_leg=upper`：

1. FOLD后比较两轮世界高度，锁存贴地侧与上侧。
2. 上侧参考增加`q_95-q_50`，作为伸长支撑腿。
3. 贴地侧两根主动轴共同摆动；单程πrad、速度6rad/s，回程前停留0.08秒。
4. 离开侧平面后再FOLD，按实际pitch选矢状面路径。

轮相对高度可由`[R_WB*(p_L_B-p_R_B)].z`求得，不需要根的绝对世界高度。
但“较低的一侧”不代表“该轮已承重”。近等高时应锁存确定的分支并记录置信度，不能每周期抖动换腿。

## 7. 六轴力矩、气簧与输出仲裁

### 7.1 脚本腿部控制

```text
tau_pd = clip(Kp*(q_ref-q_continuous) - Kd*dq, -40, 40)
```

- 普通恢复：Kp120、Kd4；stand路径：Kp80、Kd2。
- 反馈误差使用连续坐标，不在有方向回转的PD里再次包成最短角。
- CAPTURE额外加`tau_capture_i=2*omega_B.y/s_i`，再限幅到±40Nm。
  该分量在同侧两根轴上形成共同世界轴向力矩；未饱和时对机身Y轴的合反作用为`-8*omega_B.y`。

### 7.2 气簧有三种不同角色

| 角色 | 仿真 | 实机接法 |
|---|---|---|
| 被动机械力 | 每物理子步给两个P关节施加F(s) | 实体气簧自然提供，不新增电机通道 |
| 脚本气簧抵消项 | FOLD／ORBIT／SIDE_SWING等指定分支使用广义力补偿 | 通过四台DM抵消部分弹性负载；需要几何及力曲线依据 |
| RL控制 | actor学习过含气簧的动力学，部署用原PD解码 | 不额外叠加恢复脚本前馈 |

冻结仿真力曲线：`F=280+122.735918491*u²+53.969804409*u³ N`，`u=clip(s/0.08,0,1)`；
精确系数以`fit_10mpa.json`／附件为准。90–100%压缩区是明确的研究外推。

脚本补偿用`delta=q_aux-q_hip`，45点表插值得到`ds_slider/d_delta`：

```text
c = F * d_slider_d_delta
tau_hip += c
tau_aux -= c
```

导数对应**slider伸长坐标**，不是压缩量导数。真实气簧存在不意味着这项主动抵消可以省略校核，
也不意味着应在所有阶段都加上它。

| 分支 | 当前气簧抵消处理 |
|---|---|
| 回转／侧摆类FOLD、ORBIT、SIDE_SWING | 开启查表抵消 |
| THRUST | 关闭抵消，利用被动力参与蹬伸 |
| balanced PREPARE、CAPTURE、BLEND | 关闭额外抵消；静态负载已编码进参考偏置 |
| stand路径／legacy fold_plant路径 | 不走回转类的额外气簧抵消分支 |
| RL | 仅原策略PD |

复现时保留限幅顺序：**PD先限幅 → CAPTURE附加项后限幅 → 指定气簧抵消项后再次限幅**。
把所有项先相加再只限幅一次，在饱和区并不等价。

### 7.3 脚本轮部控制

轮轴Y分量在名义构型约为`[+0.9999894,-0.9999894]`，记为`w_i`。

```text
pitch_effort = 8*pitch + 1.5*omega_B.y
tau_wheel = pitch_effort*w_i - 0.2*dq_wheel_i
```

THRUST使用`-0.2*omega_wheel_world_y*w_i`制动。
启用`brake_on_body_contact`且非轮接触峰值>10N时，非stand分支改为`-0.4*omega_wheel_world_y*w_i`。
最终只在相应平衡／THRUST阶段写入轮通道；FOLD／ORBIT／SIDE_SWING等其余阶段轮主动指令为0。

轨迹启动门禁和非轮接触判断需要第9节的估计，不用电机电流大小直接替代牛顿制接触力。
世界轮角速度也不是轮关节相对速度；移植需结合姿态、连杆运动与轮速重建。

### 7.4 包络和力矩输出

训练轮包络以模型轮速绝对值乘11查询电机侧线性表：

```text
speed_motor = abs(dq_wheel_model) * 11
torque_motor_bound = interpolate([0, 999.063011765], [0.348880597015, 0], speed_motor)
bound_model = min(11 * torque_motor_bound, 3.837686567164179)
```

高于表尾速度时bound为0；它是研究先验，不是实测M3508完整四象限曲线。
脚本开启牵引限幅时还取`min(bound_model, 0.8*0.5*max(Fz,0)*r)`，本资产`r=0.06m`。

统一输出链：模型侧脚本／RL力矩、混合及模型限制处理完成后，再做`J^T`映射和真实驱动侧包络／电流限制。
四DM发送模式A的`control_torque`，MIT帧中Kp/Kd为0；两M3508也由PC计算最终力矩。
不能把中间`q_ref`改发到DM内部位置PD，同时又发送已计算的PC力矩。

## 8. ONNX 接管与时序

### 8.1 35D输入和动作约定

| 索引 | 内容与处理 |
|---|---|
| 0–3 | vx、vy=0、yaw命令；目标高度×5 |
| 4–6 | `omega_B`×0.5 |
| 7–9 | 单位重力方向`g_B` |
| 10–13 | P4主动腿角相对actor q0的包角误差 |
| 14–15 | 轮绝对角占位，恒0 |
| 16–21 | P6模型轴速度×0.1 |
| 22–27 | 上一步裁剪后的归一化policy动作P6 |
| 28–34 | 自起接管使用normal常量`[1,0,0,0,0,0,0]` |

输入最终裁剪±100，连续float32。恢复期间命令固定`vx=0,yaw=0,h=0.305`，不启用预留recover槽。

**本次12486真实合同：腿动作±3，轮动作±9。**`ONNX_INTEGRATION_V5_20260925.md`中轮±3的条目与合同不符，
应以`policy.onnx.contract.json`、`V5Control.decode()`以及RMCS `action.cpp`实际±9实现为准。

```text
q_RL_goal = q0 + 0.25 * clipped_leg_action
q_RL_target = q_now + wrap(q_RL_goal - q_now)
wheel_speed_target = 10 * clipped_wheel_action
tau_RL_leg = clip(60*(q_RL_target-q_now) - 2*dq, -40, 40)
tau_RL_wheel = clip(0.2*(wheel_speed_target-dq_wheel), -bound_model, bound_model)
```

### 8.2 历史和混合

1. 恢复开始清零上一动作、运动命令参考、jump上下文；不能保留摔倒前的行走命令。
2. Python参考每20ms做影子推理；在BLEND/RL之前不让策略力矩驱动，也不更新上一动作历史。
3. 进入BLEND时上一动作置0一次。保持独立的`recovery_targets`与`policy_targets`。
4. 每5ms用最新反馈分别计算两套模型侧力矩：
   `tau_mix=(1-alpha)*tau_script+alpha*tau_RL`，`alpha=clip(elapsed/0.2,0,1)`。
5. 混合的是力矩，不是角目标／轮速目标；不把上一帧力矩保持200ms当成混合。
6. BLEND/RL内每次策略更新保存其裁剪原始动作；不回填混合力矩、实际饱和力矩或角目标。
7. BLEND转外层RL时保留历史和推理时钟，不再次清零或插入重复推理。

严格复现Python时保持50Hz全局时钟及影子推理缓存。若改为BLEND入口立即强制新推理，
应明确该时序变化并做对照，不能宣称与原逐步轨迹完全相同。

### 8.3 在1000Hz RMCS执行器中的分工

| 频率 | 职责 |
|---|---|
| 1000Hz | 接收／发布反馈、检查取消与故障、重发最新有效六轴力矩 |
| 200Hz | 恢复子状态、参考推进、双路力矩与混合、机械／驱动限制 |
| 50Hz | 35D组装、ONNX推理、策略目标及上一动作更新 |

当前RMCS普通PREPARE在执行器每个tick推进，PD每5tick计算。移植本动作时建议恢复FSM统一落在200Hz，
使0.3秒、40ms、100ms等门禁与参考相位对齐。1000Hz重发不等于1000Hz重新计算PD。
仿真400Hz子步属于数值积分精度，不是部署PC控制频率。

发生调度超期时不连续补跑多次使用同一份旧反馈的控制；记录实际dt、跳周期及反馈年龄，按既有故障路径处理。
建议在恢复完成的稳定窗口结束后才逐渐释放上层运动指令；不能在翻正瞬间恢复摔倒前的非零vx/yaw。

## 9. 部署所需的观测与估计接口

### 9.1 现有输入

下表首先是**RMCS组件接口**。`ValueBroadcaster`只会把已选择的double/int输出转成同名ROS话题；
四元数、向量、bool和`size_t`不由该通用转发器自动导出，组件输入名也不是可直接`ros2 topic pub`的控制话题。

| 接口 | 类型／单位 | 当前用途 |
|---|---|---|
| 六电机前缀`/angle`、`/velocity`、`/torque`、`/max_torque` | double；驱动API参考端SI单位 | 标定映射及驱动限制 |
| 四DM前缀`/fault_code` | int | 故障输入 |
| `/wheel_leg/feedback_fresh` | bool | 目前要求六电机及IMU各自接收时间距当前<50ms |
| `/wheel_leg/imu/quaternion` | `Eigen::Quaterniond` | 姿态，结合外参转base |
| `/wheel_leg/imu/angular_velocity` | `Eigen::Vector3d`，rad/s | gyro |
| `/chassis/control_state` | int | 外层请求0/1/2/3 |
| `/chassis/reset_count` | `size_t` | 取消／复位序号 |
| `/chassis/control_velocity` | BaseLink方向向量 | x为vx，z承载yaw角速度命令 |
| `/chassis/control_height` | double，m | 命令高度，不是实际高度 |
| `/predefined/update_count`、`update_rate`、`timestamp` | tick、Hz、steady-clock时刻 | 调度 |

当前遥控上层从双拨杆DOWN切到任一MIDDLE时请求RL=3，RL组件内部先经过PREPARE；
双DOWN／未知拨杆触发复位并回IDLE。扩展自起后仍要由同一个组件决定PREPARE何时结束。

### 9.2 需要补充的数据

| 恢复所需量 | 仿真来源 | 部署实现要求 |
|---|---|---|
| 连续四根轴角度及速度 | PhysX主动关节 | 驱动解缠、传动标定、采样时间 |
| 真实内膝角／气簧压缩／机械余量 | 被动关节及slider | 闭链FK/LUT或实际传感器，覆盖40–110° |
| 左右轮相对高低 | 轮刚体世界位置 | 编码器FK＋姿态；不需要绝对根高度 |
| 两轮垂向载荷和支撑可信度 | 接触力矩阵 | 足端力测量或经过校验的动力学观测器 |
| 非轮接触／机壳卸载 | 全部非轮刚体接触力 | 触点传感器、接触估计及置信度；IMU倾角小不能证明离地 |
| 实际base_link高度 | 世界根位置与支撑面 | 支撑确认后的腿FK／地面估计，必要时融合测距或外部定位 |
| 根部线速度及垂向速度 | 仿真COM速度 | 轮速／IMU／姿态／接触融合；轮打滑或悬空时不可直接当里程计 |
| 世界轮角速度 | 刚体角速度 | IMU、关节运动学和轮速合成 |
| 每项反馈年龄、有效性 | 仿真同步采样 | 发布每轴序号、时间戳及估计质量，不能只依赖一个聚合fresh布尔量 |

无支撑时仅靠腿姿态无法确定绝对高度；FK“轮子落地时的高度”只是条件性估计。
电机扭矩反馈也不是轮法向力。含气簧和连杆惯性时，力观测至少涉及
`J_contact^T*f = M*ddq+C+G-tau_motor-tau_spring`；多处非轮接触使问题进一步欠定。
本文件不把这种估计器标为已经实现或已经达到5N分辨率。

建议新数据结构至少包含`timestamp/valid/confidence`，未知量不能以0冒充“无接触”或“速度为零”。
实机门禁的阈值、迟滞与驻留时间应依据估计误差和延迟重新标定，保留与仿真值的版本差异。

### 9.3 建议的纯逻辑接口草案

以下是接入设计，不是当前已发布头文件；数组统一P顺序。

```cpp
struct RecoveryFeedback {
    std::chrono::steady_clock::time_point stamp;
    std::array<double, 6> q_model_continuous;
    std::array<double, 6> dq_model;
    Eigen::Vector3d omega_base;
    Eigen::Vector3d gravity_base;
    std::array<double, 2> knee_inner_rad;
    std::array<double, 2> spring_compression_m;
    std::array<double, 2> wheel_normal_n;
    double nonwheel_contact_peak_n;
    double base_height_m;
    Eigen::Vector3d base_velocity_body;
    std::array<double, 2> wheel_relative_height_m;
    bool encoders_valid;
    bool support_estimate_valid;
    bool height_velocity_valid;
};

struct RecoveryOutput {
    std::array<double, 6> torque_model;
    RecoveryPhase phase;
    bool handover_ready;
    bool recovery_completed;
    FailureReason failure;
};
```

真实实现可将相关估计量聚合为带各自时间戳和置信度的子结构。
接入顺序：先用仿真真值适配器复现控制数学，再替换为部署估计器，重跑同一评估矩阵。

## 10. 配置及建议代码结构

### 10.1 配置应明确区分模型、恢复动作和硬件标定

建议恢复配置如下。**这是配置设计草案，当前RMCS尚不识别这些`recovery`键**；
不能粘贴到YAML后就认为已启用自起。

```yaml
recovery:
  profile: v5_232mm_40_110_capture_v1
  feedback_hz: 200
  policy_hz: 50
  active_timeout_s: 8.0
  prepare_kp: 120.0
  prepare_kd: 4.0
  stand_kp: 80.0
  stand_kd: 2.0
  fold_knee_inner_deg: 50.0
  rollover_knee_inner_deg: 70.0
  side_upper_knee_inner_deg: 95.0
  fold_speed_rad_s: 2.0
  orbit_speed_rad_s: 6.0
  thrust_speed_rad_s: 4.0
  capture_speed_rad_s: 4.0
  stand_speed_rad_s: 1.0
  normal_support_height_m: 0.40
  capture_support_height_m: 0.35
  normal_support_tilt_limit_deg: 65.0
  capture_support_tilt_limit_deg: 85.0
  capture_pitch_damping_per_root: 2.0
  blend_s: 0.20
  handover_dwell_s: 0.10
  stable_hold_s: 1.0
  side_pattern: lower_only
  side_direction: 1
```

硬件层继续使用当前已存在的`joint_control_mode: torque`。
`calibration_ready/soft_limits_ready/imu_alignment_ready`依赖实际数据，文档不提供用于绕过它们的假矩阵。

### 10.2 推荐职责边界

| 模块 | 职责 |
|---|---|
| ModelStateAdapter | 新反馈解缠、J映射、IMU外参及时间有效性 |
| MechanismModel | FK、内膝角、气簧、相对轮高、机械余量；运行时使用预计算表和轻量运动学 |
| SupportEstimator | 接触、实际高度／速度、置信度 |
| RecoveryController | 纯状态机、参考、脚本模型侧力矩、原因码；不直接访问CAN |
| PolicyRunner | 原35D组装及50Hz推理，独立策略目标和历史 |
| RlController／TorqueArbiter | 外层请求、200ms混合、统一限制和驱动映射；唯一输出拥有者 |

离线几何求解／静态SVD不放进200Hz实时路径。表、力曲线、机械资产和恢复参数一起带版本和SHA。
相邻模块传不可变的当周期快照，避免状态机读到跨时刻的四个电机角和IMU。

### 10.3 当前代码最小改造清单

1. `read_model_state_()`接入连续角与有效时间，不改ONNX包角布局。
2. 用恢复子状态机替换`update_prepare_()`的单一nominal插值路径；保留外层请求值语义。
3. 将`targets_`拆为恢复与策略两套缓存，避免BLEND时互相覆盖。
4. 抽出可在PREPARE影子运行的推理更新，避免进入RL时再清空混合期间历史。
5. `compute_motor_torques_()`接收已仲裁的模型侧力矩，统一执行限位、模型包络、J转置和驱动限制。
6. 发布恢复phase、失败原因、门禁逐项状态和饱和来源；补充支持估计接口。
7. 对真实闭链替换／验证线性膝映射，对轮端补足速度相关包络和15.8／11参考端差异。

## 11. 逐级对接验收

### 11.1 离线数值对照

不连接硬件，依次比较：原始反馈映射、35D、ONNX原始动作、裁剪历史、腿／轮目标、模型力矩、驱动力矩。

- 固定模型fixture：`models/v5_flat_12486/io_fixture.json`与`infer_example.py`。
- 恢复姿态、FF和查表：本文数值附件。
- 编码器±π／多圈／丢帧；同侧共同圈数；纯侧躺pitch奇异附近；分支左右镜像。
- 每个阶段的入口、正常转移、超时、失姿、取消；门禁中未知估计的拒绝。
- BLEND权重0/中间/1、上一动作历史及50/200/1000Hz时钟关系。
- 饱和区对比实际顺序，检查虚功一致性和软限位处理后是否再次越过其他限制。

建议初始对照容差：ONNX `atol=rtol=1e-5`、参考角1e-5rad、未饱和力矩1e-3Nm；
这是数值对照起点，不替代实机测量误差的预算。阈值附近单独测试严格不等号和驻留周期数。

```bash
PYTHONPATH=src /home/yukikaze/isaacsim60-venv/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_v5_activation.py tests/test_v54_deployment_io.py
/home/yukikaze/isaacsim60-venv/bin/python -B models/v5_flat_12486/infer_example.py
```

### 11.2 C++仿真闭环对照

先用同一PhysX机械资产与相同初态，将Python恢复输出替换为C++模块输出，其他物理条件不变。
保存逐200Hz参考／力矩及阶段事件；比较恢复方向、接管时刻、最大倾角、摆幅、接触序列、机械余量。
更换估计器前先证明控制数学一致；更换后再分别量化估计误差造成的差异。

### 11.3 硬件逐级验证

顺序按依赖进行：

1. 驱动端与模型端单位、零位、符号、比例、反馈回绕及MIT寄存器核对。
2. 单侧两根主动轴协同、小幅低能量动作；确认真实膝／气簧限位和允许扫腿空间。
3. 靠近正立的准备与RL交接，验证取消、失联及输出归属。
4. 小倾角、有明确支撑的恢复，验证接触／高度／速度估计。
5. 分别验证左右侧翻、前后倒扣及混合初态；记录失败，不用平均值掩盖单类问题。
6. 扩充摩擦、质量／负载、气簧压力、初始动量和实测时延／抖动后再建立发布能力矩阵。

现有USB-HS回环RTT约80μs的证据不等于六轴端到端闭环时延；不能上下行各加一次完整RTT，
也不能把它直接换成一个5ms或20ms控制tick。参考`V5_USB_LATENCY_CALIBRATION_20260924.md`。

## 12. Isaac Sim 仿真方法及复现

### 12.1 实际仿真手段

| 项目 | 本次设置 |
|---|---|
| 仿真器 | Isaac Sim 6.0.0.1，PhysX；Kit GUI/RTX渲染 |
| Isaac Lab来源 | `.deployment_sources/IsaacLab-3.0.0-beta2`，Git `ffff603eafc6b74264a5261cc0183d6a65390d78`；当前扩展版本6.1.14，Kit应用名3.0.0 |
| 设备 | GUI试验`--device cpu`；RTX用于渲染；正式训练在Kaiser使用CUDA |
| 机械拓扑 | 19刚体、18树关节（16R+2P）、6个真实球铰闭链约束，浮动基座 |
| 质量 | 14.642kg，保留模型左右质量差异 |
| 积分／反馈／策略 | 400Hz物理子步、200Hz反馈力矩、50HzONNX |
| 求解迭代 | articulation位置64、速度32；PhysX TGS |
| 重力 | 世界`[0,0,-9.81] m/s²` |
| 接触 | 88m×8m平地走廊，以8m箱体拼接；摩擦0.5、恢复系数0、average组合 |
| 自碰撞 | 关闭；机器人与地面碰撞开启 |
| 执行器 | 四腿和两轮主动drive stiffness/damping为0；PC显式算力矩；气簧每物理子步更新 |
| 信号扰动 | 此恢复bench关闭USB传输与观测扰动；不等同于V5.9正式训练的USB随机化验证 |
| 初态 | FK闭链构型＋碰撞支撑高度，底部初始离地约3mm；rpy标准差0.003rad，seed=190619+replica |
| 状态写入 | 仅trial reset；运行中只施加六轴力矩及被动气簧力 |
| 观测来源 | IMU/编码器供actor；高度、速度、接触真值供脚本门禁和验收 |

引用`compare_v5_spring_load.prepare_trial()`仅用于**离线静态目标计算**。
自起bench没有创建该文件`LoadRig`里的竖直导轨，也没有固定机身或施加扶正约束。

### 12.2 恢复冻结资产

以下命令在训练仓库根目录执行。`/tmp`会随重启清空；目标目录不存在时才需要重建。

```bash
ASSET_ROOT=/tmp/opencode/v5_recovery_mechanics_40_110
mkdir -p /tmp/opencode
mkdir "$ASSET_ROOT"
git archive eb4a3fd12978bda55e7636c9f95addc28cc7c302 "model/纯底盘_v5/urdf" \
  | tar -x -C "$ASSET_ROOT"
sha256sum "$ASSET_ROOT/model/纯底盘_v5/urdf/manifest.json"
```

manifest应为第1节的`dbec8e...`；入口还会逐个校验其74个依赖文件。
不能用后来修改的工作树模型替代冻结资产后仍沿用原成绩。

### 12.3 自起GUI启动

在当前桌面终端执行；输出目录必须是新的。当前机器重新登录后的DISPLAY为`:1`，
**使用会话的`$DISPLAY`，不要固定写`:0`**。若会话提供`XAUTHORITY`，也传入用户服务。

```bash
REPO=/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train
PY=/home/yukikaze/isaacsim60-venv/bin/python
BUNDLE=/tmp/opencode/v5_recovery_mechanics_40_110/model/纯底盘_v5/urdf
RUN="self-righting-$(date -u +%Y%m%dT%H%M%SZ)"
xdpyinfo -display "$DISPLAY" >/dev/null
display_env=(--setenv="DISPLAY=$DISPLAY")
if [ -n "${XAUTHORITY:-}" ]; then
  display_env+=(--setenv="XAUTHORITY=$XAUTHORITY")
fi
systemd-run --user --unit="$RUN" --property="WorkingDirectory=$REPO" \
  --property="StandardOutput=append:/tmp/opencode/$RUN.log" \
  --property="StandardError=append:/tmp/opencode/$RUN.log" \
  "${display_env[@]}" --setenv=OMNI_KIT_ACCEPT_EULA=YES \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=PYTHONUNBUFFERED=1 --setenv=TMPDIR=/tmp/opencode \
  --setenv=LD_LIBRARY_PATH=/home/yukikaze/.local/lib/compat \
  "$PY" -B "$REPO/scripts/inspect_v5_activation.py" \
  --onnx "$REPO/models/v5_flat_12486/policy.onnx" --bundle "$BUNDLE" \
  --allow-mechanical-variant --output "$REPO/reports/$RUN" --gui --device cpu \
  --poses front_down_90 back_down_90 upright crouched upright_release crouched_release \
    left_side_90 right_side_90 pitch_forward_45 pitch_backward_45 \
  --methods recovery_auto --repeats 1 --seconds 12 --recovery-budget 8 \
  --prepare-speed 2 --push-speed 6 --prepare-kp 120 --prepare-kd 4 \
  --stand-speed 1 --stand-kp 80 --stand-kd 2 \
  --orbit-speed 6 --orbit-turns 1 --thrust-on-contact --physics-substeps 2 \
  --spring-compensation --traction-cap --balanced-prepare --gravity-aligned-prepare \
  --brake-on-body-contact --thrust-knee-deg 95 --rollover-knee-deg 70 \
  --rollover-speed 4 --capture-speed 4 --capture-tilt-deg 65 \
  --capture-support-deg 85 --capture-support-height .35 --capture-pitch-damping 2 \
  --support-height .40 --side-pattern lower_only --side-direction 1 \
  --side-long-leg upper --side-speed 6 --side-angle-deg 180 --side-first-leg auto \
  --fallen-release-seconds 1 --plant-angle-deg 20 --balance-wheel-kp 8
```

界面操作：先选初态，再点`Replay all initial poses`；`Run / pause`切换运行。
所有试验同时推进，下拉框只切换观察对象／相机，不会自动从零开始。
默认在选中试验成功或失败时暂停；要跑满12秒，取消该勾选后重放。
框架的`reset`是实验重置，不是部署控制指令。

### 12.4 固定回归

用同一入口去掉`--gui`、将`--repeats`设为5，并使用新的输出目录。
要复现原50个环境的布局，`--poses`顺序必须为：

```text
upright crouched upright_release crouched_release left_side_90 right_side_90
pitch_forward_45 pitch_backward_45 front_down_90 back_down_90
```

保持相同case顺序、复制数、设备、材料和初始种子；这些都会影响数值接触结果。
GUI单例不是完整能力验收。代码、参数、运行设备或估计器发生变化后，保存新结果，不覆盖原证据。

## 13. 日志、回放与问题定位

### 13.1 当前实验自动产物

| 文件 | 含义 |
|---|---|
| `source_snapshot.py` | 本次恢复脚本原文 |
| `report.json` | 身份SHA、参数、结果、列定义；在完成窗口或正常关闭后生成 |
| `runtime.json` | GUI活跃期间的PID、时间、暂停状态、选中试验；进程退出后可能残留 |
| `traces_NNN.npz` | 完整窗口的位姿、阶段、力矩、关节／接触诊断 |
| `traces_NNN_partial.npz` | 提前关闭窗口时保存的部分轨迹 |
| `tensorboard/` | `Activation/<pose>/<method>/...`行为指标 |

NPZ诊断以50Hz采样；当前TensorBoard每0.1秒记一次且只记录各初态replica0。
TB横轴为反馈tick加重放偏移`run_index*100000`，不是直接秒数；一次反馈tick为5ms。
`handover_s`包含初始释放时间，主动恢复时间应减去该case的释放时长。
`reached_stable_rl`和末尾`success`要分别看，不能把GUI提前成功暂停当作12秒末尾验收。

部署建议另存200Hz参考／力矩环形日志、50Hz观测与动作、各阶段事件及每项门禁值，
后台桥接到TensorBoard，避免在实时线程执行磁盘写入、压缩或绘图。

### 13.2 必记字段

- 模型／机械／恢复配置／标定SHA，原始反馈参考端与版本。
- `q_raw/q_unwrapped/q_model/q_ref/dq`，同侧圈数和分支、真实膝／气簧余量。
- 姿态、gyro、实际与目标高度／速度、各估计valid及反馈年龄。
- 轮法向力、非轮接触、侧摆选腿、phase开始时刻、持续支撑／门禁时长。
- 脚本、RL、混合前后力矩，模型／机械／驱动各级限幅，实际反馈电流／力矩。
- policy原始／裁剪动作、上一动作、推理时刻与耗时、PD时刻、发送序号。
- 取消／失败原因及失败前至少1秒数据。

### 13.3 现象定位表

| 现象 | 优先核对 |
|---|---|
| 正立突然连续滚翻 | 路由、支撑valid、悬空时追世界姿态、连续角／共同圈数 |
| 翻正后又倒扣，Handover=-1 | 仍是脚本问题；看THRUST动量、CAPTURE承重域、参考和接触序列 |
| 已进入RL后过冲 | 交接速度／姿态、BLEND、上一动作、同资产策略吸引域 |
| 低位拖行 | 非轮承重、轮心与轮周速度差、准备姿态是否包含正确静态负载 |
| 单侧腿方向相反 | P/C/D顺序、J、驱动reversed与模型轴符号是否被重复处理 |
| 转过一圈后力矩暴涨 | 原始反馈周期、先解缠再映射、参考不能独立包角 |
| 命令变化慢 | 键盘／上层slew和policy实际收到的命令 |
| 命令已到而车速仍慢 | 单独评估策略跟踪、限幅和力矩包络，不用提高GUI斜率冒充策略改善 |
| GUI没有窗口 | 当前DISPLAY、XAUTHORITY、Kit日志；先确认进程与窗口，不能只看旧runtime.json |

最新普通WASD回放的UI斜率已独立为6m/s²、16rad/s²，可在界面调节；这是交互命令设置，
不改远端训练合同、恢复动作增益或六轴力矩限幅。空格将命令归零，不保证物理速度瞬间为零。

## 14. 交付清单与源码索引

### 部署方应拿到

1. 认证的ONNX及原合同、原来源manifest、运行机械manifest，分清两种资产身份。
2. 本文、恢复配置版本、数值参考JSON及其SHA。
3. 实物传动／零位／方向／回绕／MIT／电流标定，而不是配置里的零矩阵占位。
4. 支撑、实际高度／速度、机械余量估计接口及误差／延迟说明。
5. Python→C++离线fixture与逐步仿真对照结果。
6. 分初态的能力矩阵、失败原因、输出退出策略、时延和资源测量。

### 权威入口

- 恢复状态和数值算法：`scripts/inspect_v5_activation.py`。
- 恢复单元边界：`tests/test_v5_activation.py`，当前14项。
- 35D组装：`src/wheeled_tasks/chassis/scut_observation.py`。
- 原RL解码与电机／气簧模型：`src/wheeled_tasks/chassis/v5_control.py`。
- 静态几何：`tools/analyze_v5_spring_limits.py::balanced_standing_pose`。
- 静态负载：`scripts/compare_v5_spring_load.py::prepare_trial`。
- NumPy部署参考：`docs/examples/v5_policy_io.py`。
- 历史过程及参数取舍：[V5_ACTIVATION_STUDY_20260925.md](V5_ACTIVATION_STUDY_20260925.md)。
- 原控制ABI：[V54_DEPLOYMENT_INTERFACE.md](V54_DEPLOYMENT_INTERFACE.md)，具体模型数值仍以合同为准。
- 数值附件：[v5_self_righting_reference_20260926.json](evidence/v5_self_righting_reference_20260926.json)。
- 物理验收回执：[v5_capture_recovery_20260925.json](evidence/v5_capture_recovery_20260925.json)。
