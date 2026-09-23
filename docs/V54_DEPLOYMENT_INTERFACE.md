# V5.4 部署对接规范：模型结构与使用方法

本文对应本项目V5单策略、SCUT35接口，供仿真／上位机／嵌入式部署端对接。
**模型身份以导出ONNX、配套合同、manifest和SHA回执为准。**同为35D输入，不表示可以互换华南虎模型、旧V3模型或其他35D策略。

配套可执行NumPy参考：[examples/v5_policy_io.py](examples/v5_policy_io.py)。
它与训练侧观测、动作映射和关节输出端控制数学做逐项对照测试，不包含硬件通信驱动。
训练目标域和阶段见[V54_FULL_RANGE_TRAINING.md](V54_FULL_RANGE_TRAINING.md)。

## 1. 模型由哪些部分组成

### 1.1 机械／仿真模型

- 资产目录：`model/纯底盘_v5/urdf/`；`robot.usda`用于Isaac，`robot.xml`用于MuJoCo。
- 19个刚体、18个树关节（16R＋2P）、6个球闭合约束；闭合约束Jacobian已核对秩12。
- 六个内部独立控制坐标：**四个腿部根端主动轴＋两个轮轴**；基座为自由浮动。
- 膝、连杆和气簧移动副为被动机构，不作为额外六电机动作中的“膝电机”驱动。
- 模型质量约14.642kg，轮半径0.06m，轮距0.4373m；气簧额定行程80mm，使用10MPa研究力曲线。
- 接地高度指令域0.21–0.35m，指base_link原点相对轮支撑面的高度，不是车体底部离地间隙。

结构和名义位置以`manifest.json`、`model_spec.json`为准。当前模型中的电机映射／传动标定仍标记为未确认；
下述量均为**模型关节输出端量**，不是CAN-ID、转子编码器原值或电流指令。

### 1.2 策略网络

```text
Actor:  35 → Linear(256) → ELU → Linear(128) → ELU → Linear(64) → ELU → Linear(6)
Critic: 81 → Linear(256) → ELU → Linear(128) → ELU → Linear(64) → ELU → Linear(1)
```

- Actor是单帧MLP，确定性网络约50,758参数；无RNN／Transformer，无多帧堆叠。
- “上一动作”是35D输入中的六项，不是隐藏的历史帧队列。
- 训练是普通PPO＋非对称critic。critic额外使用仿真速度、高度、接触、18个树关节状态、材质和支撑高度等真值。
- **部署只运行actor。**Gaussian动作分布和critic用于训练，ONNX导出的是确定性动作均值。
- 没有运行均值／方差归一化；输入按下面固定尺度缩放，整体裁剪到`[-100,100]`。

ONNX接口：输入`obs`为`float32[batch,35]`，输出`actions`为`float32[batch,6]`；通常batch=1。
当前导出使用opset 17、动态batch、不依赖外置权重文件；输出层是线性层，**调用方需要裁剪动作**。

## 2. 两种关节顺序必须区分

### 策略顺序P：网络中的电机观测、上一动作和六维输出

| P索引 | 模型主动轴 | 功能 | 名义q₀，rad |
|---:|---|---|---:|
| 0 | `L_joint1` | 左腿主动根轴 | 0.42 |
| 1 | `LL_joint1` | 左侧连杆／曲柄主动根轴 | −0.13742282595254576 |
| 2 | `R_joint1` | 右腿主动根轴 | −0.42 |
| 3 | `RR_joint1` | 右侧连杆／曲柄主动根轴 | 0.13741557625658019 |
| 4 | `L_joint3` | 左轮 | 0 |
| 5 | `R_joint3` | 右轮 | 0 |

### 控制／采集顺序C：`manifest.control_joint_names`

```text
C = [L_joint1, LL_joint1, L_joint3, R_joint1, RR_joint1, R_joint3]
P = C[[0, 1, 3, 4, 2, 5]]
C = P[[0, 1, 4, 2, 3, 5]]
```

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
一般写作`q_model=f(q_encoder)`、`dq_model=J dq_encoder`；固定比例传动时才可简化为零位、符号和传动比。
链传动／同轴嵌套不能自动证明某一固定角差关系，也不能假定四路转子角就是四路模型关节角。
需要标定每轴的输出归属、零位、正方向、比例／耦合关系，并在动作反向映射中保持一致。

Actor不需要真实膝角、气簧位移、接触力、根部线速度或地形高度输入。电流反馈可用于驱动层监控，不直接拼入35D。

## 4. 六维动作如何变成执行器目标

网络原始输出记为`a_P`，先裁剪：前四维到`[-3,3]`，后两维到`[-9,9]`，再映射到C顺序。

### 腿部：四路位置目标

```text
q_des = q₀ + 0.25 × a_leg
q_target = q_current + atan2(sin(q_des - q_current), cos(q_des - q_current))
tau_leg = clip(60 × (q_target - q_current) - 2 × dq_current, -40, +40)
```

角度为rad，速度为rad/s，力矩为关节输出端N·m。0.25是rad/归一化动作；±3对应名义目标偏置±0.75rad。
这不是“六路力矩策略”，也不是虚拟腿长／腿角VMC策略。
零速度命令下也可能有非零腿／轮动作来维持平衡；接口校验应对照参考输出，而不是要求六维输出全为零。

### 轮部：两路速度目标

```text
wheel_speed_target = 10 × a_wheel                  # rad/s
tau_requested = 0.2 × (wheel_speed_target - dq_wheel)
tau_wheel = clip(tau_requested, -tau_limit(dq), +tau_limit(dq))
```

当前研究先验用输出端速度绝对值乘11查询电机侧线性曲线：
`(0 rad/s, 0.348880597 N·m)`到`(999.063012 rad/s, 0 N·m)`，再乘传动比11、效率1，
并限制在3.837686567 N·m以内。超过曲线速度域，模型中的力矩上限为0。
准确实现见`v40/core.py::motor_torque_limit`和配套NumPy函数。

这些是仿真关节输出端先验。实机若使用电流／转子力矩接口，还需经过已标定传动映射、效率与力矩常数转换。
若`q_joint=f(q_motor)`，理想虚功关系为`tau_motor=Jᵀ tau_joint`；不能直接把网络输出或上述关节N·m当作电流发送。

### 4.1 上一动作历史

下一帧保存的是**本周期生成并裁剪、准备发出的归一化policy动作P**。
即使底层存在通信／执行延迟，也不应再把延迟后的动作当作这六维历史，训练中的延迟分支同样保存当前裁剪指令。
控制器重新初始化时清零；普通指令变化不随意清零。

## 5. 时序和复位

- 策略周期20ms，即50Hz；每次只运行一次ONNX，持有目标直到下一策略周期。
- 训练物理／PD更新周期5ms，即200Hz，一次策略更新对应4次低层更新。
- 低层每次用最新q/dq重新计算PD力矩，腿位置目标和轮速度目标在策略周期内保持。
- 部署硬件的内环频率可以由驱动实现，但应验证等效动态、延迟和增益；不要通过改ONNX调用频率来补偿驱动差异。
- 启用前完成传感器同步与坐标转换，设置已定义的初始指令、普通上下文和零上一动作。
- 每周期检查输入／输出有限值与时间戳；无有效新输出时由部署控制器的既有停机／保持逻辑处理，避免使用未初始化数组。
- 最终鲁棒阶段训练观测与动作各0–20ms延迟及噪声；阶段是否已通过必须读取产物回执。

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

建议将同一已选模型整理成以下bundle，保留原始文件字节以便SHA验证：

```text
policy_bundle/
  policy.onnx
  policy.onnx.json                 # 导出SHA、输入输出、Torch/ORT对照结果
  policy.onnx.contract.json        # 对应的已展开阶段合同
  manifest.json                   # 同一机械资产manifest与名义位置
  own_v40_v2.json                  # control_math_source原文件，用于轮电机先验和SHA
  artifact_selection.json         # 候选／已验收角色、accepted_stage
```

仿真对接再携带完整V5资产：`robot.usda`或`robot.xml`、网格、`model_spec.json`、`fit_10mpa.json`及所需构建元数据。
训练checkpoint `.pt`用于恢复／再导出，不是ONNX推理所必需。

**`own_v40_v2.json`不是本模型的完整部署合同。**这里只复用其执行器数学先验；它的125D观测、100Hz时序、串联腿动作顺序和名义角度都不能套到V5。
范围、35D布局、50Hz时序和V5控制字段以`policy.onnx.contract.json`为准。

已结束训练块中的`policy.onnx`也可能是未通过验收的候选。读取`artifact_selection.json`：
`accepted_stage`说明已验收的能力范围；完整能力发布应对应`full_curriculum_accepted`，不能只挑最新mtime文件。
阶段目录里的原始`contract.json`也可用于校验，只要与导出metadata的contract_sha256一致。

## 8. 可运行的离线推理示例

以下只打印推理结果，不连接硬件。先安装NumPy和ONNX Runtime，将同一模型的bundle放好，在训练仓库根目录执行：

```bash
PYTHONPATH=docs/examples python - <<'PY'
from pathlib import Path
import numpy as np
import onnxruntime as ort
from v5_policy_io import CONTROL_ORDER, verify_bundle, build_observation, decode_actions

p = Path("policy_bundle")
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
print("obs:", obs.shape, obs.dtype)
print("leg targets [rad], P[0:4]:", legs)
print("wheel targets [rad/s], P[4:6]:", wheels)
# After successfully issuing these targets, use clipped.copy() as the next history.
PY
```

真实循环中，硬件适配层负责采集、坐标／传动转换、发送四路位置目标与两路速度目标，并在20ms时基上调用上述流程。
若在软件实现与训练一致的关节输出端PD，可使用`reference_joint_torques`作对照；实机驱动转换由已标定适配层承担。

## 9. 对接验收与能力边界

按同一传感器／指令fixture依次比较：35D向量 → ONNX原始6D → 裁剪和排列 → 四腿／两轮目标 → 输出端PD。
配套测试`tests/test_v54_deployment_io.py`已对照训练侧PyTorch实现，覆盖角度wrap、索引、jump上下文和饱和。
ONNX导出另要求与Torch actor在零输入和真实观测上达到`atol=rtol=1e-5`。

正式模型的技能通过情况必须看固定评测。当前训练目标包含落地恢复，但**未定义或验收任意倒地自起**。
落地后的内部RECOVERY相位不是35D中的recover命令位；不得混用。
实机主动轴映射、编码器／IMU标定、执行器动态与时延仍需由对接方提供实测绑定数据。

源码权威入口：`scut_observation.py::build_scut35`、`env.py::get_observations/step`、
`v5_control.py::decode/motor_efforts`、`chassis_export.py::export_actor`，均位于`src/`对应包下。
