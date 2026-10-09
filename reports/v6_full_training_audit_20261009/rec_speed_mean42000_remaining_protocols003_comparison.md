# REC41393 与 speed42000/actor597：四个剩余确定性协议

四协议按 vx_positive → vx_negative → yaw_positive → yaw_negative 串行真实完成；全部 OS exit0、native_mean_replay_completed_unscored、两候选身份核验通过、0 PPO、optimizer_constructed=false。nominal002 未重跑。

请求段是固定 5 Hz trace 抽样（额外终止行排除），停车段是原生 50 Hz 独立 10 s 实加零指令窗口。下面数字不能视为正式验收。

| 协议 | 模型 | 请求/实际均值 | 请求 MAE | 停车平面均速 cm/s | 停车最大漂移 m | 停车窗口完整 | 全场景闭链峰 mm | 全场景 failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vx_positive | REC41393 | +0.500 / +0.3591 | 0.14094 | 2.867 | 0.1290 | 4/4 | 0.681 | 5 |
| vx_positive | speed42000_actor597 | +0.500 / +0.3979 | 0.10209 | 2.956 | 0.1560 | 4/4 | 0.654 | 4 |
| vx_negative | REC41393 | -0.500 / -0.3300 | 0.16997 | 2.641 | 0.2312 | 4/4 | 0.681 | 5 |
| vx_negative | speed42000_actor597 | -0.500 / -0.3702 | 0.12978 | 2.806 | 0.2576 | 4/4 | 0.654 | 4 |
| yaw_positive | REC41393 | +1.000 / +1.0021 | 0.05213 | 2.280 | 0.2216 | 4/4 | 0.681 | 5 |
| yaw_positive | speed42000_actor597 | +1.000 / +1.7017 | 0.74149 | 2.457 | 0.1960 | 4/4 | 0.654 | 7 |
| yaw_negative | REC41393 | -1.000 / -0.9457 | 0.05625 | 2.435 | 0.2386 | 4/4 | 0.681 | 5 |
| yaw_negative | speed42000_actor597 | -1.000 / -1.3694 | 0.41558 | 2.144 | 0.1745 | 4/4 | 0.654 | 7 |

vx 单位 m/s；yaw 单位 rad/s（角速度）。全场景 boundary truncation 均为0；12个静站/命令副本均到正常 timeout，这不是任务合格。倒地 failure 需按 JSON 内各场景原因解读，四个协议的重复出生不能相加为独立成功率。

停车漂移从实际指令归零前一步的位置起算，不含此前行驶距离。305 mm 首次完整轨迹倾角、停车残余 yaw、逐副本窗口、原生计数与原始证据 SHA 均在同名 JSON。

当前只是固定小样本、未评分的中低速回放；未覆盖完整高速、复杂地形、普通/主动台阶、跳跃或实车。两个候选也不是正在训练的最新权重，qualification=null。

每个协议在同一原生环境内按 REC41393 → speed42000_actor597 顺序复用。相同 RNG／initial fingerprint 不覆盖 solver 全部内部状态；nominal002 独立审查已发现 script／BLEND 前的物理前缀差异，因此不能唯一归因 actor 遗忘。其余四协议的物理前缀差异尚未逐一独立核验。
