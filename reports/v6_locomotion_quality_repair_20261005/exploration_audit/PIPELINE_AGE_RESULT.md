# 信号年龄＋探索：第二组物理配对已完成

同update24500封存PT（v14新增622actor），两臂各20秒、N12/3case×N4、1000策略/4000物理步。H3源码87c174e78fbec4d6d0c894a4a2d673a7a214cb46，比较器独立冻结H4 5f2805a562d527aa3faab17bf9beb098c17c81df。111 probe/fixture/native-delay CPU检查及19比较器检查通过。推理0PPO，当前正式b4训练合同未更改，资格仍null。

每臂首cohort12/12原生timeout、failure/boundary/success/删失均0；每lane共同曝光4000tick。成功存活不是任务精度通过。两臂public出生/材料/config相同、native动作私有流1000draw/72000元素匹配，源/输入/actor finally均未改变。

| case/窗口 | mean均速 cm/s | 随机均速 cm/s | mean峰位移均值 cm | 随机峰位移均值 cm |
|---|---:|---:|---:|---:|
| stand_305mm/stand | 2.93 | 11.52 | 36.57 | 61.53 |
| forward_0p5/stop | 2.76 | 12.56 | 26.99 | 31.59 |
| backward_0p5/stop | 2.27 | 12.16 | 22.31 | 27.84 |

静止窗13秒，运动后的stop窗10秒、此前2秒预热；停车位移各自从窗口入口锚定，不包含合法行进。均值策略仍有持续漂移。正向steady的mean平面速度0.5653m/s、vx MAE8.16cm/s，反向0.5383m/s，命令±0.5m/s；没有高速行为资格。随机动作显著增大瞬时速度/姿态抖动，不能据此宣称全部长期净位移都由std解释。

实际延迟8/12启用，启用lane普通/宽域profile各4，长尾profile0个。六组有效年龄最大为35/35/35/40/30/30ms，最高观测40ms、目标30ms；并未覆盖80ms压力尾。每臂12000 policy-lag、48000 target、48000 sensor lane事件逐项验证；warmup有效历史、source tick、真实output及actor18D输入对上，最大编码误差mean5.96e-8、随机1.19e-7；PD仍为200Hz新反馈。原生raw jitter未保存，明确not_covered，没有伪造随机原始draw或重复buffer读取。

共同20s曝光下mean轮PD触界约0.18–6.99%，随机约35.25–38.66%；定义是原生PD限幅前请求达到当前限幅，不是实车过载，也不是总闭链反力。四腿触界随机约0.39–0.84%，mean0。

与上一对clean比较，mean静止均速2.59→2.93cm/s、随机10.35→11.52cm/s，是同权重下不同fixture的观察；初始化模块改变generator消费顺序，不声称跨角色完全同外生流、唯一延迟因果归因或多种子显著性。当前证据支持分别审查探索、剩余均值偏置和全DR联合困难，不支持关掉所有延迟或认为静止奖励没启用。

机器回执见pipeline_pair_comparison.json及pipeline_result_summary.json；原始JSONL本地保留、SHA绑定。曲线导出按实际pipeline_age_only role，物理tick为5ms、每第四物理帧采样50Hz，仅self-first-cohort瞬时值；不同于PPO更新时钟和共同曝光的时间平均。由Kaiser6006既有服务读取，本机仅SSH转发；无正式训练中断或重启。
