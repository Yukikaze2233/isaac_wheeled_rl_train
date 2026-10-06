# source020 原生完整窗口 reader

只读现有 recovery `attempt_004` 原生 TFRecord；不启动模拟器、模型、PPO、评测或 TensorBoard，不连接 SSH。旧 `read_recovery_window_019.py` 的冻结字节（SHA `ffefbaae2e1cbf30b520224fc286a8adbc194708570e4ce0fcb00fa7f86cdb17`）必须与 adapter 同时提供。CRC、tag、完整窗口、累计差分、驻停与 conditional-stop 加权、计时、探索和历史对比均执行原函数；没有新解析器或新的曲线数据源。

Adapter 在隔离模块内替换 source/PID/时钟与文件 pins。旧 `load_authority` 仅映射明确列出的身份字符串，强验各字符串的出现次数；原解析与 metric 函数对象保持不变。额外核对020 revision 的归档 SHA、旧／新合同计划 SHA 与019前驱路径。选择的13个源码／合同文件在读取前后重算 SHA，旧 reader 在前后重算 SHA；不会打开父 PT 或把 revision header 检查说成重新校验完整归档。

认证上下文：源码 `7f315f30741be01d70046a11848443038c14b240`；归档 `ff0c57ca2255fb0e336b28be57e0a66e7923597dd5398d0bb18c4e34f91c1b87`；唯一 worker PID `3494096`；消费 origin `29679`、任务 actor origin `2758`、新 critic warmup `0`；普通姿态课程 origin `2294`、eligible `1449`（继承含872旧 alias复习人口，没有 recovery 腿库 hold）。95组／41基础／54静态恢复标签与合同人口均强验。

root 上传两个冻结 reader 到 `/tmp/scut-window020/` 后，用远端已有环境执行：

```sh
/home/kaiser/robot-rl-sim60/env/bin/python -B \
  /tmp/scut-window020/read_recovery_window_020.py \
  --base-reader /tmp/scut-window020/read_recovery_window_019.py \
  --run-dir /home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_004/train \
  --source-root /home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_7f315f30741b \
  --updates 50 100 --end-update latest \
  --output /tmp/scut-window020/recovery_window_020.read001.json
```

每次读取使用新的 output 文件名；不会覆盖旧证据。只需50窗时使用 `--updates 50`。可加 `--reference /path/to/recovery_first100_018.json`，必须匹配封存 SHA `6227273388a30292cb60c92cd83270c83f921eecd156b0dbb1504c0497ac73a6`；不传时历史对比明确 unavailable。`--end-update` 也可指定本 attempt 的十更新端点，必须留出真实十更新 Behavior 基线；不能传 absolute actor clock。

首个50窗为新更新10→60：基线 event29688，窗口 event29689–29738，actor2769–2818。首个100窗为10→110：同基线，窗口 event29689–29788，actor2769–2868。新65时50可完整而100仍 pending，至少再需45更新。LearningAudit／Timing／Policy 每更新必须完整覆盖50或100点；Behavior 要含6或11个真实十更新端点，不能要求100个 Behavior 点或伪造零基线。`latest` 会选择最近完整十更新端点，缺观测时不得认证。

输出 `source020-recovery-native-event-window-set-v1`，包括 `windows.50/100.complete`、各窗 `missing`／`missing_required_tag_count`、实际时钟、累计差分与 ended-episode 分母、按真实 samples 加权的完整 stationary_10s 和 braking／settled_stop、Timing均值／标准差、探索 `Policy/mean_std`、旧函数 reuse、identity／before-after source pins。所有请求窗完整才有顶层 `complete:true`；不足窗返回 exit0 和 `pending_requested_actor_windows`，**exit0不代表窗口完整**。身份、CRC或数值合同错误返回 exit2 和 `read_failed`。`qualification:null` 始终保留。

解释边界：恢复54个静态场景标签混合动态采样课级，不能把它们的成功当 original-chain 成功；课级 counter 是24策略帧的累计 observer 平均，不能当精确结束计数。驻停仅代表完成 quiet2s 后10s的选择样本，样本0为 unavailable；anchors可能早于窗口。日志没有95组有符号实际 yaw rate，不能由 MAE反推；姿态 reward不是实际 pitch/roll。50与100可能重叠，禁止相加。

source020只改变 COM 不变量缓存和 disabled fixed-eval catalog边界，v18合同、reward、难度、安全阈值与时钟继承。工程等价校验不能证明不同 actor／reset／探索／DR 条件下行为相同；本地 ABBA 180.13→168.16s／6.648%不能当 Kaiser PPO配对提速。实际窗口仍受共享 GPU 和启动样本影响。这里是带探索训练数据，须与 deterministic fixed mean eval 以及历史 source712／seal2579 的零PPO mean／stochastic探针分开。

CPU回执见 `read_recovery_window_020.cpu.json`：新38项＋未改旧37项，共75 passed。只测试 synthetic CRC fixtures 与真实冻结源码 pins；尚未读取远端真实020 event。
