# V10混采续训只读审计

观察时间2026-10-04 08:07（UTC+8）。同冻结源码122d8db63f37、r11、speed attempt_001／worker3069350持续PPO，本域actor2402／3563；此次未启动新训练或评测，也未改运行合同。

[summary.json](summary.json)保存来源哈希、首／末完整100轮原生TB、各行为累计差分、计时、有效样本分母和检查点SHA。详细判读在[混采方案§8](../../../../docs/V6_MIXED_LOCOMOTION_TRAINING_20261004.md)。[remote_read.py](remote_read.py)为只读复现脚本；它需在Kaiser现有env解释器中运行，stdout输出JSON，不写远端训练文件、不创建GPU场景。

本机`snapshot.json`保留完整选定标量序列与129组差分，SHA写在summary内，未将远端1GiB event下载或提交。完整首／末窗口全部逐step确认，窗口中间未读取到的TB数据保留incomplete。停车按政策帧samples、10s驻留按完成窗samples加权；累计最大位移／倾角不能当当前趋势，缺有效窗不能当零。

结果是部分改善：后退、反向旋转下降；230／305静止基本未恢复、停车残速上升、正向yaw最近反弹。探索std同时升高，尚无同条件确定性父子评测，不认证整体遗忘或正式技能通过。记录结果不触发重启、回滚、改预算或改变自动节点。

用户追问静止原因后，08:14–08:18继续只读核查。新增[cause_summary.json](cause_summary.json)保存首／末各100轮的静止分项支付、六轴std、实际动作尺度、相位和有符号速度，以及[cause_mechanics.json](cause_mechanics.json)中的相同窗口原生物理力矩统计／有效配置／源码SHA。静止约−1.7reward/s已生效；共享探索增大、右轮参考噪声约6.86rad/s、两轮约30%近限幅请求是有依据的候选机制。低中档腿侧近限幅请求<0.7%，不支持直接归因为气簧使腿力矩不足；旧quiet／3倍字段不放大现行dense项。完整解释及未完成的因果对照见方案§9。无同卡评测、无运行修改，不能把候选机制当唯一根因。

`cause_snapshot.json`与`cause_reader.py`为本机未提交的只读原始中间数据／读取脚本，raw哈希记录在原因摘要；没有下载完整event。已逐项核对100标量点、每档132000观测帧与528000物理采样的对应窗口和源码SHA。
