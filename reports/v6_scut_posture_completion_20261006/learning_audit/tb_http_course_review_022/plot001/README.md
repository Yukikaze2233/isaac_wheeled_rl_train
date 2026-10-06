已用view_image实际检查PNG，六面板／坐标／单位／图例／说明可读。数据仅来自两份已保存Kaiser TensorBoard HTTP JSON；原JSON逐SHA保持。

- x保留原生logged update：密集12tags为30802–31846，每tag1000个reservoir样本；静站3tags为30811–31841，每tag104点。
- 淡线为raw，粗线为最近50个logged update跨度内sampled点等权均值，不插值、不冒充完整窗口均值。静站直接保留raw累计Behavior均值。
- stand/release/chain课程概率首末约56.35/32.89/10.77%→40.84/38.05/21.11%；三个当前lesson population成功率全样本恒0。
- gyro／腿门长期约20–26%，轮速门约15%；这是边际门，不是连续1s联合概率。
- 前后0.5m/s命令下实际signed vx主要约+0.2/−0.2m/s；start_stop_1驻停多数约12–14cm/s。
- 驻停图仅单组rollout条件均速，初始0缺phase samples，不能当停车成功。三高度累计速度约12.8／11.2／8.7cm/s，不能替代最新100窗口比较。

这是reservoir采样的训练曲线，不是固定评测；qualification=null。未使用SSH／GPU／PPO，未启动本机TensorBoard。
