第四完整100actor窗（311–410／task3386–3485／TB30316–30415）与前窗不重叠，原source/event身份和TFRecord CRC完整。共95组，41基础组；全部frames=固定人口×2400、结束回合分账重算通过。带探索、reset和当前难度推进，不能当同seal因果消融或最终资格。

| 指标 | 前100actor | 新100actor |
|---|---:|---:|
|stand_230mm planar_speed_mean_m_s|12.3778 cm/s|12.7758 cm/s|
|stand_305mm planar_speed_mean_m_s|10.9480 cm/s|11.3316 cm/s|
|stand_430mm planar_speed_mean_m_s|8.4242 cm/s|7.8037 cm/s|
|forward_05 vx_mae_m_s|28.6971 cm/s|29.3703 cm/s|
|backward_05 vx_mae_m_s|28.5552 cm/s|29.5565 cm/s|
|rotate_1 yaw_mae_rad_s|0.6363 rad/s|0.6038 rad/s|
|rotate_1__reverse_train yaw_mae_rad_s|0.5522 rad/s|0.5405 rad/s|

六真实驻停组加权均速11.7513→11.6284cm/s；新窗190240条件帧，92.601%高于3cm/s。54混合L0/L1/L2恢复组0/8683（2527failure/6156timeout，另88boundary分类），不是最终L2链率。Wall均20.335s，只是共享机器读数。

本地SSH等待55s超时，原远端只读结果随后已写完、reader进程已退出；直接下载原唯一结果并核全部输入及输出SHA一致，没有重启reader或训练。原远端退出码未知，保留本地exit1，完整报告状态不据此伪造exit0。

质量仍未恢复：低中静止与前后欠速偏差变差，高姿态与旋转/停车略改善；不以总reward判通过。
