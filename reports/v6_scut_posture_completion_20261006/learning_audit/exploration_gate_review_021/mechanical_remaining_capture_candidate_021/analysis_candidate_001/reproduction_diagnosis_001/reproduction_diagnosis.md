# Remaining capture：保留严格拒绝，机械观察数据可只读分析

实际进程 exit0，285.245s，4096×600 策略／2400 物理步，0 PPO。原 strict review 的拒绝不改写。

全4096首出生、3756个事件、9条全域纪录峰值状态的解压字节和解析 float bits 均与 SIDE001 相同。identity、teaching 和两份 SIDE override 回执相同；report 的行为、物理、初末 RNG、源和控制字段一致。其 gzip 文件 SHA、付款汇总 SHA、测量汇总 SHA 和时间确实不同，已逐个绑定实际文件。

付款汇总共有 2780 个 float-bit 差异：{'standard_deviation': 1868, 'sum': 234, 'mean': 232, 'sum_payment': 212, 'mean_payment': 213, 'maximum': 10, 'minimum': 11}。其中 21 个是±0 极值，其余是 sum／mean／付款 sum／付款 mean／SD 归约尾位；最大绝对差 1.8044374883174896e-09。计数、mask分区、目录、离散事件全部相同，两臂付款对账残差均为0。原观察器使用 GPU float64 scatter_add_；差异形态与归约舍入吻合，但没有逐lane density原始流，不能证明其逐样本完全一致。

三lane新trace分别889／417／1005行，firstbirth连续5ms、终止前状态与pre-autoreset逐位一致。881／1877是L2，2121严格保留L1及实际step200_upright_top出生。SIDE001并未观察这些lane完整子步，不能把这些新trace称为已有逐子步对照相同。

新增 analysis_admission 只授权描述这份真实机械数据，strict_whole_reproduction=false、qualification=null。不绕过原拒绝，不触发GPU复验、正式训练或部署。
