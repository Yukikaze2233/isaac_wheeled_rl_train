# gyro002 实际拒绝诊断

原严格准入仍拒绝；本诊断不补写通过，qualification=null，0 PPO。

实际 root 进程 exit0，墙钟 365.826s；2400次原生更新/600策略步完整，新增 hook 已在关闭前恢复。全4096第一次真正拦截发生在 update531/reference530：消费物理state530，在physics531前改变转换mask。该条记录未保留被拦截lane ID。

|已记录lane|原首差异物理步|原有序字段差异数|仅完整接触记录重排的帧数|首次phase/control变化|
|---|---:|---:|---:|---:|
|1708|352|210|6|710|
|2216|319|215|11|667|

已记录两lane在物理0..530的state、实际控制、净力／矩阵力／聚合力及所有非layout字段逐位一致。接触记录以同一point关联的force、normal、separation及派生impulse整体逐bit核对，完整记录集合一致，只有有序slot不同。2216首次319将同一摩擦force从slot0换到slot2；1708首次352将同一normal记录从slot2换到slot3。不能把这描述成力差末位舍入，也未单独排序force或point制造等价。

原始前介入事件 2682 条及全域峰记录 7 条逐位一致；4096初始出生／私有采样／std身份一致。完整4096逐步state/control未保留，因此上述投影证据只能覆盖已记录两lane与已有事件／峰记录，不能声称所有未记录lane全过程逐位一致。

未观察到时钟错位：2216/1708的第一次非接触字段变化分别是update667/710的phase PREPARE7→CAPTURE6，正好对应各自被新门拦截，输入gyro约7.015/6.967rad/s，超过原.75门。531后可存在其他lane介入／reset／共享RNG影响，本诊断不授单lane反事实因果。

实际12s两lane均pureRL但仍未释放motion、未成功、平台未保留，均右删失。1708原SIDE1.5参考在8s脚本超时；当前这条轨迹到达pureRL只是一项描述性差异，不是本冻结反事实准入通过，也不能归因成任务收益。全4096最大闭链偏差仍27.620118mm，不能称机械全域修好。

原review、rejection、candidate及所有原始输入保持不变；新增analysis_admission仅允许这一窄描述性分析，strictwhole=false/counterfactual=false。
