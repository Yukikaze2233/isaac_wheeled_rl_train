source022首100 actor完整实测：REC3872–3971／event30812–30911，以真实critic10的event30811为基线。9,830,400 lane帧，CRC完整；读取时已完成112＝10critic＋102actor，窗口只取前100actor，不把尾部2次相加。所有输入SHA前后不变。

整体质量未恢复。静站230／305／430为12.49／10.92／8.85cm/s；相较source021第五100，230略好，305几乎不变，430更差。前／后0.5速度MAE30.40／31.59cm/s，实际signed vx为＋.19357／−.18089m/s，欠速仍明显。正／反yaw角速度MAE.6368／.6051rad/s，正向改善、反向退步；不是角度位置误差，signed yaw没有原生tag。

六启停组的真实settled_stop样本加权11.6327cm/s，179,052样本、92.9948%高于3cm/s；旧11.9373cm/s／187,575样本。小幅改善仍不足以称停车达标。230／305没有完成eligible静止10s样本，不能填零漂移；430有67例，eligible子集endpoint均值.5062m／均速8.0234cm/s，不代表全部静站。

54个静态恢复组混合L0/L1/L2，共6944结束，成功0、失败2076、timeout4868（含63边界截断）。三lesson与full_chain成功日志仍0，不能把0/6944叫最终L2整链率。轮／腿／gyro hold门归一误差2.3699／1.5925／1.6341较旧小降；瞬时边际wheel15.95%、leg24.44%、gyro25.19%，不能推出连续1s接稳概率。历史stable_peak日志的lane/time均值9.58ms，不是个体最大稳定时长；L0独立release与非零goal曝光无tag，仍unavailable。

新旧两窗不是同seal或双种子配对：旧task3486–3585与新3872–3971之间有286次actor学习，原链采样概率7.405%→11.265%，还有新episode、DR、探索与10critic适应。learnable转移9,218,653→9,103,366，脚本／blend等排除611,747→727,034。结束数、失败数下降可能来自脚本更长与分布改变，不能定为修复收益。globalmean_std .463085→.463785不是各轴实测物理噪声。

实际更新wall19.0675s／env15.9551s／optimizer.1291s／diagnostics.4765s／logger.3546s，较旧wall20.2676s少1.2000s。共享Kaiser负载与早期窗口不同，不能认定源码单因提速。所有比较仅描述质量；保留3mm guard和27.620mm工程峰、qualification=null，没有新的模型／实车资格。未开启SSH、reader、GPU、模拟或PPO。
