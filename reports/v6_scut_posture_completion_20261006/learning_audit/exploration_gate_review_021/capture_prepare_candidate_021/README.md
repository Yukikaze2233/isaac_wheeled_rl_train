仅本机隔离源码候选：CAPTURE→PREPARE 增加 `gyro.norm < config.handover_gyro_max`，复用现有0.75rad/s阈值。其余函数、默认配置、ready八门、0.1s准备、0.2s融合、1s接稳、8s总时限及一次reroute逐字节保持。生产源码、003及SIDE001候选没有修改。

实际依据：SIDE1.5的1708在3.550s以输入角速度6.967rad/s进入PREPARE，随后越过水平、翻倒，最终触发原8s期限。本候选可阻止这一次高速转移；它不能保证减速或完成自起，gyro8rad/s的proxy cutoff和真实contact问题仍在。

本目录只有源码、patch、设计和聚焦CPU测试，没有原生启动器或正式合同。CPU验证不授予机械/功能/学习资格。下一原生反事实必须显式认证修改后的隔离源码，保持SIDE1.5/REC3076/RNG/64/32/3mm等，其机制接线不得静默复用未修改源码SHA。
