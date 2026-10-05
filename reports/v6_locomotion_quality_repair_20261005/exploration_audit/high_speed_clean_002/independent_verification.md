# 同PT高速 mean：clean 与 pipeline 独立核验

冻结源码 `40826af3c46b32bffd6b16f6817715b63ed1c236`，PT25000/task4685，对应v14新增1122次actor更新。两次均64环境、16组×4、32s、1600策略步/6400物理步，非最新实时模型；0PPO、非资格评测。

实际根姿态、18轴位置/速度、命令出生逐值相同；质量/惯量/COM、机器人与地面材料、重力、布局、控制时序均相同。有效合同除独立design绑定外只差 `signal_delay.enabled`。clean无延迟实例、诊断clock0；pipeline clock4685。

**出生actor观测并非完全相同**：gravity列7–9在64lane均不同，最大差0.01723/0.00885/0.000155。base reset先缓存signal_delay feedback，FixedCaseEnv随后写入最终带jitter根姿态；pipeline首次观测读旧缓存，clean读最终姿态。因此对照包含原生延迟分支的出生缓存语义，不能声称只隔离稳定运行后的帧年龄。未改源码。

| 首批真实终止 | clean | pipeline |
|---|---:|---:|
| failure | 4（全为非轮接触） | 7（4fall＋3非轮接触） |
| 横向boundary | 40 | 36 |
| 纵向boundary | 0 | 0 |
| native timeout | 20 | 21 |
| 未完成右删失 | 0 | 0 |

原生边界为纵向±40m、横向±3.7m。下面完整窗均值只计实际完整窗口；N0表示不可得，不补零。

| 用例 | clean 完整N/4 | clean读数 | pipeline 完整N/4 | pipeline读数 |
|---|---:|---:|---:|---:|
| stand_305mm / stand | 4 | 5.0943 cm/s | 4 | 5.1918 cm/s |
| stand_430mm / stand | 4 | 18.5525 cm/s | 4 | 18.4555 cm/s |
| backward_3_430mm / steady | 4 | -1.7826 m/s | 4 | -1.8973 m/s |
| backward_5_430mm / steady | 0 | null | 0 | null |
| yaw_positive_1rps_305mm / steady | 4 | 3.0725 rad/s | 4 | 3.0967 rad/s |
| yaw_negative_1rps_305mm / steady | 4 | 1.0340 rad/s | 4 | 1.0294 rad/s |
| yaw_positive_3rps_430mm / steady | 0 | null | 1 | 0.3362 rad/s |
| yaw_negative_3rps_430mm / steady | 4 | 0.4471 rad/s | 4 | 0.4406 rad/s |

正3rps高姿态clean 4/4在加速阶段非轮接触失败，故无完整steady/stop；pipeline仅1/4完整steady，但其真实高度约0.220m、目标0.430m，不能按yaw精度宣称成功。

后退3m/s高姿态两臂均4/4完成steady、随后横向出界；后退5m/s高姿态两臂steady均N0。全部±3/±5平移与取消组10s停车窗在两臂均N0/null，无法判断完整高速制动精度。

两臂延迟随机流不共享：clean没有profile/base/jitter抽样，不能伪造配对delay样本。私有动作seed93001和draw数量相同，但未重读归档actions核验跨角色RNG身份；不同存活曝光的partial均值不能直接作为总体因果差。

结论限于这个同PT、mean、名义出生、pipeline on/off回放：关闭延迟后，高姿态静止和横向出界仍存在；不能把问题全部归因于延迟，也不能据此认定唯一reward根因。普通pure-yaw不得推广到kind=spin_translate的世界参考特殊旋转平移。

独立JSON SHA256：`fca85357353629f54f0443ab4e09b785fc3b851df660ab0bb7788e7e1937a6a1`。未提交；未GPU/SSH/PPO/源码或合同修改。
