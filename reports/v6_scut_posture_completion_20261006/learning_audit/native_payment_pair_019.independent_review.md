# source712／seal2579 原生付款独立复核

两臂认证通过：相同父模型2579／PT `7f91d355…dbcca`、冻结source712、原55 wrapper。各600策略／2400物理步／12s／2,457,600 lane区间，逐lane原付款对账最大残差0，PPO／optimizer／消费／课程clock均不前进。重新核对545源文件、PT、manifest和三个wrapper字节；首出生、八门、动作/RNG及后续出生／终止诊断与原mean／stochastic报告逐值相同。

| 首出生、actor有效、仍hold区间 | mean | stochastic |
|---|---:|---:|
| lane区间数 | 943785 | 1382174 |
| 实际／actor有效即时报酬每20ms | 0.0811746 | 0.0316908 |
| 现有腿速度密度 /s | -0.0175309 | -0.099133 |
| 现有轮速度密度 /s | -0.000329944 | -0.00146818 |
| 现有水平角速度密度 /s | -0.0145299 | -0.0634483 |
| 静止broad密度 /s | -0.153155 | -0.727535 |
| 静止precision密度 /s | -0.296386 | -0.649763 |
| pitch正基线密度 /s | 0.928652 | 0.756627 |
| roll正基线密度 /s | 0.993988 | 0.969446 |
| height正基线密度 /s | 0.999901 | 0.99768 |
| 轮缘max输入均值 m/s | 0.225769 | 0.549901 |
| 腿max输入均值 rad/s | 0.992585 | 2.82676 |
| 全gyro范数均值 rad/s | 0.42676 | 1.07993 |
| 末次reference八门同时通过比例 | 0.575466 | 0.0289558 |

现有generic腿／轮／gyro与静止成本均实际付款；hold-specific takeover项未出现在active组件中。stochastic的hold区间物理噪声和成本更高，且仍有正净即时报酬；此观测不能证明策略缺少稠密信号、熵项压过奖励或决定新增权重。优势／PPO梯度没有观测，不能将解析savedσ熵与即时r跨单位直接比较。

mean首恢复出生2284／327／36分组保持979／140／15 release、908／66／9 success；stochastic恢复出生release／success均0。原始行为完整复现，L0原motion目标被改零的已知教学缺陷仍然存在。所有12s存活lane保持right-censored。普通1449人口与恢复成本分箱独立；脚本／blend排除区actor有效付款为0，终止区保留whole-interval归属。

统计按首出生actor still-holding区间加权；它与all-native区间有重叠，不可相加。八门比例为一次reference联合观测，不能当1s连续驻留概率，也不能相乘单项marginals。密度乘.02一次，success/failure事件不乘dt。

父进程回执独立记录两臂exit0；preclose报告与恢复hooks均完整。`App.close`是否返回仍未知；不推断postclose完成。当前source020 PPO行为需另读020窗口，历史2579对照不替代它。
