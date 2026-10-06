# 三 lane 首次闭链超线：实测接触、目标与力矩时序

原整体严格复现仍拒绝；本页使用独立 mechanical-only admission，0 PPO、qualification=null。

| lane／实际课 | 首次超3mm | 首次policy终止 | 前20ms核心事实 |
|---|---|---|---|
| 881／L2 front_down90 | phys887，4.435s，pin2 15.3314mm | policy222／phys888，27.6201mm | phys886 L_link1→terrain0法向4109.220N；下一setter L髋dq−90.299rad/s，实际preclip250.842Nm、帽40Nm、setter38.887Nm。已经处于PREPARE，当前目标连续约4rad/s。 |
| 1877／L2 step200_lip_near_shell | phys411，2.055s，R_rocker_P 9.9031mm | policy104／phys416，9.0361mm | phys410 RR_link1→terrain0法向31.4704N；下一setter RR轴dq73.4759rad/s、preclip−148.867Nm、帽40Nm、setter−32.799Nm。仍在ORBIT，目标±4rad/s；首次短暂超线不是最终policy终止时刻。 |
| 2121／L1实际step200_upright_top | phys998，4.990s，R_upper_E 3.7395mm | policy251／phys1004，3.4483mm | 前20ms内已有腿速约80rad/s、gyro46.315rad/s，L_link3/R_link3→terrain0接触；原preclip数百Nm，真实40Nm限幅生效。已进入PREPARE，目标±4rad/s连续；超线之后短暂回落。 |

三者在此窗均由脚本控制、learning_mask=false，均未到RL接管。因此这些首birth机械护栏事件不属于本窗PPO探索动作导致的已证实失败。881和2121的CAPTURE→PREPARE进入时刻分别是861和850；1877本窗没有此转换。

实际5ms完整轨迹可定位“接触／高速反馈／原力矩请求／下一帧闭链误差”的先后关系；这些事实不足以独立分解接触冲量、求解器误差、目标轨迹与闭链负载的因果份额。argmax pin名称不是接触对象。原6rad/s对应首birth的右删失也不证明其每子步安全。

来源SHA及全部六轴真实prephysics反馈、目标/目标斜率、preclip、帽、setter/engineeffort、terrainpatch/全netforce时序见first_guard_summary.json与完整chronological_readback.jsonl.gz；未重建encoder winding，未改原strict拒绝。
