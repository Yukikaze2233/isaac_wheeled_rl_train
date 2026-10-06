# COM static 020 local native ABBA independent review

Four sequential fresh-process arms passed the authenticated native protocol. Each used 4096 environments, 600 policy steps and 2400 physics steps, zero PPO and no qualification. Pre-close terminal sidecars and independent CPU reviews passed.

| Arm | Probe loop (s) | Warm env.step median (ms) | Warm p95 (ms) |
| --- | ---: | ---: | ---: |
| A1_baseline | 179.952933 | 300.723865 | 317.918982 |
| B1_candidate | 167.853670 | 280.065818 | 295.837140 |
| B2_candidate | 168.462290 | 281.091528 | 297.576802 |
| A2_baseline | 180.312260 | 302.247392 | 316.739923 |

Baseline mean 180.132596s; candidate mean 168.157980s. Candidate reduction 6.648% in these two local samples per variant.
Baseline endpoint spread 0.359327s; candidate spread 0.608620s. Observed ranges overlap: False.

Semantic comparisons, app startup and archive/cold-stop costs are excluded. Warm summaries use 550 samples per arm and are not pooled distributions. This native probe includes structural audit and inference; it does not measure Kaiser PPO, optimizer or logger throughput. Two samples per variant do not establish a general performance guarantee or skill/deployment qualification.

Machine-readable evidence binding: [paired_summary_independent_CPU.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/performance/com_static_candidate_020/paired_summary_independent_CPU.json).
