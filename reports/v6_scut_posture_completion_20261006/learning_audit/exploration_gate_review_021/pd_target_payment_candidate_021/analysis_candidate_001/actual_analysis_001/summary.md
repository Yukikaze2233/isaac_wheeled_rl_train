# 2216 native mean target/payment observation

Read 600 first-birth policy intervals and 2400 direct native reference samples. The last gate input is physics2399; the last payment uses physics2400. Each payment density is weighted once by20ms.

Native hold samples: 1416; joint-pass samples: 295; peak: 0.080s; breaks: 82. Gate failure counts are nonexclusive and cannot be multiplied into a1s probability.

Final done=False, motion_released=False, success=False; right-censored unfinished=True.

| Native interval regime | Policy intervals | Actor-valid | Native hold-paid | Leg payment | Wheel payment | Gyro payment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| passive_release | 1 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| mixed_passive_release_script | 1 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| script | 233 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| mixed_script_blend | 1 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| blend | 9 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| mixed_blend_pure_rl | 1 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| pure_rl | 354 | 354 | 354 | -1.209551 | -1.552033 | -5.681967 |

The three-axis values above are original native hold payments. Unmasked script/blend component values and explicit transition bins remain in JSON. PD requests are captured before recovery mixing and are not final shaft efforts. No five-peak reward, full negative ledger, critic value or new model quality conclusion follows this observation.

Input/report hashes and per-axis decoded/delayed targets, continuous PD errors, native gate breaks and original broad/fine values are retained in report.json.
