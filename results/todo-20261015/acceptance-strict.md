## Summary (means over 24 held-out patients; baseline uses the tuned threshold)

| config                        |     AP |     F1 |   fixed_F1 |   precision |   recall |     FP |   QUBO_F1 |   QUBO_FP | runtime   |
|:------------------------------|-------:|-------:|-----------:|------------:|---------:|-------:|----------:|----------:|:----------|
| phase0                        | 0.2016 | 0.2711 |     0.1463 |      0.313  |   0.4324 | 0.0514 |    0.1734 |    0.0015 | 3:39:32   |
| spw100                        | 0.2282 | 0.2716 |     0.2749 |      0.3255 |   0.4253 | 0.0508 |    0.39   |    0.0785 | 3:40:51   |
| spw25-logpower                | 0.2246 | 0.2721 |     0.238  |      0.3093 |   0.4407 | 0.0578 |    0.3407 |    0.0287 | 3:32:59   |
| spw25-mds1                    | 0.2323 | 0.272  |     0.2422 |      0.304  |   0.4502 | 0.0536 |    0.3375 |    0.0328 | 3:41:23   |
| spw25-neg025-context10        | 0.3486 | 0.3054 |     0.3142 |      0.3373 |   0.5168 | 0.0693 |    0.3425 |    0.0487 | 3:25:45   |
| spw25-neg025-context5         | 0.3291 | 0.3049 |     0.3013 |      0.3473 |   0.518  | 0.0685 |    0.341  |    0.0571 | 3:34:59   |
| spw25-neg025-logpower         | 0.2331 | 0.2758 |     0.2791 |      0.3306 |   0.4309 | 0.0489 |    0.3948 |    0.0762 | 3:14:32   |
| spw25-neg025-robust-context10 | 0.339  | 0.4654 |     0.4633 |      0.4665 |   0.5873 | 0.0224 |    0.5011 |    0.0138 | 3:25:27   |
| spw25-neg025-robust-context5  | 0.321  | 0.4508 |     0.4549 |      0.4421 |   0.5812 | 0.0219 |    0.521  |    0.0136 | 3:25:39   |
| spw25-neg025-robust           | 0.2297 | 0.384  |     0.384  |      0.3706 |   0.5003 | 0.0212 |    0.5351 |    0.0153 | 3:13:47   |
| spw25-neg025                  | 0.2337 | 0.2766 |     0.2808 |      0.3288 |   0.4344 | 0.0515 |    0.4005 |    0.0785 | 3:07:02   |
| spw25-pbw                     | 0.2171 | 0.2817 |     0.2454 |      0.3155 |   0.4684 | 0.061  |    0.3641 |    0.032  | 3:44:37   |
| spw25-relative                | 0.0242 | 0.1215 |     0.0525 |      0.1195 |   0.2194 | 0.0669 |    0.0682 |    0.016  | 3:33:19   |
| spw25-robust-context10        | 0.3343 | 0.4766 |     0.4221 |      0.4591 |   0.5971 | 0.0219 |    0.4757 |    0.0089 | 4:16:51   |
| spw25-robust-context5         | 0.3186 | 0.447  |     0.4272 |      0.4335 |   0.59   | 0.0216 |    0.4962 |    0.0089 | 4:04:50   |
| spw25-robust                  | 0.2185 | 0.3824 |     0.3754 |      0.3807 |   0.4844 | 0.0197 |    0.5105 |    0.0071 | 3:32:29   |
| spw25                         | 0.2269 | 0.2766 |     0.2396 |      0.3095 |   0.4582 | 0.0593 |    0.3329 |    0.0286 | 3:37:59   |
| spw500                        | 0.2395 | 0.2361 |     0.2361 |      0.202  |   0.5854 | 0.1121 |    0.3758 |    0.1099 | 3:44:01   |

## Sequential decisions (todo.md rules, AP tolerance 0.0)

- Phase 1: class weight: spw25 vs phase0: better=10 worse=7 tie=7; majority=Y, AP_up=Y, F1_up=Y, FP_ok=Y
- Phase 1: class weight: spw100 vs phase0: better=9 worse=9 tie=6; majority=N, AP_up=Y, F1_up=Y, FP_ok=Y
- Phase 1: class weight: spw500 vs phase0: better=5 worse=16 tie=3; majority=N, AP_up=Y, F1_up=N, FP_ok=N
  -> keep spw25
- Phase 1: max_delta_step: spw25-mds1 vs spw25: better=6 worse=13 tie=5; majority=N, AP_up=Y, F1_up=N, FP_ok=Y
  -> revert, best stays spw25
- Phase 2: patient-balanced weights: spw25-pbw vs spw25: better=10 worse=7 tie=7; majority=Y, AP_up=N, F1_up=Y, FP_ok=Y
  -> revert, best stays spw25
- Phase 2: negative downsampling: spw25-neg025 vs spw25: better=11 worse=10 tie=3; majority=Y, AP_up=Y, F1_up=Y, FP_ok=Y
  -> keep spw25-neg025
- Phase 3: log power: spw25-neg025-logpower vs spw25-neg025: better=5 worse=9 tie=10; majority=N, AP_up=N, F1_up=N, FP_ok=Y
  -> revert, best stays spw25-neg025
- Phase 3: relative power: spw25-relative vs spw25-neg025: better=0 worse=22 tie=2; majority=N, AP_up=N, F1_up=N, FP_ok=N (built on spw25, current best is spw25-neg025)
  -> revert, best stays spw25-neg025
- Phase 3: robust normalization: spw25-neg025-robust vs spw25-neg025: better=18 worse=6 tie=0; majority=Y, AP_up=N, F1_up=Y, FP_ok=Y
  -> revert, best stays spw25-neg025
- Phase 3: causal context: spw25-neg025-context5 vs spw25-neg025: better=12 worse=11 tie=1; majority=Y, AP_up=Y, F1_up=Y, FP_ok=N
- Phase 3: causal context: spw25-neg025-context10 vs spw25-neg025: better=13 worse=9 tie=2; majority=Y, AP_up=Y, F1_up=Y, FP_ok=N
  -> revert, best stays spw25-neg025

Final best baseline: spw25-neg025

## QUBO gain on the final best baseline

mean baseline F1 0.2766, QUBO F1 0.4005, gain +0.1239; better=17 worse=6; FP 0.0515 -> 0.0785
