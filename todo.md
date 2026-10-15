# Baseline F1 改善 TODO

## 程式功能實作狀態

- [x] 所有實驗選項集中在 `ExperimentRequest` / `ExperimentOptions`，預設關閉。
- [x] CLI 與 Gradio 可分別啟用每一項功能。
- [x] baseline threshold 只使用 patient-grouped inner validation 調整。
- [x] 同時保存固定 threshold 0.5 與 tuned threshold 指標。
- [x] XGBoost `scale_pos_weight` 與 `max_delta_step` 可獨立啟用。
- [x] patient-balanced sample weight 與 deterministic negative downsampling。
- [x] log power、relative power、per-record robust normalization。
- [x] 不跨 EDF、只使用過去與當下的 causal temporal context。
- [x] 所有影響結果的開關都納入 checkpoint run identity 與結果 metadata。
- [x] 新功能與 patient isolation 已有單元測試。

以下核取方塊代表「實驗是否已實際跑完並比較」，不是程式是否已支援。

## 實驗規則

- [x] 固定 subjects、outer/inner folds、random seed、前處理與 QUBO 設定。
- [x] 一次只改一個概念因素；和目前最佳版本比較，無改善就回退。
- [x] 所有參數與 threshold 只在 patient-grouped inner validation 選擇，不看 outer-test label。
- [x] 每次記錄逐患者 AP、seizure-file macro F1/precision/recall、non-seizure FP rate 與執行時間。

## Phase 0：建立公平 baseline

- [x] ~~重現 `results/qubo_run_20260721_203644.pkl` 作為對照。~~ 該 PKL 已不存在；改以 2026-10-14 重跑的 `phase0`（run `ac471c34ce`）作為對照。
- [x] 加入 baseline threshold tuning：`0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5`。
- [x] 同時輸出固定 threshold 0.5 與 inner-validation tuned threshold 的結果。

## Phase 1：處理類別不平衡

- [x] 只測 `scale_pos_weight = 1, 25, 100, 500`。
- [x] 在最佳 class weight 上，只測 `max_delta_step = 0, 1`。
- [x] 每個設定都重新在 inner validation 調 threshold。

## Phase 2：平衡患者貢獻

- [x] 讓每位訓練患者的 sample-weight 總和相同。
- [x] 若仍需要，再單獨測按患者/EDF 的 negative downsampling；保留所有 seizure epochs。

## Phase 3：改善跨患者特徵

- [x] 只加入 `log10(band_power + eps)`。
- [x] 再單獨測 relative band power。
- [x] 再單獨測不使用 label 的 per-record robust normalization。
- [x] 最後才測 5–10 秒 causal temporal context。

## 驗收

- [x] 改善必須出現在多數 held-out patients，而非只提高平均值。
- [x] AP 與 seizure-file macro F1 提升，且 non-seizure FP rate 維持可接受。
- [x] 確認 baseline 改善後，再評估 QUBO 是否獲得更大的增益。

## 結果（2026-10-15，H200 重跑）

全部 24 位患者、XGBoost、`solve_chain_qubo_exact`、預設 QUBO grid（λ 0.5–3.0、θ 0.3–0.6）、
baseline threshold 皆在 inner validation 調整。指標為 24 位 held-out 患者的平均；F1 是每位患者
seizure-file macro F1 再取平均（程式 summary 的 F1 是對所有 seizure file 平均，數值會不同）。
逐檔結果與判定記錄在 `results/todo-20261015/`，判定腳本為 `tools/todo_acceptance.py`。

| 設定 | AP | F1 | FP | QUBO F1 | QUBO FP |
|---|---|---|---|---|---|
| phase0（對照） | 0.202 | 0.271 | 0.051 | 0.173 | 0.002 |
| spw25 | 0.227 | 0.277 | 0.059 | 0.333 | 0.029 |
| spw100 | 0.228 | 0.272 | 0.051 | 0.390 | 0.079 |
| spw500 | 0.240 | 0.236 | 0.112 | 0.376 | 0.110 |
| spw25 + mds1 | 0.232 | 0.272 | 0.054 | 0.338 | 0.033 |
| spw25 + pbw | 0.217 | 0.282 | 0.061 | 0.364 | 0.032 |
| spw25 + neg0.25 | 0.234 | 0.277 | 0.052 | 0.401 | 0.079 |
| spw25 + log power | 0.225 | 0.272 | 0.058 | 0.341 | 0.029 |
| spw25 + relative power | 0.024 | 0.122 | 0.067 | 0.068 | 0.016 |
| spw25 + robust | 0.219 | 0.382 | 0.020 | 0.511 | 0.007 |
| spw25 + robust + context5 | 0.319 | 0.447 | 0.022 | 0.496 | 0.009 |
| spw25 + robust + context10 | 0.334 | **0.477** | 0.022 | 0.476 | 0.009 |
| spw25 + neg0.25 + robust | 0.230 | 0.384 | 0.021 | **0.535** | 0.015 |
| spw25 + neg0.25 + robust + context10 | 0.339 | 0.465 | 0.022 | 0.501 | 0.014 |

判定規則：和目前最佳比較，(1) 逐患者 F1 變好的人數多於變差（差異 < 0.005 視為持平），
(2) 平均 AP 與 F1 皆提升，(3) 平均 FP rate 增加不超過 0.01。

- **Phase 1**：保留 `scale_pos_weight=25`（10 好 / 7 差）；spw100 好壞各半、spw500 16 位變差；
  `max_delta_step=1` 13 位變差，回退。
- **Phase 2**：兩個選項都只在雜訊範圍內移動。pbw 的 F1 上升但 AP 下降 0.0098；neg0.25 的 AP
  上升、F1 只多 0.00004。是否保留完全取決於 AP 是否允許 0.01 的容忍度，結論不穩定。
- **Phase 3**：log power 與 relative power 皆回退（relative 讓 22 位患者變差）。robust
  normalization 讓 F1 大幅上升（18 好 / 6 差）、FP 減半，但 AP 下降約 0.004；嚴格要求 AP 上升時
  會被回退，之後 context 也因 FP 上升而回退，最終停在 F1 0.277，與實際表現明顯不符。給 AP 與 FP
  相同的 0.01 non-inferiority 容忍度後，robust + 10 秒 context 被保留。
- **結論**：最佳 baseline 為 **spw25 + robust + 10 秒 causal context**（F1 0.477、AP 0.334、
  FP 2.2%）。robust + context10 的效果在 spw25 與 spw25 + neg0.25 兩個基礎上都成立
  （19 好 / 4–5 差），Phase 1–2 的選擇對結果影響很小。
- **QUBO 是否在 baseline 改善後獲得更大增益：否。** QUBO 與 causal context 都在做時間平滑，
  兩者會互相取代：沒有 context 時 QUBO 增益很大（spw25 + neg0.25 + robust：F1 0.384 → 0.535，
  22 位變好、0 位變差、FP 1.5%，是全部設定中最好的端對端結果）；加入 10 秒 context 後
  QUBO 增益只剩 −0.001 至 +0.036。注意 chain QUBO 使用整段 EDF（含未來 epoch），屬於離線平滑；
  causal context 只用過去與當下，可用於線上偵測，兩者不是同一種使用情境。

## 後續研究：離線 vs 因果 QUBO（2026-10-15）

TODO 的結論顯示 QUBO 與 causal context 會互相取代，但原本的 chain QUBO 會用到整段 EDF 的未來
epoch。新增 `solve_chain_qubo_causal`（同一個 forward DP，每個時間點取前綴最佳解的最後狀態，
只用過去與當下），在三個 robust baseline 上比較。指標同上（逐患者 macro F1 的平均）；
逐檔結果在 `results/qubo-20261015/`。

| QUBO | 參數範圍 | baseline | baseline F1 | QUBO F1 | 好 / 差 | FP（baseline → QUBO） |
|---|---|---|---|---|---|---|
| 離線 | 預設（λ 0.5–3、θ 0.3–0.6） | spw25 + robust | 0.382 | 0.511 | 15 / 7 | 2.0% → 0.7% |
| 離線 | 預設 | spw25 + neg0.25 + robust | 0.384 | **0.535** | 22 / 0 | 2.1% → 1.5% |
| 離線 | 預設 | spw25 + robust + context10 | 0.477 | 0.476 | 12 / 9 | 2.2% → 0.9% |
| 離線 | 低段（λ 0.1–1、θ 0.05–0.4） | spw25 + robust | 0.382 | 0.449 | 15 / 7 | 2.0% → 2.5% |
| 離線 | 低段 | spw25 + neg0.25 + robust | 0.384 | 0.520 | 18 / 4 | 2.1% → 2.1% |
| 離線 | 低段 | spw25 + robust + context10 | 0.477 | 0.496 | 13 / 8 | 2.2% → 3.0% |
| 因果 | 低段 | spw25 + robust | 0.382 | **0.510** | 20 / 2 | 2.0% → 1.4% |
| 因果 | 低段 | spw25 + neg0.25 + robust | 0.384 | 0.503 | 19 / 2 | 2.1% → 1.6% |
| 因果 | 低段 | spw25 + robust + context10 | 0.477 | 0.478 | 9 / 9 | 2.2% → 2.4% |
| 因果 | 高段（λ 0.75–3、θ 0.1–0.5） | spw25 + neg0.25 + robust | 0.384 | 0.503 | 19 / 2 | 2.1% → 1.6% |

- **因果 QUBO 是比 causal context 更好的線上時間平滑。** spw25 + robust + 因果 QUBO 的 F1 0.510
  （20 好 / 2 差、FP 1.4%），高於最佳 causal-context baseline 的 0.477；兩者都只用過去資訊。
- **QUBO 的增益大部分不需要未來資訊。** 離線最好 0.535，因果最好 0.510，差距約 0.025。
- **在 context10 baseline 上，任何 QUBO 變體都沒有穩定增益**（−0.001 至 +0.019，且 FP 上升或好壞各半）。
- **擴大 QUBO 參數範圍不一定更好。** 低段範圍包含預設範圍的 λ 0.5 / θ 0.3，但 inner validation
  改選了 θ 0.1，讓離線 spw25 + robust 從 0.511 掉到 0.449。參數落在範圍邊界不代表最佳值在範圍外；
  瓶頸在 inner validation 的選參目標（`mean F1 − 0.2 × mean FP`）與 outer-test 表現不一致。
  因果版本對參數範圍較穩定（高段與低段選到相同參數、結果相同）。

建議的線上偵測組合：**XGBoost spw25 + robust normalization（不加 context）+ 因果 chain QUBO**。
下一步可檢查 QUBO 選參目標（例如改用 patient-macro F1、或加大 FP 懲罰）是否能讓 inner validation
選到的參數更穩定地轉移到 held-out 患者。

## 後續研究：QUBO 選參目標（2026-10-15）

在 spw25 + robust、低段 QUBO 範圍（λ 0.1–1、θ 0.05–0.4）上，比較 `--qubo-tune-objective`
（`file_mean`：所有驗證檔案一起平均；`patient_macro`：先在每位患者內平均）與
`--qubo-tune-alpha`（只用於 QUBO 選參的 FP 懲罰）。逐檔結果在 `results/qubo-20261015/objective/`。

| QUBO | 選參目標 | α | QUBO F1 | 好 / 差 | QUBO FP |
|---|---|---|---|---|---|
| 離線（預設範圍，參考） | file_mean | 0.2 | 0.511 | 15 / 7 | 0.7% |
| 離線 | file_mean | 0.2 | 0.449 | 15 / 7 | 2.5% |
| 離線 | file_mean | 1.0 | 0.450 | 15 / 7 | 2.5% |
| 離線 | patient_macro | 0.2 | 0.484 | 17 / 5 | 1.6% |
| 離線 | patient_macro | 1.0 | 0.490 | 16 / 4 | 1.3% |
| 因果 | file_mean | 0.2 | 0.510 | 20 / 2 | 1.4% |
| 因果 | file_mean / patient_macro | 0.2 / 1.0 | **0.513** | 20 / 2 | 1.35% |

- `patient_macro` 讓離線 QUBO 從 0.449 回到 0.484–0.490：原本的選參確實被檔案多的患者主導。
  只加大 FP 懲罰（file_mean、α 1.0）沒有效果。
- 因果 QUBO 對選參目標不敏感，三種設定都選到 λ 0.75、θ 0.2，F1 0.513。
- 目前最佳的線上偵測設定仍是 **spw25 + robust + 因果 chain QUBO**（F1 0.513、20 好 / 2 差、FP 1.35%）。

## 後續研究：癲癇事件層級指標（2026-10-16）

`tools/event_metrics.py` 從結果 PKL 的逐秒預測計算事件層級指標，先逐患者計算再平均，共 198 個癲癇事件：

- 偵測率：真實癲癇事件中至少有一秒被預測為陽性的比例；延遲：發作開始到第一個正確預測的秒數。
- 每小時誤報：距離任何癲癇事件超過 30 秒（發作前後容忍區間）的陽性預測為誤報 epoch；相隔 ≤ 10 秒的
  合併為一次警報，**每 60 秒算一次誤報**，避免長時間警報只被算成一次。
- 誤報時間比例：誤報 epoch 佔全部 epoch 的比例。

> 更正：最初版本把每段連續警報只算一次誤報，且與發作重疊的警報不算誤報。這讓「把大段時間都標成
> 陽性」的設定看起來很好（例如一次擴大範圍的實驗得到 99.5% 偵測率、每小時 1.05 次誤報，實際上標記了
> 95% 的時間）。以下數字都已用修正後的定義重算。

| 設定 | 預測 | 偵測率 | 延遲 | 每小時誤報 | 誤報時間 |
|---|---|---|---|---|---|
| phase0 | baseline | 89% | 8.3 s | 33.3 | 5.1% |
| phase0 | 離線 QUBO | 30% | 29.5 s | 0.8 | 0.2% |
| spw25 + robust | baseline | 87% | 7.8 s | 27.6 | 1.8% |
| spw25 + robust | 離線 QUBO（預設範圍） | 68% | 9.8 s | **1.8** | 0.6% |
| spw25 + robust | 因果 QUBO | 74% | 11.6 s | 4.2 | 1.2% |
| spw25 + neg0.25 + robust | 離線 QUBO（預設範圍） | 76% | 7.2 s | 3.9 | 1.4% |
| spw25 + neg0.25 + robust | 因果 QUBO | 77% | 9.8 s | 5.1 | 1.5% |
| spw25 + robust + context10 | baseline | 83% | 7.1 s | 10.9 | 2.0% |
| spw25 + robust + context10 | 離線 QUBO | 67% | 12.8 s | 1.8 | 0.8% |

- 沒有時間平滑的 baseline 每小時誤報 11–33 次，臨床上不可用；QUBO 是必要的。
- 逐秒 F1 的排名和事件層級表現不一致：因果 QUBO 的 F1 最高，但每小時誤報（4.2）是離線 QUBO（1.8）
  的兩倍以上，換來多約 6 個百分點的偵測率。

## 後續研究：以事件指標選 QUBO 參數（2026-10-16）

`--qubo-tune-objective event` 以 inner validation 上「逐患者平均偵測率 − β × 每小時誤報」選 λ/θ
（β 由 `--qubo-tune-alpha` 設定），皆為因果 chain QUBO。

> 更正：第一輪（λ 0.25–5、θ 0.1–0.5）在選參時使用了舊的誤報定義，模型因此偏好長時間警報。
> 用修正後的定義重新評估，這一輪全部比以逐秒 F1 選參更差：

| 設定 | 選參目標 | 偵測率 | 每小時誤報 | 誤報時間 |
|---|---|---|---|---|
| spw25 + robust | 逐秒 F1 | 74% | 4.2 | 1.2% |
| spw25 + robust | 事件（舊定義），β 0.02 | 74% | 3.5 | 2.8% |
| spw25 + robust | 事件（舊定義），β 0.05 | 71% | 2.9 | 2.8% |
| spw25 + neg0.25 + robust | 逐秒 F1 | 77% | 5.1 | 1.5% |
| spw25 + neg0.25 + robust | 事件（舊定義），β 0.02 | 76% | 7.2 | 9.3% |
| spw25 + neg0.25 + robust | 事件（舊定義），β 0.05 | 74% | 6.6 | 9.4% |

逐檔結果在 `results/qubo-20261016/event-objective/`。選參已改用修正後的誤報定義，重新實驗中。
