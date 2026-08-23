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

- [ ] 固定 subjects、outer/inner folds、random seed、前處理與 QUBO 設定。
- [ ] 一次只改一個概念因素；和目前最佳版本比較，無改善就回退。
- [ ] 所有參數與 threshold 只在 patient-grouped inner validation 選擇，不看 outer-test label。
- [ ] 每次記錄逐患者 AP、seizure-file macro F1/precision/recall、non-seizure FP rate 與執行時間。

## Phase 0：建立公平 baseline

- [ ] 重現 `results/qubo_run_20260721_203644.pkl` 作為對照。
- [ ] 加入 baseline threshold tuning：`0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5`。
- [ ] 同時輸出固定 threshold 0.5 與 inner-validation tuned threshold 的結果。

## Phase 1：處理類別不平衡

- [ ] 只測 `scale_pos_weight = 1, 25, 100, 500`。
- [ ] 在最佳 class weight 上，只測 `max_delta_step = 0, 1`。
- [ ] 每個設定都重新在 inner validation 調 threshold。

## Phase 2：平衡患者貢獻

- [ ] 讓每位訓練患者的 sample-weight 總和相同。
- [ ] 若仍需要，再單獨測按患者/EDF 的 negative downsampling；保留所有 seizure epochs。

## Phase 3：改善跨患者特徵

- [ ] 只加入 `log10(band_power + eps)`。
- [ ] 再單獨測 relative band power。
- [ ] 再單獨測不使用 label 的 per-record robust normalization。
- [ ] 最後才測 5–10 秒 causal temporal context。

## 驗收

- [ ] 改善必須出現在多數 held-out patients，而非只提高平均值。
- [ ] AP 與 seizure-file macro F1 提升，且 non-seizure FP rate 維持可接受。
- [ ] 確認 baseline 改善後，再評估 QUBO 是否獲得更大的增益。

下一步先重現對照組，再只啟用 Phase 0 threshold tuning；確認結果後才一次開一個後續選項。
