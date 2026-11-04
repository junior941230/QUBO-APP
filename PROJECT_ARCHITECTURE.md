# QUBO Seizure Detection 專案架構

## 目標

本專案以 CHB-MIT EDF 記錄建立 seizure epoch baseline，再用鏈狀 QUBO 做時間平滑。評估協定是 nested patient-independent leave-one-subject-out：outer test patient 完全不參與模型、baseline threshold 或 QUBO 參數選擇。

目前優先目標不是強迫 QUBO 提升，而是先改善跨患者 baseline。實驗順序與驗收條件見 `todo.md`。

## 模組責任

```text
app.py                         CLI / Gradio 入口
ui/training_tab.py             UI 元件與 ExperimentRequest 轉接
ui/viewer_tab.py               舊、新結果檢視

core/options.py                唯一的實驗設定模型與驗證
core/io.py                     資料集檔案探索
core/channels.py               canonical 18-channel montage
core/splits.py                 patient-independent split
core/checkpoint.py             依 semantic config 建立 run identity
core/results.py                PKL 儲存與讀取

features.py                    band power 與可選特徵轉換
pipeline.py                    單一 EDF 前處理、切 epoch、平行處理

models/registry.py             lazy model dispatch
models/classical.py            SVM / XGBoost 訓練與推論
models/lstm.py                 causal LSTM
models/transformer.py          sliding-window Transformer（預設因果）
models/training_data.py        patient weight / negative downsampling
models/selection.py            inner-validation baseline threshold tuning

qubo/solvers.py                SA 與 exact chain-DP solver
qubo/validation_cache.py       patient-grouped inner score cache
qubo/tuning.py                 QUBO lambda / threshold tuning

pipeline_runner/experiment.py  nested evaluation orchestration
viz/plots.py                   summary / per-file plot
tests/                         isolation、選項與演算法測試
```

`FeatureExtraction.py` 與 `pipeline.processAllFiles` 只保留為舊 notebook 的相容介面；新程式分別使用 `features.py` 與 `process_all_files`。

## 資料流

```text
subjects
  -> EDF channel preflight
  -> per-record preprocessing / optional feature transforms
  -> outer held-out patient
       -> inner patient-grouped folds
            -> train baseline with the same enabled data/model options
            -> cache validation probabilities by EDF
       -> tune baseline threshold (optional, inner labels only)
       -> tune QUBO parameters (inner labels only)
       -> train baseline on all outer-training patients
       -> infer held-out patient's EDFs
       -> fixed baseline / selected baseline / QUBO metrics
  -> checkpoint and result PKL
```

檔案邊界會一直保留到評估階段，因此 patient weighting、negative downsampling、seizure-file macro F1 與 non-seizure FP rate 不會因過早 concatenate 而失去來源資訊。

## 可選的 baseline 改善功能

所有功能預設關閉，且會寫入 run identity 與結果 metadata。

| 功能 | CLI | 套用位置 |
|---|---|---|
| Inner baseline threshold tuning | `--tune-baseline-threshold` | 每個 outer fold 的 inner cache |
| XGBoost class weight | `--xgb-class-weight --xgb-scale-pos-weight N` | inner 與 outer training |
| XGBoost max delta step | `--xgb-max-delta-step-enabled --xgb-max-delta-step N` | inner 與 outer training |
| Patient-balanced weights | `--patient-balanced-weights` | classical model training |
| Negative downsampling | `--negative-downsample --negative-keep-fraction F` | 每個 training EDF，保留全部 positives |
| Log band power | `--log-power` | feature extraction |
| Relative band power | `--relative-power` | feature extraction |
| Robust normalization | `--robust-normalize` | 每個 EDF 的 feature matrix |
| 因果 robust normalization | `--causal-robust-normalize [--causal-robust-window-seconds N] [--causal-robust-warmup-seconds N]` | 每個 EDF，只用較早的 epoch |
| Causal context | `--temporal-context-seconds N` | 每個 EDF 的 trailing mean |

若同時啟用 relative 與 log，順序是 absolute band power → relative power → log10。Robust normalization 使用整份單一記錄的無標籤統計量，適合目前的 offline evaluation；即時部署請改用 `--causal-robust-normalize`：第 t 個 epoch 以同一 EDF 中 t 之前的 epoch（N=0 為全部，否則最近 N 秒）計算 median/IQR，不使用當下或未來的 epoch，也不跨 EDF；前 warm-up 秒（預設 60）歷史不足，特徵設為 0。兩者互斥。Causal context 不使用未來 epoch，也不跨 EDF。

## 設定與 checkpoint

`ExperimentRequest` 分成兩類設定：

- Semantic：subjects、split、模型、QUBO grid、全部功能開關。任何變更都產生新的 run ID。
- Runtime：是否保存、續跑、force restart、cache 重用與 job 數。這些不改變實驗語意，因此不改 run ID。

`RUN_SCHEMA_VERSION` 在 preprocessing、evaluation 或 result schema 語意變更時遞增，避免錯誤續跑舊 checkpoint。

## 結果內容

每個 EDF 至少保存：

- `baseline_fixed_f1`：固定 threshold 0.5。
- `baseline_f1`：選定 threshold；未啟用 tuning 時等於固定版本。
- `baseline_average_precision`：不依 decision threshold 的 PR-AUC/AP。
- `patient_average_precision`：串接同一 held-out patient 全部 EDF 後計算，並附帶患者層級 macro F1 / FP rate。
- baseline / QUBO precision、recall、F1 與 non-seizure FP rate。
- baseline threshold、QUBO lambda / threshold、held-out subject 與訓練 patient 數。

Viewer 對沒有新欄位的舊 PKL 仍維持相容。

## 執行

```bash
# UI
python app.py serve

# 列出全部 CLI 選項
python app.py train --help

# 對照組：所有新功能維持關閉
python app.py train --subjects chb01 chb02 chb03 --baseline xgboost

# Phase 0：只開 baseline threshold tuning
python app.py train --subjects chb01 chb02 chb03 --baseline xgboost \
  --tune-baseline-threshold
```

## Sequence baseline

`lstm` 與 `transformer` 都把每個 EDF 當作一條逐秒特徵序列，共用 `models.registry.sequence_model()`
的訓練 / 推論介面，inner cache 與 outer 評估流程相同。Transformer 每一秒只 attend 最近
`--transformer-window-seconds` 秒（`--transformer-bidirectional` 時改為前後），以短的 depthwise
卷積取代絕對位置編碼；訓練時把 EDF 切成 `--transformer-segment-seconds` 的段，推論時分塊並帶足
前後文，結果與整段一次計算相同。Transformer 設定只在 `--baseline transformer` 時寫入 run identity。

## 驗證

```bash
python -m unittest discover -s tests -v
```

測試涵蓋 sequence baseline 的因果性與分塊推論、channel normalization、outer/inner patient isolation、checkpoint identity、GPU fallback、feature transforms、training weights/downsampling、baseline threshold selection 與 exact QUBO solver。
