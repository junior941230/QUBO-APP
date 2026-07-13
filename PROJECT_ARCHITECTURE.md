# QUBO Seizure Detection - 專案架構文檔

## 📋 項目概述

**QUBO Seizure Detection** 是一個使用量子最佳化方法（QUBO - Quadratic Unconstrained Binary Optimization）來檢測癲癇發作的機器學習應用系統。

該項目結合了：
- **傳統機器學習模型** (SVM, XGBoost)
- **深度學習模型** (LSTM)
- **量子最佳化求解器** (D-Wave Neal, QUBO)
- **互動式 Web UI** (Gradio)

---

## 🗂️ 專案目錄結構

```
QUBO-APP/
├── app.py                      # Gradio Web UI 應用入口
├── config.py                   # 全局配置參數
├── pipeline.py                 # EDF 文件預處理和特徵提取管道
├── FeatureExtraction.py        # 頻帶功率特徵提取
├── parser.py                   # 癲癇發作時間解析
├── requirements.txt            # Python 依賴包
│
├── core/                        # 核心工具模塊
│   ├── __init__.py
│   ├── checkpoint.py           # 檢查點管理（保存/載入進度）
│   ├── io.py                   # 文件 I/O 操作
│   ├── logging_utils.py        # 日誌和進度記錄
│   ├── results.py              # 結果保存和載入
│   └── ...
│
├── models/                      # 機器學習模型
│   ├── __init__.py
│   ├── classical.py            # SVM 和 XGBoost 實現
│   ├── lstm.py                 # LSTM 模型實現
│   ├── registry.py             # 模型工廠（模型選擇邏輯）
│   └── ...
│
├── qubo/                        # QUBO 最佳化模塊
│   ├── __init__.py
│   ├── solvers.py              # QUBO 求解器接口
│   ├── tuning.py               # QUBO 參數調優
│   ├── validation_cache.py     # 驗證結果快取
│   └── ...
│
├── pipeline_runner/             # 實驗執行引擎
│   ├── __init__.py
│   ├── experiment.py           # 核心實驗流程
│   └── ...
│
├── ui/                          # 用戶界面
│   ├── __init__.py
│   ├── training_tab.py         # 模型訓練標籤
│   ├── viewer_tab.py           # 結果查看標籤
│   └── ...
│
├── viz/                         # 可視化模塊
│   ├── __init__.py
│   ├── plots.py                # 繪圖和圖表生成
│   └── ...
│
├── DESTINATION/                 # 數據目錄 (EDF 文件和標籤)
│   ├── chb01/ ... chb24/       # 24 個患者的腦電圖數據
│   ├── SUBJECT-INFO            # 患者信息
│   ├── RECORDS                 # 文件清單
│   ├── RECORDS-WITH-SEIZURES   # 包含發作的文件清單
│   └── ...
│
├── checkpoints/                 # 訓練檢查點存儲
│   └── ...
│
├── results/                     # 實驗結果存儲
│   └── qubo_run_*.pkl         # 結果序列化文件
│
└── tests/                       # 單元測試
    ├── test_channels.py
    ├── test_checkpoint.py
    ├── test_experiment_isolation.py
    ├── test_splits.py
    └── test_validation_cache.py
```

---

## 🔄 數據流和核心工作流

### 1️⃣ **數據輸入層**

#### 文件結構
- **EDF 文件**: 原始腦電圖數據（每個患者多個記錄）
- **Seizure 標籤**: 每位患者的 `chbXX-summary.txt` 包含各 EDF 的發作時間段
- 資料集也包含 `.edf.seizures`，但目前實作不直接解析這些檔案

```
DESTINATION/chb01/
├── chb01_01.edf              # 腦電圖原始數據
├── chb01_02.edf
├── chb01-summary.txt         # 目前程式使用的發作時間來源
└── ...
```

#### 相關模塊
- **`parser.py`**: 解析 `chbXX-summary.txt`，提取每個 EDF 的發作時間
- **`core/io.py`**: 收集文件列表和發作時間字典

---

### 2️⃣ **數據預處理層**

#### 主要函數: `pipeline.py`

```python
def processAllFiles(fileList, seizureTimesDict, nJobs=-1)
    └─ 並行處理多個 EDF 文件
    └─ 返回特徵和標籤字典

def preprocess_one_file(edf_path, seizure_times)
    ├─ 1. 讀取 EDF 文件 (使用 MNE)
    ├─ 2. 正規化成固定 18-channel bipolar montage
    ├─ 3. 數據濾波 (帶通濾波)
    ├─ 4. 分割成 epochs (時間窗口)
    ├─ 5. 特徵提取
    └─ 返回 (特徵, 標籤)
```

通道正規化支援標準 bipolar、`T8-P8-0` 別名，以及共同參考
(`*-CS2`/單電極) 記錄；共同參考記錄會以電極相減重建 bipolar 通道。

#### 特徵提取: `FeatureExtraction.py`

```python
def extract_band_power(epoch, sfreq=256)
    # 計算每個通道在 5 個頻帶的功率
    ├─ Delta   (0.5-4 Hz)
    ├─ Theta   (4-8 Hz)
    ├─ Alpha   (8-12 Hz)
    ├─ Beta    (12-30 Hz)
    └─ Gamma   (30-40 Hz)
    
    # 使用 Welch 功率譜密度估計 (PSD)
    # 輸出: 固定 90 維向量 (18 通道 × 5 個頻帶)
```

**關鍵參數** (`config.py`)
```python
DURATION = 1.0  # epoch 持續時間 (秒)
sfreq = 256     # 採樣率 (Hz)
```

---

### 3️⃣ **模型訓練層**

#### 模型選擇工廠: `models/registry.py`

```python
def predict_scores(baseline, x_train, y_train, x_test, **kwargs)
    ├─ "svm"     → SVM 分類器 (classical.py)
    ├─ "xgboost" → XGBoost (classical.py)
    └─ "lstm"    → LSTM 時間序列模型 (lstm.py)
```

#### 支持的模型

| 模型 | 文件 | 用途 |
|------|------|------|
| **SVM** | `models/classical.py` | 快速基線模型 |
| **XGBoost** | `models/classical.py` | 梯度提升基線 |
| **LSTM** | `models/lstm.py` | 時間序列深度學習模型 |

#### 🔄 **數據前處理差異**

| 特性 | SVM/XGBoost | LSTM |
|------|-----------|------|
| 輸入結構 | 扁平 2D (合併所有文件) | 3D 時間序列 (保留序列) |
| 數據組織 | `np.concatenate([features[f] for f in train_files])` | `[features[f] for f in train_files]` |
| 標準化 | `StandardScaler().fit_transform()` | 全局統計: `mean=all.mean(axis=0)` 後廣播 |
| 序列長度 | 固定 | 動態填充 (variable-length) |
| 類別平衡 | `class_weight="balanced"` | 手動 `pos_weight=clip(neg/pos, 1.0, 20.0)` |
| 輸出 | 單一向量 | Per-epoch 向量 |
| 實現 | `models/classical.py` | `models/lstm.py` |

**LSTM 關鍵細節:**
```python
# 不合併，保留序列
seqs = [np.asarray(features[f], dtype=np.float32) for f in train_files]

# 全局標準化
all_feat = np.concatenate(seqs, axis=0)
mean, std = all_feat.mean(axis=0), all_feat.std(axis=0)
seqs = [(s - mean) / std for s in seqs]

# 類別平衡
all_y = np.concatenate(labs)
neg = len(all_y) - pos
pos_weight = np.clip(neg / max(pos, 1.0), 1.0, 20.0)

# 動態填充 (collate_pad) + per-epoch 預測
```

目前 LSTM 是雙向模型，因此每個 epoch 的輸出會使用同一 EDF 中的未來時間點；
這適合離線序列分類，不能直接視為即時、因果式 seizure detection。不同長度序列
目前以零值補齊，但尚未使用 `pack_padded_sequence`，短序列的訓練表示可能受到
padding 影響。

---

### 4️⃣ **QUBO 最佳化層**

#### QUBO 求解器: `qubo/solvers.py`

```python
def get_qubo_solver(name)
    ├─ "solve_qubo_seizure"      # 標準 QUBO 求解
    └─ "solve_chain_qubo_exact"  # 鏈式 QUBO 求解

def safe_solver_call(solver, scores, lmbda, threshold)
    ├─ 輸入: 模型預測分數、λ(拉格朗日乘數)、閾值
    └─ 輸出: 二進制分類結果 (0 或 1)
```

#### QUBO 參數調優: `qubo/tuning.py`

```
調優過程:
├─ 使用驗證集上的模型得分
├─ 掃描 λ 和 threshold 參數空間
├─ 最大化 seizure-file mean F1 - α × non-seizure-file mean FP rate
└─ 每個外層測試檔建立獨立的驗證分數快取
```

`solve_chain_qubo_exact` 是確定性的精確 DP；`solve_qubo_seizure` 使用模擬退火，
目前沒有傳入固定 seed，因此相同輸入不保證得到完全相同的輸出。

**可調參數** (`config.py`)
```python
DEFAULT_LAMBDA_LIST = [0.5, 1.0, 1.5, 2.0, 3.0]
DEFAULT_THRESHOLD_LIST = [0.3, 0.4, 0.45, 0.5, 0.6]
```

---

### 5️⃣ **實驗執行引擎**

#### 核心函數: `pipeline_runner/experiment.py`

```python
def run_experiment(
    selected_subjects,      # 選擇的患者
    baseline,               # 模型類型 (svm/xgboost/lstm)
    solver_name,            # QUBO 求解器名稱
    tune_mode,              # 調優模式 (grid/random)
    tune_n_splits,          # 交叉驗證分割數
    ...
)
    ├─ 1. 收集文件和標籤 (core/io.py)
    ├─ 2. 預處理 EDF 文件 (pipeline.py)
    ├─ 3. 以 EDF 文件為單位執行 leave-one-file-out 分割
    ├─ 4. 模型訓練 (models/registry.py)
    ├─ 5. QUBO 參數調優 (qubo/tuning.py)
    ├─ 6. 應用 QUBO 求解器 (qubo/solvers.py)
    ├─ 7. 評估指標 (F1, Precision, Recall)
    ├─ 8. 繪製結果圖表 (viz/plots.py)
    └─ 9. 保存結果 (core/results.py)
```

---

### 6️⃣ **用戶界面層**

#### 主應用: `app.py`

```python
def build_ui()
    ├─ 使用 Gradio 框架
    └─ 包含兩個主要標籤:
        ├─ "🧪 Training Tab" (training_tab.py)
        │   ├─ 輸入參數選擇
        │   ├─ 模型訓練執行
        │   ├─ 實時進度顯示
        │   └─ 結果表格和圖表
        │
        └─ "📂 Viewer Tab" (viewer_tab.py)
            ├─ 載入先前的結果
            ├─ 查看詳細指標
            └─ 下載結果文件
```

#### 訓練標籤: `ui/training_tab.py`

**輸入參數:**
- 患者選擇（多選）
- 模型選擇（SVM/XGBoost/LSTM）
- QUBO 求解器選擇
- 調優配置（λ 和 threshold 值）
- 系統參數（並行任務數、批量大小等）
- LSTM 超參數（隱藏層大小、層數、epochs 等）

**輸出:**
- 每個 EDF 的結果表格（F1、Precision、Recall、FP rate、最佳 QUBO 參數）
- 總結圖表（seizure-file F1、non-seizure-file FP rate、F1 improvement）
- 詳細對比圖表

#### 查看器標籤: `ui/viewer_tab.py`

- 列表加載以前的 `.pkl` 結果文件
- 顯示實驗元數據
- 渲染結果圖表

---

## 🔧 核心配置

### `config.py` 配置

```python
# 目錄配置
DESTINATION_DIR = Path("DESTINATION")      # 數據目錄
RESULTS_DIR = Path("results")              # 結果輸出目錄
CHECKPOINT_DIR = Path("checkpoints")       # 檢查點目錄

# QUBO 參數預設值
DEFAULT_LAMBDA_LIST = [0.5, 1.0, 1.5, 2.0, 3.0]
DEFAULT_THRESHOLD_LIST = [0.3, 0.4, 0.45, 0.5, 0.6]

# 調優參數
TUNE_ALPHA = 0.2                           # non-seizure FP rate 懲罰權重
BASELINE_THRESHOLD = 0.5                   # 基線分類閾值

# 種子參數
RANDOM_SEED = 42                           # SVM/XGBoost 與資料切分種子
RUN_SCHEMA_VERSION = 2                     # checkpoint/result 相容性版本
```

`RANDOM_SEED` 目前尚未套用到 PyTorch LSTM 與 Neal 模擬退火，因此它不能保證
整條實驗流程完全可重現。

---

## 📦 依賴包

### `requirements.txt`

| 包名 | 版本 | 用途 |
|------|------|------|
| `dwave-neal` | 0.6.0 | QUBO 模擬退火求解器 |
| `gradio` | 6.13.0 | Web UI 框架 |
| `joblib` | 1.5.3 | EDF 並行前處理 |
| `matplotlib` | 3.10.9 | 數據可視化 |
| `mne` | 1.12.1 | 腦電圖數據處理 |
| `numpy` | 2.4.3 | 數值計算 |
| `pandas` | 2.3.3 | 結果表格 |
| `scikit-learn` | 1.8.0 | CPU SVM fallback、切分與指標計算 |
| `scipy` | 1.16.3 | 頻帶功率特徵 |
| `torch` | 2.10.0 | LSTM baseline |
| `xgboost` | 3.2.0 | 梯度提升模型 |

GPU SVM 使用 RAPIDS/cuML，請透過 `environment-rapids.yml` 建立 conda 環境；`requirements.txt` 只列 pip 可安裝的核心依賴。若沒有安裝 cuML，SVM 會自動降級使用 scikit-learn。

---

## 🎯 工作流示例

### 典型實驗流程

```
用戶在 UI 中輸入參數
    ↓
run_experiment() 開始執行
    ↓
1. 收集患者文件列表
    ↓
2. 並行預處理 EDF 文件 (pipeline.py)
    ├─ 讀取 EDF 數據
    ├─ 分割成 epochs
    ├─ 提取頻帶功率特徵
    └─ 獲得發作/非發作標籤
    ↓
3. 外層 Leave-One-File-Out 分割
    ├─ 每次保留一個 EDF 作為測試檔
    └─ 其餘 EDF 作為該 fold 的訓練候選集
    ↓
4. 訓練選定模型 (models/registry.py)
    ├─ SVM.fit(x_train, y_train)
    ├─ 或 XGBoost.fit(x_train, y_train)
    ├─ 或 LSTM.fit(train_files, lstm_params)
    └─ 在測試集上生成預測概率
    ↓
5. QUBO 參數調優 (qubo/tuning.py)
    ├─ 只使用當前外層訓練檔建立 inner-validation 快取
    ├─ 掃描 λ 和 threshold 網格
    ├─ 找到最佳參數組合
    └─ 不讓外層測試檔進入 inner-validation model fit
    ↓
6. 應用 QUBO 求解器 (qubo/solvers.py)
    ├─ solve_qubo_seizure(scores, λ, threshold)
    └─ 生成最終二進制預測
    ↓
7. 評估結果
    ├─ 計算 F1, Precision, Recall
    └─ 與基線比較
    ↓
8. 可視化 (viz/plots.py)
    ├─ 總結圖表
    └─ 詳細對比圖表
    ↓
9. 保存結果 (core/results.py)
    ├─ 保存為 qubo_run_*.pkl
    └─ 結果 = {meta, result_df, detail_cache}
    ↓
UI 顯示結果
```

---

## 💾 檢查點和結果管理

### 檢查點系統 (`core/checkpoint.py`)

- **保存進度**: 每完成或跳過一個測試 EDF，保存 rows、detail cache 與 skipped 清單
- **恢復功能**: 允許略過已完成的測試 EDF
- **Run ID**: 由完整執行設定產生；LSTM 時包含所有 `lstm_params`
- **目前限制**: EDF 前處理與未完成 fold 的 validation cache 不會持久化，恢復時仍會重算
- **恢復驗證**: 載入 checkpoint 時會比對完整執行設定；不一致或舊 checkpoint
  缺少設定時，會捨棄該 checkpoint 並從頭開始

### 結果存儲 (`core/results.py`)

**結果文件結構:**
```python
{
    "meta": {
        "timestamp": str,
        "subjects": list,
        "baseline": str,
        "solver": str,
        ...
    },
    "result_df": DataFrame,  # 詳細結果表
    "detail_cache": dict,    # 詳細預測快取
    "saved_at": str
}
```

**文件命名**: `qubo_run_YYYYMMDD_HHMMSS.pkl`

---

## 📊 評估指標

系統使用以下指標評估模型性能：

| 指標 | 計算方式 | 用途 |
|------|---------|------|
| **F1 Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | 主要優化目標 |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | 避免假陽性 |
| **Recall** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | 避免漏檢 |
| **False-positive rate** | 非 seizure 檔案中預測為 1 的 epoch 比例 | 評估正常記錄的誤報 |

目前結果表沒有輸出 Accuracy。F1 只在包含 seizure 的測試檔上具有主要解讀價值；
對完全沒有 seizure 的檔案，摘要使用預測陽性比例評估誤報。

---

## 🚀 運行應用

### 啟動 UI 伺服器

```bash
conda activate rapids-26.04
python app.py
```

訪問：`http://localhost:7860`

### 命令行使用

```bash
# 直接執行訓練／評估
python app.py train \
    --subjects chb01 chb02 \
    --baseline svm \
    --solver solve_chain_qubo_exact \
    --tune-mode nfold \
    --tune-n-splits 5

# 查看所有 CLI 選項
python app.py train --help
```

---

## 🔍 關鍵設計模式

### 1. **工廠模式** (Factory Pattern)
- `models/registry.py`: 動態選擇模型
- `qubo/solvers.py`: 動態選擇 QUBO 求解器

### 2. **並行化** (Parallelization)
- `pipeline.py`: 使用 `joblib.Parallel` 並行預處理 EDF 文件
- 可配置 CPU 核心數；`n_jobs=-1` 會使用所有可用核心
- 每個 worker 都會 preload EDF，處理大量檔案時應限制 worker 數以避免記憶體壓力

### 3. **快取機制** (Caching)
- `qubo/validation_cache.py`: 每個外層測試 fold 分別建立驗證快取；
  快取訓練資料不包含該外層測試檔，避免資料洩漏
- `core/checkpoint.py`: 保存已完成 EDF 的評估結果，支持部分恢復

### 4. **模塊化設計** (Modularity)
- 各層功能分離（數據→預處理→模型→優化→UI）
- 易於擴展新模型或求解器

---

## ⚠️ 已知限制與結果解讀

1. **切分單位是 EDF，不是患者**：同一患者的其他 EDF 可能同時出現在訓練集。
   若要衡量對未見患者的泛化能力，需另行實作 leave-one-subject-out 或 group split。
2. **LSTM 為雙向、非因果模型**：預測會使用未來時間點，只適合離線評估。
3. **完整流程尚未完全可重現**：LSTM 與 Neal solver 尚未使用固定 seed。
4. **變長 LSTM 序列直接補零**：目前沒有 packed sequence，padding 可能影響反向 LSTM。
5. **checkpoint identity 不完整**：LSTM 超參數尚未納入 Run ID；變更設定時需 Force Restart。
6. **前處理缺少逐檔容錯**：任一 EDF 前處理失敗可能中止整批 `joblib` 工作。
7. **QUBO 調優的 solver 失敗會被略過**：目前所有 validation solver call 都失敗時，
   仍可能回傳第一組參數與 0 分，應在後續版本加入成功次數驗證。
8. **Pickle 僅適用於可信檔案**：result viewer 與 checkpoint 會呼叫 `pickle.load`，
   不應載入來源不明或經第三方修改的檔案。

---

## 🎓 擴展點

該架構支持以下擴展：

1. **新模型**: 在 `models/` 目錄添加新模型類，在 `registry.py` 註冊
2. **新求解器**: 在 `qubo/solvers.py` 添加新求解器
3. **新數據源**: 擴展 `parser.py` 支持其他格式
4. **新評估指標**: 在實驗流程中添加自定義指標
5. **可視化增強**: 在 `viz/plots.py` 添加新圖表類型

---

## 📝 文件清單

| 文件 | 行數 | 用途 |
|------|------|------|
| `app.py` | ~140 | Gradio UI 與 CLI 入口 |
| `config.py` | ~20 | 全局配置 |
| `pipeline.py` | ~200 | EDF 預處理與 QUBO solver |
| `FeatureExtraction.py` | ~30 | 特徵提取邏輯 |
| `parser.py` | ~70 | 癲癇標籤解析 |
| `core/` | ~250 | 核心工具集 |
| `models/` | ~220 | 機器學習模型 |
| `qubo/` | ~190 | QUBO 最佳化與驗證快取 |
| `pipeline_runner/` | ~380 | 實驗執行引擎 |
| `ui/` | ~230 | Web UI 組件 |
| `viz/` | ~70 | 可視化邏輯 |
| `tests/` | ~220 | 核心單元與隔離測試 |

---

## 🤝 貢獻指南

要擴展該項目：

1. **新模型**: 
   - 在 `models/new_model.py` 實現
   - 在 `models/registry.py` 註冊

2. **新功能**:
   - 在相應模塊中實現邏輯
   - 更新相關單元測試
   - 在此文檔中記錄

3. **代碼風格**:
   - 遵循 PEP 8 標準
   - 添加文檔字符串
   - 使用類型提示

---

**最後更新**: 2026-07-13

**版本**: 1.1

**作者**: Project Team
