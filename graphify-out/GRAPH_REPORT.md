# Graph Report - .  (2026-07-21)

## Corpus Check
- Corpus is ~9,240 words - fits in a single context window. You may not need a graph.

## Summary
- 169 nodes · 352 edges · 14 communities
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- LSTM Models and Validation
- EEG Channel Preprocessing
- Architecture and Dependencies
- QUBO Experiment Pipeline
- CLI and Training UI
- Checkpoint Identity
- Channel Audit Tools
- Results Visualization
- Data Splitting Integrity

## God Nodes (most connected - your core abstractions)
1. `run_experiment()` - 25 edges
2. `log_step()` - 18 edges
3. `build_channel_plan()` - 14 edges
4. `_train_lstm_on_files()` - 10 edges
5. `make_run_id()` - 9 edges
6. `load_checkpoint()` - 9 edges
7. `predict_scores()` - 9 edges
8. `scan_chb_channels()` - 7 edges
9. `validate_edf_channels()` - 7 edges
10. `leave_one_file_out_train_sets()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Core Python Dependencies` --semantically_similar_to--> `Pip Application Stack`  [INFERRED] [semantically similar]
  requirements.txt → environment-rapids.yml
- `build_training_tab()` --indirect_call--> `run_experiment()`  [INFERRED]
  ui/training_tab.py → pipeline_runner/experiment.py
- `scan_chb_channels()` --calls--> `build_channel_plan()`  [EXTRACTED]
  analysisFile.py → core/channels.py
- `build_ui()` --calls--> `build_viewer_tab()`  [EXTRACTED]
  app.py → ui/viewer_tab.py
- `train()` --calls--> `run_experiment()`  [EXTRACTED]
  app.py → pipeline_runner/experiment.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **End-to-End Seizure Detection Workflow** — project_architecture_edf_preprocessing_pipeline, project_architecture_model_registry, project_architecture_qubo_optimization, project_architecture_experiment_engine, project_architecture_checkpoint_and_result_persistence [EXTRACTED 1.00]
- **RAPIDS GPU Scientific Environment** — environment_rapids_cuda_12_9, environment_rapids_cuml_gpu_svm_backend, environment_rapids_scientific_python_stack, environment_rapids_pip_application_stack [EXTRACTED 1.00]

## Communities (14 total, 0 thin omitted)

### Community 0 - "LSTM Models and Validation"
Cohesion: 0.10
Nodes (23): Dataset, predict_svm(), predict_xgboost(), collate_pad(), predict_lstm(), _predict_lstm_sequence(), Train LSTM treating each file as one sequence., Predict per-epoch probability for one sequence. (+15 more)

### Community 1 - "EEG Channel Preprocessing"
Cohesion: 0.14
Nodes (15): build_channel_plan(), _find_electrode_source(), _first_present(), CHB-MIT channel normalization helpers.  The dataset contains both bipolar record, Return instructions for constructing the canonical bipolar montage.      Each re, Split EDF paths into channel-compatible files and validation failures.      Only, validate_edf_channels(), extract_band_power() (+7 more)

### Community 2 - "Architecture and Dependencies"
Cohesion: 0.10
Nodes (21): CUDA 12.9, cuML GPU SVM Backend, Pip Application Stack, RAPIDS 26.04 Environment, Scientific Python Stack, Band-Power Features, Canonical 18-Channel Montage, Checkpoint and Result Persistence (+13 more)

### Community 3 - "QUBO Experiment Pipeline"
Cohesion: 0.23
Nodes (14): clear_checkpoint(), log_step(), parse_float_list(), save_results_pkl(), 輸入:          all_scores: SVM 產出的機率向量 (E,)         lmbda: 平滑係數 (控制連續性的強弱), 針對鏈狀 QUBO 的精確 DP 解法，O(E) 時間複雜度     比 SA 快且保證全局最優解, run_experiment(), solve_chain_qubo_exact() (+6 more)

### Community 4 - "CLI and Training UI"
Cohesion: 0.23
Nodes (12): build_ui(), CliProgress, _env_bool(), _float_grid(), parse_args(), serve(), train(), collect_files_and_seizures() (+4 more)

### Community 5 - "Checkpoint Identity"
Cohesion: 0.25
Nodes (9): checkpoint_path(), _config_signature(), load_checkpoint(), make_run_id(), Return a stable representation of the settings that define a run., Create a deterministic run id from experiment config., Load a checkpoint only when it was written for the expected config., save_checkpoint() (+1 more)

### Community 6 - "Channel Audit Tools"
Cohesion: 0.26
Nodes (9): _discover_edf_files(), main(), Write the validation details to CSV., Scan EDF headers and validate the canonical CHB bipolar montage.      Returns on, Print raw channel variants and canonical-montage validation results., report_channel_diff(), scan_chb_channels(), write_channel_audit_csv() (+1 more)

### Community 7 - "Results Visualization"
Cohesion: 0.41
Nodes (9): format_meta(), list_result_pkls(), load_result_pkl(), build_viewer_tab(), load_and_display_pkl(), refresh_pkl_list(), show_file_detail(), build_detail_plot() (+1 more)

### Community 8 - "Data Splitting Integrity"
Cohesion: 0.43
Nodes (3): leave_one_file_out_train_sets(), Map each outer test file to a training set that excludes that file., LeaveOneFileOutTests

## Knowledge Gaps
- **4 isolated node(s):** `QUBO Seizure Detection Architecture`, `Canonical 18-Channel Montage`, `Band-Power Features`, `CUDA 12.9`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_experiment()` connect `QUBO Experiment Pipeline` to `LSTM Models and Validation`, `EEG Channel Preprocessing`, `CLI and Training UI`, `Checkpoint Identity`, `Results Visualization`, `Data Splitting Integrity`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `build_channel_plan()` connect `EEG Channel Preprocessing` to `Channel Audit Tools`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `log_step()` connect `QUBO Experiment Pipeline` to `LSTM Models and Validation`, `Checkpoint Identity`, `Results Visualization`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **What connects `QUBO Seizure Detection Architecture`, `Canonical 18-Channel Montage`, `Band-Power Features` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `LSTM Models and Validation` be split into smaller, more focused modules?**
  _Cohesion score 0.09682539682539683 - nodes in this community are weakly interconnected._
- **Should `EEG Channel Preprocessing` be split into smaller, more focused modules?**
  _Cohesion score 0.14492753623188406 - nodes in this community are weakly interconnected._
- **Should `Architecture and Dependencies` be split into smaller, more focused modules?**
  _Cohesion score 0.10476190476190476 - nodes in this community are weakly interconnected._