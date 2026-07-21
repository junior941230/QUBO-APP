---
type: "query"
date: "2026-07-21T06:49:10.429713+00:00"
question: "將專案改成 patient-independent evaluation"
contributor: "graphify"
outcome: "useful"
source_nodes: ["run_experiment()", "splits.py", "build_validation_score_cache_kfold()", "training_tab.py"]
---

# Q: 將專案改成 patient-independent evaluation

## Answer

Expanded from original query via graph vocab: [subjects, split, train, test, validation, dataset, record, edf, fold, kfold, seizure, files]. The graph showed file-level outer splitting in run_experiment and file-level inner validation caches. Implemented nested leave-one-subject-out evaluation, subject-grouped LOSO/group N-fold tuning, explicit file-to-subject mapping, leakage assertions, subject-aware output, CLI/UI documentation, and isolation tests.

## Outcome

- Signal: useful

## Source Nodes

- run_experiment()
- splits.py
- build_validation_score_cache_kfold()
- training_tab.py