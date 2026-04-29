# Job Config Spec

## 1. Objectif

Definir le format YAML des configurations de jobs utilisees par le projet.

Les jobs concernes sont:

- dataset generation
- dataset build
- dataset merge final
- train
- benchmark
- evaluation

## 2. Structure generale

Chaque config YAML contient:

- `project`
- `runtime`
- `storage`
- `job`
- une section specifique au type de job

## 3. Schema commun minimal

Exemple:

```yaml
project:
  name: songo-model-stockfish-for-google-collab

runtime:
  device: cuda
  seed: 42
  num_workers: 2

storage:
  drive_root: /content/drive/MyDrive/songo-stockfish
  repo_root: /content/songo-model-stockfish-for-google-collab
  jobs_root: /content/songo-stockfish-runtime/jobs
  jobs_backup_root: /content/drive/MyDrive/songo-stockfish/runtime_backup/jobs
  runtime_state_backup_mode: minimal
  runtime_state_backup_events_enabled: false
  runtime_state_backup_metrics_enabled: false
  runtime_state_backup_artifact_patterns:
    - "*_summary.json"

job:
  run_type: train
  job_id: auto
  resume: true
  save_every_minutes: 5
```

Bloc Firestore recommande (reprise cross-Colab train/eval/benchmark incluse):

```yaml
firestore:
  job_firestore_backend: firestore
  job_firestore_enabled: true
  job_firestore_strict: true
  job_firestore_project_id: songo-model-ai
  worker_checkpoints_firestore_collection: worker_checkpoints
  job_firestore_credentials_path: /content/drive/MyDrive/songo-stockfish/secrets/service-account.json
  job_firestore_api_key: ""
  job_firestore_checkpoint_min_interval_seconds: 30
  job_firestore_checkpoint_state_only_on_change: true
```

## 4. Config `dataset_generation`

Exemple runtime multi-Colab quota-first:

```yaml
project:
  name: songo-model-stockfish-for-google-collab

runtime:
  seed: 42
  num_workers: 16
  multiprocessing_start_method: spawn
  max_tasks_per_child: 25

storage:
  drive_root: /content/drive/MyDrive/songo-stockfish
  repo_root: /content/songo-model-stockfish-for-google-collab

job:
  run_type: dataset_generation
  job_id: auto
  resume: true
  auto_rollover_completed_job: true

dataset_generation:
  source_mode: benchmatch
  dataset_source_id: sampled_full_matrix_colab_pro
  source_dataset_id: ""
  source_dataset_ids: []
  matchups:
    - minimax:medium vs minimax:hard
    - minimax:hard vs mcts:hard
  games: 400
  target_samples: 20000000
  sample_every_n_plies: 2
  max_moves: 400
  output_raw_dir: /content/songo-stockfish-runtime/data/raw_full_matrix_colab_pro
  output_sampled_dir: data/sampled_full_matrix_colab_pro
  completed_game_detection_mode: sampled_only
  progress_update_every_n_games: 20
  max_pending_futures: 32

  global_progress_backend: firestore
  global_progress_firestore_project_id: songo-model-ai
  global_progress_firestore_collection: global_generation_progress
  global_progress_firestore_document: bench_models_20m_global
  global_progress_firestore_credentials_path: /content/drive/MyDrive/songo-stockfish/secrets/service-account.json
  global_budget_enforcement_mode: batched
  global_progress_flush_every_n_games: 200
  global_target_poll_interval_seconds: 60

  # Si source_mode=augment_existing (Passe B):
  # source_mode: augment_existing
  # source_dataset_id: sampled_full_matrix_colab_pro_worker_x
  # derivation_params:
  #   include_original_samples: true
  #   max_depth: 2
  #   max_branching: 3
  #   max_generated_per_source_sample: 8
  #   counterfactual_teacher_engine: minimax
  #   counterfactual_teacher_level: insane
  #   counterfactual_top_k: 2
  #   counterfactual_include_exploration: true
  #   counterfactual_exploration_seed_offset: 17
```

`completed_game_detection_mode`:
- `raw_and_sampled` (defaut historique): une game est "complete" si raw + sampled existent.
- `sampled_only` (recommande en Colab volatile): une game est "complete" si sampled existe.
- `raw_only`: reserve au debug.

## 5. Config `dataset_build`

Exemple runtime multi-Colab:

```yaml
project:
  name: songo-model-stockfish-for-google-collab

runtime:
  multiprocessing_start_method: fork
  max_tasks_per_child: 200

storage:
  drive_root: /content/drive/MyDrive/songo-stockfish
  repo_root: /content/songo-model-stockfish-for-google-collab

job:
  run_type: dataset_build
  job_id: auto
  resume: true
  auto_rollover_completed_job: true

dataset_build:
  source_dataset_id: sampled_full_matrix_colab_pro
  input_sampled_dir: data/sampled_full_matrix_colab_pro
  dataset_id: dataset_v2_full_matrix_colab_pro_insane_20m
  output_dir: data/datasets/dataset_v2_full_matrix_colab_pro_insane_20m
  label_cache_dir: /content/songo-stockfish-runtime/data/label_cache/dataset_v2_full_matrix_colab_pro_insane_20m
  target_labeled_samples: 20000000
  follow_source_updates: true
  source_poll_interval_seconds: 45
  num_workers: 16
  max_pending_futures: 32
  export_partial_every_n_files: 200
  include_tactical_analysis: true
  value_target_mix_teacher_weight: 0.85
  hard_examples_enabled: true
  hard_examples_margin_threshold: 0.08
  hard_examples_outcome_focus: 0.35
  hard_examples_weight_multiplier: 2.0
  dedupe_sample_ids: true
  stop_when_global_target_reached: true
  global_target_id: bench_models_20m_global
  global_target_samples: 20000000
  global_target_stabilization_polls: 3
  split:
    train: 0.8
    validation: 0.1
    test: 0.1
  teacher:
    engine: minimax
    level: insane

  global_progress_backend: firestore
  global_progress_firestore_project_id: songo-model-ai
  global_progress_firestore_collection: global_generation_progress
  global_progress_firestore_document: bench_models_20m_global
  global_progress_firestore_credentials_path: /content/drive/MyDrive/songo-stockfish/secrets/service-account.json

  dataset_registry_backend: firestore
  dataset_registry_firestore_project_id: songo-model-ai
  dataset_registry_firestore_collection: dataset_registry
  dataset_registry_firestore_document: primary
  dataset_registry_firestore_credentials_path: /content/drive/MyDrive/songo-stockfish/secrets/service-account.json
```

## 6. Config `train`

Exemple:

```yaml
project:
  name: songo-model-stockfish-for-google-collab

runtime:
  device: cuda
  seed: 42
  num_workers: 12
  pin_memory: true
  persistent_workers: true
  mixed_precision: true

storage:
  drive_root: /content/drive/MyDrive/songo-stockfish
  repo_root: /content/songo-model-stockfish-for-google-collab

job:
  run_type: train
  job_id: auto
  resume: true
  save_every_minutes: 5

train:
  dataset_selection_mode: largest_built
  dataset_id: auto
  model_id: auto
  model_id_prefix: songo_policy_value_colab_pro
  init_from_promoted_best: true
  batch_size: 4096
  epochs: 40
  learning_rate: 0.0005
  gradient_clip_norm: 1.0
  early_stopping_patience: 6
  hard_example_oversampling_enabled: true
  hard_example_weight_exponent: 1.0
  hard_example_weight_min: 1.0
  hard_example_weight_max: 4.0
  scheduler:
    type: cosine
    min_lr: 0.00005
  checkpoint_dir: models/checkpoints
  final_dir: models/final
```

## 7. Config `benchmark`

Exemple:

```yaml
project:
  name: songo-model-stockfish-for-google-collab

runtime:
  seed: 42

storage:
  drive_root: /content/drive/MyDrive/songo-stockfish

job:
  run_type: benchmark
  job_id: auto
  resume: true

benchmark:
  target: auto_latest
  model_search_enabled: true
  model_search_profile: fort_plusplus
  model_search_depth: 3
  model_search_top_k: 6
  model_search_top_k_child: 4
  model_search_alpha_beta: true
  model_search_policy_weight: 0.35
  model_search_value_weight: 1.0
  matchups:
    - minimax:medium
    - minimax:hard
    - mcts:medium
    - mcts:hard
  games_per_matchup: 50
  alternate_first_player: true
  output_dir: reports/benchmarks
```

Champs benchmark model search:

- `model_search_enabled`: active la recherche cote `ModelAgent` (desactivee => prior policy seule)
- `model_search_profile`: profil de recherche. Valeur supportee: `fort_plusplus`
- `model_search_depth`: profondeur de recherche minimax (plies)
- `model_search_top_k`: nombre de coups candidats explores a la racine (tries par prior policy)
- `model_search_top_k_child`: largeur de branchement sur les noeuds internes
- `model_search_alpha_beta`: active le pruning alpha-beta
- `model_search_policy_weight`: poids du prior policy dans le score final a la racine
- `model_search_value_weight`: poids de la value de recherche dans le score final a la racine

## 8. Config `evaluation`

Exemple:

```yaml
project:
  name: songo-model-stockfish-for-google-collab

runtime:
  device: cuda

storage:
  drive_root: /content/drive/MyDrive/songo-stockfish

job:
  run_type: evaluation
  job_id: auto
  resume: true

evaluation:
  dataset_selection_mode: largest_built
  checkpoint_path: models/final/latest.pt
  output_dir: reports/evaluations
```

## 9. Regles communes

Toutes les configs doivent:

- etre auto-suffisantes
- pouvoir etre copiees dans le dossier du job
- servir de source de verite de l'execution

En mode Firestore:

- ne pas utiliser API key seule avec `google-cloud-firestore`
- fournir un `credentials_path` service account lisible depuis Colab

## 10. Decision V1

Les scripts du projet lisent une config YAML unique par job.
