# Job Config Spec

## 1. Objectif

Definir le format YAML des futures configurations de jobs.

Les jobs concernes sont:

- dataset generation
- dataset build
- train
- benchmark
- evaluation

## 2. Structure generale

Chaque config YAML doit contenir:

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
  git_commit: auto

runtime:
  device: cuda
  seed: 42
  num_workers: 2

storage:
  drive_root: /content/drive/MyDrive/songo-stockfish
  repo_root: /content/songo-model-stockfish-for-google-collab

job:
  run_type: train
  job_id: auto
  resume: true
  save_every_minutes: 5
```

## 4. Config `dataset_generation`

Exemple:

```yaml
project:
  name: songo-model-stockfish-for-google-collab

runtime:
  seed: 42

storage:
  drive_root: /content/drive/MyDrive/songo-stockfish

job:
  run_type: dataset_generation
  job_id: auto
  resume: true

dataset_generation:
  matchups:
    - minimax:medium vs minimax:hard
    - minimax:medium vs mcts:medium
  games: 200
  sample_strategy: uniform_by_ply
  sample_every_n_plies: 2
  output_raw_dir: data/raw
  output_sampled_dir: data/sampled
```

## 5. Config `dataset_build`

Exemple:

```yaml
project:
  name: songo-model-stockfish-for-google-collab

storage:
  drive_root: /content/drive/MyDrive/songo-stockfish

job:
  run_type: dataset_build
  job_id: auto
  resume: true

dataset_build:
  input_labeled_dir: data/labeled
  dataset_id: dataset_v1_20260402
  output_dir: data/datasets/dataset_v1_20260402
  split:
    train: 0.8
    validation: 0.1
    test: 0.1
```

## 6. Config `train`

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

job:
  run_type: train
  job_id: auto
  resume: true
  save_every_minutes: 5

train:
  dataset_id: dataset_v1_20260402
  dataset_path: data/datasets/dataset_v1_20260402/train.npz
  validation_path: data/datasets/dataset_v1_20260402/validation.npz
  model_family: policy_value
  backbone: mlp
  batch_size: 64
  epochs: 20
  learning_rate: 0.0003
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
  target: engine_v1
  matchups:
    - minimax:medium
    - minimax:hard
    - mcts:medium
    - mcts:hard
  games_per_matchup: 50
  alternate_first_player: true
  output_dir: reports/benchmarks
```

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
  model_id: songo_policy_value_v1
  checkpoint_path: models/final/songo_policy_value_v1.pt
  dataset_id: dataset_v1_20260402
  test_dataset_path: data/datasets/dataset_v1_20260402/test.npz
  output_dir: reports/evaluations
```

## 9. Regles communes

Toutes les configs doivent:

- etre auto-suffisantes
- pouvoir etre copiees dans le dossier du job
- servir de source de verite de l'execution

## 10. Decision V1

Les scripts du projet devront lire une config YAML unique par job.
