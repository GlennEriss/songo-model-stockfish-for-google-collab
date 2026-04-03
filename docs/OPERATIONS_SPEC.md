# Operations Spec

## 1. Objectif

Definir les specifications d'exploitation du projet:

- schemas de logs
- schemas des fichiers de job
- schemas des resumes de modele
- conventions de reprise
- arborescence cible du repo

## 2. Conventions globales

Toutes les executions importantes du projet doivent avoir:

- un `job_id`
- un `run_type`
- un `config.yaml`
- un `run_status.json`
- un `state.json`
- un `events.jsonl`
- un `metrics.jsonl`

## 3. `job_id`

Format recommande:

```text
<run_type>_<yyyyMMdd_HHmmss>_<short_id>
```

Exemples:

- `train_20260402_153000_a1b2`
- `benchmark_20260402_181500_c8f1`
- `dataset_generation_20260402_210400_z9k4`

## 4. Schema `run_status.json`

Objectif:

- donner l'etat global courant du job

Exemple:

```json
{
  "job_id": "train_20260402_153000_a1b2",
  "run_type": "train",
  "status": "running",
  "phase": "epoch_03_validation",
  "created_at": "2026-04-02T15:30:00Z",
  "updated_at": "2026-04-02T16:04:21Z",
  "resume_supported": true,
  "last_checkpoint_path": "models/checkpoints/model_epoch_03.pt",
  "last_state_path": "jobs/train_20260402_153000_a1b2/state.json",
  "git_commit": "<commit_sha>",
  "dataset_id": "dataset_v1_20260402"
}
```

## 5. Schema `state.json`

Objectif:

- stocker le minimum necessaire a la reprise

### 5.1 Pour `train`

Exemple:

```json
{
  "job_id": "train_20260402_153000_a1b2",
  "run_type": "train",
  "epoch": 3,
  "global_step": 1840,
  "best_metric": 0.61,
  "last_completed_phase": "validation",
  "checkpoint_path": "models/checkpoints/model_epoch_03.pt"
}
```

### 5.2 Pour `benchmark`

Exemple:

```json
{
  "job_id": "benchmark_20260402_181500_c8f1",
  "run_type": "benchmark",
  "matchup_id": "engine_v1_vs_minimax_medium",
  "completed_games": 18,
  "remaining_games": 32,
  "last_completed_game_id": "game_000018"
}
```

### 5.3 Pour `dataset_generation`

Exemple:

```json
{
  "job_id": "dataset_generation_20260402_210400_z9k4",
  "run_type": "dataset_generation",
  "completed_games": 120,
  "written_shards": 4,
  "last_shard_path": "data/raw/shards/shard_0004.jsonl",
  "sample_count": 12840
}
```

## 6. Schema `events.jsonl`

Objectif:

- journal detaille chronologique

Chaque ligne doit etre un objet JSON.

Exemple:

```json
{
  "timestamp": "2026-04-02T16:01:04Z",
  "job_id": "train_20260402_153000_a1b2",
  "run_type": "train",
  "level": "INFO",
  "phase": "training",
  "message": "checkpoint saved",
  "epoch": 3,
  "step": 1800,
  "artifact_path": "models/checkpoints/model_epoch_03.pt"
}
```

## 7. Schema `metrics.jsonl`

Objectif:

- stocker les mesures quantitatives progressives

### 7.1 Pour `train`

Exemple:

```json
{
  "timestamp": "2026-04-02T16:00:00Z",
  "job_id": "train_20260402_153000_a1b2",
  "metric_type": "train_step",
  "epoch": 3,
  "step": 1720,
  "loss_total": 1.284,
  "loss_policy": 0.932,
  "loss_value": 0.352,
  "learning_rate": 0.0003
}
```

### 7.2 Pour `benchmark`

Exemple:

```json
{
  "timestamp": "2026-04-02T18:24:10Z",
  "job_id": "benchmark_20260402_181500_c8f1",
  "metric_type": "game_result",
  "matchup_id": "engine_v1_vs_minimax_medium",
  "game_id": "game_000018",
  "winner": "engine_v1",
  "moves": 84,
  "avg_move_time_ms": 61.2
}
```

## 8. Schema `model_card.json`

Objectif:

- decrire un modele produit par le projet

Exemple:

```json
{
  "model_id": "songo_policy_value_v1",
  "created_at": "2026-04-03T09:12:00Z",
  "git_commit": "<commit_sha>",
  "dataset_id": "dataset_v1_20260402",
  "training_job_id": "train_20260402_153000_a1b2",
  "architecture": {
    "family": "policy_value",
    "backbone": "mlp",
    "hidden_sizes": [256, 256, 128]
  },
  "checkpoint_path": "models/final/songo_policy_value_v1.pt",
  "best_validation_metric": 0.61,
  "benchmark_summary_path": "reports/benchmarks/songo_policy_value_v1_summary.json"
}
```

## 9. Schema `benchmark_summary.json`

Objectif:

- resumer les performances d'un benchmark

Exemple:

```json
{
  "job_id": "benchmark_20260402_181500_c8f1",
  "model_or_engine": "engine_v1",
  "matchups": [
    {
      "opponent": "minimax:medium",
      "games": 50,
      "wins": 29,
      "losses": 18,
      "draws": 3,
      "winrate": 0.58
    },
    {
      "opponent": "mcts:medium",
      "games": 50,
      "wins": 21,
      "losses": 25,
      "draws": 4,
      "winrate": 0.42
    }
  ]
}
```

## 10. Convention de reprise

Un job est resumable si:

- `run_status.json` existe
- `state.json` existe
- les artefacts partiels precedents existent encore

Regles:

- ne jamais relancer les unites deja marquees `completed`
- conserver le meme `job_id`
- ajouter de nouveaux evenements dans les memes fichiers JSONL

## 11. Etats standards d'un job

Etats recommandes:

- `pending`
- `running`
- `interrupted`
- `failed`
- `completed`

## 12. Arborescence cible du repo

```text
songo-model-stockfish-for-google-collab/
  docs/
  notebooks/
  config/
  requirements/
    requirements.in
    requirements.txt
    dev.in
    dev.txt
  scripts/
  src/
    songo_model_stockfish/
      adapters/
      engine/
      evaluation/
      data/
      training/
      benchmark/
      ops/
      cli/
```

## 13. Arborescence cible Drive

```text
MyDrive/songo-stockfish/
  code/
  data/
    raw/
    sampled/
    labeled/
    datasets/
  jobs/
  models/
    checkpoints/
    final/
    cards/
  logs/
  reports/
    benchmarks/
    evaluations/
```

## 14. Exigence de qualite

Cette specification est respectee si:

- tous les scripts ecrivent les memes types de fichiers
- tous les jobs sont tracables
- les reprises sont coherentes
- les resumes de modele et benchmark sont exploitables
