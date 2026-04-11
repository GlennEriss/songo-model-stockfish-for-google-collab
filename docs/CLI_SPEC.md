# CLI Spec

## 1. Objectif

Definir les commandes CLI actuellement supportees par le projet.

## 2. Point d'entree

Commande recommandee:

```bash
python -m songo_model_stockfish.cli.main <command> ...
```

## 3. Principes communs

Les commandes de job (`dataset-generate`, `dataset-build`, `dataset-merge-final`, `train`, `benchmark`, `evaluate`) supportent:

- `--config`
- `--job-id`
- `--resume`
- `--dry-run`

Elles creent un `JobContext`, ecrivent `run_status.json` / `state.json`, puis executent le handler associe.

## 4. Commandes supportees

### 4.1 `dataset-generate`

Usage:

```bash
python -m songo_model_stockfish.cli.main dataset-generate --config config/dataset_generation.full_matrix.colab_pro.yaml
```

Overrides supportes:

- `--generation-mode` (`benchmatch`, `clone_existing`, `derive_existing`, `augment_existing`, `merge_existing`)
- `--dataset-source-id`
- `--source-dataset-id`
- `--source-dataset-ids`
- `--derivation-strategy` (`unique_positions`, `endgame_focus`, `high_branching`)
- `--target-samples`
- options d'augmentation et de dedup merge

### 4.2 `dataset-build`

Usage:

```bash
python -m songo_model_stockfish.cli.main dataset-build --config config/dataset_build.full_matrix.colab_pro.yaml
```

Overrides supportes:

- `--source-dataset-id`
- `--dataset-id-override`
- `--target-labeled-samples`

### 4.3 `dataset-merge-final`

Usage:

```bash
python -m songo_model_stockfish.cli.main dataset-merge-final --config config/dataset_merge_final.colab_pro.yaml --dataset-id dataset_merged_final_v1
```

Options:

- `--dataset-id` (requis)
- `--source-dataset-ids`
- `--include-all-built`
- `--dedupe-sample-ids`
- `--keep-duplicate-sample-ids`

### 4.4 `train`

Usage:

```bash
python -m songo_model_stockfish.cli.main train --config config/train.full_matrix.colab_pro.yaml
```

### 4.5 `benchmark`

Usage:

```bash
python -m songo_model_stockfish.cli.main benchmark --config config/benchmark.colab_pro.yaml
```

### 4.6 `evaluate`

Usage:

```bash
python -m songo_model_stockfish.cli.main evaluate --config config/evaluation.full_matrix.colab_pro.yaml
```

### 4.7 `resume`

Usage:

```bash
python -m songo_model_stockfish.cli.main resume --job-id train_colab_pro_continue_compact_001
```

Comportement:

- relit `jobs/<job_id>/config.yaml`
- detecte `run_type`
- relance le handler correspondant avec le meme `job_id`

### 4.8 `status`

Usage:

```bash
python -m songo_model_stockfish.cli.main status --job-id dataset_benchmatch_xxx
```

Comportement:

- lit `jobs/<job_id>/run_status.json`
- affiche le contenu brut

### 4.9 `dataset-list`

Usage:

```bash
python -m songo_model_stockfish.cli.main dataset-list --config config/dataset_generation.full_matrix.colab_pro.yaml --kind all
```

Options:

- `--kind` (`all`, `sources`, `built`)
- `--json`

Note:

- cette commande lit le registre dataset local (`data/dataset_registry.json`) via `build_project_paths`.
- en monitoring Firestore runtime, le notebook compact utilise des helpers dedies pour lire `dataset_registry/primary`.

## 5. Etats d'execution

Les jobs ecrivent classiquement:

- `pending`
- `running`
- `failed`
- `completed`

dans `run_status.json`, avec events/metrics JSONL associes.

## 6. Exigence de coherence

La CLI doit rester:

- stable en Colab
- resumable
- compatible multi-Colab avec backend Firestore cote jobs
