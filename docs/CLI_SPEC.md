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

`storage-cleanup` est une commande d'operations de stockage:

- par defaut: dry-run (aucune suppression)
- `--apply`: active les suppressions effectives
- `--all`: active toutes les familles de cleanup en une passe

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
- `--derivation-strategy` (`unique_positions`, `endgame_focus`, `high_branching`, `balanced_score_gap`, `balanced_legal_moves`, `rare_seed_profiles`)
- `--target-samples`
- options d'augmentation et de dedup merge

Notes derive_existing:
- `balanced_score_gap` equilibre les positions selon l'ecart de score (`derivation_params.score_gap_boundaries`)
- `balanced_legal_moves` equilibre selon le nombre de coups legaux (`derivation_params.legal_moves_boundaries`)
- `rare_seed_profiles` sur-echantillonne les profils de plateau rares (ponderation inverse-sqrt)
- `derivation_params.balanced_dedupe_positions=true` evite de dupliquer la meme position dans les strategies balancees

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

### 4.10 `storage-cleanup`

Usage:

```bash
python -m songo_model_stockfish.cli.main storage-cleanup --config config/train.full_matrix.colab_pro.yaml --all
```

Options:

- `--apply`: applique les suppressions (sinon dry-run)
- `--all`: active toutes les purges supportees
- `--purge-drive-runtime`: migration hash-verifiee Drive `jobs/` + `logs/pipeline/` -> runtime local puis purge source
- `--purge-runtime-backup-streams`: suppression `events.jsonl` / `metrics.jsonl` dans `runtime_backup/jobs` (jobs actifs ignores)
- `--purge-drive-raw`: suppression des `raw_dir` Drive deja migres/confirmes
- `--purge-drive-label-cache`: suppression `data/label_cache/*` hors datasets conserves
- `--purge-models`: suppression des artefacts modeles hors keep set puis resync `model_registry` + `promoted/best`
- `--keep-model-id` / `--keep-model-ids`: ids modeles a conserver
- `--keep-top-models`: nombre de top models (registry) a conserver
- `--keep-dataset-id`: ids dataset a conserver pour le label cache
- `--allow-purge-without-manifest`: autorise purge runtime Drive sans manifest de migration

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
