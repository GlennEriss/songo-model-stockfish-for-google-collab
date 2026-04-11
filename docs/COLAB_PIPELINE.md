# Colab Pipeline

## Objectif

Definir une organisation solide pour executer le pipeline dataset/train dans Google Colab, avec coordination multi-workers via Redis + Firestore et artefacts lourds sur Drive.

## Schema cible

1. cloner le repo GitHub
2. installer `requirements.txt`
3. configurer Drive + Firestore credentials
4. generer les configs actives runtime
5. lancer `dataset-generate` en parallele
6. lancer `dataset-build` en parallele
7. monitorer la progression globale (Redis-first, Firestore snapshot)
8. reprendre automatiquement si session interrompue
9. lancer train/evaluation/benchmark sur dataset final

## Execution type par worker

- un worker prend un bloc matchup (ex: `500` games)
- il execute ses games en parallele local (`8..16` parties selon CPU)
- il ecrit son mini-dataset dans son espace worker Drive
- il met a jour l'etat global en micro-batch
- en fin de bloc, il merge vers dataset principal puis passe au bloc suivant

## Organisation recommandee

- GitHub:
  - code source
  - configs
  - notebooks
  - scripts
  - documentation
- Google Drive:
  - datasets volumineux
  - checkpoints
  - logs
  - rapports
  - etats de jobs
- Firestore:
  - etat vivant global (`global_generation_progress`)
  - registre datasets (`dataset_registry`)
  - coordination workers (`worker_leases`)
  - checkpoints runtime (`worker_checkpoints`)
  - manifests pipeline (`pipeline_manifests`)
- Redis:
  - compteurs frequents et heartbeat workers
  - cache de monitoring temps reel

## Notebook principal actuel

- `notebooks/colab_compact.ipynb`

Il couvre:

- bootstrap runtime Colab
- generation des YAML actifs
- lancement parallele `dataset-generate` + `dataset-build`
- monitoring source/build/global/workers/health
- monitoring Redis-first + consolidation periodique Firestore
- mode quota economique (`LOW_QUOTA_PROFILE`)
- reprise auto via `job_id` + checkpoints

## Point de cadrage important

Le notebook Colab de ce projet ne doit pas reutiliser les anciens notebooks ou anciens scripts de train de `songo-ai`.

Il peut seulement s'appuyer sur `songo-ai` pour:

- la logique du jeu
- `minimax`
- `mcts`
- les benchmatchs

## Contraintes a garder en tete

- temps limite des sessions Colab
- GPU non garanti selon l'offre
- necessite de reprendre facilement un run interrompu
- importance de sauvegarder regulièrement les artefacts
- separation stricte entre code mis a jour et artefacts existants
- quotas Firestore read/write en multi-Colab
- gouvernance de sync Redis -> Firestore

## Parametres quota-first recommandes

- `GLOBAL_PROGRESS_BACKEND='firestore'`
- `GLOBAL_BUDGET_ENFORCEMENT_MODE='batched'`
- `GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES=200`
- `GLOBAL_TARGET_POLL_INTERVAL_SECONDS=60`
- `SOURCE_POLL_INTERVAL_SECONDS=45` a `60`
- `DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES=200` a `500`
- `MONITOR_REFRESH_SECONDS=90` a `120`
- `PIPELINE_MANIFEST_FIRESTORE_WRITE_ENABLED=False`
- `REDIS_SYNC_FLUSH_SECONDS=60..120`

## Documents complementaires

Voir aussi:

- `docs/COLAB_OPERATIONS.md`
- `docs/SYSTEM_ARCHITECTURE.md`
- `docs/FIRESTORE_ARCHITECTURE_20M.md`
- `docs/MODEL_STRATEGY.md`
- `docs/DATASET_AND_BENCHMARK_ARCHITECTURE.md`
