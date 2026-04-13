# Colab Operations

## 1. Objectif

Definir les operations standard pour executer le projet dans Google Colab de maniere robuste.

## 2. Regle centrale

Le code vit sur GitHub.

Les artefacts persistants vivent sur Google Drive.

Les artefacts runtime volatils (jobs/logs live) vivent en local Colab.

Cette separation ne doit jamais etre rompue.

L'etat vivant multi-Colab utilise Redis (temps reel) + Firestore (durable).

## 3. Arborescence recommandee

### 3.1 Drive (persistant)

```text
MyDrive/songo-stockfish/
  code/
  data/
    raw/
    processed/
    datasets/
  models/
  reports/
  exports/
```

### 3.2 Runtime local Colab (volatile, recommande)

```text
/content/songo-stockfish-runtime/
  jobs/
  logs/
    pipeline/
```

## 4. Ce qui doit etre stocke dans Drive

- datasets
- checkpoints
- rapports benchmark
- rapports evaluation
- model cards

## 4.1 Ce qui doit etre stocke en runtime local Colab

- `jobs/<job_id>/state.json`
- `jobs/<job_id>/run_status.json`
- `jobs/<job_id>/*_summary.json`
- `logs/pipeline/*.log`
- `logs/pipeline/latest_dataset_pipeline_<worker_tag>.json`
- snapshots monitoring (`workers_status_snapshot_*.json`, `health_snapshot_*.json`)

## 4.2 Ce qui doit etre stocke dans Firestore

- `global_generation_progress/{global_target_id}`
- `dataset_registry/primary`
- `worker_leases/{global_target_id}`
- `worker_checkpoints/{job_id}`
- `pipeline_manifests/{worker_tag}`

Important:

- Firestore Python (`google-cloud-firestore`) doit etre utilise avec un service account (`FIRESTORE_CREDENTIALS_PATH`)
- le mode API key seul n'est pas supporte pour ce client serveur

## 4.3 Ce qui doit etre stocke dans Redis

- compteurs frequents globaux (`samples`, `games`)
- heartbeat workers (TTL)
- compteurs frequents par worker
- etat temps reel de monitoring

## 5. Ce qui doit rester dans GitHub

- `src/`
- `docs/`
- `notebooks/`
- `scripts/`
- `config/`
- `requirements/`

## 6. Workflow Colab recommande

1. monter Google Drive
2. cloner ou mettre a jour le repo GitHub
3. installer les dependances
4. definir `DRIVE_ROOT`, `RUNTIME_STATE_MODE`, `RUNTIME_LOCAL_ROOT` et `FIRESTORE_CREDENTIALS_PATH`
5. activer Redis pour le temps reel + Firestore pour le durable
6. lancer le pipeline (`dataset-generate` + `dataset-build`) avec `job_id` dedie worker
7. activer le mode quota economique (`LOW_QUOTA_PROFILE=True`) en multi-Colab
8. garder un seul notebook de monitoring live (Redis-first) pour eviter les lectures Firestore redondantes
9. reprendre si la session tombe

## 6.1 Workflow par bloc matchup (recommande)

Pour chaque worker:

1. reserver un matchup libre via lease
2. executer un bloc de `200..500` games pour ce matchup
3. ecrire les fichiers worker-local dans le runtime local (pas de melange direct)
4. publier progression globale en micro-batch (N games ou intervalle)
5. a la fin du bloc, merger le mini-dataset worker vers le dataset principal
6. marquer le bloc comme `merged`
7. reserver le matchup/bloc suivant

Regle anti-collision:

- un seul worker actif par bloc matchup
- si worker stale, la lease expire et le bloc devient reprenable
- la fusion du bloc doit etre idempotente (`block_id`) pour eviter double merge

## 7. Strategie de mise a jour du code

Quand le code change:

- on met a jour le repo clone
- on ne touche pas aux dossiers d'artefacts
- on conserve les `job_id`, checkpoints, datasets et rapports

Le repo clone ne doit pas etre l'endroit ou les artefacts critiques sont ecrits.

## 8. Reprise apres perte de session

Chaque notebook et chaque script doit pouvoir:

- retrouver le dernier `job_id`
- relire `run_status.json`
- relire `state.json`
- relire `worker_checkpoints/{job_id}` si reprise cross-Colab
- relire `global_generation_progress/{global_target_id}` pour l'etat global
- reprendre depuis le dernier checkpoint ou la derniere unite terminee

## 9. Notebook principal

Notebook principal actuel:

- `notebooks/colab_compact.ipynb`
  - bootstrap runtime
  - generation des configs actives
  - lancement parallele `dataset-generate` + `dataset-build`
  - monitoring global Redis-first + Firestore snapshot
  - health check workers actifs/inactifs
  - train/eval configures sur dataset global prioritaire (sinon plus gros shard famille)
  - benchmark modele avec recherche legere configurable (`model_search_enabled`, `model_search_top_k`)

## 10. Commandes Git a standardiser

Le projet devra documenter clairement:

- clone initial
- pull de mise a jour
- checkout de branche si besoin

L'objectif est:

- mettre a jour le code sans impacter les artefacts Drive

## 11. Sauvegardes regulieres

Les scripts devront sauvegarder:

- checkpoints frequents pour le train
- etat de progression pour benchmark et dataset
- metriques progressives
- resume final de job
- checkpoint runtime Firestore pour reprise multi-session
- buffer runtime Redis pour temps reel

## 11.1 Profil quota recommande (multi-Colab)

Par defaut sur plusieurs workers:

- `global_budget_enforcement_mode='batched'`
- `global_progress_flush_every_n_games >= 200`
- `source_poll_interval_seconds >= 45`
- `export_partial_every_n_files >= 200`
- `monitor_refresh_seconds >= 90`
- `pipeline_manifest_firestore_write_enabled=false`
- `redis_sync_flush_seconds=60..120`
- `matchup_block_games=500` (ou `200` si tests)
- `worker_checkpoint_flush_seconds=60..120`
- `dataset_registry_update_mode=micro_batch_plus_end_of_block`
- `single_consolidator_lock_enabled=true`

## 11.2 Regles d'ecriture Firestore (obligatoires)

1. `worker_checkpoints`:
- pas de write par fichier
- write par intervalle + transitions critiques

2. `dataset_registry`:
- write micro-batch + fin de bloc uniquement

3. `global_generation_progress`:
- write batch (`N games` ou intervalle), plus write fin de bloc

4. consolidation Redis -> Firestore:
- un seul consolidateur actif (lock distribue)

## 12. Exigence de robustesse

Une execution Colab sera consideree robuste si:

- une session interrompue peut reprendre
- les artefacts precedents ne sont pas perdus
- une mise a jour de code reste propre
- les logs permettent de comprendre l'etat du run
- les compteurs globaux convergent (Redis temps reel, Firestore durable) sans divergence durable entre Colabs
- les merges de bloc sont idempotents et sans double comptage
