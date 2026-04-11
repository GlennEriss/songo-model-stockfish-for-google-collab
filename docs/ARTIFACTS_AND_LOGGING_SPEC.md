# Artifacts And Logging Spec

## 1. Objectif

Definir les artefacts persistants et les conventions de logging du projet.

## 2. Artefacts majeurs

Le projet produira principalement:

- checkpoints
- datasets
- logs d'evenements
- logs de metriques
- resumes benchmark
- resumes evaluation
- model cards

En mode multi-Colab, Firestore conserve aussi des artefacts runtime de coordination:

- `global_generation_progress/{global_target_id}`
- `dataset_registry/primary`
- `worker_leases/{global_target_id}`
- `worker_checkpoints/{job_id}`
- `pipeline_manifests/{worker_tag}`

## 3. Niveaux de logs

Niveaux recommandes:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`

Regle runtime multi-Colab:

- les erreurs Firestore doivent inclure un `hint` actionnable
- les logs doivent contenir le contexte minimal de diagnostic:
  - `project_id`
  - `collection`
  - `auth_mode`
  - `credentials_path_exists`
  - `strict`

## 4. Sorties de logs

Chaque job doit produire:

- log console temps reel
- `events.jsonl`
- `metrics.jsonl`

En plus, les jobs doivent produire des evenements runtime explicites pour la sync Firestore:

- `firestore_checkpoint_sync_config_resolved`
- `firestore_worker_checkpoint_sync_failed`
- metric `firestore_checkpoint_sync_summary` en fin de job (`completed|failed|cancelled`)

## 5. Exigences de lisibilite

Les logs console doivent etre:

- lisibles
- synthetiques
- orientes progression

Les logs persistants doivent etre:

- structures
- complets
- faciles a parser

## 6. Evenements critiques a logger

### Pour tous les jobs

- start job
- load config
- load state
- resume detected
- checkpoint saved
- artifact written
- warning notable
- failure
- completion

### Pour le runtime Firestore

- auth/firestore config resolved
- sync checkpoint write success/failure
- global progress update success/failure
- dataset registry update success/failure
- worker lease assignment
- hint d'erreur standardise (quota/auth/timeout/permission)

### Pour `benchmark`

- start matchup
- game completed
- matchup completed

### Pour `dataset_generation`

- game sampled
- shard written
- labeling completed

### Pour `train`

- epoch started
- validation completed
- best checkpoint updated

## 7. Artefacts minimaux par type de job

### `train`

- `run_status.json`
- `state.json`
- `events.jsonl`
- `metrics.jsonl`
- checkpoints
- `training_summary.json`

### `benchmark`

- `run_status.json`
- `state.json`
- `events.jsonl`
- `metrics.jsonl`
- `benchmark_summary.json`

### `dataset_generation`

- `run_status.json`
- `state.json`
- `events.jsonl`
- `metrics.jsonl`
- shards

## 8. Resume final standard

Chaque job doit produire un resume final:

- identite du job
- config utilisee
- statut final
- artefacts crees
- metriques principales
- stats de sync Firestore checkpoint:
  - attempted
  - written
  - skipped_unchanged
  - skipped_min_interval
  - failed

## 9. Conservation

Les artefacts existants ne doivent pas etre supprimes automatiquement lors d'une mise a jour du code.

Toute politique de nettoyage devra etre explicite et manuelle.

Regle complementaire:

- Drive reste la persistance des gros artefacts.
- Firestore reste la source de verite de coordination runtime.
