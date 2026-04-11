# Colab Operations

## 1. Objectif

Definir les operations standard pour executer le projet dans Google Colab de maniere robuste.

## 2. Regle centrale

Le code vit sur GitHub.

Les artefacts vivent sur Google Drive.

Cette separation ne doit jamais etre rompue.

L'etat vivant multi-Colab est centralise dans Firestore.

## 3. Arborescence Drive recommandee

```text
MyDrive/songo-stockfish/
  code/
  data/
    raw/
    processed/
    datasets/
  jobs/
  models/
  logs/
  reports/
  exports/
```

## 4. Ce qui doit etre stocke dans Drive

- datasets
- checkpoints
- logs
- manifests de jobs
- rapports benchmark
- rapports evaluation
- model cards

## 4.1 Ce qui doit etre stocke dans Firestore

- `global_generation_progress/{global_target_id}`
- `dataset_registry/primary`
- `worker_leases/{global_target_id}`
- `worker_checkpoints/{job_id}`
- `pipeline_manifests/{worker_tag}`

Important:

- Firestore Python (`google-cloud-firestore`) doit etre utilise avec un service account (`FIRESTORE_CREDENTIALS_PATH`)
- le mode API key seul n'est pas supporte pour ce client serveur

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
4. definir les chemins Drive + `FIRESTORE_CREDENTIALS_PATH`
5. activer le backend runtime Firestore (`GLOBAL_PROGRESS_BACKEND='firestore'`)
6. lancer le pipeline (`dataset-generate` + `dataset-build`) avec `job_id` dedie worker
7. activer le mode quota economique (`LOW_QUOTA_PROFILE=True`) en multi-Colab
8. garder un seul notebook de monitoring live pour eviter les lectures redondantes
9. reprendre si la session tombe

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
  - monitoring global Firestore
  - health check workers actifs/inactifs

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

## 11.1 Profil quota recommande (multi-Colab)

Par defaut sur plusieurs workers:

- `global_budget_enforcement_mode='batched'`
- `global_progress_flush_every_n_games >= 200`
- `source_poll_interval_seconds >= 45`
- `export_partial_every_n_files >= 200`
- `monitor_refresh_seconds >= 90`
- `pipeline_manifest_firestore_write_enabled=false`

## 12. Exigence de robustesse

Une execution Colab sera consideree robuste si:

- une session interrompue peut reprendre
- les artefacts precedents ne sont pas perdus
- une mise a jour de code reste propre
- les logs permettent de comprendre l'etat du run
- les compteurs globaux convergent dans Firestore sans divergence durable entre Colabs
