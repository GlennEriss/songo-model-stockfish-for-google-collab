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

### 3.3 Backup hybride des etats essentiels (Drive)

```text
MyDrive/songo-stockfish/
  runtime_backup/
    jobs/
      <job_id>/
        config.yaml
        run_status.json
        state.json
        events.jsonl
        metrics.jsonl
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
- ces fichiers peuvent etre perdus si la VM est recreee

## 4.1.1 Sauvegarde hybride recommandee (anti-perte VM)

- activer `runtime_state_backup_enabled=true`
- pointer `jobs_backup_root` vers `MyDrive/songo-stockfish/runtime_backup/jobs`
- garder un intervalle de sync raisonnable (ex: `runtime_state_backup_min_interval_seconds=30`)
- definir aussi un filet de securite `runtime_state_backup_force_interval_seconds` (ex: 180)
- a la reprise, restaurer le runtime local depuis ce backup avant relance

### 4.1.2 Migration Drive -> runtime local (cellule 3bis)

- utiliser un lock de migration partage pour eviter les courses
- verifier la copie avec hash SHA256 (pas seulement la taille)
- ne jamais purger un job detecte actif:
  - `run_status` actif
  - `updated_at` recent
  - PID encore vivant via manifest
- purge en deux temps: rename en quarantaine puis recheck actif avant suppression

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
6. lancer le pipeline continu dataset:
   - `notebook_step.py streaming-pipeline --disable-auto-train`
   - execute `dataset-generate` + `dataset-build` en parallele
7. monitorer les logs live en notebook:
   - `/content/songo_streaming_pipeline.log`
8. fusionner les datasets builds `colab_*`:
   - `notebook_step.py merge-built-datasets`
   - dedupe `sample_ids` + ecrasement de la fusion precedente
9. monitorer les logs live de fusion:
   - `/content/songo_merge_built_datasets.log`
10. declencher manuellement train/eval/benchmark:
   - `notebook_step.py run-job train-eval-benchmark`
11. monitorer les logs live train/eval/benchmark:
   - `/content/songo_train_eval_benchmark.log`
12. lancer si besoin un tournoi modeles:
   - `notebook_step.py model-tournament --games-per-pair 10`
   - logs live: `/content/songo_model_tournament.log`
13. reprendre si la session tombe

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
  - cellule 5: `streaming-pipeline --disable-auto-train` (generate + build en parallele)
  - cellule 5: logs live fichier `/content/songo_streaming_pipeline.log`
  - cellule 6: `merge-built-datasets` (fusion globale des builds colabs, dedupe, overwrite)
  - cellule 6: logs live fichier `/content/songo_merge_built_datasets.log`
  - cellule 7: `run-job train-eval-benchmark` (declenchement manuel)
  - cellule 7: logs live fichier `/content/songo_train_eval_benchmark.log`
  - cellule 7: preflight train visible (dataset, taille, split, epochs, batch size)
  - cellule 8 (optionnelle): `model-tournament --games-per-pair 10`
  - cellule 8: logs live fichier `/content/songo_model_tournament.log`

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
