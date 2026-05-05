Scripts Colab pour separer proprement le code GitHub et les artefacts Drive.

Fichiers principaux:

- `bootstrap_workspace.py`
  Bootstrap Python centralise pour Colab: detection identite, layout workspace Drive, clone/pull du repo et installation des dependances.

- `generate_active_configs.py`
  Genere les configs actives (`dataset_generation`, `dataset_build`, `train`, `evaluation`, `benchmark`) pour une identite de workspace (`colab_1`, `colab_2`, ...).

- `run_job.py`
  Runner live (heartbeat + logs stream) pour lancer `dataset-generate`, `dataset-build`, `train`, `evaluate`, `benchmark`, `train-eval`, ou la pipeline `train-eval-benchmark`.

- `notebook_step.py`
  Wrapper CLI pour cellules Colab compactes: `bootstrap`, `generate-configs`, `audit-storage`, `run-job`, `streaming-pipeline`, `merge-built-datasets`, `publish-merged-dataset-gcs`, `vertex-custom-job`, `model-tournament`.

- `run_streaming_pipeline.py`
  Orchestrateur continu: lance `dataset-generate` et `dataset-build` en parallele.
  Option `--disable-auto-train` pour laisser le declenchement train/eval/benchmark hors pipeline dataset.

- `run_merge_built_datasets.py`
  Fusionne les datasets builds de tous les workspaces Colab (`colab_*`), dedupe les `sample_ids`, ecrase la fusion precedente, puis repointe les configs train/eval actives vers le dataset fusionne.

- `publish_merged_dataset_to_gcs.py`
  Publie automatiquement le dataset fusionne vers GCS, met a jour un pointeur `latest.json`, et peut synchroniser `models/` vers GCS pour les jobs Vertex.

- `submit_vertex_custom_job.py`
  Genere des configs runtime Vertex (storage sous `/gcs/...`), publie un package Python sur GCS (`python-package-uris`), puis soumet des Custom Jobs Vertex AI pour `train-eval` et `benchmark` sans dependre de Docker local cote Colab.

- `run_model_tournament.py`
  Lance un tournoi round-robin entre tous les modeles du registre (`model_registry.json`) avec score 3/1/0, logs live par partie et export JSON detaille.

- `init_drive_layout.sh`
  Prepare l'arborescence Drive persistante pour `data/`, `jobs/`, `models/`, `reports/` et `logs/`.

- `bootstrap_from_github.sh`
  Monte le layout Drive, clone ou met a jour le repo dans `/content`, cree le `.venv` et installe les dependances.

- `update_repo_from_github.sh`
  Met a jour proprement le code depuis GitHub sans toucher aux artefacts persistants sur Drive.

- `snapshot_code_to_drive.sh`
  Copie une photo du code courant vers `Drive/code_snapshots/` pour audit ou reprise manuelle.

- `status_watch.sh`
  Observe un `job_id` et affiche l'evolution de `run_status.json`, `state.json` et `events.jsonl`.

Principe de base:

- le code s'execute depuis `/content/songo-model-stockfish-for-google-collab`
- les artefacts critiques locaux vivent dans `/content/drive/MyDrive/songo-stockfish`
- un `git pull` ne doit jamais supprimer datasets, checkpoints ou resumes de jobs

Workflow notebook compact actuel (`notebooks/colab_compact.ipynb`):

1. monter Drive
2. bootstrap workspace (`notebook_step.py bootstrap`)
3. generer les configs actives (`notebook_step.py generate-configs`)
4. audit stockage (`notebook_step.py audit-storage`)
5. lancer le pipeline continu dataset (`notebook_step.py streaming-pipeline`)
   - `dataset-generate` + `dataset-build` en parallele
   - auto-train desactive via `--disable-auto-train`
6. fusionner les datasets builds des colabs (`notebook_step.py merge-built-datasets`)
   - dedupe des `sample_ids`
   - ecrasement de l'ancien dataset fusionne
   - patch auto des configs train/eval actives pour utiliser le dataset fusionne
7. configurer GCP / Vertex (project, bucket, machine, accel)
8. authentifier gcloud dans Colab (obligatoire avant GCS/Vertex)
9. publier dataset fusionne + models vers GCS (`notebook_step.py publish-merged-dataset-gcs`)
10. lancer train/eval sur Vertex AI (`notebook_step.py vertex-custom-job train-eval`)
11. lancer benchmatch sur Vertex AI (`notebook_step.py vertex-custom-job benchmark`)

Logs live notebook:

- cellule 5:
  - fichier: `/content/songo_streaming_pipeline.log`
  - affichage live par lecture continue du fichier (tail)
- cellule 6:
  - fichier: `/content/songo_merge_built_datasets.log`
  - affichage live par lecture continue du fichier (tail)
- cellule 7:
  - pas de job long (validation variables GCP/Vertex)
- cellule 8:
  - authentification gcloud dans Colab (`auth.authenticate_user` + verification bucket)
- cellule 9:
  - fichier: `/content/songo_publish_dataset_gcs.log`
  - affichage live par lecture continue du fichier (tail)
- cellule 10:
  - fichier: `/content/songo_vertex_train_eval.log`
  - affichage live par lecture continue du fichier (tail)
- cellule 11:
  - fichier: `/content/songo_vertex_benchmark.log`
  - affichage live par lecture continue du fichier (tail)

Notes:

- ancien comportement (avant migration) : train/eval/benchmark/tournoi executes localement sur Colab
- nouveau comportement : le notebook conserve la generation + fusion dataset, puis delègue train/eval/benchmark a Vertex AI

Detail utile pour cellule 9:

- le mode `train-eval` execute dans Vertex via `songo_model_stockfish.ops.vertex_entrypoint`:
  - train (`python -m songo_model_stockfish.cli.main train --config ...`)
  - eval (`python -m songo_model_stockfish.cli.main evaluate --config ...`)
  - promotion appliquee selon la logique existante du registre modeles (pendant l'eval)
