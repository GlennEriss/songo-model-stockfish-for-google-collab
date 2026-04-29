Scripts Colab pour separer proprement le code GitHub et les artefacts Drive.

Fichiers principaux:

- `bootstrap_workspace.py`
  Bootstrap Python centralise pour Colab: detection identite, layout workspace Drive, clone/pull du repo et installation des dependances.

- `generate_active_configs.py`
  Genere les configs actives (`dataset_generation`, `dataset_build`, `train`, `evaluation`, `benchmark`) pour une identite de workspace (`colab_1`, `colab_2`, ...).

- `run_job.py`
  Runner live (heartbeat + logs stream) pour lancer `dataset-generate`, `dataset-build`, `train`, `evaluate`, `benchmark`, ou la pipeline `train-eval-benchmark`.

- `notebook_step.py`
  Wrapper CLI pour cellules Colab compactes: `bootstrap`, `generate-configs`, `audit-storage`, `run-job`, `streaming-pipeline`, `merge-built-datasets`, `model-tournament`.

- `run_streaming_pipeline.py`
  Orchestrateur continu: lance `dataset-generate` et `dataset-build` en parallele.
  Option `--disable-auto-train` pour laisser le declenchement `train-eval-benchmark` en manuel.

- `run_merge_built_datasets.py`
  Fusionne les datasets builds de tous les workspaces Colab (`colab_*`), dedupe les `sample_ids`, ecrase la fusion precedente, puis repointe les configs train/eval actives vers le dataset fusionne.

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
- les artefacts critiques vivent dans `/content/drive/MyDrive/songo-stockfish`
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
7. lancer train/eval/benchmark manuellement (`notebook_step.py run-job train-eval-benchmark`)
8. lancer le tournoi modeles (`notebook_step.py model-tournament`)

Logs live notebook:

- cellule 5:
  - fichier: `${SONGO_DRIVE_WORKSPACE_ROOT}/logs/notebook/songo_streaming_pipeline.log`
  - affichage live par lecture continue du fichier (tail)
- cellule 6:
  - fichier: `${SONGO_DRIVE_WORKSPACE_ROOT}/logs/notebook/songo_merge_built_datasets.log`
  - affichage live par lecture continue du fichier (tail)
- cellule 7:
  - fichier: `${SONGO_DRIVE_WORKSPACE_ROOT}/logs/notebook/songo_train_eval_benchmark.log`
  - affichage live par lecture continue du fichier (tail)
- cellule 8:
  - fichier: `${SONGO_DRIVE_WORKSPACE_ROOT}/logs/notebook/songo_model_tournament.log`
  - affichage live par lecture continue du fichier (tail)

Note:

- ancien comportement (avant migration) : logs notebook sous `/content/songo_*.log` (ephemeres runtime Colab)

Detail utile pour cellule 7:

- le mode `train-eval-benchmark` affiche un preflight train avant lancement:
  - dataset resolu
  - taille (`labeled_samples`, `target_labeled_samples`, split train/val/test)
  - mode de selection dataset
  - epochs et batch size planifies
