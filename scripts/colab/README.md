Scripts Colab pour separer proprement le code GitHub et les artefacts Drive.

Fichiers principaux:

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
