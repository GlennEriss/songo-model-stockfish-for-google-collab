from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "colab_compact.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip("\n"),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip("\n"),
    }


cells = [
    md(
        """
        # Songo Stockfish - Multi Colab Minimal

        Flux minimal:
        1. Monter Drive
        2. Préparer / mettre à jour le projet (script centralisé)
        3. Générer configs actives par identité Colab (script centralisé)
        4. Audit stockage (aucune purge)
        5. Lancer le pipeline continu dataset (generate + build, sans auto-train)
        6. Fusionner les datasets builds des colabs
        7. Configurer GCP / Vertex (project, bucket, compute)
        8. Authentifier gcloud dans Colab
        9. Publier dataset fusionne + models vers GCS
        10. Déclencher Train + Eval sur Vertex AI (sans Docker local Colab)
        11. Déclencher Benchmatch sur Vertex AI (cellule séparée)
        """
    ),
    md("## 1) Monter Drive"),
    code(
        """
        from google.colab import drive
        drive.mount('/content/drive')
        """
    ),
    md("## 2) Setup projet + workspace par identité Google Drive"),
    code(
        """
        import json
        import os
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        GIT_REPO_URL = 'https://github.com/GlennEriss/songo-model-stockfish-for-google-collab.git'
        GIT_BRANCH = 'main'
        REPO_NAME = 'songo-model-stockfish-for-google-collab'
        DRIVE_PROJECT_NAME = 'songo-stockfish'
        # Change cette valeur sur chaque Colab: colab_1, colab_2, colab_3, ...
        # Si vide, le script tentera une detection auto Drive.
        COLAB_IDENTITY = 'colab_1'

        WORKTREE = Path(f'/content/{REPO_NAME}')
        PYTHON_BIN = sys.executable or 'python3'

        # On clone juste assez pour avoir accès aux scripts centralisés.
        if not (WORKTREE / '.git').exists():
            if WORKTREE.exists():
                shutil.rmtree(WORKTREE)
            subprocess.run(['git', 'clone', '--branch', GIT_BRANCH, GIT_REPO_URL, str(WORKTREE)], check=True)

        step_script = WORKTREE / 'scripts' / 'colab' / 'notebook_step.py'
        summary_path = Path('/tmp/songo_bootstrap_summary.json')
        subprocess.run(
            [
                PYTHON_BIN,
                '-u',
                str(step_script),
                'bootstrap',
                '--git-repo-url',
                GIT_REPO_URL,
                '--git-branch',
                GIT_BRANCH,
                '--worktree',
                str(WORKTREE),
                '--drive-project-name',
                DRIVE_PROJECT_NAME,
                '--colab-identity',
                COLAB_IDENTITY,
                '--python-bin',
                PYTHON_BIN,
                '--summary-path',
                str(summary_path),
            ],
            check=True,
        )

        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        DRIVE_ROOT = Path(summary['drive_root'])
        DRIVE_IDENTITY_KEY = str(summary.get('drive_identity_key', '')).strip() or 'unknown_drive_identity'
        DRIVE_WORKSPACE_ROOT = Path(summary['drive_workspace_root'])

        # Propagation kernel notebook pour les cellules suivantes.
        env_updates = {
            'SONGO_DRIVE_ROOT': str(DRIVE_ROOT),
            'SONGO_ENFORCE_DRIVE_ROOT_WRITES': '1',
            'SONGO_DRIVE_IDENTITY_KEY': DRIVE_IDENTITY_KEY,
            'SONGO_DRIVE_WORKSPACE_ROOT': str(DRIVE_WORKSPACE_ROOT),
            'SONGO_WORKTREE': str(WORKTREE),
            'SONGO_PYTHON_BIN': PYTHON_BIN,
        }
        for _k, _v in env_updates.items():
            os.environ[_k] = _v

        print('DRIVE_ROOT           =', DRIVE_ROOT)
        print('COLAB_IDENTITY       =', COLAB_IDENTITY or '<auto>')
        print('DRIVE_IDENTITY_KEY   =', DRIVE_IDENTITY_KEY)
        print('DRIVE_WORKSPACE_ROOT =', DRIVE_WORKSPACE_ROOT)
        print('WORKTREE             =', WORKTREE)
        """
    ),
    md("## 3) Générer configs actives minimalistes"),
    code(
        """
        import os
        import subprocess
        import sys

        WORKTREE = os.environ.get('SONGO_WORKTREE', '/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('SONGO_PYTHON_BIN', (sys.executable or 'python3'))
        subprocess.run(
            [
                PYTHON_BIN,
                '-u',
                f'{WORKTREE}/scripts/colab/notebook_step.py',
                'generate-configs',
                '--worktree',
                WORKTREE,
            ],
            check=True,
        )
        """
    ),
    md("## 4) Audit stockage (aucune purge)"),
    code(
        """
        import os
        import subprocess
        import sys

        WORKTREE = os.environ.get('SONGO_WORKTREE', '/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('SONGO_PYTHON_BIN', (sys.executable or 'python3'))
        subprocess.run(
            [
                PYTHON_BIN,
                '-u',
                f'{WORKTREE}/scripts/colab/notebook_step.py',
                'audit-storage',
            ],
            check=True,
        )
        """
    ),
    md("## 5) Pipeline continu dataset (sans auto-train)"),
    code(
        """
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        WORKTREE = os.environ.get('SONGO_WORKTREE', '/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('SONGO_PYTHON_BIN', (sys.executable or 'python3'))
        LOG_PATH = Path('/content/songo_streaming_pipeline.log')
        cmd = [
            PYTHON_BIN,
            '-u',
            f'{WORKTREE}/scripts/colab/notebook_step.py',
            'streaming-pipeline',
            '--worktree',
            WORKTREE,
            '--disable-auto-train',
            '--heartbeat-seconds',
            '30',
            '--poll-seconds',
            '20',
        ]

        print('Streaming pipeline log file =', LOG_PATH)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing_size = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
        with LOG_PATH.open('a', encoding='utf-8', buffering=1) as log_file:
            log_file.write('\\n=== START streaming-pipeline ===\\n')
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            )

        cursor = existing_size
        while True:
            if LOG_PATH.exists():
                with LOG_PATH.open('r', encoding='utf-8', errors='replace') as reader:
                    reader.seek(cursor)
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end='')
                    cursor = reader.tell()
            rc = proc.poll()
            if rc is not None:
                if LOG_PATH.exists():
                    with LOG_PATH.open('r', encoding='utf-8', errors='replace') as reader:
                        reader.seek(cursor)
                        tail = reader.read()
                        if tail:
                            print(tail, end='')
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
                break
            time.sleep(2.0)
        """
    ),
    md("## 6) Fusion globale des datasets builds Colab"),
    code(
        """
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        WORKTREE = os.environ.get('SONGO_WORKTREE', '/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('SONGO_PYTHON_BIN', (sys.executable or 'python3'))
        LOG_PATH = Path('/content/songo_merge_built_datasets.log')
        MERGE_SUMMARY_PATH = Path('/tmp/songo_merge_built_datasets_summary.json')
        NOTEBOOK_STEP = f'{WORKTREE}/scripts/colab/notebook_step.py'
        MERGE_SCRIPT = f'{WORKTREE}/scripts/colab/run_merge_built_datasets.py'

        help_proc = subprocess.run(
            [PYTHON_BIN, NOTEBOOK_STEP, '-h'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        help_text = help_proc.stdout or ''
        has_merge_subcommand = ('merge-built-datasets' in help_text)

        if has_merge_subcommand:
            cmd = [
                PYTHON_BIN,
                '-u',
                NOTEBOOK_STEP,
                'merge-built-datasets',
                '--worktree',
                WORKTREE,
                '--summary-path',
                str(MERGE_SUMMARY_PATH),
                '--heartbeat-seconds',
                '30',
            ]
        elif Path(MERGE_SCRIPT).exists():
            cmd = [
                PYTHON_BIN,
                '-u',
                MERGE_SCRIPT,
                '--worktree',
                WORKTREE,
                '--summary-path',
                str(MERGE_SUMMARY_PATH),
                '--heartbeat-seconds',
                '30',
            ]
        else:
            raise RuntimeError(
                'merge-built-datasets indisponible dans ce repo. '
                'Mets le repo a jour (cellule 2) puis relance.'
            )

        print('Merge built datasets log file =', LOG_PATH)
        print('Command =', cmd)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing_size = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
        with LOG_PATH.open('a', encoding='utf-8', buffering=1) as log_file:
            log_file.write('\\n=== START merge-built-datasets ===\\n')
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            )

        cursor = existing_size
        while True:
            if LOG_PATH.exists():
                with LOG_PATH.open('r', encoding='utf-8', errors='replace') as reader:
                    reader.seek(cursor)
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end='')
                    cursor = reader.tell()
            rc = proc.poll()
            if rc is not None:
                if LOG_PATH.exists():
                    with LOG_PATH.open('r', encoding='utf-8', errors='replace') as reader:
                        reader.seek(cursor)
                        tail = reader.read()
                        if tail:
                            print(tail, end='')
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
                if MERGE_SUMMARY_PATH.exists():
                    os.environ['SONGO_MERGE_SUMMARY_PATH'] = str(MERGE_SUMMARY_PATH)
                    print('MERGE_SUMMARY_PATH =', MERGE_SUMMARY_PATH)
                break
            time.sleep(2.0)
        """
    ),
    md("## 7) Config Vertex / GCS"),
    code(
        """
        import os
        # A ajuster une fois (ou relancer avec de nouvelles valeurs).
        os.environ.setdefault('SONGO_VERTEX_PROJECT_ID', 'songo-model-ai')
        os.environ.setdefault('SONGO_VERTEX_REGION', 'us-central1')
        os.environ.setdefault('SONGO_VERTEX_GCS_BUCKET', 'songo-model-ai-vertex-bucket-001')
        os.environ.setdefault('SONGO_VERTEX_GCS_PREFIX', 'songo-stockfish')

        # Compute train/eval Vertex.
        # Defaut robuste: CPU (evite les erreurs de quota GPU T4).
        os.environ.setdefault('SONGO_VERTEX_MACHINE_TYPE', 'e2-standard-16')
        os.environ.setdefault('SONGO_VERTEX_ACCELERATOR_TYPE', '')
        os.environ.setdefault('SONGO_VERTEX_ACCELERATOR_COUNT', '0')
        os.environ.setdefault('SONGO_VERTEX_EXECUTOR_IMAGE_URI', 'us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest')

        # Optionnel.
        os.environ.setdefault('SONGO_VERTEX_SERVICE_ACCOUNT', 'songo-vertex-runtime@songo-model-ai.iam.gserviceaccount.com')
        os.environ.setdefault('SONGO_VERTEX_STREAM_LOGS', '0')

        print('SONGO_VERTEX_PROJECT_ID =', os.environ.get('SONGO_VERTEX_PROJECT_ID', ''))
        print('SONGO_VERTEX_REGION     =', os.environ.get('SONGO_VERTEX_REGION', ''))
        print('SONGO_VERTEX_GCS_BUCKET =', os.environ.get('SONGO_VERTEX_GCS_BUCKET', ''))
        print('SONGO_VERTEX_GCS_PREFIX =', os.environ.get('SONGO_VERTEX_GCS_PREFIX', ''))
        print('SONGO_VERTEX_MACHINE    =', os.environ.get('SONGO_VERTEX_MACHINE_TYPE', ''))
        print('SONGO_VERTEX_ACCEL_TYPE =', os.environ.get('SONGO_VERTEX_ACCELERATOR_TYPE', ''))
        print('SONGO_VERTEX_ACCEL_CNT  =', os.environ.get('SONGO_VERTEX_ACCELERATOR_COUNT', ''))
        print('SONGO_VERTEX_IMAGE      =', os.environ.get('SONGO_VERTEX_EXECUTOR_IMAGE_URI', ''))
        print('SONGO_VERTEX_SA         =', os.environ.get('SONGO_VERTEX_SERVICE_ACCOUNT', ''))
        print('SONGO_VERTEX_STREAM_LOGS=', os.environ.get('SONGO_VERTEX_STREAM_LOGS', '0'))

        if not os.environ.get('SONGO_VERTEX_PROJECT_ID', '').strip():
            raise RuntimeError('SONGO_VERTEX_PROJECT_ID vide. Renseigne la variable dans cette cellule.')
        if not os.environ.get('SONGO_VERTEX_GCS_BUCKET', '').strip():
            raise RuntimeError('SONGO_VERTEX_GCS_BUCKET vide. Renseigne la variable dans cette cellule.')
        """
    ),
    md("## 8) Auth GCP dans Colab (obligatoire avant GCS/Vertex)"),
    code(
        """
        import os
        import subprocess
        from google.colab import auth

        project_id = os.environ.get('SONGO_VERTEX_PROJECT_ID', '').strip()
        bucket = os.environ.get('SONGO_VERTEX_GCS_BUCKET', '').strip()
        account = os.environ.get('SONGO_GCLOUD_ACCOUNT', '').strip()

        if not project_id:
            raise RuntimeError('SONGO_VERTEX_PROJECT_ID vide. Execute d abord la cellule 7.')
        if not bucket:
            raise RuntimeError('SONGO_VERTEX_GCS_BUCKET vide. Execute d abord la cellule 7.')

        auth.authenticate_user()

        subprocess.run(['gcloud', 'config', 'set', 'project', project_id], check=True)
        if account:
            subprocess.run(['gcloud', 'config', 'set', 'account', account], check=True)

        print('Verification auth gcloud:')
        subprocess.run(['gcloud', 'auth', 'list'], check=True)
        print('Verification acces bucket:')
        subprocess.run(['gcloud', 'storage', 'ls', f'gs://{bucket}'], check=True)
        print('Auth GCP Colab OK')
        """
    ),
    md("## 9) Publier le dataset fusionne vers GCS"),
    code(
        """
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        WORKTREE = os.environ.get('SONGO_WORKTREE', '/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('SONGO_PYTHON_BIN', (sys.executable or 'python3'))
        LOG_PATH = Path('/content/songo_publish_dataset_gcs.log')
        SUMMARY_PATH = Path('/tmp/songo_publish_dataset_gcs_summary.json')
        MERGE_SUMMARY_PATH = os.environ.get('SONGO_MERGE_SUMMARY_PATH', '/tmp/songo_merge_built_datasets_summary.json')
        cmd = [
            PYTHON_BIN,
            '-u',
            f'{WORKTREE}/scripts/colab/notebook_step.py',
            'publish-merged-dataset-gcs',
            '--worktree',
            WORKTREE,
            '--merge-summary-path',
            MERGE_SUMMARY_PATH,
            '--summary-path',
            str(SUMMARY_PATH),
            '--heartbeat-seconds',
            '30',
        ]

        print('Publish dataset->GCS log file =', LOG_PATH)
        print('Publish summary path =', SUMMARY_PATH)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing_size = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
        with LOG_PATH.open('a', encoding='utf-8', buffering=1) as log_file:
            log_file.write('\\n=== START publish-merged-dataset-gcs ===\\n')
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            )

        cursor = existing_size
        while True:
            if LOG_PATH.exists():
                with LOG_PATH.open('r', encoding='utf-8', errors='replace') as reader:
                    reader.seek(cursor)
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end='')
                    cursor = reader.tell()
            rc = proc.poll()
            if rc is not None:
                if LOG_PATH.exists():
                    with LOG_PATH.open('r', encoding='utf-8', errors='replace') as reader:
                        reader.seek(cursor)
                        tail = reader.read()
                        if tail:
                            print(tail, end='')
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
                if SUMMARY_PATH.exists():
                    os.environ['SONGO_PUBLISH_DATASET_GCS_SUMMARY_PATH'] = str(SUMMARY_PATH)
                    print('PUBLISH_SUMMARY_PATH =', SUMMARY_PATH)
                break
            time.sleep(2.0)
        """
    ),
    md("## 10) Train + Eval sur Vertex AI"),
    code(
        """
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        WORKTREE = os.environ.get('SONGO_WORKTREE', '/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('SONGO_PYTHON_BIN', (sys.executable or 'python3'))
        LOG_PATH = Path('/content/songo_vertex_train_eval.log')
        SUMMARY_PATH = Path('/tmp/songo_vertex_train_eval_summary.json')
        cmd = [
            PYTHON_BIN,
            '-u',
            f'{WORKTREE}/scripts/colab/notebook_step.py',
            'vertex-custom-job',
            'train-eval',
            '--worktree',
            WORKTREE,
            '--summary-path',
            str(SUMMARY_PATH),
            '--heartbeat-seconds',
            '30',
        ]
        if os.environ.get('SONGO_VERTEX_STREAM_LOGS', '0').strip().lower() in {'1', 'true', 'yes', 'on'}:
            cmd.append('--stream-logs')

        print('Vertex train+eval log file =', LOG_PATH)
        print('Vertex summary path =', SUMMARY_PATH)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing_size = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
        with LOG_PATH.open('a', encoding='utf-8', buffering=1) as log_file:
            log_file.write('\\n=== START vertex-custom-job train-eval ===\\n')
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            )

        cursor = existing_size
        while True:
            if LOG_PATH.exists():
                with LOG_PATH.open('r', encoding='utf-8', errors='replace') as reader:
                    reader.seek(cursor)
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end='')
                    cursor = reader.tell()
            rc = proc.poll()
            if rc is not None:
                if LOG_PATH.exists():
                    with LOG_PATH.open('r', encoding='utf-8', errors='replace') as reader:
                        reader.seek(cursor)
                        tail = reader.read()
                        if tail:
                            print(tail, end='')
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
                if SUMMARY_PATH.exists():
                    os.environ['SONGO_VERTEX_TRAIN_EVAL_SUMMARY_PATH'] = str(SUMMARY_PATH)
                    print('VERTEX_TRAIN_EVAL_SUMMARY_PATH =', SUMMARY_PATH)
                break
            time.sleep(2.0)
        """
    ),
    md("## 11) Benchmatch sur Vertex AI (manuel, apres train+eval)"),
    code(
        """
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        WORKTREE = os.environ.get('SONGO_WORKTREE', '/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('SONGO_PYTHON_BIN', (sys.executable or 'python3'))
        LOG_PATH = Path('/content/songo_vertex_benchmark.log')
        SUMMARY_PATH = Path('/tmp/songo_vertex_benchmark_summary.json')
        cmd = [
            PYTHON_BIN,
            '-u',
            f'{WORKTREE}/scripts/colab/notebook_step.py',
            'vertex-custom-job',
            'benchmark',
            '--worktree',
            WORKTREE,
            '--machine-type',
            'e2-standard-8',
            '--accelerator-count',
            '0',
            '--executor-image-uri',
            'us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest',
            '--summary-path',
            str(SUMMARY_PATH),
            '--heartbeat-seconds',
            '30',
        ]
        if os.environ.get('SONGO_VERTEX_STREAM_LOGS', '0').strip().lower() in {'1', 'true', 'yes', 'on'}:
            cmd.append('--stream-logs')

        print('Vertex benchmark log file =', LOG_PATH)
        print('Vertex summary path =', SUMMARY_PATH)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing_size = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
        with LOG_PATH.open('a', encoding='utf-8', buffering=1) as log_file:
            log_file.write('\\n=== START vertex-custom-job benchmark ===\\n')
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            )

        cursor = existing_size
        while True:
            if LOG_PATH.exists():
                with LOG_PATH.open('r', encoding='utf-8', errors='replace') as reader:
                    reader.seek(cursor)
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end='')
                    cursor = reader.tell()
            rc = proc.poll()
            if rc is not None:
                if LOG_PATH.exists():
                    with LOG_PATH.open('r', encoding='utf-8', errors='replace') as reader:
                        reader.seek(cursor)
                        tail = reader.read()
                        if tail:
                            print(tail, end='')
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
                if SUMMARY_PATH.exists():
                    os.environ['SONGO_VERTEX_BENCHMARK_SUMMARY_PATH'] = str(SUMMARY_PATH)
                    print('VERTEX_BENCHMARK_SUMMARY_PATH =', SUMMARY_PATH)
                break
            time.sleep(2.0)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(NOTEBOOK_PATH)
