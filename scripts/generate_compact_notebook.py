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
        7. Déclencher train/eval/benchmark manuellement
        8. Lancer un tournoi live entre tous les modèles (optionnel)
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
        cmd = [
            PYTHON_BIN,
            '-u',
            f'{WORKTREE}/scripts/colab/notebook_step.py',
            'merge-built-datasets',
            '--worktree',
            WORKTREE,
            '--heartbeat-seconds',
            '30',
        ]

        print('Merge built datasets log file =', LOG_PATH)
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
                break
            time.sleep(2.0)
        """
    ),
    md("## 7) Train -> Eval -> Benchmark (manuel)"),
    code(
        """
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        WORKTREE = os.environ.get('SONGO_WORKTREE', '/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('SONGO_PYTHON_BIN', (sys.executable or 'python3'))
        LOG_PATH = Path('/content/songo_train_eval_benchmark.log')
        cmd = [
            PYTHON_BIN,
            '-u',
            f'{WORKTREE}/scripts/colab/notebook_step.py',
            'run-job',
            'train-eval-benchmark',
            '--worktree',
            WORKTREE,
            '--heartbeat-seconds',
            '30',
        ]

        print('Train/Eval/Benchmark log file =', LOG_PATH)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing_size = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
        with LOG_PATH.open('a', encoding='utf-8', buffering=1) as log_file:
            log_file.write('\\n=== START train-eval-benchmark ===\\n')
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
    md("## 8) Tournoi des modèles (10 matchs par paire, live)"),
    code(
        """
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        WORKTREE = os.environ.get('SONGO_WORKTREE', '/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('SONGO_PYTHON_BIN', (sys.executable or 'python3'))
        LOG_PATH = Path('/content/songo_model_tournament.log')
        cmd = [
            PYTHON_BIN,
            '-u',
            f'{WORKTREE}/scripts/colab/notebook_step.py',
            'model-tournament',
            '--worktree',
            WORKTREE,
            '--games-per-pair',
            '10',
            '--max-moves',
            '400',
        ]

        print('Model tournament log file =', LOG_PATH)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing_size = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
        with LOG_PATH.open('a', encoding='utf-8', buffering=1) as log_file:
            log_file.write('\\n=== START model-tournament ===\\n')
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
