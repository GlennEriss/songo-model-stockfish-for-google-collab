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
        6. Déclencher train/eval/benchmark manuellement
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

        WORKTREE = os.environ.get('SONGO_WORKTREE', '/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('SONGO_PYTHON_BIN', (sys.executable or 'python3'))
        subprocess.run(
            [
                PYTHON_BIN,
                f'{WORKTREE}/scripts/colab/notebook_step.py',
                'streaming-pipeline',
                '--worktree',
                WORKTREE,
                '--disable-auto-train',
                '--heartbeat-seconds',
                '30',
                '--poll-seconds',
                '20',
            ],
            check=True,
        )
        """
    ),
    md("## 6) Train -> Eval -> Benchmark (manuel)"),
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
                f'{WORKTREE}/scripts/colab/notebook_step.py',
                'run-job',
                'train-eval-benchmark',
                '--worktree',
                WORKTREE,
                '--heartbeat-seconds',
                '30',
            ],
            check=True,
        )
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
