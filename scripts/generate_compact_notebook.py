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
        4. Lancer dataset (génération + build)
        5. Entraîner, évaluer, benchmarker, promotion globale
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

        bootstrap_script = WORKTREE / 'scripts' / 'colab' / 'bootstrap_workspace.py'
        summary_path = Path('/tmp/songo_bootstrap_summary.json')
        subprocess.run(
            [
                PYTHON_BIN,
                str(bootstrap_script),
                '--git-repo-url', GIT_REPO_URL,
                '--git-branch', GIT_BRANCH,
                '--worktree', str(WORKTREE),
                '--drive-project-name', DRIVE_PROJECT_NAME,
                '--colab-identity', COLAB_IDENTITY,
                '--python-bin', PYTHON_BIN,
                '--summary-path', str(summary_path),
            ],
            check=True,
        )

        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        DRIVE_ROOT = Path(summary['drive_root'])
        DRIVE_IDENTITY_KEY = str(summary.get('drive_identity_key', '')).strip() or 'unknown_drive_identity'
        DRIVE_WORKSPACE_ROOT = Path(summary['drive_workspace_root'])

        # Propagation kernel notebook pour les cellules suivantes.
        os.environ['SONGO_DRIVE_ROOT'] = str(DRIVE_ROOT)
        os.environ['SONGO_ENFORCE_DRIVE_ROOT_WRITES'] = '1'
        os.environ['SONGO_DRIVE_IDENTITY_KEY'] = DRIVE_IDENTITY_KEY
        os.environ['SONGO_DRIVE_WORKSPACE_ROOT'] = str(DRIVE_WORKSPACE_ROOT)

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
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        WORKTREE = Path('/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = sys.executable or 'python3'
        DRIVE_ROOT = Path(os.environ.get('SONGO_DRIVE_ROOT', '/content/drive/MyDrive/songo-stockfish'))
        DRIVE_IDENTITY_KEY = str(os.environ.get('SONGO_DRIVE_IDENTITY_KEY', '')).strip() or 'unknown_drive_identity'

        script = WORKTREE / 'scripts' / 'colab' / 'generate_active_configs.py'
        summary_path = Path('/tmp/songo_active_configs_summary.json')
        subprocess.run(
            [
                PYTHON_BIN,
                str(script),
                '--worktree', str(WORKTREE),
                '--drive-root', str(DRIVE_ROOT),
                '--identity', DRIVE_IDENTITY_KEY,
                '--summary-path', str(summary_path),
            ],
            check=True,
        )

        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        print('Configs actives:')
        for _name, _path in summary.get('active_configs', {}).items():
            print(' -', _path)
        """
    ),
    md("## 4) Audit stockage (aucune purge)"),
    code(
        """
        from pathlib import Path
        import os

        DRIVE_ROOT = Path(os.environ.get('SONGO_DRIVE_ROOT', '/content/drive/MyDrive/songo-stockfish'))

        print('Aucun nettoyage automatique dans ce notebook (purge desactivee).')
        print('Drive root =', DRIVE_ROOT)

        if not DRIVE_ROOT.exists():
            raise RuntimeError(f'Drive root introuvable: {DRIVE_ROOT}')

        print('\\nContenu racine:')
        for item in sorted(DRIVE_ROOT.iterdir(), key=lambda p: p.name):
            typ = 'DIR ' if item.is_dir() else 'FILE'
            print(f' - [{typ}] {item.name}')

        print('\\nWorkspaces Colab detectes:')
        workspaces = [
            p for p in DRIVE_ROOT.iterdir()
            if p.is_dir() and (p.name.startswith('colab_') or p.name == 'unknown_drive_identity')
        ]
        if not workspaces:
            print(' - aucun')
        else:
            for ws in sorted(workspaces, key=lambda p: p.name):
                print(' -', ws)
        """
    ),
    md("## 5) Lancer la génération de dataset (long run, cumulatif)"),
    code(
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        WORKTREE = Path('/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = sys.executable or 'python3'
        DRIVE_IDENTITY_KEY = str(os.environ.get('SONGO_DRIVE_IDENTITY_KEY', '')).strip() or 'unknown_drive_identity'

        subprocess.run(
            [
                PYTHON_BIN,
                str(WORKTREE / 'scripts' / 'colab' / 'run_job.py'),
                'dataset-generate',
                '--worktree', str(WORKTREE),
                '--identity', DRIVE_IDENTITY_KEY,
                '--heartbeat-seconds', '30',
            ],
            check=True,
        )
        """
    ),
    md("## 6) Lancer le build dataset (long run, cumulatif)"),
    code(
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        WORKTREE = Path('/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = sys.executable or 'python3'
        DRIVE_IDENTITY_KEY = str(os.environ.get('SONGO_DRIVE_IDENTITY_KEY', '')).strip() or 'unknown_drive_identity'

        subprocess.run(
            [
                PYTHON_BIN,
                str(WORKTREE / 'scripts' / 'colab' / 'run_job.py'),
                'dataset-build',
                '--worktree', str(WORKTREE),
                '--identity', DRIVE_IDENTITY_KEY,
                '--heartbeat-seconds', '30',
            ],
            check=True,
        )
        """
    ),
    md("## 7) Train -> Eval -> Benchmark (promotion globale incluse)"),
    code(
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        WORKTREE = Path('/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = sys.executable or 'python3'
        DRIVE_IDENTITY_KEY = str(os.environ.get('SONGO_DRIVE_IDENTITY_KEY', '')).strip() or 'unknown_drive_identity'
        DRIVE_ROOT = Path(os.environ.get('SONGO_DRIVE_ROOT', '/content/drive/MyDrive/songo-stockfish'))

        subprocess.run(
            [
                PYTHON_BIN,
                str(WORKTREE / 'scripts' / 'colab' / 'run_job.py'),
                'train-eval-benchmark',
                '--worktree', str(WORKTREE),
                '--identity', DRIVE_IDENTITY_KEY,
                '--drive-root', str(DRIVE_ROOT),
                '--heartbeat-seconds', '30',
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
