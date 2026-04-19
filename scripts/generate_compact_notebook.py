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
        2. Préparer / mettre à jour le projet
        3. Générer configs actives par identité Colab
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
        import os
        import re
        import sys
        import shutil
        import subprocess
        from pathlib import Path

        GIT_REPO_URL = 'https://github.com/GlennEriss/songo-model-stockfish-for-google-collab.git'
        GIT_BRANCH = 'main'
        REPO_NAME = 'songo-model-stockfish-for-google-collab'
        DRIVE_PROJECT_NAME = 'songo-stockfish'
        # Change cette valeur sur chaque Colab: colab_1, colab_2, colab_3, ...
        # Si vide, le notebook tentera une detection auto Drive.
        COLAB_IDENTITY = 'colab_1'

        DRIVE_ROOT = Path('/content/drive/MyDrive') / DRIVE_PROJECT_NAME
        MYDRIVE_ROOT = Path('/content/drive/MyDrive')
        WORKTREE = Path(f'/content/{REPO_NAME}')
        PYTHON_BIN = sys.executable or 'python3'

        if not MYDRIVE_ROOT.exists():
            raise RuntimeError("MyDrive non visible. Exécute d'abord drive.mount('/content/drive').")
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

        def _sanitize_identity(value: str) -> str:
            text = str(value or '').strip().lower()
            text = text.replace('@', '_at_')
            text = re.sub(r'[^a-z0-9._-]+', '_', text)
            text = re.sub(r'_{2,}', '_', text).strip('._-')
            return text[:120] if text else ''

        def _detect_drive_identity_key() -> str:
            local_override = _sanitize_identity(COLAB_IDENTITY)
            if local_override:
                return local_override
            override = _sanitize_identity(os.environ.get('SONGO_DRIVE_IDENTITY_KEY', ''))
            if override:
                return override
            try:
                import google.auth
                from googleapiclient.discovery import build

                creds, _ = google.auth.default(
                    scopes=['https://www.googleapis.com/auth/drive.metadata.readonly']
                )
                svc = build('drive', 'v3', credentials=creds, cache_discovery=False)
                about = svc.about().get(fields='user(emailAddress,permissionId)').execute()
                user = dict(about.get('user', {}) or {})
                email = _sanitize_identity(user.get('emailAddress', ''))
                if email:
                    return email
                permission_id = _sanitize_identity(user.get('permissionId', ''))
                if permission_id:
                    return f'perm_{permission_id}'
            except Exception:
                pass
            return 'unknown_drive_identity'

        DRIVE_IDENTITY_KEY = _detect_drive_identity_key()
        DRIVE_WORKSPACE_ROOT = DRIVE_ROOT / DRIVE_IDENTITY_KEY
        DRIVE_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

        os.environ['SONGO_DRIVE_ROOT'] = str(DRIVE_ROOT)
        os.environ['SONGO_ENFORCE_DRIVE_ROOT_WRITES'] = '1'
        os.environ['SONGO_DRIVE_IDENTITY_KEY'] = DRIVE_IDENTITY_KEY
        os.environ['SONGO_DRIVE_WORKSPACE_ROOT'] = str(DRIVE_WORKSPACE_ROOT)

        for rel in [
            'secrets',
            'models',
            f'{DRIVE_IDENTITY_KEY}/jobs',
            f'{DRIVE_IDENTITY_KEY}/logs',
            f'{DRIVE_IDENTITY_KEY}/reports',
            f'{DRIVE_IDENTITY_KEY}/data',
            f'{DRIVE_IDENTITY_KEY}/data/datasets',
            f'{DRIVE_IDENTITY_KEY}/data/label_cache',
            f'{DRIVE_IDENTITY_KEY}/runtime_backup/jobs',
        ]:
            (DRIVE_ROOT / rel).mkdir(parents=True, exist_ok=True)

        if not (WORKTREE / '.git').exists():
            if WORKTREE.exists():
                shutil.rmtree(WORKTREE)
            subprocess.run(['git', 'clone', '--branch', GIT_BRANCH, GIT_REPO_URL, str(WORKTREE)], check=True)
        else:
            subprocess.run(['git', '-C', str(WORKTREE), 'fetch', 'origin', GIT_BRANCH], check=True)
            subprocess.run(['git', '-C', str(WORKTREE), 'checkout', GIT_BRANCH], check=True)
            subprocess.run(['git', '-C', str(WORKTREE), 'pull', '--ff-only', 'origin', GIT_BRANCH], check=True)

        os.chdir(WORKTREE)
        subprocess.run([PYTHON_BIN, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([PYTHON_BIN, '-m', 'pip', 'install', '-r', str(WORKTREE / 'requirements.txt')], check=True)

        print('DRIVE_ROOT           =', DRIVE_ROOT)
        print('COLAB_IDENTITY       =', COLAB_IDENTITY)
        print('DRIVE_IDENTITY_KEY   =', DRIVE_IDENTITY_KEY)
        print('DRIVE_WORKSPACE_ROOT =', DRIVE_WORKSPACE_ROOT)
        print('WORKTREE             =', WORKTREE)
        """
    ),
    md("## 3) Générer configs actives minimalistes"),
    code(
        """
        import copy
        import time
        import yaml
        from pathlib import Path

        WORKTREE = Path('/content/songo-model-stockfish-for-google-collab')
        DRIVE_ROOT = Path('/content/drive/MyDrive/songo-stockfish')
        DRIVE_IDENTITY_KEY = str(os.environ.get('SONGO_DRIVE_IDENTITY_KEY', '')).strip() or 'unknown_drive_identity'
        WORKSPACE_REL = DRIVE_IDENTITY_KEY

        TARGET_SAMPLES = int(os.environ.get('SONGO_TARGET_SAMPLES', '1000000000'))  # "infini" pratique
        TARGET_LABELED_SAMPLES = int(os.environ.get('SONGO_TARGET_LABELED_SAMPLES', str(TARGET_SAMPLES)))
        GAMES_PER_MATCHUP = int(os.environ.get('SONGO_GAMES_PER_MATCHUP', '400'))

        BASE_SOURCE_ID = f'sampled_full_matrix_{DRIVE_IDENTITY_KEY}'
        BASE_DATASET_ID = f'dataset_full_matrix_{DRIVE_IDENTITY_KEY}'
        RUN_TS = int(time.time())
        TRAIN_JOB_ID = f'train_{DRIVE_IDENTITY_KEY}_{RUN_TS}'
        EVAL_JOB_ID = f'eval_{DRIVE_IDENTITY_KEY}_{RUN_TS}'
        BENCH_JOB_ID = f'benchmark_{DRIVE_IDENTITY_KEY}_{RUN_TS}'

        generated = WORKTREE / 'config' / 'generated'
        generated.mkdir(parents=True, exist_ok=True)

        def _load_yaml(rel_path: str) -> dict:
            return yaml.safe_load((WORKTREE / rel_path).read_text(encoding='utf-8')) or {}

        def _save_yaml(path: Path, payload: dict) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')

        def _inject_storage(cfg: dict) -> dict:
            out = copy.deepcopy(cfg)
            out.setdefault('storage', {})
            out['storage']['drive_root'] = str(DRIVE_ROOT)
            out['storage']['jobs_root'] = f'{WORKSPACE_REL}/jobs'
            out['storage']['jobs_backup_root'] = f'{WORKSPACE_REL}/runtime_backup/jobs'
            out['storage']['logs_root'] = f'{WORKSPACE_REL}/logs'
            out['storage']['reports_root'] = f'{WORKSPACE_REL}/reports'
            out['storage']['data_root'] = f'{WORKSPACE_REL}/data'
            # Global models partagés pour promotion globale entre Colabs.
            out['storage']['models_root'] = 'models'
            out['storage']['runtime_state_backup_enabled'] = True
            out.setdefault('runtime', {})
            out['runtime']['device'] = out['runtime'].get('device', 'cuda')
            return out

        # dataset-generate
        gen = _inject_storage(_load_yaml('config/dataset_generation.full_matrix.colab_pro.yaml'))
        gen.setdefault('job', {})
        gen['job']['resume'] = True
        gen['job']['run_type'] = 'dataset_generate'
        gen['job']['job_id'] = f'dataset_generate_{DRIVE_IDENTITY_KEY}'
        gen.setdefault('dataset_generation', {})
        dgen = gen['dataset_generation']
        dgen['generation_mode'] = 'benchmatch'
        dgen['dataset_source_id'] = BASE_SOURCE_ID
        dgen['raw_dir'] = f'{WORKSPACE_REL}/data/raw/{BASE_SOURCE_ID}'
        dgen['sampled_dir'] = f'{WORKSPACE_REL}/data/sampled/{BASE_SOURCE_ID}'
        dgen['target_samples'] = TARGET_SAMPLES
        dgen['games_per_matchup'] = GAMES_PER_MATCHUP
        dgen['cycle_matchups_until_target'] = True
        dgen['max_matchup_cycles'] = 0
        dgen['matchups'] = [
            'minimax:medium', 'minimax:hard', 'minimax:insane',
            'mcts:medium', 'mcts:hard', 'mcts:insane',
        ]
        gen_active = generated / f'dataset_generation.{DRIVE_IDENTITY_KEY}.active.yaml'
        _save_yaml(gen_active, gen)

        # dataset-build
        build = _inject_storage(_load_yaml('config/dataset_build.full_matrix.colab_pro.yaml'))
        build.setdefault('job', {})
        build['job']['resume'] = True
        build['job']['run_type'] = 'dataset_build'
        build['job']['job_id'] = f'dataset_build_{DRIVE_IDENTITY_KEY}'
        build.setdefault('dataset_build', {})
        db = build['dataset_build']
        db['build_mode'] = 'auto'
        db['source_dataset_id'] = BASE_SOURCE_ID
        db['input_sampled_dir'] = f'{WORKSPACE_REL}/data/sampled/{BASE_SOURCE_ID}'
        db['dataset_id'] = BASE_DATASET_ID
        db['output_dir'] = f'{WORKSPACE_REL}/data/datasets/{BASE_DATASET_ID}'
        db['label_cache_dir'] = f'{WORKSPACE_REL}/data/label_cache/{BASE_DATASET_ID}'
        db['target_labeled_samples'] = TARGET_LABELED_SAMPLES
        db['follow_source_updates'] = True
        db.setdefault('teacher', {})
        db['teacher']['engine'] = 'minimax'
        db['teacher']['level'] = 'insane'
        build_active = generated / f'dataset_build.{DRIVE_IDENTITY_KEY}.active.yaml'
        _save_yaml(build_active, build)

        # train
        train = _inject_storage(_load_yaml('config/train.full_matrix.colab_pro.yaml'))
        train.setdefault('job', {})
        train['job']['run_type'] = 'train'
        train['job']['resume'] = True
        train['job']['job_id'] = TRAIN_JOB_ID
        train.setdefault('train', {})
        tr = train['train']
        tr['dataset_selection_mode'] = 'largest_built'
        tr['dataset_id'] = 'auto'
        tr['init_from_promoted_best'] = True
        tr['promoted_best_checkpoint_path'] = 'models/promoted/best/model.pt'
        tr['model_id_prefix'] = f'songo_policy_value_colab_pro_{DRIVE_IDENTITY_KEY}'
        tr['checkpoint_dir'] = f'models/checkpoints/{DRIVE_IDENTITY_KEY}'
        tr['lineage_dir'] = f'models/lineage/{DRIVE_IDENTITY_KEY}'
        train_active = generated / f'train.{DRIVE_IDENTITY_KEY}.active.yaml'
        _save_yaml(train_active, train)

        # eval
        eval_cfg = _inject_storage(_load_yaml('config/evaluation.full_matrix.colab_pro.yaml'))
        eval_cfg.setdefault('job', {})
        eval_cfg['job']['run_type'] = 'evaluation'
        eval_cfg['job']['resume'] = True
        eval_cfg['job']['job_id'] = EVAL_JOB_ID
        eval_cfg.setdefault('evaluation', {})
        eval_cfg['evaluation']['model_id'] = 'auto_latest'
        eval_cfg['evaluation']['dataset_selection_mode'] = 'largest_built'
        eval_cfg['evaluation']['output_dir'] = f'{WORKSPACE_REL}/reports/evaluations'
        eval_active = generated / f'evaluation.{DRIVE_IDENTITY_KEY}.active.yaml'
        _save_yaml(eval_active, eval_cfg)

        # benchmark
        bench = _inject_storage(_load_yaml('config/benchmark.colab_pro.yaml'))
        bench.setdefault('job', {})
        bench['job']['run_type'] = 'benchmark'
        bench['job']['resume'] = True
        bench['job']['job_id'] = BENCH_JOB_ID
        bench.setdefault('benchmark', {})
        b = bench['benchmark']
        b['target'] = 'auto_latest'
        b['model_search_profile'] = 'fort_plusplus'
        b['model_search_depth'] = 3
        b['model_search_top_k'] = 6
        b['model_search_top_k_child'] = 4
        b['model_search_alpha_beta'] = True
        b['games_per_matchup'] = 50
        b['matchups'] = ['minimax:medium', 'minimax:hard', 'mcts:medium', 'mcts:hard', 'mcts:insane']
        b['output_dir'] = f'{WORKSPACE_REL}/reports/benchmarks'
        bench_active = generated / f'benchmark.{DRIVE_IDENTITY_KEY}.active.yaml'
        _save_yaml(bench_active, bench)

        print('Configs actives:')
        print(' -', gen_active)
        print(' -', build_active)
        print(' -', train_active)
        print(' -', eval_active)
        print(' -', bench_active)
        """
    ),
    md("## 4) Purge datasets existants (optionnel, garde les modèles)"),
    code(
        """
        import shutil
        from pathlib import Path

        DRIVE_ROOT = Path('/content/drive/MyDrive/songo-stockfish')
        DRY_RUN = True  # Passe a False pour appliquer.

        candidates = [
            DRIVE_ROOT / 'data' / 'datasets',
            DRIVE_ROOT / 'data' / 'raw',
            DRIVE_ROOT / 'data' / 'sampled',
            DRIVE_ROOT / 'data' / 'label_cache',
        ]
        # Datasets des workspaces utilisateurs.
        for child in DRIVE_ROOT.iterdir():
            if not child.is_dir():
                continue
            if child.name in {'models', 'reports', 'secrets'}:
                continue
            candidates.extend(
                [
                    child / 'data' / 'datasets',
                    child / 'data' / 'raw',
                    child / 'data' / 'sampled',
                    child / 'data' / 'label_cache',
                ]
            )

        removed = 0
        for p in candidates:
            if not p.exists():
                continue
            print('[candidate]', p)
            if not DRY_RUN:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1

        print('DRY_RUN =', DRY_RUN)
        if not DRY_RUN:
            print('Suppression terminee. Dossiers supprimes =', removed)
        """
    ),
    md("## 5) Lancer la génération de dataset (long run, cumulatif)"),
    code(
        """
        import os
        import subprocess
        from pathlib import Path

        WORKTREE = Path('/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('PYTHON_BIN', 'python3')
        DRIVE_IDENTITY_KEY = os.environ.get('SONGO_DRIVE_IDENTITY_KEY', 'unknown_drive_identity')
        CFG = WORKTREE / 'config' / 'generated' / f'dataset_generation.{DRIVE_IDENTITY_KEY}.active.yaml'

        cmd = [
            PYTHON_BIN, '-m', 'songo_model_stockfish.cli.main',
            'dataset-generate',
            '--config', str(CFG),
        ]
        print('RUN:', cmd)
        subprocess.run(cmd, cwd=str(WORKTREE), check=True)
        """
    ),
    md("## 6) Lancer le build dataset (long run, cumulatif)"),
    code(
        """
        import os
        import subprocess
        from pathlib import Path

        WORKTREE = Path('/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('PYTHON_BIN', 'python3')
        DRIVE_IDENTITY_KEY = os.environ.get('SONGO_DRIVE_IDENTITY_KEY', 'unknown_drive_identity')
        CFG = WORKTREE / 'config' / 'generated' / f'dataset_build.{DRIVE_IDENTITY_KEY}.active.yaml'

        cmd = [
            PYTHON_BIN, '-m', 'songo_model_stockfish.cli.main',
            'dataset-build',
            '--config', str(CFG),
        ]
        print('RUN:', cmd)
        subprocess.run(cmd, cwd=str(WORKTREE), check=True)
        """
    ),
    md("## 7) Train -> Eval -> Benchmark (promotion globale incluse)"),
    code(
        """
        import os
        import json
        import yaml
        import subprocess
        from pathlib import Path

        WORKTREE = Path('/content/songo-model-stockfish-for-google-collab')
        PYTHON_BIN = os.environ.get('PYTHON_BIN', 'python3')
        DRIVE_IDENTITY_KEY = os.environ.get('SONGO_DRIVE_IDENTITY_KEY', 'unknown_drive_identity')

        TRAIN_CFG = WORKTREE / 'config' / 'generated' / f'train.{DRIVE_IDENTITY_KEY}.active.yaml'
        EVAL_CFG = WORKTREE / 'config' / 'generated' / f'evaluation.{DRIVE_IDENTITY_KEY}.active.yaml'
        BENCH_CFG = WORKTREE / 'config' / 'generated' / f'benchmark.{DRIVE_IDENTITY_KEY}.active.yaml'

        def _run(cmd):
            print('RUN:', cmd)
            subprocess.run(cmd, cwd=str(WORKTREE), check=True)

        # 1) Train
        _run([PYTHON_BIN, '-m', 'songo_model_stockfish.cli.main', 'train', '--config', str(TRAIN_CFG)])

        # 2) Resolve latest trained model id from registry
        registry_path = Path('/content/drive/MyDrive/songo-stockfish/models/model_registry.json')
        registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {'models': []}
        models = list(registry.get('models', []))
        if not models:
            raise RuntimeError('Aucun modele trouve dans model_registry apres train.')
        latest = max(models, key=lambda item: float(item.get('sort_ts', 0.0)))
        model_id = str(latest.get('model_id', '')).strip()
        if not model_id:
            raise RuntimeError('model_id vide dans le registre.')
        print('model_id entrainé =', model_id)

        # 3) Eval ciblé sur ce model
        eval_payload = yaml.safe_load(EVAL_CFG.read_text(encoding='utf-8')) or {}
        eval_payload.setdefault('evaluation', {})
        eval_payload['evaluation']['model_id'] = model_id
        eval_runtime = EVAL_CFG.with_name(EVAL_CFG.stem + '.runtime.yaml')
        eval_runtime.write_text(yaml.safe_dump(eval_payload, sort_keys=False), encoding='utf-8')
        _run([PYTHON_BIN, '-m', 'songo_model_stockfish.cli.main', 'evaluate', '--config', str(eval_runtime)])

        # 4) Benchmark ciblé sur ce model
        bench_payload = yaml.safe_load(BENCH_CFG.read_text(encoding='utf-8')) or {}
        bench_payload.setdefault('benchmark', {})
        bench_payload['benchmark']['target'] = model_id
        bench_runtime = BENCH_CFG.with_name(BENCH_CFG.stem + '.runtime.yaml')
        bench_runtime.write_text(yaml.safe_dump(bench_payload, sort_keys=False), encoding='utf-8')
        _run([PYTHON_BIN, '-m', 'songo_model_stockfish.cli.main', 'benchmark', '--config', str(bench_runtime)])

        # La promotion globale est gérée par le pipeline benchmark/registry.
        promoted_meta = Path('/content/drive/MyDrive/songo-stockfish/models/promoted/best/metadata.json')
        if promoted_meta.exists():
            meta = json.loads(promoted_meta.read_text(encoding='utf-8'))
            print('promoted_model_id =', meta.get('model_id', '<none>'))
            print('promoted_checkpoint =', meta.get('promoted_checkpoint_path', '<none>'))
        else:
            print('Aucun metadata de promotion trouvé.')
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
