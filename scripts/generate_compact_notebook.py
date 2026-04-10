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
        # songo-model-stockfish-for-google-collab - Compact

        Notebook compact pour:
        1. monter Drive
        2. mettre a jour le code
        3. generer les configs actives
        4. lancer `dataset-generate` + `dataset-build` en parallele
        5. lister les datasets par taille ou par date
        6. entrainer
        7. evaluer
        8. benchmarker le dernier modele
        """
    ),
    md("## 1. Monter Drive"),
    code(
        """
        from google.colab import drive
        drive.mount('/content/drive')
        """
    ),
    md("## 2. Preparer le projet"),
    code(
        """
        import os
        import sys
        import shutil
        import subprocess
        from pathlib import Path

        GIT_REPO_URL = 'https://github.com/GlennEriss/songo-model-stockfish-for-google-collab.git'
        GIT_BRANCH = 'main'
        PROJECT_NAME = 'songo-model-stockfish-for-google-collab'
        DRIVE_ROOT = '/content/drive/MyDrive/songo-stockfish'
        DEFAULT_WORKTREE = f'/content/{PROJECT_NAME}'

        WORKTREE = os.environ.get('WORKTREE', DEFAULT_WORKTREE)
        PYTHON_BIN = sys.executable or 'python3'

        for relative in [
            'jobs',
            'data',
            'data/datasets',
            'data/label_cache',
            'logs/pipeline',
            'models',
            'reports/evaluations',
            'reports/benchmarks',
        ]:
            Path(DRIVE_ROOT, relative).mkdir(parents=True, exist_ok=True)

        if not (Path(WORKTREE) / '.git').exists():
            if Path(WORKTREE).exists():
                shutil.rmtree(WORKTREE)
            subprocess.run(['git', 'clone', '--branch', GIT_BRANCH, GIT_REPO_URL, WORKTREE], check=True)
        else:
            subprocess.run(['git', '-C', WORKTREE, 'fetch', 'origin', GIT_BRANCH], check=True)
            subprocess.run(['git', '-C', WORKTREE, 'checkout', GIT_BRANCH], check=True)
            subprocess.run(['git', '-C', WORKTREE, 'pull', '--ff-only', 'origin', GIT_BRANCH], check=True)

        subprocess.run([PYTHON_BIN, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([PYTHON_BIN, '-m', 'pip', 'install', '-r', f'{WORKTREE}/requirements.txt'], check=True)

        print('DRIVE_ROOT =', DRIVE_ROOT)
        print('WORKTREE   =', WORKTREE)
        print('PYTHON_BIN =', PYTHON_BIN)
        print('Repo et dependances prets')
        """
    ),
    md("## 3. Parametres"),
    code(
        """
        DATASET_GENERATE_CONFIG = 'config/dataset_generation.full_matrix.colab_pro.yaml'
        DATASET_BUILD_CONFIG = 'config/dataset_build.full_matrix.colab_pro.yaml'
        TRAIN_CONTINUE_CONFIG = 'config/train.full_matrix.colab_pro.yaml'
        TRAIN_SCRATCH_CONFIG = 'config/train.full_matrix.colab_pro.scratch.yaml'
        EVALUATION_CONFIG = 'config/evaluation.full_matrix.colab_pro.yaml'
        BENCHMARK_CONFIG = 'config/benchmark.colab_pro.yaml'

        DATASET_SOURCE_ID = 'sampled_full_matrix_colab_pro_bench_models_20m'
        DATASET_BUILD_ID = 'dataset_v5_full_matrix_colab_pro_insane_20m'
        TARGET_SAMPLES = 20000000
        TARGET_LABELED_SAMPLES = 20000000
        BENCHMATCH_GAMES = 400
        BENCHMATCH_CLASSIC_AGENTS = [
            'minimax:medium',
            'minimax:hard',
            'mcts:medium',
            'mcts:hard',
        ]
        BENCHMATCH_INCLUDE_ALL_REGISTERED_MODELS = True
        BENCHMATCH_MODEL_PREFIX = 'songo_policy_value_colab_pro_'
        BENCHMATCH_MODEL_IDS = []
        BENCHMATCH_EXCLUDE_MODEL_IDS = []
        BENCHMATCH_MODEL_LIMIT = 0
        BENCHMATCH_INCLUDE_SELF_PLAY = True
        BENCHMATCH_ORDERED_MATCHUPS = True
        BENCHMATCH_MODEL_AGENT_DEVICE = 'cpu'
        SOURCE_POLL_INTERVAL_SECONDS = 20
        DATASET_GENERATE_WORKERS = 16
        DATASET_GENERATE_MAX_PENDING_FUTURES = 32
        DATASET_BUILD_WORKERS = 16
        DATASET_BUILD_MAX_PENDING_FUTURES = 32
        AUTO_TUNE_RESOURCES = True

        CPU_SAFE_NUM_WORKERS = 2
        CPU_SAFE_PREFETCH_FACTOR = 2

        DATASET_GENERATE_CONFIG_ACTIVE_PATH = f'{WORKTREE}/config/generated/dataset_generation.compact.active.yaml'
        DATASET_BUILD_CONFIG_ACTIVE_PATH = f'{WORKTREE}/config/generated/dataset_build.compact.active.yaml'
        TRAIN_CONTINUE_CPU_CONFIG_PATH = f'{WORKTREE}/config/generated/train.full_matrix.colab_pro.cpu.yaml'
        TRAIN_SCRATCH_CPU_CONFIG_PATH = f'{WORKTREE}/config/generated/train.full_matrix.colab_pro.scratch.cpu.yaml'
        EVALUATION_CPU_CONFIG_PATH = f'{WORKTREE}/config/generated/evaluation.full_matrix.colab_pro.cpu.yaml'
        BENCHMARK_CPU_CONFIG_PATH = f'{WORKTREE}/config/generated/benchmark.colab_pro.cpu.yaml'
        TRAIN_CONTINUE_TPU_CONFIG_PATH = f'{WORKTREE}/config/generated/train.full_matrix.colab_pro.tpu.yaml'
        TRAIN_SCRATCH_TPU_CONFIG_PATH = f'{WORKTREE}/config/generated/train.full_matrix.colab_pro.scratch.tpu.yaml'
        EVALUATION_TPU_CONFIG_PATH = f'{WORKTREE}/config/generated/evaluation.full_matrix.colab_pro.tpu.yaml'
        BENCHMARK_TPU_CONFIG_PATH = f'{WORKTREE}/config/generated/benchmark.colab_pro.tpu.yaml'
        TRAIN_CONTINUE_20M_CONFIG_ACTIVE_PATH = f'{WORKTREE}/config/generated/train.full_matrix.colab_pro.continue.dataset20m.active.yaml'
        TRAIN_SCRATCH_20M_CONFIG_ACTIVE_PATH = f'{WORKTREE}/config/generated/train.full_matrix.colab_pro.scratch.dataset20m.active.yaml'
        EVALUATION_20M_CONFIG_ACTIVE_PATH = f'{WORKTREE}/config/generated/evaluation.full_matrix.colab_pro.dataset20m.active.yaml'

        DATASET_LIST_KIND = 'all'       # 'sources', 'built', 'all'
        DATASET_LIST_SORT_BY = 'size'   # 'size', 'created_at', 'updated_at'
        DATASET_LIST_LIMIT = 20

        TRAIN_CONTINUE_JOB_ID = 'train_colab_pro_continue_compact_001'
        TRAIN_SCRATCH_JOB_ID = 'train_colab_pro_scratch_compact_001'
        EVALUATION_JOB_ID = 'eval_colab_pro_compact_001'
        BENCHMARK_JOB_ID = 'benchmark_colab_pro_compact_001'

        def _auto_dataset_generate_job_id(dataset_source_id: str, target_samples: int) -> str:
            safe_source = dataset_source_id.replace('-', '_')
            return f'dataset_benchmatch_{safe_source}_{target_samples}_001'

        def _auto_dataset_build_job_id(dataset_id: str, target_samples: int) -> str:
            safe_dataset = dataset_id.replace('-', '_')
            return f'build_{safe_dataset}_{target_samples}_001'

        DATASET_GENERATE_JOB_ID = _auto_dataset_generate_job_id(DATASET_SOURCE_ID, TARGET_SAMPLES)
        DATASET_BUILD_JOB_ID = _auto_dataset_build_job_id(DATASET_BUILD_ID, TARGET_LABELED_SAMPLES)

        print('DATASET_SOURCE_ID       =', DATASET_SOURCE_ID)
        print('DATASET_BUILD_ID        =', DATASET_BUILD_ID)
        print('TARGET_SAMPLES          =', TARGET_SAMPLES)
        print('TARGET_LABELED_SAMPLES  =', TARGET_LABELED_SAMPLES)
        print('AUTO_TUNE_RESOURCES     =', AUTO_TUNE_RESOURCES)
        print('DATASET_GENERATE_WORKERS =', DATASET_GENERATE_WORKERS)
        print('DATASET_BUILD_WORKERS    =', DATASET_BUILD_WORKERS)
        print('DATASET_GENERATE_JOB_ID =', DATASET_GENERATE_JOB_ID)
        print('DATASET_BUILD_JOB_ID    =', DATASET_BUILD_JOB_ID)
        print('TRAIN_CONTINUE_JOB_ID   =', TRAIN_CONTINUE_JOB_ID)
        print('TRAIN_SCRATCH_JOB_ID    =', TRAIN_SCRATCH_JOB_ID)
        print('EVALUATION_JOB_ID       =', EVALUATION_JOB_ID)
        print('BENCHMARK_JOB_ID        =', BENCHMARK_JOB_ID)
        """
    ),
    md("## 4. Generer les configs actives"),
    code(
        """
        import json
        import os
        import re
        import math
        import subprocess
        from pathlib import Path

        import torch
        import yaml

        RUNTIME_PROFILE = 'cpu'
        TPU_ENV_PRESENT = bool(os.environ.get('COLAB_TPU_ADDR'))
        TPU_RUNTIME_READY = False
        try:
            import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
            _ = xm.xla_device()
            TPU_RUNTIME_READY = True
        except Exception:
            TPU_RUNTIME_READY = False

        RUNTIME_HAS_CUDA = bool(torch.cuda.is_available())
        if TPU_RUNTIME_READY:
            RUNTIME_PROFILE = 'tpu'
        elif RUNTIME_HAS_CUDA:
            RUNTIME_PROFILE = 'gpu'
        else:
            RUNTIME_PROFILE = 'cpu'

        TRAIN_CONTINUE_CONFIG_ACTIVE = TRAIN_CONTINUE_CONFIG
        TRAIN_SCRATCH_CONFIG_ACTIVE = TRAIN_SCRATCH_CONFIG
        EVALUATION_CONFIG_ACTIVE = EVALUATION_CONFIG
        BENCHMARK_CONFIG_ACTIVE = BENCHMARK_CONFIG
        DATASET_GENERATE_CONFIG_ACTIVE = DATASET_GENERATE_CONFIG_ACTIVE_PATH
        DATASET_BUILD_CONFIG_ACTIVE = DATASET_BUILD_CONFIG_ACTIVE_PATH
        TRAIN_CONTINUE_20M_CONFIG_ACTIVE = TRAIN_CONTINUE_20M_CONFIG_ACTIVE_PATH
        TRAIN_SCRATCH_20M_CONFIG_ACTIVE = TRAIN_SCRATCH_20M_CONFIG_ACTIVE_PATH
        EVALUATION_20M_CONFIG_ACTIVE = EVALUATION_20M_CONFIG_ACTIVE_PATH

        def _version_sort_key(model_id: str):
            match = re.search(r'_v(\\d+)$', model_id)
            return (0, int(match.group(1))) if match else (1, model_id)

        def _make_cpu_safe_runtime_cfg(cfg: dict) -> dict:
            runtime_cfg = dict(cfg.get('runtime', {}) or {})
            runtime_cfg['device'] = 'cpu'
            runtime_cfg['num_workers'] = int(CPU_SAFE_NUM_WORKERS)
            runtime_cfg['pin_memory'] = False
            runtime_cfg['persistent_workers'] = bool(CPU_SAFE_NUM_WORKERS > 0)
            runtime_cfg['prefetch_factor'] = int(CPU_SAFE_PREFETCH_FACTOR) if CPU_SAFE_NUM_WORKERS > 0 else 0
            runtime_cfg['mixed_precision'] = False
            cfg['runtime'] = runtime_cfg
            return cfg

        def _make_tpu_runtime_cfg(cfg: dict) -> dict:
            runtime_cfg = dict(cfg.get('runtime', {}) or {})
            runtime_cfg['device'] = 'xla'
            runtime_cfg['num_workers'] = int(max(2, CPU_SAFE_NUM_WORKERS))
            runtime_cfg['pin_memory'] = False
            runtime_cfg['persistent_workers'] = bool(runtime_cfg['num_workers'] > 0)
            runtime_cfg['prefetch_factor'] = int(max(2, CPU_SAFE_PREFETCH_FACTOR)) if runtime_cfg['num_workers'] > 0 else 0
            runtime_cfg['mixed_precision'] = False
            cfg['runtime'] = runtime_cfg
            return cfg

        def _write_yaml(payload: dict, target_path_str: str) -> str:
            target_path = Path(target_path_str)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding='utf-8')
            return str(target_path)

        def _detect_ram_gb() -> float:
            try:
                import psutil  # type: ignore[import-not-found]
                return float(psutil.virtual_memory().total) / (1024 ** 3)
            except Exception:
                try:
                    pages = os.sysconf('SC_PHYS_PAGES')
                    page_size = os.sysconf('SC_PAGE_SIZE')
                    return float(pages * page_size) / (1024 ** 3)
                except Exception:
                    return 0.0

        def _detect_gpu_mem_gb() -> float:
            if not torch.cuda.is_available():
                return 0.0
            try:
                return float(torch.cuda.get_device_properties(0).total_memory) / (1024 ** 3)
            except Exception:
                return 0.0

        def _safe_loadavg() -> float:
            try:
                return float(os.getloadavg()[0])
            except Exception:
                return 0.0

        cpu_count = int(os.cpu_count() or 2)
        ram_gb = _detect_ram_gb()
        gpu_mem_gb = _detect_gpu_mem_gb()
        load1 = _safe_loadavg()

        runtime_summary = {
            'cpu_count': cpu_count,
            'ram_gb': round(ram_gb, 2),
            'loadavg_1m': round(load1, 2),
            'runtime_profile': RUNTIME_PROFILE,
            'gpu_available': bool(RUNTIME_HAS_CUDA),
            'gpu_mem_gb': round(gpu_mem_gb, 2),
            'tpu_env_present': bool(TPU_ENV_PRESENT),
            'tpu_runtime_ready': bool(TPU_RUNTIME_READY),
        }

        if AUTO_TUNE_RESOURCES:
            if RUNTIME_PROFILE == 'tpu':
                suggested_workers = max(8, min(24, cpu_count))
                suggested_train_workers = max(4, min(12, cpu_count // 2 if cpu_count > 1 else 1))
                suggested_batch_size = 4096
                BENCHMATCH_MODEL_AGENT_DEVICE = 'cpu'
            elif RUNTIME_PROFILE == 'gpu':
                suggested_workers = max(8, min(24, cpu_count))
                suggested_train_workers = max(4, min(12, cpu_count // 2 if cpu_count > 1 else 1))
                if gpu_mem_gb >= 30:
                    suggested_batch_size = 8192
                elif gpu_mem_gb >= 15:
                    suggested_batch_size = 4096
                else:
                    suggested_batch_size = 2048
                BENCHMATCH_MODEL_AGENT_DEVICE = 'cuda'
            else:
                if load1 > 0 and cpu_count > 0 and (load1 / float(cpu_count)) > 0.85:
                    cpu_budget = max(4, int(math.floor(cpu_count * 0.6)))
                else:
                    cpu_budget = max(4, int(math.floor(cpu_count * 0.8)))
                mem_cap = max(4, int(ram_gb // 0.8)) if ram_gb > 0 else 8
                suggested_workers = max(4, min(24, cpu_budget, mem_cap))
                suggested_train_workers = max(2, min(8, suggested_workers // 2))
                if ram_gb >= 24:
                    suggested_batch_size = 4096
                elif ram_gb >= 12:
                    suggested_batch_size = 2048
                else:
                    suggested_batch_size = 1024
                BENCHMATCH_MODEL_AGENT_DEVICE = 'cpu'

            DATASET_GENERATE_WORKERS = int(suggested_workers)
            DATASET_BUILD_WORKERS = int(suggested_workers)
            DATASET_GENERATE_MAX_PENDING_FUTURES = int(DATASET_GENERATE_WORKERS * 2)
            DATASET_BUILD_MAX_PENDING_FUTURES = int(DATASET_BUILD_WORKERS * 2)
            CPU_SAFE_NUM_WORKERS = int(suggested_train_workers)
            tuned_batch_size = int(suggested_batch_size)
        else:
            tuned_batch_size = None

        if RUNTIME_PROFILE == 'cpu':
            for base_relative, target_path, assign_name in [
                (TRAIN_CONTINUE_CONFIG, TRAIN_CONTINUE_CPU_CONFIG_PATH, 'TRAIN_CONTINUE_CONFIG_ACTIVE'),
                (TRAIN_SCRATCH_CONFIG, TRAIN_SCRATCH_CPU_CONFIG_PATH, 'TRAIN_SCRATCH_CONFIG_ACTIVE'),
                (EVALUATION_CONFIG, EVALUATION_CPU_CONFIG_PATH, 'EVALUATION_CONFIG_ACTIVE'),
                (BENCHMARK_CONFIG, BENCHMARK_CPU_CONFIG_PATH, 'BENCHMARK_CONFIG_ACTIVE'),
            ]:
                base_cfg = yaml.safe_load((Path(WORKTREE) / base_relative).read_text(encoding='utf-8')) or {}
                globals()[assign_name] = _write_yaml(_make_cpu_safe_runtime_cfg(base_cfg), target_path)
        elif RUNTIME_PROFILE == 'tpu':
            for base_relative, target_path, assign_name in [
                (TRAIN_CONTINUE_CONFIG, TRAIN_CONTINUE_TPU_CONFIG_PATH, 'TRAIN_CONTINUE_CONFIG_ACTIVE'),
                (TRAIN_SCRATCH_CONFIG, TRAIN_SCRATCH_TPU_CONFIG_PATH, 'TRAIN_SCRATCH_CONFIG_ACTIVE'),
                (EVALUATION_CONFIG, EVALUATION_TPU_CONFIG_PATH, 'EVALUATION_CONFIG_ACTIVE'),
                (BENCHMARK_CONFIG, BENCHMARK_TPU_CONFIG_PATH, 'BENCHMARK_CONFIG_ACTIVE'),
            ]:
                base_cfg = yaml.safe_load((Path(WORKTREE) / base_relative).read_text(encoding='utf-8')) or {}
                globals()[assign_name] = _write_yaml(_make_tpu_runtime_cfg(base_cfg), target_path)

        if AUTO_TUNE_RESOURCES and tuned_batch_size is not None:
            for key_name in ['TRAIN_CONTINUE_CONFIG_ACTIVE', 'TRAIN_SCRATCH_CONFIG_ACTIVE']:
                current_path = Path(str(globals()[key_name]))
                cfg_payload = yaml.safe_load(current_path.read_text(encoding='utf-8')) or {}
                runtime_payload = dict(cfg_payload.get('runtime', {}) or {})
                runtime_payload['num_workers'] = int(CPU_SAFE_NUM_WORKERS)
                runtime_payload['persistent_workers'] = bool(CPU_SAFE_NUM_WORKERS > 0)
                runtime_payload['prefetch_factor'] = int(max(2, CPU_SAFE_PREFETCH_FACTOR)) if CPU_SAFE_NUM_WORKERS > 0 else 0
                runtime_payload['pin_memory'] = bool(runtime_payload.get('device', 'cpu') == 'cuda')
                cfg_payload['runtime'] = runtime_payload
                train_payload = dict(cfg_payload.get('train', {}) or {})
                train_payload['batch_size'] = int(tuned_batch_size)
                cfg_payload['train'] = train_payload
                current_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False, allow_unicode=True), encoding='utf-8')

            eval_path = Path(str(EVALUATION_CONFIG_ACTIVE))
            eval_cfg_payload = yaml.safe_load(eval_path.read_text(encoding='utf-8')) or {}
            eval_runtime_payload = dict(eval_cfg_payload.get('runtime', {}) or {})
            eval_runtime_payload['num_workers'] = int(max(1, CPU_SAFE_NUM_WORKERS))
            eval_runtime_payload['persistent_workers'] = bool(eval_runtime_payload['num_workers'] > 0)
            eval_runtime_payload['prefetch_factor'] = int(max(2, CPU_SAFE_PREFETCH_FACTOR)) if eval_runtime_payload['num_workers'] > 0 else 0
            eval_runtime_payload['pin_memory'] = bool(eval_runtime_payload.get('device', 'cpu') == 'cuda')
            eval_cfg_payload['runtime'] = eval_runtime_payload
            eval_path.write_text(yaml.safe_dump(eval_cfg_payload, sort_keys=False, allow_unicode=True), encoding='utf-8')

        model_registry_path = Path(DRIVE_ROOT) / 'models' / 'model_registry.json'
        model_registry = json.loads(model_registry_path.read_text(encoding='utf-8')) if model_registry_path.exists() else {'models': []}
        selected_model_ids = []
        seen_model_ids = set()

        def _maybe_add_model(model_id: str):
            model_id = str(model_id).strip()
            if not model_id:
                return
            if BENCHMATCH_MODEL_PREFIX and not model_id.startswith(BENCHMATCH_MODEL_PREFIX):
                return
            if model_id in BENCHMATCH_EXCLUDE_MODEL_IDS or model_id in seen_model_ids:
                return
            seen_model_ids.add(model_id)
            selected_model_ids.append(model_id)

        if BENCHMATCH_INCLUDE_ALL_REGISTERED_MODELS:
            discovered = [str(item.get('model_id', '')).strip() for item in model_registry.get('models', [])]
            for model_id in sorted(discovered, key=_version_sort_key):
                _maybe_add_model(model_id)

        for model_id in BENCHMATCH_MODEL_IDS:
            _maybe_add_model(model_id)

        if BENCHMATCH_MODEL_LIMIT > 0:
            selected_model_ids = selected_model_ids[:BENCHMATCH_MODEL_LIMIT]

        bench_agents = list(BENCHMATCH_CLASSIC_AGENTS) + [f'model:{model_id}' for model_id in selected_model_ids]
        matchups = []
        if BENCHMATCH_ORDERED_MATCHUPS:
            for agent_a in bench_agents:
                for agent_b in bench_agents:
                    if not BENCHMATCH_INCLUDE_SELF_PLAY and agent_a == agent_b:
                        continue
                    matchups.append(f'{agent_a} vs {agent_b}')
        else:
            for idx, agent_a in enumerate(bench_agents):
                start = idx if BENCHMATCH_INCLUDE_SELF_PLAY else idx + 1
                for agent_b in bench_agents[start:]:
                    if not BENCHMATCH_INCLUDE_SELF_PLAY and agent_a == agent_b:
                        continue
                    matchups.append(f'{agent_a} vs {agent_b}')

        generate_cfg = yaml.safe_load((Path(WORKTREE) / DATASET_GENERATE_CONFIG).read_text(encoding='utf-8')) or {}
        generate_block = dict(generate_cfg.get('dataset_generation', {}) or {})
        if DATASET_SOURCE_ID.startswith('sampled_'):
            output_sampled_dir = f'data/{DATASET_SOURCE_ID}'
            output_raw_dir = f"data/raw_{DATASET_SOURCE_ID[len('sampled_'):]}"
        else:
            output_sampled_dir = f'data/{DATASET_SOURCE_ID}'
            output_raw_dir = f'data/raw_{DATASET_SOURCE_ID}'
        generate_block['source_mode'] = 'benchmatch'
        generate_block['dataset_source_id'] = DATASET_SOURCE_ID
        generate_block['target_samples'] = int(TARGET_SAMPLES)
        generate_block['output_sampled_dir'] = output_sampled_dir
        generate_block['output_raw_dir'] = output_raw_dir
        generate_block['workers'] = int(DATASET_GENERATE_WORKERS)
        generate_block['max_pending_futures'] = int(DATASET_GENERATE_MAX_PENDING_FUTURES)
        generate_block['games'] = int(BENCHMATCH_GAMES)
        generate_block['matchups'] = matchups
        generate_block['model_agent_device'] = str(BENCHMATCH_MODEL_AGENT_DEVICE)
        generate_block['progress_update_every_n_games'] = 1
        generate_cfg['dataset_generation'] = generate_block
        DATASET_GENERATE_CONFIG_ACTIVE = _write_yaml(generate_cfg, DATASET_GENERATE_CONFIG_ACTIVE_PATH)

        build_cfg = yaml.safe_load((Path(WORKTREE) / DATASET_BUILD_CONFIG).read_text(encoding='utf-8')) or {}
        build_block = dict(build_cfg.get('dataset_build', {}) or {})
        build_block['source_dataset_id'] = DATASET_SOURCE_ID
        build_block['input_sampled_dir'] = f'data/{DATASET_SOURCE_ID}'
        build_block['dataset_id'] = DATASET_BUILD_ID
        build_block['target_labeled_samples'] = int(TARGET_LABELED_SAMPLES)
        build_block['follow_source_updates'] = True
        build_block['source_poll_interval_seconds'] = float(SOURCE_POLL_INTERVAL_SECONDS)
        build_block['workers'] = int(DATASET_BUILD_WORKERS)
        build_block['max_pending_futures'] = int(DATASET_BUILD_MAX_PENDING_FUTURES)
        build_block['export_partial_every_n_files'] = int(min(20, int(build_block.get('export_partial_every_n_files', 100) or 100)))
        build_cfg['dataset_build'] = build_block
        DATASET_BUILD_CONFIG_ACTIVE = _write_yaml(build_cfg, DATASET_BUILD_CONFIG_ACTIVE_PATH)

        train_continue_cfg = yaml.safe_load(Path(TRAIN_CONTINUE_CONFIG_ACTIVE).read_text(encoding='utf-8')) or {}
        train_continue_block = dict(train_continue_cfg.get('train', {}) or {})
        train_continue_block['dataset_selection_mode'] = 'configured'
        train_continue_block['dataset_id'] = DATASET_BUILD_ID
        train_continue_cfg['train'] = train_continue_block
        TRAIN_CONTINUE_20M_CONFIG_ACTIVE = _write_yaml(train_continue_cfg, TRAIN_CONTINUE_20M_CONFIG_ACTIVE_PATH)

        train_scratch_cfg = yaml.safe_load(Path(TRAIN_SCRATCH_CONFIG_ACTIVE).read_text(encoding='utf-8')) or {}
        train_scratch_block = dict(train_scratch_cfg.get('train', {}) or {})
        train_scratch_block['dataset_selection_mode'] = 'configured'
        train_scratch_block['dataset_id'] = DATASET_BUILD_ID
        train_scratch_cfg['train'] = train_scratch_block
        TRAIN_SCRATCH_20M_CONFIG_ACTIVE = _write_yaml(train_scratch_cfg, TRAIN_SCRATCH_20M_CONFIG_ACTIVE_PATH)

        eval_cfg = yaml.safe_load(Path(EVALUATION_CONFIG_ACTIVE).read_text(encoding='utf-8')) or {}
        eval_block = dict(eval_cfg.get('evaluation', {}) or {})
        eval_block['dataset_selection_mode'] = 'configured'
        eval_block['dataset_id'] = DATASET_BUILD_ID
        eval_cfg['evaluation'] = eval_block
        EVALUATION_20M_CONFIG_ACTIVE = _write_yaml(eval_cfg, EVALUATION_20M_CONFIG_ACTIVE_PATH)

        print('RUNTIME_PROFILE                =', RUNTIME_PROFILE)
        print('TPU_ENV_PRESENT                =', TPU_ENV_PRESENT)
        print('TPU_RUNTIME_READY              =', TPU_RUNTIME_READY)
        print('RUNTIME_HAS_CUDA               =', RUNTIME_HAS_CUDA)
        print('RUNTIME_SUMMARY                =', runtime_summary)
        print('DATASET_GENERATE_CONFIG_ACTIVE =', DATASET_GENERATE_CONFIG_ACTIVE)
        print('DATASET_BUILD_CONFIG_ACTIVE    =', DATASET_BUILD_CONFIG_ACTIVE)
        print('TRAIN_CONTINUE_CONFIG_ACTIVE   =', TRAIN_CONTINUE_CONFIG_ACTIVE)
        print('TRAIN_SCRATCH_CONFIG_ACTIVE    =', TRAIN_SCRATCH_CONFIG_ACTIVE)
        print('TRAIN_CONTINUE_20M_CONFIG_ACTIVE =', TRAIN_CONTINUE_20M_CONFIG_ACTIVE)
        print('TRAIN_SCRATCH_20M_CONFIG_ACTIVE  =', TRAIN_SCRATCH_20M_CONFIG_ACTIVE)
        print('EVALUATION_20M_CONFIG_ACTIVE     =', EVALUATION_20M_CONFIG_ACTIVE)
        print('EVALUATION_CONFIG_ACTIVE       =', EVALUATION_CONFIG_ACTIVE)
        print('BENCHMARK_CONFIG_ACTIVE        =', BENCHMARK_CONFIG_ACTIVE)
        print('output_raw_dir                 =', output_raw_dir)
        print('output_sampled_dir             =', output_sampled_dir)
        print('selected_model_ids             =', selected_model_ids)
        print('total_agents                   =', len(bench_agents))
        print('total_matchups                 =', len(matchups))
        print('export_partial_every_n_files   =', build_block.get('export_partial_every_n_files'))
        print('autotuned_generate_workers     =', DATASET_GENERATE_WORKERS)
        print('autotuned_build_workers        =', DATASET_BUILD_WORKERS)
        print('autotuned_train_workers        =', CPU_SAFE_NUM_WORKERS)
        print('autotuned_train_batch_size     =', tuned_batch_size)
        """
    ),
    md("## 5. Lancer le pipeline dataset en parallele"),
    code(
        """
        import json
        import shlex
        import subprocess
        from datetime import UTC, datetime
        from pathlib import Path

        logs_dir = Path(DRIVE_ROOT) / 'logs' / 'pipeline'
        logs_dir.mkdir(parents=True, exist_ok=True)
        jobs_dir = Path(DRIVE_ROOT) / 'jobs'
        jobs_dir.mkdir(parents=True, exist_ok=True)
        (jobs_dir / DATASET_GENERATE_JOB_ID).mkdir(parents=True, exist_ok=True)
        (jobs_dir / DATASET_BUILD_JOB_ID).mkdir(parents=True, exist_ok=True)

        generate_log_path = logs_dir / f'{DATASET_GENERATE_JOB_ID}.log'
        build_log_path = logs_dir / f'{DATASET_BUILD_JOB_ID}.log'

        def _launch_background(cmd: str, log_path: Path) -> int:
            handle = log_path.open('a', encoding='utf-8')
            proc = subprocess.Popen(
                ['/bin/bash', '-lc', cmd],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            handle.close()
            return int(proc.pid)

        generate_cmd = (
            f'cd {shlex.quote(WORKTREE)} && '
            f'PYTHONPATH={shlex.quote(f"{WORKTREE}/src")} '
            f'{shlex.quote(PYTHON_BIN)} -m songo_model_stockfish.cli.main '
            f'dataset-generate --config {shlex.quote(str(DATASET_GENERATE_CONFIG_ACTIVE))} '
            f'--job-id {shlex.quote(DATASET_GENERATE_JOB_ID)}'
        )

        build_cmd = (
            f'cd {shlex.quote(WORKTREE)} && '
            f'PYTHONPATH={shlex.quote(f"{WORKTREE}/src")} '
            f'{shlex.quote(PYTHON_BIN)} -m songo_model_stockfish.cli.main '
            f'dataset-build --config {shlex.quote(str(DATASET_BUILD_CONFIG_ACTIVE))} '
            f'--job-id {shlex.quote(DATASET_BUILD_JOB_ID)}'
        )

        generate_pid = _launch_background(generate_cmd, generate_log_path)
        build_pid = _launch_background(build_cmd, build_log_path)

        manifest = {
            'launched_at': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            'dataset_source_id': DATASET_SOURCE_ID,
            'dataset_id': DATASET_BUILD_ID,
            'generate_job_id': DATASET_GENERATE_JOB_ID,
            'build_job_id': DATASET_BUILD_JOB_ID,
            'generate_pid': generate_pid,
            'build_pid': build_pid,
            'generate_log_path': str(generate_log_path),
            'build_log_path': str(build_log_path),
        }
        latest_path = logs_dir / 'latest_dataset_pipeline.json'
        latest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding='utf-8')

        print('Pipeline lance en parallele')
        print('  dataset-generate pid =', generate_pid)
        print('  dataset-build pid    =', build_pid)
        print('  generate log         =', generate_log_path)
        print('  build log            =', build_log_path)
        print('  manifest             =', latest_path)
        """
    ),
    md("## 5bis. Suivre le pipeline dataset"),
    code(
        """
        import json
        import os
        import subprocess
        from pathlib import Path

        logs_dir = Path(DRIVE_ROOT) / 'logs' / 'pipeline'
        manifest_path = logs_dir / 'latest_dataset_pipeline.json'

        if not manifest_path.exists():
            print('Manifest introuvable:', manifest_path)
        else:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            generate_pid = int(manifest.get('generate_pid', 0) or 0)
            build_pid = int(manifest.get('build_pid', 0) or 0)
            print('Manifest:')
            print(json.dumps(manifest, indent=2, ensure_ascii=True))

            for label, pid in [('dataset-generate', generate_pid), ('dataset-build', build_pid)]:
                if pid <= 0:
                    print(f'\\n{label}: pid absent')
                    continue
                proc = subprocess.run(
                    ['ps', '-p', str(pid), '-o', 'pid=,ppid=,etime=,state=,command='],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = proc.stdout.strip()
                print(f'\\n{label}:')
                print(output if output else 'processus non trouve')
        """
    ),
    code(
        """
        import json
        import time
        from pathlib import Path

        registry_path = Path(DRIVE_ROOT) / 'data' / 'dataset_registry.json'

        def _load_json_retry(path: Path, retries: int = 6, wait_seconds: float = 0.25):
            last_exc = None
            for _ in range(retries):
                try:
                    return json.loads(path.read_text(encoding='utf-8'))
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    time.sleep(wait_seconds)
            raise last_exc

        if not registry_path.exists():
            print('dataset_registry.json introuvable:', registry_path)
        else:
            registry = _load_json_retry(registry_path)
            source = next((item for item in registry.get('dataset_sources', []) if item.get('dataset_source_id') == DATASET_SOURCE_ID), None)
            built = next((item for item in registry.get('built_datasets', []) if item.get('dataset_id') == DATASET_BUILD_ID), None)

            print('Source courante:')
            if source is None:
                print('- aucune entree source pour', DATASET_SOURCE_ID)
            else:
                print('  dataset_source_id =', source.get('dataset_source_id'))
                print('  source_mode       =', source.get('source_mode'))
                print('  source_status     =', source.get('source_status'))
                print('  sampled_positions =', source.get('sampled_positions'))
                print('  sampled_files     =', source.get('sampled_files'))
                print('  updated_at        =', source.get('updated_at'))

            print('\\nDataset final courant:')
            if built is None:
                print('- aucune entree built pour', DATASET_BUILD_ID)
            else:
                print('  dataset_id        =', built.get('dataset_id'))
                print('  build_status      =', built.get('build_status'))
                print('  labeled_samples   =', built.get('labeled_samples'))
                print('  target_samples    =', built.get('target_labeled_samples'))
                print('  output_dir        =', built.get('output_dir'))
                print('  updated_at        =', built.get('updated_at'))
        """
    ),
    code(
        """
        # KPI live: progression source + build vers la cible
        import json
        import time
        from pathlib import Path

        REFRESH_SECONDS = 15
        MAX_LOOPS = 40

        registry_path = Path(DRIVE_ROOT) / 'data' / 'dataset_registry.json'
        jobs_root = Path(DRIVE_ROOT) / 'jobs'

        def _load_json_retry(path: Path, retries: int = 6, wait_seconds: float = 0.25):
            last_exc = None
            for _ in range(retries):
                try:
                    return json.loads(path.read_text(encoding='utf-8'))
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    time.sleep(wait_seconds)
            raise last_exc

        def _safe_pct(value: int, target: int) -> float:
            if target <= 0:
                return 0.0
            return (100.0 * float(value)) / float(target)

        def _latest_job_dir(job_id: str) -> Path | None:
            if not jobs_root.exists():
                return None
            candidates = [p for p in jobs_root.iterdir() if p.is_dir() and p.name.startswith(job_id.rsplit('_', 1)[0])]
            if not candidates:
                return None
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

        for loop_idx in range(MAX_LOOPS):
            if not registry_path.exists():
                print('dataset_registry.json introuvable:', registry_path)
                break

            registry = _load_json_retry(registry_path)
            source = next((item for item in registry.get('dataset_sources', []) if item.get('dataset_source_id') == DATASET_SOURCE_ID), None)
            built = next((item for item in registry.get('built_datasets', []) if item.get('dataset_id') == DATASET_BUILD_ID), None)

            source_samples = int(source.get('sampled_positions', 0)) if source else 0
            source_status = str(source.get('source_status', '<none>')) if source else '<none>'
            source_files = int(source.get('sampled_files', 0)) if source else 0

            build_samples = int(built.get('labeled_samples', 0)) if built else 0
            build_status = str(built.get('build_status', '<none>')) if built else '<none>'
            build_target = int(built.get('target_labeled_samples', TARGET_LABELED_SAMPLES)) if built else int(TARGET_LABELED_SAMPLES)

            played_games = 0
            latest_generate_dir = _latest_job_dir(DATASET_GENERATE_JOB_ID)
            if latest_generate_dir is not None:
                summary_path = latest_generate_dir / 'dataset_generation' / 'dataset_generation_summary.json'
                if summary_path.exists():
                    summary = _load_json_retry(summary_path)
                    played_games = int(summary.get('added_games', 0))

            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f'[{ts}] source_samples={source_samples}/{TARGET_SAMPLES} ({_safe_pct(source_samples, int(TARGET_SAMPLES)):.2f}%) | source_files={source_files} | source_status={source_status}')
            print(f'[{ts}] played_games={played_games} | build_samples={build_samples}/{build_target} ({_safe_pct(build_samples, build_target):.2f}%) | build_status={build_status}')
            print('-' * 120)

            if loop_idx >= (MAX_LOOPS - 1):
                break
            time.sleep(REFRESH_SECONDS)
        """
    ),
    code(
        """
        import json
        from pathlib import Path

        LOG_TAIL_LINES = 40
        logs_dir = Path(DRIVE_ROOT) / 'logs' / 'pipeline'
        manifest_path = logs_dir / 'latest_dataset_pipeline.json'

        if not manifest_path.exists():
            print('Manifest introuvable:', manifest_path)
        else:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            for label, key in [('dataset-generate', 'generate_log_path'), ('dataset-build', 'build_log_path')]:
                log_path = Path(str(manifest.get(key, '')))
                print(f'\\n===== {label} | {log_path} =====')
                if not log_path.exists():
                    print('log introuvable')
                    continue
                lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
                for line in lines[-LOG_TAIL_LINES:]:
                    print(line)
        """
    ),
    code(
        """
        # Optionnel: suivi live du log dataset-build. Interrompre la cellule quand tu veux.
        import json
        from pathlib import Path

        logs_dir = Path(DRIVE_ROOT) / 'logs' / 'pipeline'
        manifest_path = logs_dir / 'latest_dataset_pipeline.json'
        fallback_path = logs_dir / f'{DATASET_BUILD_JOB_ID}.log'

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            log_path = Path(str(manifest.get('build_log_path', fallback_path)))
        else:
            log_path = fallback_path

        print('tailing:', log_path)
        if not log_path.exists():
            print('log introuvable:', log_path)
        else:
            !tail -f {log_path}
        """
    ),
    md("## 6. Lister les datasets"),
    code(
        """
        import json
        from datetime import datetime
        from pathlib import Path

        registry_path = Path(DRIVE_ROOT) / 'data' / 'dataset_registry.json'
        if not registry_path.exists():
            print('dataset_registry.json introuvable:', registry_path)
        else:
            registry = json.loads(registry_path.read_text(encoding='utf-8'))

            def _fmt_ts(value: str) -> str:
                value = str(value or '').strip()
                if not value:
                    return '<na>'
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    return value

            def _size_value(item: dict, kind: str) -> int:
                if kind == 'source':
                    return int(item.get('sampled_positions', 0))
                return int(item.get('labeled_samples', 0))

            def _sort_key(item: dict, kind: str):
                if DATASET_LIST_SORT_BY == 'size':
                    return (_size_value(item, kind), str(item.get('updated_at', '')))
                if DATASET_LIST_SORT_BY == 'created_at':
                    return (str(item.get('dataset_version', '')), _size_value(item, kind))
                return (str(item.get('updated_at', '')), _size_value(item, kind))

            if DATASET_LIST_KIND in ('sources', 'all'):
                print('Sources datasets:')
                sources = sorted(registry.get('dataset_sources', []), key=lambda item: _sort_key(item, 'source'), reverse=True)
                for item in sources[:DATASET_LIST_LIMIT]:
                    print(
                        '-', item.get('dataset_source_id'),
                        '| mode=', item.get('source_mode'),
                        '| status=', item.get('source_status', 'completed'),
                        '| samples=', item.get('sampled_positions', 0),
                        '| created=', _fmt_ts(item.get('dataset_version', '')),
                        '| updated=', _fmt_ts(item.get('updated_at', '')),
                    )
                if not sources:
                    print('- none')

            if DATASET_LIST_KIND in ('built', 'all'):
                print('\\nBuilt datasets:')
                built = sorted(registry.get('built_datasets', []), key=lambda item: _sort_key(item, 'built'), reverse=True)
                for item in built[:DATASET_LIST_LIMIT]:
                    print(
                        '-', item.get('dataset_id'),
                        '| build_mode=', item.get('build_mode', 'teacher_label'),
                        '| status=', item.get('build_status', 'completed'),
                        '| labeled=', item.get('labeled_samples', 0),
                        '| schema=', item.get('feature_schema_version', '<na>'),
                        '| created=', _fmt_ts(item.get('dataset_version', '')),
                        '| updated=', _fmt_ts(item.get('updated_at', '')),
                    )
                if not built:
                    print('- none')
        """
    ),
    md("## 7. Entrainement"),
    code(
        """
        print('Cellule A: continue depuis best sur dataset 20M partiel')
        print('config =', TRAIN_CONTINUE_20M_CONFIG_ACTIVE)
        !bash -lc "cd $WORKTREE && PYTHONPATH=$WORKTREE/src $PYTHON_BIN -m songo_model_stockfish.cli.main train --config $TRAIN_CONTINUE_20M_CONFIG_ACTIVE --job-id $TRAIN_CONTINUE_JOB_ID"
        """
    ),
    code(
        """
        print('Cellule B: from scratch sur dataset 20M partiel')
        print('config =', TRAIN_SCRATCH_20M_CONFIG_ACTIVE)
        !bash -lc "cd $WORKTREE && PYTHONPATH=$WORKTREE/src $PYTHON_BIN -m songo_model_stockfish.cli.main train --config $TRAIN_SCRATCH_20M_CONFIG_ACTIVE --job-id $TRAIN_SCRATCH_JOB_ID"
        """
    ),
    md("## 8. Evaluation"),
    code(
        """
        print('Evaluation sur dataset 20M partiel configure')
        print('config =', EVALUATION_20M_CONFIG_ACTIVE)
        !bash -lc "cd $WORKTREE && PYTHONPATH=$WORKTREE/src $PYTHON_BIN -m songo_model_stockfish.cli.main evaluate --config $EVALUATION_20M_CONFIG_ACTIVE --job-id $EVALUATION_JOB_ID"
        """
    ),
    code(
        """
        import json
        import re
        from pathlib import Path

        jobs_root = Path(DRIVE_ROOT) / 'jobs'
        requested_job_id = globals().get('EVALUATION_JOB_ID', '')

        if requested_job_id:
            match = re.match(r'^(.*?)(\\d+)$', requested_job_id)
            if match:
                prefix = match.group(1)
                candidates = [path for path in jobs_root.iterdir() if path.is_dir() and re.match(rf'^{re.escape(prefix)}\\d+$', path.name)]
            else:
                candidates = [path for path in jobs_root.iterdir() if path.is_dir() and path.name.startswith(requested_job_id)]
        else:
            candidates = [path for path in jobs_root.iterdir() if path.is_dir() and path.name.startswith('eval_')]

        if not candidates:
            print('Aucun job evaluation trouve')
        else:
            latest_job_dir = sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]
            summary_path = latest_job_dir / 'evaluation_summary.json'
            print('Evaluation job lu =', latest_job_dir.name)
            if not summary_path.exists():
                print('Resume introuvable:', summary_path)
            else:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                print(json.dumps(summary, indent=2, ensure_ascii=True))
        """
    ),
    md("## 9. Benchmark"),
    code(
        """
        !bash -lc "cd $WORKTREE && PYTHONPATH=$WORKTREE/src $PYTHON_BIN -m songo_model_stockfish.cli.main benchmark --config $BENCHMARK_CONFIG_ACTIVE --job-id $BENCHMARK_JOB_ID"
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
