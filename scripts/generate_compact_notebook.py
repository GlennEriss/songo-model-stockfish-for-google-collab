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

        BASE_DATASET_SOURCE_ID = 'sampled_full_matrix_colab_pro_bench_models_20m'
        BASE_DATASET_BUILD_ID = 'dataset_v5_full_matrix_colab_pro_insane_20m'
        MULTI_COLAB_SAFE_MODE = True
        WORKER_TAG = ''  # Laisse vide pour auto, ou fixe manuellement: 'w1', 'w2', etc.
        WORKER_COUNT = 1  # Mets 2,3,4... quand tu lances plusieurs Colab
        WORKER_INDEX = -1  # -1 => auto assignment
        AUTO_ASSIGN_WORKER_INDEX = True
        AUTO_REUSE_CHECKPOINT_WORKER_TAG = True
        CHECKPOINT_STALE_MINUTES = 20  # Reprise auto d'un worker stoppe apres inactivite
        ACTIVE_WORKER_GUARD_MINUTES = 10  # N'assigne pas un tag deja actif dans cette fenetre
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
        BENCHMATCH_MODEL_SCAN_DIR = 'models/final'
        BENCHMATCH_MODEL_ORDER = 'newest_first'  # 'newest_first' ou 'oldest_first'
        BENCHMATCH_MODEL_IDS = []
        BENCHMATCH_EXCLUDE_MODEL_IDS = []
        BENCHMATCH_MODEL_LIMIT = 0
        BENCHMATCH_INCLUDE_SELF_PLAY = True
        BENCHMATCH_ORDERED_MATCHUPS = True
        BENCHMATCH_SHUFFLE_MATCHUPS = True
        BENCHMATCH_MODEL_AGENT_DEVICE = 'cpu'
        GLOBAL_TARGET_ENABLED = True
        GLOBAL_TARGET_ID = 'bench_models_20m_global'
        GLOBAL_TARGET_SAMPLES = 20000000
        GLOBAL_PROGRESS_PATH = f'data/global_generation_progress/{GLOBAL_TARGET_ID}.json'
        GLOBAL_PROGRESS_BACKEND = 'firestore'  # Firestore est la source de verite
        FIRESTORE_PROJECT_ID = 'songo-model-ai'
        FIRESTORE_COLLECTION = 'global_generation_progress'
        FIRESTORE_DOCUMENT = GLOBAL_TARGET_ID
        FIRESTORE_CREDENTIALS_PATH = ''  # Optionnel: chemin JSON service account
        FIRESTORE_API_KEY = 'AIzaSyA0I4zJMpBElpwyae0tLlNpnMG0fnF07ys'
        if str(GLOBAL_PROGRESS_BACKEND).strip().lower() != 'firestore':
            raise ValueError("GLOBAL_PROGRESS_BACKEND doit rester 'firestore' (fallback fichier desactive).")
        SOURCE_POLL_INTERVAL_SECONDS = 20
        DATASET_GENERATE_WORKERS = 16
        DATASET_GENERATE_MAX_PENDING_FUTURES = 32
        DATASET_BUILD_WORKERS = 16
        DATASET_BUILD_MAX_PENDING_FUTURES = 32
        DATASET_BUILD_DEDUPE_SAMPLE_IDS = True
        DATASET_BUILD_ADAPTIVE_POLLING = True
        GLOBAL_TARGET_STABILIZATION_POLLS = 3
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
        EVALUATION_20M_RUNTIME_CONFIG_PATH = f'{WORKTREE}/config/generated/evaluation.full_matrix.colab_pro.dataset20m.runtime.active.yaml'

        DATASET_LIST_KIND = 'all'       # 'sources', 'built', 'all'
        DATASET_LIST_SORT_BY = 'size'   # 'size', 'created_at', 'updated_at'
        DATASET_LIST_LIMIT = 20

        TRAIN_CONTINUE_JOB_ID = 'train_colab_pro_continue_compact_001'
        TRAIN_SCRATCH_JOB_ID = 'train_colab_pro_scratch_compact_001'
        EVALUATION_JOB_ID = 'eval_colab_pro_compact_001'
        BENCHMARK_JOB_ID = 'benchmark_colab_pro_compact_001'

        import hashlib
        import json
        import re
        import socket
        import time
        from datetime import datetime
        from pathlib import Path

        def _acquire_lock_dir(lock_dir: Path, timeout_seconds: float = 20.0, poll_seconds: float = 0.1) -> bool:
            deadline = time.time() + max(1.0, float(timeout_seconds))
            while time.time() < deadline:
                try:
                    lock_dir.mkdir(parents=True, exist_ok=False)
                    return True
                except FileExistsError:
                    time.sleep(max(0.01, float(poll_seconds)))
            return False

        def _release_lock_dir(lock_dir: Path) -> None:
            try:
                lock_dir.rmdir()
            except Exception:
                pass

        def _auto_assign_worker_index(*, drive_root: str, global_target_id: str, worker_tag: str, worker_count: int) -> int:
            worker_count = max(1, int(worker_count))
            leases_path = Path(drive_root) / 'data' / 'global_generation_progress' / f'{global_target_id}.worker_leases.json'
            lock_dir = Path(str(leases_path) + '.lock')
            lock_ok = _acquire_lock_dir(lock_dir)
            if not lock_ok:
                return abs(hash(worker_tag)) % worker_count
            try:
                payload = {}
                if leases_path.exists():
                    try:
                        payload = json.loads(leases_path.read_text(encoding='utf-8'))
                    except Exception:
                        payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                leases = payload.get('leases', {})
                if not isinstance(leases, dict):
                    leases = {}

                existing = leases.get(worker_tag, {})
                if isinstance(existing, dict):
                    existing_index = existing.get('index')
                    if isinstance(existing_index, int) and 0 <= existing_index < worker_count:
                        leases[worker_tag] = {'index': existing_index, 'updated_at': time.time()}
                        payload['leases'] = leases
                        leases_path.parent.mkdir(parents=True, exist_ok=True)
                        leases_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')
                        return int(existing_index)

                used = set()
                for info in leases.values():
                    if isinstance(info, dict):
                        idx = info.get('index')
                        if isinstance(idx, int) and 0 <= idx < worker_count:
                            used.add(idx)
                assigned = next((idx for idx in range(worker_count) if idx not in used), None)
                if assigned is None:
                    assigned = abs(hash(worker_tag)) % worker_count
                leases[worker_tag] = {'index': int(assigned), 'updated_at': time.time()}
                payload['leases'] = leases
                leases_path.parent.mkdir(parents=True, exist_ok=True)
                leases_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')
                return int(assigned)
            finally:
                _release_lock_dir(lock_dir)

        def _parse_iso_to_epoch(value: object) -> float:
            text = str(value or '').strip()
            if not text:
                return 0.0
            try:
                return float(datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp())
            except Exception:
                return 0.0

        def _load_json_retry(path: Path, retries: int = 6, wait_seconds: float = 0.25, default=None):
            fallback = {} if default is None else default
            last_exc = None
            for attempt in range(retries):
                try:
                    return json.loads(path.read_text(encoding='utf-8'))
                except (FileNotFoundError, OSError):
                    if attempt + 1 >= retries:
                        return fallback
                    time.sleep(wait_seconds)
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    time.sleep(wait_seconds)
            if last_exc is not None:
                return fallback
            return fallback

        def _normalize_global_payload(payload: object, target_samples: int) -> dict:
            raw = payload if isinstance(payload, dict) else {}
            workers = raw.get('workers', {})
            if not isinstance(workers, dict):
                workers = {}
            return {
                'global_target_id': str(raw.get('global_target_id', GLOBAL_TARGET_ID)),
                'target_samples': int(raw.get('target_samples', target_samples) or target_samples),
                'total_samples': int(raw.get('total_samples', 0) or 0),
                'total_games': int(raw.get('total_games', 0) or 0),
                'workers': workers,
                'updated_at': str(raw.get('updated_at', '<none>') or '<none>'),
            }

        def _firestore_monitor_debug_context(*, global_target_id: str) -> dict:
            credentials_path = str(FIRESTORE_CREDENTIALS_PATH).strip()
            api_key = str(FIRESTORE_API_KEY).strip()
            auth_mode = 'adc'
            if credentials_path:
                auth_mode = 'service_account_file'
            elif api_key:
                auth_mode = 'api_key_anonymous'
            return {
                'backend': str(GLOBAL_PROGRESS_BACKEND).strip().lower(),
                'project_id': str(FIRESTORE_PROJECT_ID).strip(),
                'collection': str(FIRESTORE_COLLECTION).strip() or 'global_generation_progress',
                'document': str(FIRESTORE_DOCUMENT).strip() or str(global_target_id),
                'auth_mode': auth_mode,
                'credentials_path': credentials_path,
                'credentials_path_exists': bool(Path(credentials_path).exists()) if credentials_path else False,
                'api_key_set': bool(api_key),
            }

        def _firestore_monitor_error_hint(exc: Exception, debug_context: dict) -> str:
            text = f'{type(exc).__name__}: {exc}'.lower()
            auth_mode = str(debug_context.get('auth_mode', '')).strip().lower()
            if auth_mode == 'adc' and ('metadata.google.internal' in text or 'compute engine metadata' in text):
                return 'ADC indisponible ici; renseigne FIRESTORE_API_KEY ou FIRESTORE_CREDENTIALS_PATH.'
            if 'permissiondenied' in text or 'permission denied' in text:
                return 'Acces refuse par les regles Firestore.'
            if 'unauthenticated' in text or 'invalid authentication credentials' in text:
                return 'Authentification Firestore invalide.'
            if 'deadlineexceeded' in text or 'timeout' in text:
                return 'Timeout reseau vers Firestore.'
            if 'serviceunavailable' in text:
                return 'Service Firestore temporairement indisponible.'
            return ''

        def _load_global_progress_payload(*, drive_root: str, global_target_id: str, target_samples: int) -> dict:
            fallback_payload = {
                'global_target_id': str(global_target_id),
                'target_samples': int(target_samples),
                'total_samples': 0,
                'total_games': 0,
                'workers': {},
                'updated_at': '<none>',
            }
            backend = str(GLOBAL_PROGRESS_BACKEND).strip().lower()
            if backend == 'firestore':
                debug_context = _firestore_monitor_debug_context(global_target_id=str(global_target_id))
                try:
                    from google.cloud import firestore
                    from google.auth.credentials import AnonymousCredentials
                    from google.api_core.client_options import ClientOptions
                    credentials_path = str(FIRESTORE_CREDENTIALS_PATH).strip()
                    project_id = str(FIRESTORE_PROJECT_ID).strip()
                    collection = str(FIRESTORE_COLLECTION).strip() or 'global_generation_progress'
                    document = str(FIRESTORE_DOCUMENT).strip() or str(global_target_id)
                    if credentials_path:
                        if not Path(credentials_path).exists():
                            raise FileNotFoundError(f'Credentials introuvables: {credentials_path}')
                        from google.oauth2 import service_account
                        creds = service_account.Credentials.from_service_account_file(credentials_path)
                        client = firestore.Client(project=(project_id or None), credentials=creds)
                    elif str(FIRESTORE_API_KEY).strip():
                        creds = AnonymousCredentials()
                        client = firestore.Client(
                            project=(project_id or None),
                            credentials=creds,
                            client_options=ClientOptions(api_key=str(FIRESTORE_API_KEY).strip()),
                        )
                    else:
                        client = firestore.Client(project=(project_id or None))
                    snap = client.collection(collection).document(document).get()
                    if snap.exists:
                        return _normalize_global_payload(snap.to_dict(), int(target_samples))
                    return fallback_payload
                except Exception as exc:
                    hint = _firestore_monitor_error_hint(exc, debug_context)
                    context_text = json.dumps(debug_context, ensure_ascii=True, sort_keys=True)
                    message = (
                        f'Lecture Firestore impossible pour global progress (fallback fichier desactive) | '
                        f'context={context_text} | cause={type(exc).__name__}: {exc}'
                    )
                    if hint:
                        message = f'{message} | hint={hint}'
                    raise RuntimeError(message) from exc
            raise RuntimeError(f'GLOBAL_PROGRESS_BACKEND non supporte: {backend} (mode firestore requis).')

        def _active_worker_tags_from_global_progress(
            *,
            drive_root: str,
            global_target_id: str,
            base_source_id: str,
            active_minutes: float,
        ) -> set[str]:
            payload = _load_global_progress_payload(
                drive_root=drive_root,
                global_target_id=global_target_id,
                target_samples=int(GLOBAL_TARGET_SAMPLES),
            )
            workers = payload.get('workers', {})
            if not isinstance(workers, dict):
                return set()
            active_seconds = max(60.0, float(active_minutes) * 60.0)
            now_ts = time.time()
            tags: set[str] = set()
            source_prefix = f'{base_source_id}_'
            for _job_id, info in workers.items():
                if not isinstance(info, dict):
                    continue
                updated_ts = _parse_iso_to_epoch(info.get('updated_at'))
                if updated_ts <= 0:
                    continue
                age_seconds = now_ts - updated_ts
                if age_seconds > active_seconds:
                    continue
                source_id = str(info.get('dataset_source_id', '')).strip()
                if not source_id.startswith(source_prefix):
                    continue
                tag = source_id[len(source_prefix):].strip()
                if tag:
                    tags.add(tag)
            return tags

        def _detect_resumable_worker_tag(
            *,
            drive_root: str,
            base_source_id: str,
            base_build_id: str,
            stale_minutes: float,
            excluded_tags: set[str] | None = None,
        ) -> str | None:
            registry_path = Path(drive_root) / 'data' / 'dataset_registry.json'
            if not registry_path.exists():
                return None
            try:
                registry = json.loads(registry_path.read_text(encoding='utf-8'))
            except Exception:
                return None
            if not isinstance(registry, dict):
                return None

            candidates: dict[str, dict[str, float]] = {}
            source_prefix = f'{base_source_id}_'
            build_prefix = f'{base_build_id}_'

            for item in registry.get('dataset_sources', []):
                if not isinstance(item, dict):
                    continue
                source_id = str(item.get('dataset_source_id', ''))
                if not source_id.startswith(source_prefix):
                    continue
                tag = source_id[len(source_prefix):].strip()
                if not tag:
                    continue
                source_status = str(item.get('source_status', 'partial')).strip().lower()
                if source_status == 'completed':
                    continue
                updated = _parse_iso_to_epoch(item.get('updated_at'))
                samples = float(int(item.get('sampled_positions', 0)))
                cur = candidates.get(tag, {'updated': 0.0, 'samples': 0.0})
                cur['updated'] = max(float(cur.get('updated', 0.0)), float(updated))
                cur['samples'] = max(float(cur.get('samples', 0.0)), float(samples))
                candidates[tag] = cur

            for item in registry.get('built_datasets', []):
                if not isinstance(item, dict):
                    continue
                dataset_id = str(item.get('dataset_id', ''))
                if not dataset_id.startswith(build_prefix):
                    continue
                tag = dataset_id[len(build_prefix):].strip()
                if not tag:
                    continue
                build_status = str(item.get('build_status', 'partial')).strip().lower()
                if build_status == 'completed':
                    continue
                updated = _parse_iso_to_epoch(item.get('updated_at'))
                samples = float(int(item.get('labeled_samples', 0)))
                cur = candidates.get(tag, {'updated': 0.0, 'samples': 0.0})
                cur['updated'] = max(float(cur.get('updated', 0.0)), float(updated))
                cur['samples'] = max(float(cur.get('samples', 0.0)), float(samples))
                candidates[tag] = cur

            if not candidates:
                return None

            stale_seconds = max(60.0, float(stale_minutes) * 60.0)
            now_ts = time.time()
            eligible: list[tuple[str, dict[str, float]]] = []
            excluded = set(excluded_tags or set())
            for tag, meta in candidates.items():
                if tag in excluded:
                    continue
                updated = float(meta.get('updated', 0.0))
                if updated <= 0:
                    continue
                if (now_ts - updated) >= stale_seconds:
                    eligible.append((tag, meta))

            if not eligible:
                return None

            eligible.sort(key=lambda item: (float(item[1].get('samples', 0.0)), float(item[1].get('updated', 0.0))), reverse=True)
            return str(eligible[0][0])

        RESUMED_FROM_CHECKPOINT = False
        active_worker_tags = _active_worker_tags_from_global_progress(
            drive_root=DRIVE_ROOT,
            global_target_id=GLOBAL_TARGET_ID,
            base_source_id=BASE_DATASET_SOURCE_ID,
            active_minutes=float(ACTIVE_WORKER_GUARD_MINUTES),
        )

        if not WORKER_TAG:
            if AUTO_REUSE_CHECKPOINT_WORKER_TAG:
                resumed = _detect_resumable_worker_tag(
                    drive_root=DRIVE_ROOT,
                    base_source_id=BASE_DATASET_SOURCE_ID,
                    base_build_id=BASE_DATASET_BUILD_ID,
                    stale_minutes=float(CHECKPOINT_STALE_MINUTES),
                    excluded_tags=active_worker_tags,
                )
                if resumed:
                    WORKER_TAG = resumed
                    RESUMED_FROM_CHECKPOINT = True
            if not WORKER_TAG:
                WORKER_TAG = socket.gethostname().strip().lower().replace('-', '_')
                WORKER_TAG = re.sub(r'[^a-z0-9_]+', '_', WORKER_TAG).strip('_') or 'worker'
                if WORKER_TAG in active_worker_tags:
                    suffix = 1
                    candidate = f'{WORKER_TAG}_{suffix}'
                    while candidate in active_worker_tags:
                        suffix += 1
                        candidate = f'{WORKER_TAG}_{suffix}'
                    WORKER_TAG = candidate

        if AUTO_ASSIGN_WORKER_INDEX and int(WORKER_COUNT) > 1 and int(WORKER_INDEX) < 0:
            WORKER_INDEX = _auto_assign_worker_index(
                drive_root=DRIVE_ROOT,
                global_target_id=GLOBAL_TARGET_ID,
                worker_tag=WORKER_TAG,
                worker_count=int(WORKER_COUNT),
            )
        elif int(WORKER_INDEX) < 0:
            WORKER_INDEX = 0

        if MULTI_COLAB_SAFE_MODE:
            DATASET_SOURCE_ID = f'{BASE_DATASET_SOURCE_ID}_{WORKER_TAG}'
            DATASET_BUILD_ID = f'{BASE_DATASET_BUILD_ID}_{WORKER_TAG}'
            TRAIN_CONTINUE_JOB_ID = f'{TRAIN_CONTINUE_JOB_ID}_{WORKER_TAG}'
            TRAIN_SCRATCH_JOB_ID = f'{TRAIN_SCRATCH_JOB_ID}_{WORKER_TAG}'
            EVALUATION_JOB_ID = f'{EVALUATION_JOB_ID}_{WORKER_TAG}'
            BENCHMARK_JOB_ID = f'{BENCHMARK_JOB_ID}_{WORKER_TAG}'
        else:
            DATASET_SOURCE_ID = BASE_DATASET_SOURCE_ID
            DATASET_BUILD_ID = BASE_DATASET_BUILD_ID

        PIPELINE_MANIFEST_PATH = f'logs/pipeline/latest_dataset_pipeline_{WORKER_TAG}.json'

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
        print('MULTI_COLAB_SAFE_MODE   =', MULTI_COLAB_SAFE_MODE)
        print('WORKER_TAG              =', WORKER_TAG)
        print('WORKER_COUNT            =', WORKER_COUNT)
        print('WORKER_INDEX            =', WORKER_INDEX)
        print('AUTO_ASSIGN_WORKER_INDEX =', AUTO_ASSIGN_WORKER_INDEX)
        print('AUTO_REUSE_CHECKPOINT_WORKER_TAG =', AUTO_REUSE_CHECKPOINT_WORKER_TAG)
        print('ACTIVE_WORKER_GUARD_MINUTES =', ACTIVE_WORKER_GUARD_MINUTES)
        print('CHECKPOINT_STALE_MINUTES =', CHECKPOINT_STALE_MINUTES)
        print('active_worker_tags_before_assign =', sorted(active_worker_tags))
        print('RESUMED_FROM_CHECKPOINT =', RESUMED_FROM_CHECKPOINT)
        print('PIPELINE_MANIFEST_PATH  =', PIPELINE_MANIFEST_PATH)
        print('TARGET_SAMPLES          =', TARGET_SAMPLES)
        print('TARGET_LABELED_SAMPLES  =', TARGET_LABELED_SAMPLES)
        print('AUTO_TUNE_RESOURCES     =', AUTO_TUNE_RESOURCES)
        print('DATASET_GENERATE_WORKERS =', DATASET_GENERATE_WORKERS)
        print('DATASET_BUILD_WORKERS    =', DATASET_BUILD_WORKERS)
        print('BENCHMATCH_SHUFFLE_MATCHUPS =', BENCHMATCH_SHUFFLE_MATCHUPS)
        print('DATASET_BUILD_DEDUPE_SAMPLE_IDS =', DATASET_BUILD_DEDUPE_SAMPLE_IDS)
        print('DATASET_BUILD_ADAPTIVE_POLLING =', DATASET_BUILD_ADAPTIVE_POLLING)
        print('GLOBAL_TARGET_ENABLED    =', GLOBAL_TARGET_ENABLED)
        print('GLOBAL_TARGET_ID         =', GLOBAL_TARGET_ID)
        print('GLOBAL_TARGET_SAMPLES    =', GLOBAL_TARGET_SAMPLES)
        print('GLOBAL_PROGRESS_BACKEND  =', GLOBAL_PROGRESS_BACKEND)
        print('FIRESTORE_PROJECT_ID     =', FIRESTORE_PROJECT_ID)
        print('FIRESTORE_COLLECTION     =', FIRESTORE_COLLECTION)
        print('FIRESTORE_DOCUMENT       =', FIRESTORE_DOCUMENT)
        print('FIRESTORE_API_KEY_SET    =', bool(str(FIRESTORE_API_KEY).strip()))
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
        skipped_unavailable_model_ids = []
        discovered_model_rows = []
        discovered_model_map = {}
        registry_sort_ts_by_id = {}
        for item in model_registry.get('models', []):
            model_id = str(item.get('model_id', '')).strip()
            if not model_id:
                continue
            sort_ts = float(item.get('sort_ts', 0.0) or 0.0)
            if sort_ts > registry_sort_ts_by_id.get(model_id, 0.0):
                registry_sort_ts_by_id[model_id] = sort_ts

        model_scan_dir = Path(DRIVE_ROOT) / str(BENCHMATCH_MODEL_SCAN_DIR).strip()
        if model_scan_dir.exists():
            for ckpt_path in model_scan_dir.glob('*.pt'):
                model_id = ckpt_path.stem.strip()
                if not model_id:
                    continue
                if BENCHMATCH_MODEL_PREFIX and not model_id.startswith(BENCHMATCH_MODEL_PREFIX):
                    continue
                if model_id in BENCHMATCH_EXCLUDE_MODEL_IDS:
                    continue
                sort_ts = float(registry_sort_ts_by_id.get(model_id, float(ckpt_path.stat().st_mtime)))
                existing = discovered_model_map.get(model_id)
                if existing is None or sort_ts >= float(existing.get('sort_ts', 0.0)):
                    row = {
                        'model_id': model_id,
                        'checkpoint_path': str(ckpt_path),
                        'sort_ts': sort_ts,
                        'source': 'filesystem',
                    }
                    discovered_model_map[model_id] = row

        # Fallback: si le dossier scanne est vide/incomplet, on complete avec le registre (checkpoint existant).
        for item in model_registry.get('models', []):
            model_id = str(item.get('model_id', '')).strip()
            checkpoint_value = str(item.get('checkpoint_path', '')).strip()
            if not model_id or not checkpoint_value:
                continue
            if BENCHMATCH_MODEL_PREFIX and not model_id.startswith(BENCHMATCH_MODEL_PREFIX):
                continue
            if model_id in BENCHMATCH_EXCLUDE_MODEL_IDS:
                continue
            checkpoint_path = Path(checkpoint_value)
            if not checkpoint_path.exists():
                continue
            sort_ts = float(item.get('sort_ts', checkpoint_path.stat().st_mtime) or checkpoint_path.stat().st_mtime)
            existing = discovered_model_map.get(model_id)
            if existing is None or sort_ts >= float(existing.get('sort_ts', 0.0)):
                row = {
                    'model_id': model_id,
                    'checkpoint_path': str(checkpoint_path),
                    'sort_ts': sort_ts,
                    'source': 'registry',
                }
                discovered_model_map[model_id] = row

        discovered_model_rows = list(discovered_model_map.values())
        order_value = str(BENCHMATCH_MODEL_ORDER).strip().lower()
        newest_first = order_value != 'oldest_first'
        discovered_model_rows = sorted(
            discovered_model_rows,
            key=lambda item: (float(item.get('sort_ts', 0.0)), str(item.get('model_id', ''))),
            reverse=newest_first,
        )
        available_discovered_model_ids = {str(item.get('model_id', '')).strip() for item in discovered_model_rows if str(item.get('model_id', '')).strip()}

        def _maybe_add_model(model_id: str):
            model_id = str(model_id).strip()
            if not model_id:
                return
            if BENCHMATCH_MODEL_PREFIX and not model_id.startswith(BENCHMATCH_MODEL_PREFIX):
                return
            if model_id in BENCHMATCH_EXCLUDE_MODEL_IDS or model_id in seen_model_ids:
                return
            if model_id not in available_discovered_model_ids:
                skipped_unavailable_model_ids.append(model_id)
                return
            seen_model_ids.add(model_id)
            selected_model_ids.append(model_id)

        if BENCHMATCH_INCLUDE_ALL_REGISTERED_MODELS:
            discovered_ids = [str(item.get('model_id', '')).strip() for item in discovered_model_rows]
            for model_id in discovered_ids:
                _maybe_add_model(model_id)

        for model_id in BENCHMATCH_MODEL_IDS:
            _maybe_add_model(model_id)

        if BENCHMATCH_MODEL_LIMIT > 0:
            selected_model_ids = selected_model_ids[:BENCHMATCH_MODEL_LIMIT]
        skipped_unavailable_model_ids = sorted({mid for mid in skipped_unavailable_model_ids if mid}, key=_version_sort_key)

        bench_agents = list(BENCHMATCH_CLASSIC_AGENTS) + [f'model:{model_id}' for model_id in selected_model_ids]
        all_matchups = []
        if BENCHMATCH_ORDERED_MATCHUPS:
            for agent_a in bench_agents:
                for agent_b in bench_agents:
                    if not BENCHMATCH_INCLUDE_SELF_PLAY and agent_a == agent_b:
                        continue
                    all_matchups.append(f'{agent_a} vs {agent_b}')
        else:
            for idx, agent_a in enumerate(bench_agents):
                start = idx if BENCHMATCH_INCLUDE_SELF_PLAY else idx + 1
                for agent_b in bench_agents[start:]:
                    if not BENCHMATCH_INCLUDE_SELF_PLAY and agent_a == agent_b:
                        continue
                    all_matchups.append(f'{agent_a} vs {agent_b}')
        if BENCHMATCH_SHUFFLE_MATCHUPS:
            shuffle_seed = int(hashlib.sha1(f'{GLOBAL_TARGET_ID}:matchups'.encode('utf-8')).hexdigest()[:8], 16)
            all_matchups = sorted(
                all_matchups,
                key=lambda spec: hashlib.sha1(f'{shuffle_seed}:{spec}'.encode('utf-8')).hexdigest(),
            )

        worker_count = max(1, int(WORKER_COUNT))
        worker_index = int(WORKER_INDEX)
        if worker_count > 1:
            if worker_index < 0 or worker_index >= worker_count:
                raise ValueError(f'WORKER_INDEX doit etre dans [0, {worker_count - 1}], recu={worker_index}')
            matchups = [spec for idx, spec in enumerate(all_matchups) if (idx % worker_count) == worker_index]
            if not matchups:
                raise ValueError(
                    f'Ce worker n a recu aucun matchup (worker_index={worker_index}, worker_count={worker_count}, total_matchups={len(all_matchups)}).'
                )
        else:
            matchups = list(all_matchups)

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
        generate_block['global_target_enabled'] = bool(GLOBAL_TARGET_ENABLED)
        generate_block['global_target_id'] = str(GLOBAL_TARGET_ID)
        generate_block['global_target_samples'] = int(GLOBAL_TARGET_SAMPLES)
        generate_block['global_progress_path'] = str(GLOBAL_PROGRESS_PATH)
        generate_block['global_progress_backend'] = str(GLOBAL_PROGRESS_BACKEND)
        generate_block['global_progress_firestore_enabled'] = str(GLOBAL_PROGRESS_BACKEND).strip().lower() == 'firestore'
        generate_block['global_progress_firestore_project_id'] = str(FIRESTORE_PROJECT_ID)
        generate_block['global_progress_firestore_collection'] = str(FIRESTORE_COLLECTION)
        generate_block['global_progress_firestore_document'] = str(FIRESTORE_DOCUMENT)
        generate_block['global_progress_firestore_credentials_path'] = str(FIRESTORE_CREDENTIALS_PATH)
        generate_block['global_progress_firestore_api_key'] = str(FIRESTORE_API_KEY)
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
        build_block['dedupe_sample_ids'] = bool(DATASET_BUILD_DEDUPE_SAMPLE_IDS)
        build_block['adaptive_source_polling'] = bool(DATASET_BUILD_ADAPTIVE_POLLING)
        build_block['source_poll_interval_min_seconds'] = float(max(5.0, SOURCE_POLL_INTERVAL_SECONDS / 2.0))
        build_block['source_poll_interval_max_seconds'] = float(max(60.0, SOURCE_POLL_INTERVAL_SECONDS * 4.0))
        build_block['stop_when_global_target_reached'] = bool(GLOBAL_TARGET_ENABLED)
        build_block['global_target_id'] = str(GLOBAL_TARGET_ID)
        build_block['global_target_progress_path'] = str(GLOBAL_PROGRESS_PATH)
        build_block['global_target_progress_backend'] = str(GLOBAL_PROGRESS_BACKEND)
        build_block['global_target_progress_firestore_enabled'] = str(GLOBAL_PROGRESS_BACKEND).strip().lower() == 'firestore'
        build_block['global_target_progress_firestore_project_id'] = str(FIRESTORE_PROJECT_ID)
        build_block['global_target_progress_firestore_collection'] = str(FIRESTORE_COLLECTION)
        build_block['global_target_progress_firestore_document'] = str(FIRESTORE_DOCUMENT)
        build_block['global_target_progress_firestore_credentials_path'] = str(FIRESTORE_CREDENTIALS_PATH)
        build_block['global_target_progress_firestore_api_key'] = str(FIRESTORE_API_KEY)
        build_block['global_target_samples'] = int(GLOBAL_TARGET_SAMPLES)
        build_block['global_target_stabilization_polls'] = int(GLOBAL_TARGET_STABILIZATION_POLLS)
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
        print('GLOBAL_PROGRESS_BACKEND        =', GLOBAL_PROGRESS_BACKEND)
        print('FIRESTORE_PROJECT_ID           =', FIRESTORE_PROJECT_ID)
        print('FIRESTORE_COLLECTION           =', FIRESTORE_COLLECTION)
        print('FIRESTORE_DOCUMENT             =', FIRESTORE_DOCUMENT)
        print('FIRESTORE_API_KEY_SET          =', bool(str(FIRESTORE_API_KEY).strip()))
        print('TRAIN_CONTINUE_CONFIG_ACTIVE   =', TRAIN_CONTINUE_CONFIG_ACTIVE)
        print('TRAIN_SCRATCH_CONFIG_ACTIVE    =', TRAIN_SCRATCH_CONFIG_ACTIVE)
        print('TRAIN_CONTINUE_20M_CONFIG_ACTIVE =', TRAIN_CONTINUE_20M_CONFIG_ACTIVE)
        print('TRAIN_SCRATCH_20M_CONFIG_ACTIVE  =', TRAIN_SCRATCH_20M_CONFIG_ACTIVE)
        print('EVALUATION_20M_CONFIG_ACTIVE     =', EVALUATION_20M_CONFIG_ACTIVE)
        print('EVALUATION_CONFIG_ACTIVE       =', EVALUATION_CONFIG_ACTIVE)
        print('BENCHMARK_CONFIG_ACTIVE        =', BENCHMARK_CONFIG_ACTIVE)
        print('output_raw_dir                 =', output_raw_dir)
        print('output_sampled_dir             =', output_sampled_dir)
        print('model_scan_dir                 =', model_scan_dir)
        print('model_order                    =', 'newest_first' if newest_first else 'oldest_first')
        print('discovered_model_ids           =', [str(item.get('model_id', '')) for item in discovered_model_rows])
        print('selected_model_ids             =', selected_model_ids)
        print('skipped_missing_model_ids      =', skipped_unavailable_model_ids)
        print('total_agents                   =', len(bench_agents))
        print('total_matchups_all             =', len(all_matchups))
        print('total_matchups_worker          =', len(matchups))
        print('export_partial_every_n_files   =', build_block.get('export_partial_every_n_files'))
        print('dedupe_sample_ids              =', build_block.get('dedupe_sample_ids'))
        print('adaptive_source_polling        =', build_block.get('adaptive_source_polling'))
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
        latest_path = Path(DRIVE_ROOT) / PIPELINE_MANIFEST_PATH
        latest_path.parent.mkdir(parents=True, exist_ok=True)
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
        manifest_path = Path(DRIVE_ROOT) / PIPELINE_MANIFEST_PATH

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

        def _load_json_retry(path: Path, retries: int = 6, wait_seconds: float = 0.25, default=None):
            fallback = {} if default is None else default
            last_exc = None
            for attempt in range(retries):
                try:
                    return json.loads(path.read_text(encoding='utf-8'))
                except (FileNotFoundError, OSError):
                    if attempt + 1 >= retries:
                        return fallback
                    time.sleep(wait_seconds)
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    time.sleep(wait_seconds)
            if last_exc is not None:
                return fallback
            return fallback

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

        def _load_json_retry(path: Path, retries: int = 6, wait_seconds: float = 0.25, default=None):
            fallback = {} if default is None else default
            last_exc = None
            for attempt in range(retries):
                try:
                    return json.loads(path.read_text(encoding='utf-8'))
                except (FileNotFoundError, OSError):
                    if attempt + 1 >= retries:
                        return fallback
                    time.sleep(wait_seconds)
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    time.sleep(wait_seconds)
            if last_exc is not None:
                return fallback
            return fallback

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
        # KPI global: progression agregee de tous les workers
        import json
        import time
        from pathlib import Path

        REFRESH_SECONDS = 15
        MAX_LOOPS = 40

        registry_path = Path(DRIVE_ROOT) / 'data' / 'dataset_registry.json'

        def _load_json_retry(path: Path, retries: int = 6, wait_seconds: float = 0.25, default=None):
            fallback = {} if default is None else default
            last_exc = None
            for attempt in range(retries):
                try:
                    return json.loads(path.read_text(encoding='utf-8'))
                except (FileNotFoundError, OSError):
                    if attempt + 1 >= retries:
                        return fallback
                    time.sleep(wait_seconds)
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    time.sleep(wait_seconds)
            if last_exc is not None:
                return fallback
            return fallback

        def _safe_pct(value: int, target: int) -> float:
            if target <= 0:
                return 0.0
            return (100.0 * float(value)) / float(target)

        for loop_idx in range(MAX_LOOPS):
            global_payload = _load_global_progress_payload(
                drive_root=DRIVE_ROOT,
                global_target_id=GLOBAL_TARGET_ID,
                target_samples=int(GLOBAL_TARGET_SAMPLES),
            )

            global_total_samples = int(global_payload.get('total_samples', 0))
            global_total_games = int(global_payload.get('total_games', 0))
            global_target_samples = int(global_payload.get('target_samples', GLOBAL_TARGET_SAMPLES))
            workers_payload = global_payload.get('workers', {})
            if not isinstance(workers_payload, dict):
                workers_payload = {}
            global_workers = len(workers_payload)

            built_total_samples = 0
            built_worker_datasets = 0
            if registry_path.exists():
                registry = _load_json_retry(registry_path)
                for item in registry.get('built_datasets', []):
                    dataset_id = str(item.get('dataset_id', ''))
                    if dataset_id == BASE_DATASET_BUILD_ID or dataset_id.startswith(f'{BASE_DATASET_BUILD_ID}_'):
                        built_total_samples += int(item.get('labeled_samples', 0))
                        built_worker_datasets += 1

            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            print(
                f'[{ts}] GLOBAL generate_samples={global_total_samples}/{global_target_samples} '
                f'({_safe_pct(global_total_samples, global_target_samples):.2f}%) | '
                f'global_games={global_total_games} | workers={global_workers} | '
                f'updated_at={global_payload.get("updated_at", "<none>")}'
            )
            print(f'[{ts}] GLOBAL build_labeled_samples_sum={built_total_samples} | build_worker_datasets={built_worker_datasets}')
            print('-' * 120)

            if loop_idx >= (MAX_LOOPS - 1):
                break
            time.sleep(REFRESH_SECONDS)
        """
    ),
    code(
        """
        # Workers status: actif/inactif (vision globale multi-Colab)
        import json
        import time
        from datetime import datetime
        from pathlib import Path

        ACTIVE_THRESHOLD_SECONDS = 600

        def _load_json_retry(path: Path, retries: int = 6, wait_seconds: float = 0.25, default=None):
            fallback = {} if default is None else default
            last_exc = None
            for attempt in range(retries):
                try:
                    return json.loads(path.read_text(encoding='utf-8'))
                except (FileNotFoundError, OSError):
                    if attempt + 1 >= retries:
                        return fallback
                    time.sleep(wait_seconds)
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    time.sleep(wait_seconds)
            if last_exc is not None:
                return fallback
            return fallback

        def _parse_iso_to_epoch(value: object) -> float:
            text = str(value or '').strip()
            if not text:
                return 0.0
            try:
                return float(datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp())
            except Exception:
                return 0.0

        payload = _load_global_progress_payload(
            drive_root=DRIVE_ROOT,
            global_target_id=GLOBAL_TARGET_ID,
            target_samples=int(GLOBAL_TARGET_SAMPLES),
        )
        workers = payload.get('workers', {})
        if not isinstance(workers, dict) or not workers:
            print('Aucun worker global enregistre')
        else:
            now_ts = time.time()
            rows = []
            for worker_job_id, info in workers.items():
                if not isinstance(info, dict):
                    continue
                updated_at = str(info.get('updated_at', ''))
                updated_ts = _parse_iso_to_epoch(updated_at)
                age_seconds = (now_ts - updated_ts) if updated_ts > 0 else float('inf')
                status = 'active' if age_seconds <= ACTIVE_THRESHOLD_SECONDS else 'inactive'
                rows.append(
                    {
                        'status': status,
                        'age_seconds': age_seconds,
                        'worker_job_id': worker_job_id,
                        'dataset_source_id': str(info.get('dataset_source_id', '')),
                        'contributed_samples': int(info.get('contributed_samples', 0)),
                        'contributed_games': int(info.get('contributed_games', 0)),
                        'updated_at': updated_at or '<none>',
                    }
                )

            rows.sort(key=lambda row: (0 if row['status'] == 'active' else 1, row['age_seconds']))

            active_count = sum(1 for row in rows if row['status'] == 'active')
            inactive_count = len(rows) - active_count
            print(
                f'Workers global: total={len(rows)} | active={active_count} | inactive={inactive_count} '
                f'| active_threshold_seconds={ACTIVE_THRESHOLD_SECONDS}'
            )
            for row in rows:
                age_display = 'inf' if row['age_seconds'] == float('inf') else str(int(row['age_seconds']))
                print(
                    f"- status={row['status']:8s} | age_s={age_display:>5s} | "
                    f"job={row['worker_job_id']} | source={row['dataset_source_id']} | "
                    f"samples={row['contributed_samples']} | games={row['contributed_games']} | "
                    f"updated_at={row['updated_at']}"
                )
        """
    ),
    code(
        """
        # Health check compact: un seul bloc pour voir si le pipeline avance vraiment
        import json
        import time
        from datetime import datetime
        from pathlib import Path

        ACTIVE_THRESHOLD_SECONDS = 600
        registry_path = Path(DRIVE_ROOT) / 'data' / 'dataset_registry.json'
        health_state_path = Path(DRIVE_ROOT) / 'logs' / 'pipeline' / f'health_snapshot_{GLOBAL_TARGET_ID}_{WORKER_TAG}.json'

        def _load_json_retry(path: Path, retries: int = 6, wait_seconds: float = 0.25, default=None):
            fallback = {} if default is None else default
            last_exc = None
            for attempt in range(retries):
                try:
                    return json.loads(path.read_text(encoding='utf-8'))
                except (FileNotFoundError, OSError):
                    if attempt + 1 >= retries:
                        return fallback
                    time.sleep(wait_seconds)
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    time.sleep(wait_seconds)
            if last_exc is not None:
                return fallback
            return fallback

        def _parse_iso_to_epoch(value: object) -> float:
            text = str(value or '').strip()
            if not text:
                return 0.0
            try:
                return float(datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp())
            except Exception:
                return 0.0

        def _safe_pct(value: int, target: int) -> float:
            if target <= 0:
                return 0.0
            return (100.0 * float(value)) / float(target)

        payload = _load_global_progress_payload(
            drive_root=DRIVE_ROOT,
            global_target_id=GLOBAL_TARGET_ID,
            target_samples=int(GLOBAL_TARGET_SAMPLES),
        )
        workers = payload.get('workers', {})
        if not isinstance(workers, dict):
            workers = {}

        now_ts = time.time()
        active_rows = []
        inactive_rows = []
        for worker_job_id, info in workers.items():
            if not isinstance(info, dict):
                continue
            updated_at = str(info.get('updated_at', ''))
            updated_ts = _parse_iso_to_epoch(updated_at)
            age_seconds = (now_ts - updated_ts) if updated_ts > 0 else float('inf')
            row = {
                'job': worker_job_id,
                'source': str(info.get('dataset_source_id', '')),
                'samples': int(info.get('contributed_samples', 0)),
                'games': int(info.get('contributed_games', 0)),
                'updated_at': updated_at or '<none>',
                'age_seconds': age_seconds,
            }
            if age_seconds <= ACTIVE_THRESHOLD_SECONDS:
                active_rows.append(row)
            else:
                inactive_rows.append(row)

        active_rows.sort(key=lambda r: r['age_seconds'])
        inactive_rows.sort(key=lambda r: r['age_seconds'])

        build_total = 0
        build_workers = 0
        if registry_path.exists():
            registry = _load_json_retry(registry_path, default={'built_datasets': []})
            for item in registry.get('built_datasets', []):
                dataset_id = str(item.get('dataset_id', ''))
                if dataset_id == BASE_DATASET_BUILD_ID or dataset_id.startswith(f'{BASE_DATASET_BUILD_ID}_'):
                    build_total += int(item.get('labeled_samples', 0))
                    build_workers += 1

        current = {
            'ts': now_ts,
            'global_samples': int(payload.get('total_samples', 0)),
            'global_games': int(payload.get('total_games', 0)),
            'build_total': int(build_total),
            'active_workers': len(active_rows),
            'inactive_workers': len(inactive_rows),
        }
        previous = _load_json_retry(health_state_path, default={})
        prev_ts = float(previous.get('ts', 0.0) or 0.0)
        elapsed = max(0.0, current['ts'] - prev_ts) if prev_ts > 0 else 0.0
        previous_global_samples = int(previous.get('global_samples', 0) or 0)
        previous_global_games = int(previous.get('global_games', 0) or 0)
        previous_build_total = int(previous.get('build_total', 0) or 0)
        delta_global_samples = int(current['global_samples'] - previous_global_samples)
        delta_global_games = int(current['global_games'] - previous_global_games)
        delta_build_total = int(current['build_total'] - previous_build_total)
        regression_detected = (
            delta_global_samples < 0
            or delta_global_games < 0
            or delta_build_total < 0
        )

        health_state_path.parent.mkdir(parents=True, exist_ok=True)
        health_state_path.write_text(json.dumps(current, indent=2, ensure_ascii=True), encoding='utf-8')

        target = int(payload.get('target_samples', GLOBAL_TARGET_SAMPLES))
        global_updated_at = str(payload.get('updated_at', '<none>'))
        global_age = now_ts - _parse_iso_to_epoch(global_updated_at)
        global_age_display = 'inf' if global_age == float('inf') else str(int(max(0.0, global_age)))

        health = 'ok'
        if len(active_rows) == 0:
            health = 'critical'
        elif regression_detected:
            health = 'warning'
        elif elapsed > 0 and delta_global_samples <= 0 and delta_build_total <= 0:
            health = 'warning'

        print(
            f"HEALTH={health.upper()} | global={current['global_samples']}/{target} ({_safe_pct(current['global_samples'], target):.2f}%) "
            f"| games={current['global_games']} | build_sum={current['build_total']} | workers_active={len(active_rows)} "
            f"| workers_inactive={len(inactive_rows)} | global_age_s={global_age_display}"
        )
        if elapsed > 0:
            print(
                f"TREND last_{int(elapsed)}s | delta_global_samples={delta_global_samples} | "
                f"delta_global_games={delta_global_games} | delta_build_sum={delta_build_total}"
            )
        else:
            print('TREND: premier snapshot (relance la cellule pour voir les deltas)')
        if regression_detected:
            print(
                'TREND_NOTE: regression detectee (lecture/lag multi-Colab possible). '
                'Verifier avec la cellule Workers global qui reste la reference principale.'
            )

        for row in active_rows[:5]:
            print(
                f"ACTIVE | age_s={int(row['age_seconds'])} | samples={row['samples']} | games={row['games']} "
                f"| job={row['job']}"
            )
        if len(active_rows) > 5:
            print(f'ACTIVE | +{len(active_rows) - 5} autres workers actifs')
        for row in inactive_rows[:3]:
            age_display = 'inf' if row['age_seconds'] == float('inf') else str(int(row['age_seconds']))
            print(
                f"INACTIVE | age_s={age_display} | samples={row['samples']} | games={row['games']} "
                f"| job={row['job']}"
            )
        if len(inactive_rows) > 3:
            print(f'INACTIVE | +{len(inactive_rows) - 3} autres workers inactifs')
        """
    ),
    code(
        """
        import json
        from pathlib import Path

        LOG_TAIL_LINES = 40
        logs_dir = Path(DRIVE_ROOT) / 'logs' / 'pipeline'
        manifest_path = Path(DRIVE_ROOT) / PIPELINE_MANIFEST_PATH

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
        manifest_path = Path(DRIVE_ROOT) / PIPELINE_MANIFEST_PATH
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
        print('Cellule A: continue depuis best sur le plus grand dataset disponible')
        print('config =', TRAIN_CONTINUE_CONFIG_ACTIVE)
        !bash -lc "cd $WORKTREE && PYTHONPATH=$WORKTREE/src $PYTHON_BIN -m songo_model_stockfish.cli.main train --config $TRAIN_CONTINUE_CONFIG_ACTIVE --job-id $TRAIN_CONTINUE_JOB_ID"
        """
    ),
    code(
        """
        print('Cellule B: from scratch sur le plus grand dataset disponible')
        print('config =', TRAIN_SCRATCH_CONFIG_ACTIVE)
        !bash -lc "cd $WORKTREE && PYTHONPATH=$WORKTREE/src $PYTHON_BIN -m songo_model_stockfish.cli.main train --config $TRAIN_SCRATCH_CONFIG_ACTIVE --job-id $TRAIN_SCRATCH_JOB_ID"
        """
    ),
    md("## 8. Evaluation"),
    code(
        """
        # Selection automatique d'un modele compatible avec le dataset cible (input_dim identique)
        import json
        from pathlib import Path
        import numpy as np
        import torch
        import yaml

        base_eval_cfg_path = Path(EVALUATION_20M_CONFIG_ACTIVE)
        runtime_eval_cfg_path = Path(EVALUATION_20M_RUNTIME_CONFIG_PATH)
        eval_cfg = yaml.safe_load(base_eval_cfg_path.read_text(encoding='utf-8')) or {}
        eval_block = dict(eval_cfg.get('evaluation', {}) or {})
        requested_dataset_id = str(eval_block.get('dataset_id', DATASET_BUILD_ID))
        dataset_id = requested_dataset_id

        registry_path = Path(DRIVE_ROOT) / 'data' / 'dataset_registry.json'
        registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {'built_datasets': []}
        built_entries = [item for item in registry.get('built_datasets', []) if isinstance(item, dict)]

        def _entry_output_dir(entry: dict) -> Path:
            return Path(str(entry.get('output_dir', '')).strip())

        def _has_test_npz(entry: dict) -> bool:
            output_dir = _entry_output_dir(entry)
            return output_dir.exists() and (output_dir / 'test.npz').exists()

        built = next(
            (
                item
                for item in built_entries
                if str(item.get('dataset_id', '')).strip() == dataset_id and _has_test_npz(item)
            ),
            None,
        )

        if built is None:
            prefix = f'{BASE_DATASET_BUILD_ID}_'
            shard_candidates = [
                item
                for item in built_entries
                if _has_test_npz(item)
                and (
                    str(item.get('dataset_id', '')).strip() == BASE_DATASET_BUILD_ID
                    or str(item.get('dataset_id', '')).strip().startswith(prefix)
                )
            ]
            if shard_candidates:
                shard_candidates = sorted(
                    shard_candidates,
                    key=lambda item: (
                        int(item.get('labeled_samples', 0) or 0),
                        str(item.get('updated_at', '')),
                    ),
                    reverse=True,
                )
                built = shard_candidates[0]
                dataset_id = str(built.get('dataset_id', dataset_id)).strip() or dataset_id

        if built is None:
            datasets_root = Path(DRIVE_ROOT) / 'data' / 'datasets'
            dir_candidates = []
            if datasets_root.exists():
                for path in datasets_root.iterdir():
                    if not path.is_dir():
                        continue
                    name = path.name
                    if name == requested_dataset_id or name == BASE_DATASET_BUILD_ID or name.startswith(f'{BASE_DATASET_BUILD_ID}_'):
                        if (path / 'test.npz').exists():
                            dir_candidates.append(path)
            if dir_candidates:
                selected_dir = sorted(dir_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                built = {
                    'dataset_id': selected_dir.name,
                    'output_dir': str(selected_dir),
                    'input_dim': 0,
                    'labeled_samples': 0,
                    'updated_at': '',
                }
                dataset_id = selected_dir.name

        if built is None:
            raise ValueError(
                f'Dataset introuvable pour evaluation (requested={requested_dataset_id}, base={BASE_DATASET_BUILD_ID}). '
                'Attends un snapshot build avec test.npz ou relance la cellule de configuration.'
            )

        dataset_input_dim = int(built.get('input_dim') or 0)
        test_npz_path = Path(str(built.get('output_dir', ''))) / 'test.npz'
        if dataset_input_dim <= 0:
            with np.load(test_npz_path, allow_pickle=True) as test_npz:
                dataset_input_dim = int(test_npz['x'].shape[1])

        models_root = Path(DRIVE_ROOT) / 'models'
        model_registry_path = models_root / 'model_registry.json'
        model_registry = json.loads(model_registry_path.read_text(encoding='utf-8')) if model_registry_path.exists() else {'models': []}
        candidates = sorted(model_registry.get('models', []), key=lambda item: float(item.get('sort_ts', 0.0)), reverse=True)

        selected = None
        for item in candidates:
            checkpoint_path = Path(str(item.get('checkpoint_path', '')).strip())
            if not checkpoint_path.exists():
                continue
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            except Exception:
                continue
            model_config = checkpoint.get('model_config', {})
            model_input_dim = int(model_config.get('input_dim', 0) or 0)
            if model_input_dim == dataset_input_dim:
                selected = {
                    'model_id': str(item.get('model_id', '')),
                    'checkpoint_path': str(checkpoint_path),
                    'input_dim': model_input_dim,
                }
                break

        if selected is None:
            raise ValueError(
                f'Aucun checkpoint compatible trouve pour dataset_id={dataset_id} (dataset_input_dim={dataset_input_dim}). '
                'Entraine d abord un modele sur ce dataset/schema.'
            )

        eval_block['dataset_selection_mode'] = 'configured'
        eval_block['dataset_id'] = dataset_id
        eval_block['test_dataset_path'] = str(test_npz_path)
        eval_block['model_id'] = selected['model_id']
        eval_block['checkpoint_path'] = selected['checkpoint_path']
        eval_cfg['evaluation'] = eval_block
        runtime_eval_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_eval_cfg_path.write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding='utf-8')

        print('Evaluation sur dataset 20M partiel configure')
        print('dataset_id =', dataset_id)
        print('dataset_input_dim =', dataset_input_dim)
        print('selected_model_id =', selected['model_id'])
        print('selected_checkpoint =', selected['checkpoint_path'])
        print('runtime config =', runtime_eval_cfg_path)
        !bash -lc "cd $WORKTREE && PYTHONPATH=$WORKTREE/src $PYTHON_BIN -m songo_model_stockfish.cli.main evaluate --config $EVALUATION_20M_RUNTIME_CONFIG_PATH --job-id $EVALUATION_JOB_ID"
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
