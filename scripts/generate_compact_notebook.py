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
        9. lancer un tournoi inter-modeles (3/1/0)
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
        import json
        import os
        from pathlib import Path

        def _as_bool(value, default=False):
            if isinstance(value, bool):
                return value
            if value is None:
                return bool(default)
            text = str(value).strip().lower()
            if text in {'1', 'true', 'yes', 'on', 'y', 't'}:
                return True
            if text in {'0', 'false', 'no', 'off', 'n', 'f'}:
                return False
            return bool(default)

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
        BENCHMATCH_CYCLE_MATCHUPS_UNTIL_TARGET = True
        BENCHMATCH_MAX_MATCHUP_CYCLES = 0  # 0 = illimite
        BENCHMATCH_MODEL_AGENT_DEVICE = 'cpu'
        DATASET_GENERATE_SOURCE_MODE = 'benchmatch'  # 'benchmatch' ou 'self_play_puct'
        SELF_PLAY_MODEL = 'auto_best'
        SELF_PLAY_MODEL_DEVICE = 'cpu'
        SELF_PLAY_GAMES_PER_CYCLE = 400
        SELF_PLAY_CYCLE_UNTIL_TARGET = True
        SELF_PLAY_MAX_MATCHUP_CYCLES = 0  # 0 = illimite
        SELF_PLAY_NUM_SIMULATIONS = 96
        SELF_PLAY_C_PUCT = 1.5
        SELF_PLAY_TEMPERATURE = 1.0
        SELF_PLAY_TEMPERATURE_END = 0.1
        SELF_PLAY_TEMPERATURE_DROP_PLY = 12
        SELF_PLAY_ROOT_DIRICHLET_ALPHA = 0.3
        SELF_PLAY_ROOT_DIRICHLET_EPSILON = 0.25
        SELF_PLAY_DETERMINISTIC = False
        DATASET_BUILD_MODE_OVERRIDE = ''  # '', 'auto', 'teacher_label', 'source_prelabeled'
        LOW_QUOTA_PROFILE = True
        LOW_QUOTA_GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES = 200
        LOW_QUOTA_GLOBAL_TARGET_POLL_INTERVAL_SECONDS = 60.0
        LOW_QUOTA_SOURCE_POLL_INTERVAL_SECONDS = 45
        LOW_QUOTA_DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES = 200
        LOW_QUOTA_MONITOR_REFRESH_SECONDS = 90
        LOW_QUOTA_MONITOR_MAX_LOOPS = 20
        LOW_QUOTA_WRITE_PIPELINE_MANIFEST_FIRESTORE = False
        GLOBAL_TARGET_ENABLED = True
        GLOBAL_TARGET_ID = 'bench_models_20m_global'
        GLOBAL_TARGET_SAMPLES = 20000000
        GLOBAL_PROGRESS_PATH = f'data/global_generation_progress/{GLOBAL_TARGET_ID}.json'
        GLOBAL_PROGRESS_BACKEND = 'firestore'  # Firestore est la source de verite
        GLOBAL_BUDGET_ENFORCEMENT_MODE = 'batched'  # 'batched' (recommande pour quota Firestore) ou 'strict'
        GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES = 20
        GLOBAL_TARGET_POLL_INTERVAL_SECONDS = 15.0
        FIRESTORE_PROJECT_ID = 'songo-model-ai'
        FIRESTORE_COLLECTION = 'global_generation_progress'
        FIRESTORE_DOCUMENT = GLOBAL_TARGET_ID
        FIRESTORE_CREDENTIALS_PATH = '/content/drive/MyDrive/songo-stockfish/secrets/songo-model-ai-firebase-adminsdk-fbsvc-b3ef3bdacb.json'  # Chemin JSON service account (requis pour Python Firestore)
        FIRESTORE_API_KEY = ''  # Non supporte par google-cloud-firestore (serveur)
        FIRESTORE_DATASET_REGISTRY_COLLECTION = 'dataset_registry'
        FIRESTORE_DATASET_REGISTRY_DOCUMENT = 'primary'
        FIRESTORE_WORKER_LEASES_COLLECTION = 'worker_leases'
        FIRESTORE_PIPELINE_MANIFESTS_COLLECTION = 'pipeline_manifests'
        FIRESTORE_WORKER_CHECKPOINTS_COLLECTION = 'worker_checkpoints'
        REDIS_SECRET_JSON_PATH = '/content/drive/MyDrive/songo-stockfish/secrets/upstash_redis.json'
        REDIS_URL_OVERRIDE = 'https://touching-sculpin-69695.upstash.io'
        REDIS_TOKEN_OVERRIDE = ''
        REDIS_SECRET_JSON_LOADED = False
        redis_secret_json_path = Path(REDIS_SECRET_JSON_PATH)
        if redis_secret_json_path.exists():
            try:
                secret_payload = json.loads(redis_secret_json_path.read_text(encoding='utf-8'))
                if isinstance(secret_payload, dict):
                    url_candidate = str(
                        secret_payload.get('UPSTASH_REDIS_REST_URL', '') or secret_payload.get('url', '')
                    ).strip()
                    token_candidate = str(
                        secret_payload.get('UPSTASH_REDIS_REST_TOKEN', '') or secret_payload.get('token', '')
                    ).strip()
                    if url_candidate and not os.environ.get('UPSTASH_REDIS_REST_URL'):
                        os.environ['UPSTASH_REDIS_REST_URL'] = url_candidate
                    if token_candidate and not os.environ.get('UPSTASH_REDIS_REST_TOKEN'):
                        os.environ['UPSTASH_REDIS_REST_TOKEN'] = token_candidate
                    REDIS_SECRET_JSON_LOADED = True
            except Exception:
                REDIS_SECRET_JSON_LOADED = False
        if str(REDIS_URL_OVERRIDE).strip():
            os.environ['UPSTASH_REDIS_REST_URL'] = str(REDIS_URL_OVERRIDE).strip()
        if str(REDIS_TOKEN_OVERRIDE).strip():
            os.environ['UPSTASH_REDIS_REST_TOKEN'] = str(REDIS_TOKEN_OVERRIDE).strip()
        GLOBAL_PROGRESS_REDIS_ENABLED = _as_bool(os.environ.get('GLOBAL_PROGRESS_REDIS_ENABLED', '1'), default=True)
        GLOBAL_PROGRESS_REDIS_URL = os.environ.get('UPSTASH_REDIS_REST_URL', '').strip()
        GLOBAL_PROGRESS_REDIS_TOKEN = os.environ.get('UPSTASH_REDIS_REST_TOKEN', '').strip()
        GLOBAL_PROGRESS_REDIS_KEY_PREFIX = f'songo:{GLOBAL_TARGET_ID}'
        GLOBAL_PROGRESS_REDIS_CACHE_TTL_SECONDS = 120
        GLOBAL_PROGRESS_WORKERS_RETENTION_SECONDS = 86400
        GLOBAL_PROGRESS_WORKERS_MAX_ENTRIES = 5000
        WORKER_CHECKPOINTS_MIN_INTERVAL_SECONDS = 60.0
        WORKER_CHECKPOINTS_STATE_ONLY_ON_CHANGE = True
        SOURCE_POLL_INTERVAL_SECONDS = 20
        DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES = 20
        MONITOR_REFRESH_SECONDS = 15
        MONITOR_MAX_LOOPS = 40
        PIPELINE_MANIFEST_FIRESTORE_WRITE_ENABLED = True
        if LOW_QUOTA_PROFILE:
            GLOBAL_BUDGET_ENFORCEMENT_MODE = 'batched'
            GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES = int(LOW_QUOTA_GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES)
            GLOBAL_TARGET_POLL_INTERVAL_SECONDS = float(LOW_QUOTA_GLOBAL_TARGET_POLL_INTERVAL_SECONDS)
            SOURCE_POLL_INTERVAL_SECONDS = int(LOW_QUOTA_SOURCE_POLL_INTERVAL_SECONDS)
            DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES = int(LOW_QUOTA_DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES)
            MONITOR_REFRESH_SECONDS = float(LOW_QUOTA_MONITOR_REFRESH_SECONDS)
            MONITOR_MAX_LOOPS = int(LOW_QUOTA_MONITOR_MAX_LOOPS)
            PIPELINE_MANIFEST_FIRESTORE_WRITE_ENABLED = bool(LOW_QUOTA_WRITE_PIPELINE_MANIFEST_FIRESTORE)
            GLOBAL_PROGRESS_REDIS_CACHE_TTL_SECONDS = max(120, int(GLOBAL_PROGRESS_REDIS_CACHE_TTL_SECONDS))
            WORKER_CHECKPOINTS_MIN_INTERVAL_SECONDS = max(60.0, float(WORKER_CHECKPOINTS_MIN_INTERVAL_SECONDS))
        if str(GLOBAL_PROGRESS_BACKEND).strip().lower() != 'firestore':
            raise ValueError("GLOBAL_PROGRESS_BACKEND doit rester 'firestore' (fallback fichier desactive).")
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

        if str(GLOBAL_PROGRESS_BACKEND).strip().lower() == 'firestore':
            if not str(FIRESTORE_CREDENTIALS_PATH).strip():
                raise RuntimeError('FIRESTORE_CREDENTIALS_PATH est vide en mode Firestore.')
            if not Path(str(FIRESTORE_CREDENTIALS_PATH)).exists():
                raise FileNotFoundError(f'FIRESTORE_CREDENTIALS_PATH introuvable: {FIRESTORE_CREDENTIALS_PATH}')

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
            try:
                from google.cloud import firestore
                credentials_path = str(FIRESTORE_CREDENTIALS_PATH).strip()
                project_id = str(FIRESTORE_PROJECT_ID).strip()
                collection = str(FIRESTORE_WORKER_LEASES_COLLECTION).strip() or 'worker_leases'
                document = str(global_target_id).strip() or 'default'
                if credentials_path:
                    from google.oauth2 import service_account
                    creds = service_account.Credentials.from_service_account_file(credentials_path)
                    client = firestore.Client(project=(project_id or None), credentials=creds)
                elif str(FIRESTORE_API_KEY).strip():
                    raise RuntimeError(
                        'Mode API key non supporte avec google-cloud-firestore; '
                        'renseigne FIRESTORE_CREDENTIALS_PATH.'
                    )
                else:
                    raise RuntimeError(
                        'Credentials Firestore absents; renseigne FIRESTORE_CREDENTIALS_PATH.'
                    )
                doc_ref = client.collection(collection).document(document)
                tx = client.transaction()

                @firestore.transactional
                def _assign(transaction):
                    snap = doc_ref.get(transaction=transaction)
                    payload = snap.to_dict() if snap.exists else {}
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
                            transaction.set(doc_ref, payload)
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
                    transaction.set(doc_ref, payload)
                    return int(assigned)

                return int(_assign(tx))
            except Exception:
                return abs(hash(worker_tag)) % worker_count

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

        def _read_global_progress_from_redis(*, global_target_id: str, target_samples: int) -> dict | None:
            if not bool(GLOBAL_PROGRESS_REDIS_ENABLED):
                return None
            redis_url = str(GLOBAL_PROGRESS_REDIS_URL).strip()
            redis_token = str(GLOBAL_PROGRESS_REDIS_TOKEN).strip()
            if not redis_url or not redis_token:
                return None
            try:
                from upstash_redis import Redis
                redis_client = Redis(url=redis_url, token=redis_token)
                key_prefix = str(GLOBAL_PROGRESS_REDIS_KEY_PREFIX).strip() or f'songo:{global_target_id}'
                key = f'{key_prefix}:global_progress'
                raw_payload = redis_client.get(key)
                if raw_payload is None:
                    return None
                if isinstance(raw_payload, str):
                    payload = json.loads(raw_payload)
                elif isinstance(raw_payload, dict):
                    payload = raw_payload
                else:
                    return None
                if not isinstance(payload, dict):
                    return None
                return _normalize_global_payload(payload, int(target_samples))
            except Exception:
                return None

        def _firestore_monitor_debug_context(*, global_target_id: str) -> dict:
            credentials_path = str(FIRESTORE_CREDENTIALS_PATH).strip()
            api_key = str(FIRESTORE_API_KEY).strip()
            auth_mode = 'missing_credentials'
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
            if auth_mode == 'api_key_anonymous':
                return 'Firestore Python ne supporte pas API key seule; configure FIRESTORE_CREDENTIALS_PATH (service account JSON).'
            if auth_mode == 'missing_credentials':
                return 'Credentials Firestore absents; configure FIRESTORE_CREDENTIALS_PATH (service account JSON).'
            if auth_mode == 'adc' and ('metadata.google.internal' in text or 'compute engine metadata' in text):
                return 'ADC indisponible ici; configure FIRESTORE_CREDENTIALS_PATH (service account JSON).'
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
            redis_payload = _read_global_progress_from_redis(
                global_target_id=str(global_target_id),
                target_samples=int(target_samples),
            )
            if redis_payload is not None:
                return redis_payload
            backend = str(GLOBAL_PROGRESS_BACKEND).strip().lower()
            if backend == 'firestore':
                debug_context = _firestore_monitor_debug_context(global_target_id=str(global_target_id))
                try:
                    from google.cloud import firestore
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
                        raise RuntimeError(
                            'Mode API key non supporte avec google-cloud-firestore; '
                            'renseigne FIRESTORE_CREDENTIALS_PATH.'
                        )
                    else:
                        raise RuntimeError(
                            'Credentials Firestore absents; renseigne FIRESTORE_CREDENTIALS_PATH.'
                        )
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

        def _get_firestore_client():
            from google.cloud import firestore
            credentials_path = str(FIRESTORE_CREDENTIALS_PATH).strip()
            project_id = str(FIRESTORE_PROJECT_ID).strip()
            if credentials_path:
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_file(credentials_path)
                return firestore.Client(project=(project_id or None), credentials=creds)
            if str(FIRESTORE_API_KEY).strip():
                raise RuntimeError(
                    'Mode API key non supporte avec google-cloud-firestore; '
                    'renseigne FIRESTORE_CREDENTIALS_PATH.'
                )
            raise RuntimeError(
                'Credentials Firestore absents; renseigne FIRESTORE_CREDENTIALS_PATH.'
            )

        def _read_firestore_doc(collection: str, document: str, default: dict | None = None) -> dict:
            fallback = {} if default is None else default
            client = _get_firestore_client()
            snap = client.collection(str(collection)).document(str(document)).get()
            if not snap.exists:
                return dict(fallback)
            payload = snap.to_dict() or {}
            return payload if isinstance(payload, dict) else dict(fallback)

        def _write_firestore_doc(collection: str, document: str, payload: dict) -> None:
            retries = 4
            wait_seconds = 1.0
            last_exc = None
            for attempt in range(retries):
                try:
                    client = _get_firestore_client()
                    client.collection(str(collection)).document(str(document)).set(dict(payload))
                    return
                except Exception as exc:
                    last_exc = exc
                    text = f'{type(exc).__name__}: {exc}'.lower()
                    retryable = (
                        'resourceexhausted' in text
                        or 'quota exceeded' in text
                        or 'deadlineexceeded' in text
                        or 'serviceunavailable' in text
                        or 'temporarily unavailable' in text
                    )
                    if (not retryable) or (attempt + 1 >= retries):
                        break
                    time.sleep(wait_seconds)
                    wait_seconds = min(10.0, wait_seconds * 2.0)
            if last_exc is None:
                raise RuntimeError('Ecriture Firestore impossible (erreur inconnue).')
            raise RuntimeError(f'Ecriture Firestore impossible: {type(last_exc).__name__}: {last_exc}') from last_exc

        def _load_dataset_registry_payload() -> dict:
            payload = _read_firestore_doc(
                FIRESTORE_DATASET_REGISTRY_COLLECTION,
                FIRESTORE_DATASET_REGISTRY_DOCUMENT,
                default={'dataset_sources': [], 'built_datasets': []},
            )
            payload.setdefault('dataset_sources', [])
            payload.setdefault('built_datasets', [])
            return payload

        def _load_pipeline_manifest_payload(worker_tag: str) -> dict:
            return _read_firestore_doc(
                FIRESTORE_PIPELINE_MANIFESTS_COLLECTION,
                str(worker_tag),
                default={},
            )

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
            try:
                registry = _load_dataset_registry_payload()
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
        print('DATASET_GENERATE_SOURCE_MODE =', DATASET_GENERATE_SOURCE_MODE)
        print('SELF_PLAY_MODEL         =', SELF_PLAY_MODEL)
        print('SELF_PLAY_GAMES_PER_CYCLE =', SELF_PLAY_GAMES_PER_CYCLE)
        print('SELF_PLAY_NUM_SIMULATIONS =', SELF_PLAY_NUM_SIMULATIONS)
        print('DATASET_BUILD_MODE_OVERRIDE =', DATASET_BUILD_MODE_OVERRIDE or '<auto>')
        print('LOW_QUOTA_PROFILE       =', LOW_QUOTA_PROFILE)
        print('SOURCE_POLL_INTERVAL_SECONDS =', SOURCE_POLL_INTERVAL_SECONDS)
        print('DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES =', DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES)
        print('MONITOR_REFRESH_SECONDS =', MONITOR_REFRESH_SECONDS)
        print('MONITOR_MAX_LOOPS       =', MONITOR_MAX_LOOPS)
        print('PIPELINE_MANIFEST_FIRESTORE_WRITE_ENABLED =', PIPELINE_MANIFEST_FIRESTORE_WRITE_ENABLED)
        print('AUTO_TUNE_RESOURCES     =', AUTO_TUNE_RESOURCES)
        print('DATASET_GENERATE_WORKERS =', DATASET_GENERATE_WORKERS)
        print('DATASET_BUILD_WORKERS    =', DATASET_BUILD_WORKERS)
        print('BENCHMATCH_SHUFFLE_MATCHUPS =', BENCHMATCH_SHUFFLE_MATCHUPS)
        print('BENCHMATCH_CYCLE_MATCHUPS_UNTIL_TARGET =', BENCHMATCH_CYCLE_MATCHUPS_UNTIL_TARGET)
        print('BENCHMATCH_MAX_MATCHUP_CYCLES =', BENCHMATCH_MAX_MATCHUP_CYCLES)
        print('DATASET_BUILD_DEDUPE_SAMPLE_IDS =', DATASET_BUILD_DEDUPE_SAMPLE_IDS)
        print('DATASET_BUILD_ADAPTIVE_POLLING =', DATASET_BUILD_ADAPTIVE_POLLING)
        print('GLOBAL_TARGET_ENABLED    =', GLOBAL_TARGET_ENABLED)
        print('GLOBAL_TARGET_ID         =', GLOBAL_TARGET_ID)
        print('GLOBAL_TARGET_SAMPLES    =', GLOBAL_TARGET_SAMPLES)
        print('GLOBAL_PROGRESS_BACKEND  =', GLOBAL_PROGRESS_BACKEND)
        print('GLOBAL_BUDGET_ENFORCEMENT_MODE =', GLOBAL_BUDGET_ENFORCEMENT_MODE)
        print('GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES =', GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES)
        print('GLOBAL_TARGET_POLL_INTERVAL_SECONDS =', GLOBAL_TARGET_POLL_INTERVAL_SECONDS)
        print('FIRESTORE_PROJECT_ID     =', FIRESTORE_PROJECT_ID)
        print('FIRESTORE_COLLECTION     =', FIRESTORE_COLLECTION)
        print('FIRESTORE_DOCUMENT       =', FIRESTORE_DOCUMENT)
        print('FIRESTORE_CREDENTIALS_PATH =', FIRESTORE_CREDENTIALS_PATH)
        print('FIRESTORE_CREDENTIALS_EXISTS =', Path(str(FIRESTORE_CREDENTIALS_PATH)).exists())
        print('FIRESTORE_API_KEY_SET    =', bool(str(FIRESTORE_API_KEY).strip()))
        print('GLOBAL_PROGRESS_REDIS_ENABLED =', GLOBAL_PROGRESS_REDIS_ENABLED)
        print('REDIS_SECRET_JSON_PATH   =', REDIS_SECRET_JSON_PATH)
        print('REDIS_SECRET_JSON_LOADED =', REDIS_SECRET_JSON_LOADED)
        print('GLOBAL_PROGRESS_REDIS_URL_SET =', bool(str(GLOBAL_PROGRESS_REDIS_URL).strip()))
        print('GLOBAL_PROGRESS_REDIS_TOKEN_SET =', bool(str(GLOBAL_PROGRESS_REDIS_TOKEN).strip()))
        print('GLOBAL_PROGRESS_REDIS_KEY_PREFIX =', GLOBAL_PROGRESS_REDIS_KEY_PREFIX)
        print('GLOBAL_PROGRESS_REDIS_CACHE_TTL_SECONDS =', GLOBAL_PROGRESS_REDIS_CACHE_TTL_SECONDS)
        print('GLOBAL_PROGRESS_WORKERS_RETENTION_SECONDS =', GLOBAL_PROGRESS_WORKERS_RETENTION_SECONDS)
        print('GLOBAL_PROGRESS_WORKERS_MAX_ENTRIES =', GLOBAL_PROGRESS_WORKERS_MAX_ENTRIES)
        print('WORKER_CHECKPOINTS_MIN_INTERVAL_SECONDS =', WORKER_CHECKPOINTS_MIN_INTERVAL_SECONDS)
        print('WORKER_CHECKPOINTS_STATE_ONLY_ON_CHANGE =', WORKER_CHECKPOINTS_STATE_ONLY_ON_CHANGE)
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
        generate_source_mode = str(DATASET_GENERATE_SOURCE_MODE).strip().lower() or 'benchmatch'
        if generate_source_mode not in {'benchmatch', 'self_play_puct'}:
            raise ValueError(f"DATASET_GENERATE_SOURCE_MODE non supporte: {generate_source_mode}")
        generate_block['source_mode'] = generate_source_mode
        generate_block['dataset_source_id'] = DATASET_SOURCE_ID
        generate_block['target_samples'] = int(TARGET_SAMPLES)
        generate_block['output_sampled_dir'] = output_sampled_dir
        generate_block['output_raw_dir'] = output_raw_dir
        generate_block['workers'] = int(DATASET_GENERATE_WORKERS)
        generate_block['max_pending_futures'] = int(DATASET_GENERATE_MAX_PENDING_FUTURES)
        generate_block['model_agent_device'] = str(BENCHMATCH_MODEL_AGENT_DEVICE)
        if generate_source_mode == 'benchmatch':
            generate_block['games'] = int(BENCHMATCH_GAMES)
            generate_block['matchups'] = matchups
            generate_block['cycle_matchups_until_target'] = bool(BENCHMATCH_CYCLE_MATCHUPS_UNTIL_TARGET)
            generate_block['max_matchup_cycles'] = int(BENCHMATCH_MAX_MATCHUP_CYCLES)
        else:
            generate_block.pop('matchups', None)
            generate_block['games'] = int(SELF_PLAY_GAMES_PER_CYCLE)
            generate_block['cycle_matchups_until_target'] = bool(SELF_PLAY_CYCLE_UNTIL_TARGET)
            generate_block['max_matchup_cycles'] = int(SELF_PLAY_MAX_MATCHUP_CYCLES)
            generate_block['self_play_model'] = str(SELF_PLAY_MODEL)
            generate_block['self_play_model_device'] = str(SELF_PLAY_MODEL_DEVICE)
            generate_block['self_play_num_simulations'] = int(SELF_PLAY_NUM_SIMULATIONS)
            generate_block['self_play_c_puct'] = float(SELF_PLAY_C_PUCT)
            generate_block['self_play_temperature'] = float(SELF_PLAY_TEMPERATURE)
            generate_block['self_play_temperature_end'] = float(SELF_PLAY_TEMPERATURE_END)
            generate_block['self_play_temperature_drop_ply'] = int(SELF_PLAY_TEMPERATURE_DROP_PLY)
            generate_block['self_play_root_dirichlet_alpha'] = float(SELF_PLAY_ROOT_DIRICHLET_ALPHA)
            generate_block['self_play_root_dirichlet_epsilon'] = float(SELF_PLAY_ROOT_DIRICHLET_EPSILON)
            generate_block['self_play_deterministic'] = bool(SELF_PLAY_DETERMINISTIC)
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
        generate_block['global_progress_redis_enabled'] = bool(GLOBAL_PROGRESS_REDIS_ENABLED)
        generate_block['global_progress_redis_url'] = str(GLOBAL_PROGRESS_REDIS_URL)
        generate_block['global_progress_redis_key_prefix'] = str(GLOBAL_PROGRESS_REDIS_KEY_PREFIX)
        generate_block['global_progress_redis_cache_ttl_seconds'] = int(GLOBAL_PROGRESS_REDIS_CACHE_TTL_SECONDS)
        generate_block['global_progress_workers_retention_seconds'] = int(GLOBAL_PROGRESS_WORKERS_RETENTION_SECONDS)
        generate_block['global_progress_workers_max_entries'] = int(GLOBAL_PROGRESS_WORKERS_MAX_ENTRIES)
        generate_block['worker_checkpoints_min_interval_seconds'] = float(WORKER_CHECKPOINTS_MIN_INTERVAL_SECONDS)
        generate_block['job_firestore_checkpoint_state_only_on_change'] = bool(WORKER_CHECKPOINTS_STATE_ONLY_ON_CHANGE)
        generate_block['dataset_registry_backend'] = str(GLOBAL_PROGRESS_BACKEND)
        generate_block['dataset_registry_firestore_project_id'] = str(FIRESTORE_PROJECT_ID)
        generate_block['dataset_registry_firestore_collection'] = str(FIRESTORE_DATASET_REGISTRY_COLLECTION)
        generate_block['dataset_registry_firestore_document'] = str(FIRESTORE_DATASET_REGISTRY_DOCUMENT)
        generate_block['dataset_registry_firestore_credentials_path'] = str(FIRESTORE_CREDENTIALS_PATH)
        generate_block['dataset_registry_firestore_api_key'] = str(FIRESTORE_API_KEY)
        generate_block['progress_update_every_n_games'] = int(GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES)
        generate_block['global_budget_enforcement_mode'] = str(GLOBAL_BUDGET_ENFORCEMENT_MODE)
        generate_block['global_progress_flush_every_n_games'] = int(GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES)
        generate_block['global_target_poll_interval_seconds'] = float(GLOBAL_TARGET_POLL_INTERVAL_SECONDS)
        generate_cfg['dataset_generation'] = generate_block
        DATASET_GENERATE_CONFIG_ACTIVE = _write_yaml(generate_cfg, DATASET_GENERATE_CONFIG_ACTIVE_PATH)

        build_cfg = yaml.safe_load((Path(WORKTREE) / DATASET_BUILD_CONFIG).read_text(encoding='utf-8')) or {}
        build_block = dict(build_cfg.get('dataset_build', {}) or {})
        build_block['source_dataset_id'] = DATASET_SOURCE_ID
        build_block['input_sampled_dir'] = f'data/{DATASET_SOURCE_ID}'
        build_block['dataset_id'] = DATASET_BUILD_ID
        build_block['target_labeled_samples'] = int(TARGET_LABELED_SAMPLES)
        requested_build_mode = str(DATASET_BUILD_MODE_OVERRIDE).strip().lower()
        if requested_build_mode:
            if requested_build_mode not in {'auto', 'teacher_label', 'source_prelabeled'}:
                raise ValueError(f'DATASET_BUILD_MODE_OVERRIDE non supporte: {requested_build_mode}')
            build_mode = requested_build_mode
        else:
            build_mode = 'source_prelabeled' if generate_source_mode == 'self_play_puct' else 'teacher_label'
        build_block['build_mode'] = build_mode
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
        build_block['global_target_progress_redis_enabled'] = bool(GLOBAL_PROGRESS_REDIS_ENABLED)
        build_block['global_target_progress_redis_url'] = str(GLOBAL_PROGRESS_REDIS_URL)
        build_block['global_target_progress_redis_key_prefix'] = str(GLOBAL_PROGRESS_REDIS_KEY_PREFIX)
        build_block['global_target_progress_redis_cache_ttl_seconds'] = int(GLOBAL_PROGRESS_REDIS_CACHE_TTL_SECONDS)
        build_block['global_target_progress_workers_retention_seconds'] = int(GLOBAL_PROGRESS_WORKERS_RETENTION_SECONDS)
        build_block['global_target_progress_workers_max_entries'] = int(GLOBAL_PROGRESS_WORKERS_MAX_ENTRIES)
        build_block['worker_checkpoints_min_interval_seconds'] = float(WORKER_CHECKPOINTS_MIN_INTERVAL_SECONDS)
        build_block['job_firestore_checkpoint_state_only_on_change'] = bool(WORKER_CHECKPOINTS_STATE_ONLY_ON_CHANGE)
        build_block['dataset_registry_backend'] = str(GLOBAL_PROGRESS_BACKEND)
        build_block['dataset_registry_firestore_project_id'] = str(FIRESTORE_PROJECT_ID)
        build_block['dataset_registry_firestore_collection'] = str(FIRESTORE_DATASET_REGISTRY_COLLECTION)
        build_block['dataset_registry_firestore_document'] = str(FIRESTORE_DATASET_REGISTRY_DOCUMENT)
        build_block['dataset_registry_firestore_credentials_path'] = str(FIRESTORE_CREDENTIALS_PATH)
        build_block['dataset_registry_firestore_api_key'] = str(FIRESTORE_API_KEY)
        build_block['global_target_samples'] = int(GLOBAL_TARGET_SAMPLES)
        build_block['global_target_stabilization_polls'] = int(GLOBAL_TARGET_STABILIZATION_POLLS)
        build_block['workers'] = int(DATASET_BUILD_WORKERS)
        build_block['max_pending_futures'] = int(DATASET_BUILD_MAX_PENDING_FUTURES)
        build_block['export_partial_every_n_files'] = int(max(1, int(DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES)))
        build_cfg['dataset_build'] = build_block
        DATASET_BUILD_CONFIG_ACTIVE = _write_yaml(build_cfg, DATASET_BUILD_CONFIG_ACTIVE_PATH)

        def _resolve_train_eval_dataset_id() -> str:
            # Priorite: dataset global fusionne (BASE_DATASET_BUILD_ID), sinon plus gros shard de la famille.
            try:
                registry_payload = _load_dataset_registry_payload()
            except Exception:
                return str(DATASET_BUILD_ID)

            built_entries = registry_payload.get('built_datasets', []) if isinstance(registry_payload, dict) else []
            if not isinstance(built_entries, list):
                return str(DATASET_BUILD_ID)

            def _entry_output_dir(entry: dict) -> Path:
                output_text = str(entry.get('output_dir', '')).strip()
                if not output_text:
                    return Path('')
                output_path = Path(output_text)
                if output_path.is_absolute():
                    return output_path
                return Path(DRIVE_ROOT) / output_text

            def _entry_has_required_npz(entry: dict) -> bool:
                out = _entry_output_dir(entry)
                return out.exists() and (out / 'train.npz').exists() and (out / 'validation.npz').exists()

            def _entry_sort_key(entry: dict):
                return (
                    int(entry.get('labeled_samples', 0) or 0),
                    str(entry.get('updated_at', '')),
                )

            exact_global = next(
                (
                    item
                    for item in built_entries
                    if isinstance(item, dict)
                    and str(item.get('dataset_id', '')).strip() == str(BASE_DATASET_BUILD_ID)
                    and _entry_has_required_npz(item)
                ),
                None,
            )
            if exact_global is not None:
                return str(exact_global.get('dataset_id', BASE_DATASET_BUILD_ID))

            family_prefix = f'{BASE_DATASET_BUILD_ID}_'
            family = [
                item
                for item in built_entries
                if isinstance(item, dict)
                and _entry_has_required_npz(item)
                and (
                    str(item.get('dataset_id', '')).strip() == str(BASE_DATASET_BUILD_ID)
                    or str(item.get('dataset_id', '')).strip().startswith(family_prefix)
                )
            ]
            if family:
                family.sort(key=_entry_sort_key, reverse=True)
                return str(family[0].get('dataset_id', DATASET_BUILD_ID))

            return str(DATASET_BUILD_ID)

        TRAIN_EVAL_DATASET_ID = _resolve_train_eval_dataset_id()

        train_continue_cfg = yaml.safe_load(Path(TRAIN_CONTINUE_CONFIG_ACTIVE).read_text(encoding='utf-8')) or {}
        train_continue_block = dict(train_continue_cfg.get('train', {}) or {})
        train_continue_block['dataset_selection_mode'] = 'configured'
        train_continue_block['dataset_id'] = TRAIN_EVAL_DATASET_ID
        train_continue_block['dataset_registry_backend'] = str(GLOBAL_PROGRESS_BACKEND)
        train_continue_block['dataset_registry_firestore_project_id'] = str(FIRESTORE_PROJECT_ID)
        train_continue_block['dataset_registry_firestore_collection'] = str(FIRESTORE_DATASET_REGISTRY_COLLECTION)
        train_continue_block['dataset_registry_firestore_document'] = str(FIRESTORE_DATASET_REGISTRY_DOCUMENT)
        train_continue_block['dataset_registry_firestore_credentials_path'] = str(FIRESTORE_CREDENTIALS_PATH)
        train_continue_block['dataset_registry_firestore_api_key'] = str(FIRESTORE_API_KEY)
        train_continue_cfg['train'] = train_continue_block
        TRAIN_CONTINUE_20M_CONFIG_ACTIVE = _write_yaml(train_continue_cfg, TRAIN_CONTINUE_20M_CONFIG_ACTIVE_PATH)

        train_scratch_cfg = yaml.safe_load(Path(TRAIN_SCRATCH_CONFIG_ACTIVE).read_text(encoding='utf-8')) or {}
        train_scratch_block = dict(train_scratch_cfg.get('train', {}) or {})
        train_scratch_block['dataset_selection_mode'] = 'configured'
        train_scratch_block['dataset_id'] = TRAIN_EVAL_DATASET_ID
        train_scratch_block['dataset_registry_backend'] = str(GLOBAL_PROGRESS_BACKEND)
        train_scratch_block['dataset_registry_firestore_project_id'] = str(FIRESTORE_PROJECT_ID)
        train_scratch_block['dataset_registry_firestore_collection'] = str(FIRESTORE_DATASET_REGISTRY_COLLECTION)
        train_scratch_block['dataset_registry_firestore_document'] = str(FIRESTORE_DATASET_REGISTRY_DOCUMENT)
        train_scratch_block['dataset_registry_firestore_credentials_path'] = str(FIRESTORE_CREDENTIALS_PATH)
        train_scratch_block['dataset_registry_firestore_api_key'] = str(FIRESTORE_API_KEY)
        train_scratch_cfg['train'] = train_scratch_block
        TRAIN_SCRATCH_20M_CONFIG_ACTIVE = _write_yaml(train_scratch_cfg, TRAIN_SCRATCH_20M_CONFIG_ACTIVE_PATH)

        eval_cfg = yaml.safe_load(Path(EVALUATION_CONFIG_ACTIVE).read_text(encoding='utf-8')) or {}
        eval_block = dict(eval_cfg.get('evaluation', {}) or {})
        eval_block['dataset_selection_mode'] = 'configured'
        eval_block['dataset_id'] = TRAIN_EVAL_DATASET_ID
        eval_block['dataset_registry_backend'] = str(GLOBAL_PROGRESS_BACKEND)
        eval_block['dataset_registry_firestore_project_id'] = str(FIRESTORE_PROJECT_ID)
        eval_block['dataset_registry_firestore_collection'] = str(FIRESTORE_DATASET_REGISTRY_COLLECTION)
        eval_block['dataset_registry_firestore_document'] = str(FIRESTORE_DATASET_REGISTRY_DOCUMENT)
        eval_block['dataset_registry_firestore_credentials_path'] = str(FIRESTORE_CREDENTIALS_PATH)
        eval_block['dataset_registry_firestore_api_key'] = str(FIRESTORE_API_KEY)
        eval_cfg['evaluation'] = eval_block
        EVALUATION_20M_CONFIG_ACTIVE = _write_yaml(eval_cfg, EVALUATION_20M_CONFIG_ACTIVE_PATH)

        print('RUNTIME_PROFILE                =', RUNTIME_PROFILE)
        print('TPU_ENV_PRESENT                =', TPU_ENV_PRESENT)
        print('TPU_RUNTIME_READY              =', TPU_RUNTIME_READY)
        print('RUNTIME_HAS_CUDA               =', RUNTIME_HAS_CUDA)
        print('RUNTIME_SUMMARY                =', runtime_summary)
        print('DATASET_GENERATE_CONFIG_ACTIVE =', DATASET_GENERATE_CONFIG_ACTIVE)
        print('DATASET_BUILD_CONFIG_ACTIVE    =', DATASET_BUILD_CONFIG_ACTIVE)
        print('TRAIN_EVAL_DATASET_ID          =', TRAIN_EVAL_DATASET_ID)
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
        print('generate_source_mode           =', generate_source_mode)
        print('dataset_build_mode             =', build_mode)
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
    md("## 5. Creation Dataset (Generate + Build)"),
    md("### 5.A Lancer Le Pipeline Dataset En Parallele"),
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
        firestore_manifest_written = False
        firestore_manifest_error = ''
        if PIPELINE_MANIFEST_FIRESTORE_WRITE_ENABLED:
            try:
                _write_firestore_doc(FIRESTORE_PIPELINE_MANIFESTS_COLLECTION, WORKER_TAG, manifest)
                firestore_manifest_written = True
            except Exception as exc:
                firestore_manifest_error = f'{type(exc).__name__}: {exc}'
        else:
            firestore_manifest_error = 'disabled_by_low_quota_profile'

        print('Pipeline lance en parallele')
        print('  dataset-generate pid =', generate_pid)
        print('  dataset-build pid    =', build_pid)
        print('  generate log         =', generate_log_path)
        print('  build log            =', build_log_path)
        print('  manifest             =', latest_path)
        print('  firestore_manifest   =', f'{FIRESTORE_PIPELINE_MANIFESTS_COLLECTION}/{WORKER_TAG}')
        print('  firestore_written    =', firestore_manifest_written)
        if firestore_manifest_error:
            print('  firestore_error      =', firestore_manifest_error)
        """
    ),
    md("## 5bis. Suivi Pipeline Et Logs"),
    md(
        """
        Vue par source:
        - `5bis.A` = Manifest pipeline (Drive prioritaire, Firestore fallback)
        - `5bis.B` = Drive local worker (etat local des jobs)
        - `5bis.C` = Redis cache (et ecart vs Firestore)
        - `5bis.D` = Logs worker (fichiers Drive)
        - `5bis.E` = Metriques checkpoint sync Firestore (events/metrics de run)
        """
    ),
    md("### 5bis.A Suivi Manifest Pipeline (Drive prioritaire, Firestore fallback)"),
    code(
        """
        import json
        import os
        import subprocess
        from pathlib import Path

        logs_dir = Path(DRIVE_ROOT) / 'logs' / 'pipeline'

        manifest = {}
        manifest_source = 'none'
        local_manifest_path = Path(DRIVE_ROOT) / PIPELINE_MANIFEST_PATH

        # Priorite: manifest local Drive (toujours ecrit par la cellule de lancement)
        if local_manifest_path.exists():
            try:
                manifest = json.loads(local_manifest_path.read_text(encoding='utf-8'))
                if isinstance(manifest, dict) and manifest:
                    manifest_source = f'drive:{local_manifest_path}'
            except Exception:
                manifest = {}

        # Fallback: Firestore (utile si local indisponible)
        if not manifest:
            try:
                manifest = _load_pipeline_manifest_payload(WORKER_TAG)
                if isinstance(manifest, dict) and manifest:
                    manifest_source = f'firestore:{FIRESTORE_PIPELINE_MANIFESTS_COLLECTION}/{WORKER_TAG}'
            except Exception:
                manifest = {}

        if not manifest:
            print('Manifest introuvable (Drive + Firestore)')
            print('  drive_path =', local_manifest_path)
            print('  firestore  =', f'{FIRESTORE_PIPELINE_MANIFESTS_COLLECTION}/{WORKER_TAG}')
        else:
            generate_pid = int(manifest.get('generate_pid', 0) or 0)
            build_pid = int(manifest.get('build_pid', 0) or 0)
            print('Manifest source =', manifest_source)
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
        # Snapshot Firestore: global progress + dataset_registry (worker courant)
        try:
            global_payload = _read_firestore_doc(
                FIRESTORE_COLLECTION,
                FIRESTORE_DOCUMENT,
                default={
                    'global_target_id': GLOBAL_TARGET_ID,
                    'target_samples': int(GLOBAL_TARGET_SAMPLES),
                    'total_samples': 0,
                    'total_games': 0,
                    'workers': {},
                    'updated_at': '<none>',
                },
            )
            registry = _load_dataset_registry_payload()
        except Exception as exc:
            print('Lecture Firestore impossible:', f'{type(exc).__name__}: {exc}')
            global_payload = {}
            registry = {'dataset_sources': [], 'built_datasets': []}

        if not isinstance(global_payload, dict):
            global_payload = {}
        workers_payload = global_payload.get('workers', {})
        if not isinstance(workers_payload, dict):
            workers_payload = {}

        print('Global Firestore:')
        print('  collection/document =', f'{FIRESTORE_COLLECTION}/{FIRESTORE_DOCUMENT}')
        print('  total_samples       =', int(global_payload.get('total_samples', 0)))
        print('  target_samples      =', int(global_payload.get('target_samples', GLOBAL_TARGET_SAMPLES)))
        print('  total_games         =', int(global_payload.get('total_games', 0)))
        print('  workers             =', len(workers_payload))
        print('  updated_at          =', global_payload.get('updated_at', '<none>'))

        source = next((item for item in registry.get('dataset_sources', []) if item.get('dataset_source_id') == DATASET_SOURCE_ID), None)
        built = next((item for item in registry.get('built_datasets', []) if item.get('dataset_id') == DATASET_BUILD_ID), None)

        print('\\nSource courante (Firestore dataset_registry):')
        if source is None:
            print('- aucune entree source pour', DATASET_SOURCE_ID)
        else:
            print('  dataset_source_id =', source.get('dataset_source_id'))
            print('  source_mode       =', source.get('source_mode'))
            print('  source_status     =', source.get('source_status'))
            print('  sampled_positions =', source.get('sampled_positions'))
            print('  sampled_files     =', source.get('sampled_files'))
            print('  updated_at        =', source.get('updated_at'))

        print('\\nDataset final courant (Firestore dataset_registry):')
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
        # KPI live (Firestore dataset_registry + Drive job summary)
        import json
        import time
        from pathlib import Path

        REFRESH_SECONDS = float(MONITOR_REFRESH_SECONDS)
        MAX_LOOPS = int(MONITOR_MAX_LOOPS)

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
            registry = _load_dataset_registry_payload()
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
            print(f'[{ts}] [FIRESTORE dataset_registry] source_samples={source_samples}/{TARGET_SAMPLES} ({_safe_pct(source_samples, int(TARGET_SAMPLES)):.2f}%) | source_files={source_files} | source_status={source_status}')
            print(f'[{ts}] [DRIVE summary + FIRESTORE build] played_games={played_games} | build_samples={build_samples}/{build_target} ({_safe_pct(build_samples, build_target):.2f}%) | build_status={build_status}')
            print('-' * 120)

            if loop_idx >= (MAX_LOOPS - 1):
                break
            time.sleep(REFRESH_SECONDS)
        """
    ),
    code(
        """
        # KPI global (Firestore direct): progression agregee de tous les workers
        import json
        import time
        from pathlib import Path

        REFRESH_SECONDS = float(MONITOR_REFRESH_SECONDS)
        MAX_LOOPS = int(MONITOR_MAX_LOOPS)

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
            global_payload = _read_firestore_doc(
                FIRESTORE_COLLECTION,
                FIRESTORE_DOCUMENT,
                default={
                    'global_target_id': GLOBAL_TARGET_ID,
                    'target_samples': int(GLOBAL_TARGET_SAMPLES),
                    'total_samples': 0,
                    'total_games': 0,
                    'workers': {},
                    'updated_at': '<none>',
                },
            )
            if not isinstance(global_payload, dict):
                global_payload = {}

            global_total_samples = int(global_payload.get('total_samples', 0))
            global_total_games = int(global_payload.get('total_games', 0))
            global_target_samples = int(global_payload.get('target_samples', GLOBAL_TARGET_SAMPLES))
            workers_payload = global_payload.get('workers', {})
            if not isinstance(workers_payload, dict):
                workers_payload = {}
            global_workers = len(workers_payload)

            built_total_samples = 0
            built_worker_datasets = 0
            registry = _load_dataset_registry_payload()
            for item in registry.get('built_datasets', []):
                dataset_id = str(item.get('dataset_id', ''))
                if dataset_id == BASE_DATASET_BUILD_ID or dataset_id.startswith(f'{BASE_DATASET_BUILD_ID}_'):
                    built_total_samples += int(item.get('labeled_samples', 0))
                    built_worker_datasets += 1

            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            print(
                f'[{ts}] [FIRESTORE global] generate_samples={global_total_samples}/{global_target_samples} '
                f'({_safe_pct(global_total_samples, global_target_samples):.2f}%) | '
                f'global_games={global_total_games} | workers={global_workers} | '
                f'updated_at={global_payload.get("updated_at", "<none>")}'
            )
            print(f'[{ts}] [FIRESTORE registry] build_labeled_samples_sum={built_total_samples} | build_worker_datasets={built_worker_datasets}')
            print('-' * 120)

            if loop_idx >= (MAX_LOOPS - 1):
                break
            time.sleep(REFRESH_SECONDS)
        """
    ),
    code(
        """
        # Workers status (Firestore direct): actif/inactif (vision globale multi-Colab)
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

        payload = _read_firestore_doc(
            FIRESTORE_COLLECTION,
            FIRESTORE_DOCUMENT,
            default={
                'global_target_id': GLOBAL_TARGET_ID,
                'target_samples': int(GLOBAL_TARGET_SAMPLES),
                'total_samples': 0,
                'total_games': 0,
                'workers': {},
                'updated_at': '<none>',
            },
        )
        if not isinstance(payload, dict):
            payload = {}
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
                f'[FIRESTORE global] Workers: total={len(rows)} | active={active_count} | inactive={inactive_count} '
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
        # Health check compact (Firestore direct): un seul bloc pour voir si le pipeline avance vraiment
        import json
        import time
        from datetime import datetime
        from pathlib import Path

        ACTIVE_THRESHOLD_SECONDS = 600
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

        payload = _read_firestore_doc(
            FIRESTORE_COLLECTION,
            FIRESTORE_DOCUMENT,
            default={
                'global_target_id': GLOBAL_TARGET_ID,
                'target_samples': int(GLOBAL_TARGET_SAMPLES),
                'total_samples': 0,
                'total_games': 0,
                'workers': {},
                'updated_at': '<none>',
            },
        )
        if not isinstance(payload, dict):
            payload = {}
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
        registry = _load_dataset_registry_payload()
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
    md("### 5bis.B Suivi Drive (fichiers worker locaux)"),
    code(
        """
        # Vue Drive: progression locale du worker (fichiers jobs/*)
        import json
        import time
        from pathlib import Path

        REFRESH_SECONDS = float(MONITOR_REFRESH_SECONDS)
        MAX_LOOPS = int(MONITOR_MAX_LOOPS)
        jobs_root = Path(DRIVE_ROOT) / 'jobs'

        def _load_json_retry(path: Path, retries: int = 4, wait_seconds: float = 0.2, default=None):
            fallback = {} if default is None else default
            for attempt in range(retries):
                try:
                    return json.loads(path.read_text(encoding='utf-8'))
                except (FileNotFoundError, OSError, json.JSONDecodeError):
                    if attempt + 1 >= retries:
                        return fallback
                    time.sleep(wait_seconds)
            return fallback

        def _latest_job_dir(job_id: str) -> Path | None:
            if not jobs_root.exists():
                return None
            requested = str(job_id).strip()
            if not requested:
                return None
            parts = requested.rsplit('_', 1)
            prefix = parts[0] if len(parts) == 2 and parts[1].isdigit() else requested
            candidates = []
            for path in jobs_root.iterdir():
                if not path.is_dir():
                    continue
                name = path.name
                if name == requested or name == prefix or (prefix and name.startswith(prefix + '_')):
                    candidates.append(path)
            if not candidates:
                return None
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

        for loop_idx in range(MAX_LOOPS):
            generate_dir = _latest_job_dir(DATASET_GENERATE_JOB_ID)
            build_dir = _latest_job_dir(DATASET_BUILD_JOB_ID)

            generate_status = _load_json_retry(generate_dir / 'run_status.json', default={}) if generate_dir else {}
            generate_state = _load_json_retry(generate_dir / 'state.json', default={}) if generate_dir else {}
            generate_summary = _load_json_retry(generate_dir / 'dataset_generation' / 'dataset_generation_summary.json', default={}) if generate_dir else {}

            build_status = _load_json_retry(build_dir / 'run_status.json', default={}) if build_dir else {}
            build_state = _load_json_retry(build_dir / 'state.json', default={}) if build_dir else {}
            build_summary = _load_json_retry(build_dir / 'dataset_build' / 'dataset_build_summary.json', default={}) if build_dir else {}

            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            print(
                f'[{ts}] [DRIVE generate] dir={generate_dir} | status={generate_status.get("status", "<none>")} '
                f'| phase={generate_status.get("phase", "<none>")} | sample_count={generate_state.get("sample_count", "<none>")} '
                f'| total_samples_summary={generate_summary.get("total_samples", "<none>")} | added_games_summary={generate_summary.get("added_games", "<none>")}'
            )
            print(
                f'[{ts}] [DRIVE build] dir={build_dir} | status={build_status.get("status", "<none>")} '
                f'| phase={build_status.get("phase", "<none>")} | labeled_samples_state={build_state.get("labeled_samples", "<none>")} '
                f'| processed_files_state={build_state.get("processed_files", "<none>")} | labeled_samples_summary={build_summary.get("labeled_samples", "<none>")}'
            )
            print('-' * 120)

            if loop_idx >= (MAX_LOOPS - 1):
                break
            time.sleep(REFRESH_SECONDS)
        """
    ),
    md("### 5bis.C Suivi Redis (cache de progression globale)"),
    code(
        """
        # Vue Redis cache: snapshot + ecart vs Firestore
        import json

        redis_key = f'{GLOBAL_PROGRESS_REDIS_KEY_PREFIX}:global_progress'
        print('Redis enabled =', bool(GLOBAL_PROGRESS_REDIS_ENABLED))
        print('Redis url set =', bool(str(GLOBAL_PROGRESS_REDIS_URL).strip()))
        print('Redis token set =', bool(str(GLOBAL_PROGRESS_REDIS_TOKEN).strip()))
        print('Redis key =', redis_key)

        if not bool(GLOBAL_PROGRESS_REDIS_ENABLED):
            print('Redis desactive, rien a lire.')
        elif not str(GLOBAL_PROGRESS_REDIS_URL).strip() or not str(GLOBAL_PROGRESS_REDIS_TOKEN).strip():
            print('Redis active mais URL/token manquants.')
        else:
            try:
                from upstash_redis import Redis

                redis_client = Redis(
                    url=str(GLOBAL_PROGRESS_REDIS_URL).strip(),
                    token=str(GLOBAL_PROGRESS_REDIS_TOKEN).strip(),
                )
                raw_payload = redis_client.get(redis_key)
                if raw_payload is None:
                    print('Aucune entree Redis pour cette key.')
                else:
                    if isinstance(raw_payload, str):
                        cache_payload = json.loads(raw_payload)
                    elif isinstance(raw_payload, dict):
                        cache_payload = raw_payload
                    else:
                        cache_payload = {}
                    if not isinstance(cache_payload, dict):
                        cache_payload = {}
                    print('Redis payload:')
                    print('  total_samples =', int(cache_payload.get('total_samples', 0)))
                    print('  target_samples =', int(cache_payload.get('target_samples', GLOBAL_TARGET_SAMPLES)))
                    print('  total_games =', int(cache_payload.get('total_games', 0)))
                    print('  updated_at =', cache_payload.get('updated_at', '<none>'))
                    print('  redis_cached_at =', cache_payload.get('_redis_cached_at', '<none>'))

                    firestore_payload = _read_firestore_doc(
                        FIRESTORE_COLLECTION,
                        FIRESTORE_DOCUMENT,
                        default={
                            'total_samples': 0,
                            'target_samples': int(GLOBAL_TARGET_SAMPLES),
                            'total_games': 0,
                            'updated_at': '<none>',
                        },
                    )
                    if not isinstance(firestore_payload, dict):
                        firestore_payload = {}
                    delta_samples = int(cache_payload.get('total_samples', 0)) - int(firestore_payload.get('total_samples', 0))
                    delta_games = int(cache_payload.get('total_games', 0)) - int(firestore_payload.get('total_games', 0))
                    print('Ecart Redis - Firestore:')
                    print('  delta_samples =', delta_samples)
                    print('  delta_games   =', delta_games)
            except Exception as exc:
                print('Lecture Redis impossible:', f'{type(exc).__name__}: {exc}')
        """
    ),
    md("### 5bis.D Logs worker (Drive)"),
    code(
        """
        import json
        from pathlib import Path

        LOG_TAIL_LINES = 40
        logs_dir = Path(DRIVE_ROOT) / 'logs' / 'pipeline'
        local_manifest_path = Path(DRIVE_ROOT) / PIPELINE_MANIFEST_PATH
        manifest = {}
        manifest_source = 'none'

        if local_manifest_path.exists():
            try:
                manifest = json.loads(local_manifest_path.read_text(encoding='utf-8'))
                if isinstance(manifest, dict) and manifest:
                    manifest_source = f'drive:{local_manifest_path}'
            except Exception:
                manifest = {}

        if not manifest:
            try:
                manifest = _load_pipeline_manifest_payload(WORKER_TAG)
                if isinstance(manifest, dict) and manifest:
                    manifest_source = f'firestore:{FIRESTORE_PIPELINE_MANIFESTS_COLLECTION}/{WORKER_TAG}'
            except Exception:
                manifest = {}

        print('manifest source =', manifest_source)
        for label, key, fallback in [
            ('dataset-generate', 'generate_log_path', logs_dir / f'{DATASET_GENERATE_JOB_ID}.log'),
            ('dataset-build', 'build_log_path', logs_dir / f'{DATASET_BUILD_JOB_ID}.log'),
        ]:
            manifest_path = Path(str(manifest.get(key, '')).strip()) if manifest else Path('')
            log_path = manifest_path if str(manifest_path).strip() else fallback
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
        fallback_path = logs_dir / f'{DATASET_BUILD_JOB_ID}.log'

        local_manifest_path = Path(DRIVE_ROOT) / PIPELINE_MANIFEST_PATH
        manifest = {}
        if local_manifest_path.exists():
            try:
                manifest = json.loads(local_manifest_path.read_text(encoding='utf-8'))
            except Exception:
                manifest = {}
        if not manifest:
            try:
                manifest = _load_pipeline_manifest_payload(WORKER_TAG)
            except Exception:
                manifest = {}

        if manifest:
            manifest_path = Path(str(manifest.get('build_log_path', '')).strip())
            log_path = manifest_path if str(manifest_path).strip() else fallback_path
        else:
            log_path = fallback_path

        print('tailing:', log_path)
        if not log_path.exists():
            print('log introuvable:', log_path)
        else:
            !tail -f {log_path}
        """
    ),
    md("### 5bis.E Metriques Checkpoint Sync Firestore"),
    code(
        """
        # Metriques firestore_checkpoint_sync_summary dans les metrics des jobs worker.
        import json
        from pathlib import Path

        jobs_root = Path(DRIVE_ROOT) / 'jobs'

        def _latest_job_dir(job_id: str) -> Path | None:
            if not jobs_root.exists():
                return None
            requested = str(job_id).strip()
            if not requested:
                return None
            parts = requested.rsplit('_', 1)
            prefix = parts[0] if len(parts) == 2 and parts[1].isdigit() else requested
            candidates = []
            for path in jobs_root.iterdir():
                if not path.is_dir():
                    continue
                name = path.name
                if name == requested or name == prefix or (prefix and name.startswith(prefix + '_')):
                    candidates.append(path)
            if not candidates:
                return None
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

        def _read_jsonl(path: Path) -> list[dict]:
            if not path.exists():
                return []
            rows: list[dict] = []
            for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                    if isinstance(row, dict):
                        rows.append(row)
                except Exception:
                    continue
            return rows

        for label, job_id in [('dataset-generate', DATASET_GENERATE_JOB_ID), ('dataset-build', DATASET_BUILD_JOB_ID)]:
            job_dir = _latest_job_dir(job_id)
            print(f'\\n===== {label} | job_id={job_id} | dir={job_dir} =====')
            if job_dir is None:
                print('job dir introuvable')
                continue
            metrics_path = job_dir / 'metrics.jsonl'
            events_path = job_dir / 'events.jsonl'

            metrics_rows = _read_jsonl(metrics_path)
            sync_rows = [row for row in metrics_rows if str(row.get('metric_type', '')) == 'firestore_checkpoint_sync_summary']
            if sync_rows:
                latest = sync_rows[-1]
                print('Derniere metric firestore_checkpoint_sync_summary:')
                print('  timestamp            =', latest.get('timestamp', '<none>'))
                print('  status               =', latest.get('status', '<none>'))
                print('  phase                =', latest.get('phase', '<none>'))
                print('  attempted            =', latest.get('attempted', 0))
                print('  written              =', latest.get('written', 0))
                print('  skipped_unchanged    =', latest.get('skipped_unchanged', 0))
                print('  skipped_min_interval =', latest.get('skipped_min_interval', 0))
                print('  failed               =', latest.get('failed', 0))
                print('  auth_mode            =', latest.get('auth_mode', '<none>'))
                print('  collection           =', latest.get('collection', '<none>'))
            else:
                print('Aucune metric firestore_checkpoint_sync_summary pour ce job.')

            event_rows = _read_jsonl(events_path)
            failed_events = [row for row in event_rows if str(row.get('message', '')) == 'firestore_worker_checkpoint_sync_failed']
            print('failed sync events =', len(failed_events))
            if failed_events:
                last_failed = failed_events[-1]
                print('Dernier echec:')
                print('  timestamp =', last_failed.get('timestamp', '<none>'))
                print('  error     =', last_failed.get('error', '<none>'))
                print('  hint      =', last_failed.get('hint', '<none>'))
        """
    ),
    md("## 6. Lister les datasets"),
    code(
        """
        import json
        from datetime import datetime
        from pathlib import Path

        try:
            registry = _load_dataset_registry_payload()

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
        except Exception as exc:
            print('dataset_registry Firestore introuvable:', f'{type(exc).__name__}: {exc}')
        """
    ),
    md("## 6bis. Sanity Check Dataset (10 echantillons aleatoires)"),
    code(
        """
        import json
        from pathlib import Path
        import numpy as np

        DATASET_PREVIEW_ROWS = 10
        DATASET_PREVIEW_SPLIT = 'train'  # 'train', 'validation', 'test'
        DATASET_PREVIEW_SEED = 42
        DATASET_PREVIEW_FEATURES_HEAD = 16

        def _resolve_preview_dataset_entry() -> dict:
            registry = _load_dataset_registry_payload()
            built_entries = [item for item in registry.get('built_datasets', []) if isinstance(item, dict)]
            requested = str(DATASET_BUILD_ID).strip()
            split_name = str(DATASET_PREVIEW_SPLIT).strip().lower() or 'train'

            def _has_split(entry: dict) -> bool:
                output_dir = Path(str(entry.get('output_dir', '')).strip())
                return output_dir.exists() and (output_dir / f'{split_name}.npz').exists()

            direct = next(
                (
                    item
                    for item in built_entries
                    if str(item.get('dataset_id', '')).strip() == requested and _has_split(item)
                ),
                None,
            )
            if direct is not None:
                return direct

            prefix = f'{BASE_DATASET_BUILD_ID}_'
            shard_candidates = [
                item
                for item in built_entries
                if _has_split(item)
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
                return shard_candidates[0]

            datasets_root = Path(DRIVE_ROOT) / 'data' / 'datasets'
            if datasets_root.exists():
                dir_candidates = []
                for path in datasets_root.iterdir():
                    if not path.is_dir():
                        continue
                    name = path.name
                    if name == requested or name == BASE_DATASET_BUILD_ID or name.startswith(prefix):
                        if (path / f'{split_name}.npz').exists():
                            dir_candidates.append(path)
                if dir_candidates:
                    selected = sorted(dir_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                    return {
                        'dataset_id': selected.name,
                        'output_dir': str(selected),
                        'labeled_samples': 0,
                        'build_mode': '<unknown>',
                        'feature_schema_version': '<unknown>',
                    }
            raise ValueError(
                f'Aucun dataset build avec {split_name}.npz trouve '
                f'(requested={requested}, base={BASE_DATASET_BUILD_ID}).'
            )

        dataset_entry = _resolve_preview_dataset_entry()
        dataset_id = str(dataset_entry.get('dataset_id', '<unknown>'))
        output_dir = Path(str(dataset_entry.get('output_dir', '')).strip())
        split_name = str(DATASET_PREVIEW_SPLIT).strip().lower() or 'train'
        npz_path = output_dir / f'{split_name}.npz'
        if not npz_path.exists():
            raise FileNotFoundError(f'Split introuvable: {npz_path}')

        with np.load(npz_path, allow_pickle=True) as data:
            arrays = {key: data[key] for key in data.files}

        x = arrays.get('x', np.zeros((0, 0), dtype=np.float32))
        legal_mask = arrays.get('legal_mask', np.zeros((0, 7), dtype=np.float32))
        policy_index = arrays.get('policy_index', np.zeros((0,), dtype=np.int64))
        policy_target_full = arrays.get('policy_target_full')
        value_target = arrays.get('value_target', np.zeros((0,), dtype=np.float32))
        sample_ids = arrays.get('sample_ids', np.asarray([], dtype=object))
        game_ids = arrays.get('game_ids', np.asarray([], dtype=object))

        total = int(x.shape[0]) if x.ndim >= 1 else 0
        if policy_target_full is None:
            policy_target_full = np.zeros((total, 7), dtype=np.float32)
            if total > 0:
                rows = np.arange(total, dtype=np.int64)
                valid = np.logical_and(policy_index >= 0, policy_index < 7)
                policy_target_full[rows[valid], policy_index[valid]] = 1.0

        policy_sum = policy_target_full.sum(axis=1) if total > 0 else np.asarray([], dtype=np.float32)
        legal_count = legal_mask.sum(axis=1) if total > 0 else np.asarray([], dtype=np.float32)
        invalid_policy_sum = int(np.sum(np.abs(policy_sum - 1.0) > 1e-3)) if total > 0 else 0
        invalid_policy_index = int(np.sum(np.logical_or(policy_index < 0, policy_index >= 7))) if total > 0 else 0
        invalid_policy_not_legal = int(
            np.sum(
                np.logical_and(
                    np.logical_and(policy_index >= 0, policy_index < 7),
                    legal_mask[np.arange(total), np.clip(policy_index, 0, 6)] <= 0.0,
                )
            )
        ) if total > 0 else 0
        invalid_legal_moves = int(np.sum(legal_count <= 0.0)) if total > 0 else 0
        invalid_value_range = int(np.sum(np.logical_or(value_target < -1.0, value_target > 1.0))) if total > 0 else 0
        nan_count = int(np.isnan(x).sum()) if total > 0 else 0
        inf_count = int(np.isinf(x).sum()) if total > 0 else 0

        print('=== Dataset Sanity Check ===')
        print('dataset_id             =', dataset_id)
        print('split                  =', split_name)
        print('npz_path               =', npz_path)
        print('build_mode             =', dataset_entry.get('build_mode', '<unknown>'))
        print('schema_version         =', dataset_entry.get('feature_schema_version', '<unknown>'))
        print('samples                =', total)
        print('input_dim              =', int(x.shape[1]) if x.ndim == 2 else 0)
        print('keys                   =', sorted(arrays.keys()))
        print('value_min_max          =', (
            float(value_target.min()) if total > 0 else 0.0,
            float(value_target.max()) if total > 0 else 0.0,
        ))
        print('anomalies.policy_sum   =', invalid_policy_sum)
        print('anomalies.policy_index =', invalid_policy_index)
        print('anomalies.policy_legal =', invalid_policy_not_legal)
        print('anomalies.legal_moves  =', invalid_legal_moves)
        print('anomalies.value_range  =', invalid_value_range)
        print('anomalies.nan_features =', nan_count)
        print('anomalies.inf_features =', inf_count)

        if total <= 0:
            raise ValueError(f'Dataset vide pour split={split_name}: {npz_path}')

        rng = np.random.default_rng(int(DATASET_PREVIEW_SEED))
        sample_count = min(int(DATASET_PREVIEW_ROWS), total)
        indices = sorted(rng.choice(total, size=sample_count, replace=False).tolist())

        print('')
        print(f'=== Echantillons Aleatoires ({sample_count}) ===')
        for rank, idx in enumerate(indices, start=1):
            sid = str(sample_ids[idx]) if idx < len(sample_ids) else '<na>'
            gid = str(game_ids[idx]) if idx < len(game_ids) else '<na>'
            pidx = int(policy_index[idx]) if idx < len(policy_index) else -1
            legal_vec = legal_mask[idx].astype(np.int64).tolist() if idx < len(legal_mask) else []
            policy_vec = policy_target_full[idx].astype(np.float64)
            top_moves = np.argsort(-policy_vec)[:3].tolist()
            top_moves_1b = [int(move + 1) for move in top_moves]
            top_probs = [float(policy_vec[move]) for move in top_moves]
            value = float(value_target[idx]) if idx < len(value_target) else 0.0
            x_head = x[idx, : int(DATASET_PREVIEW_FEATURES_HEAD)].astype(np.float64).tolist() if x.ndim == 2 else []
            print('-' * 120)
            print(f'#{rank} idx={idx} | sample_id={sid} | game_id={gid}')
            print(' policy_index(1-based)=', (pidx + 1) if pidx >= 0 else '<invalid>')
            print(' top3_moves(1-based)  =', top_moves_1b, '| top3_probs=', [round(v, 6) for v in top_probs])
            print(' legal_mask(7)        =', legal_vec)
            print(' value_target         =', round(value, 6))
            print(f' x_head(first {int(DATASET_PREVIEW_FEATURES_HEAD)}) =', [round(v, 6) for v in x_head])
        print('-' * 120)
        """
    ),
    md("## 7. Entrainement + Evaluation Automatique"),
    code(
        """
        # Utilitaire commun: selection auto dataset/checkpoint compatibles puis lancement evaluate.
        import json
        import shlex
        import subprocess
        from pathlib import Path
        import numpy as np
        import torch
        import yaml

        def _prepare_eval_runtime_config(
            *,
            locked_dataset_id: str = '',
            locked_model_id: str = '',
            locked_checkpoint_path: str = '',
        ) -> tuple[Path, str]:
            base_eval_cfg_path = Path(EVALUATION_20M_CONFIG_ACTIVE)
            runtime_eval_cfg_path = Path(EVALUATION_20M_RUNTIME_CONFIG_PATH)
            eval_cfg = yaml.safe_load(base_eval_cfg_path.read_text(encoding='utf-8')) or {}
            eval_block = dict(eval_cfg.get('evaluation', {}) or {})
            requested_dataset_id = str(locked_dataset_id).strip() or str(eval_block.get('dataset_id', DATASET_BUILD_ID))
            dataset_id = requested_dataset_id
            dataset_locked = bool(str(locked_dataset_id).strip())

            registry = _load_dataset_registry_payload()
            built_entries = [item for item in registry.get('built_datasets', []) if isinstance(item, dict)]

            def _entry_output_dir(entry: dict) -> Path:
                return Path(str(entry.get('output_dir', '')).strip())

            def _resolve_eval_npz_from_output_dir(output_dir: Path) -> Path | None:
                test_npz = output_dir / 'test.npz'
                if test_npz.exists():
                    return test_npz
                validation_npz = output_dir / 'validation.npz'
                if validation_npz.exists():
                    return validation_npz
                return None

            def _resolve_eval_npz(entry: dict) -> Path | None:
                output_dir = _entry_output_dir(entry)
                if not output_dir.exists():
                    return None
                return _resolve_eval_npz_from_output_dir(output_dir)

            def _has_eval_npz(entry: dict) -> bool:
                return _resolve_eval_npz(entry) is not None

            built = next(
                (
                    item
                    for item in built_entries
                    if str(item.get('dataset_id', '')).strip() == dataset_id and _has_eval_npz(item)
                ),
                None,
            )

            if built is None and (not dataset_locked):
                prefix = f'{BASE_DATASET_BUILD_ID}_'
                shard_candidates = [
                    item
                    for item in built_entries
                    if _has_eval_npz(item)
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
                        if dataset_locked:
                            if name != requested_dataset_id:
                                continue
                            if _resolve_eval_npz_from_output_dir(path) is not None:
                                dir_candidates.append(path)
                        else:
                            if name == requested_dataset_id or name == BASE_DATASET_BUILD_ID or name.startswith(f'{BASE_DATASET_BUILD_ID}_'):
                                if _resolve_eval_npz_from_output_dir(path) is not None:
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
                if dataset_locked:
                    raise ValueError(
                        f'Dataset verrouille introuvable pour evaluation: {requested_dataset_id}. '
                        'Le train a utilise ce dataset mais aucun split eval (test/validation) est disponible.'
                    )
                raise ValueError(
                    f'Dataset introuvable pour evaluation (requested={requested_dataset_id}, base={BASE_DATASET_BUILD_ID}). '
                    'Attends un snapshot build avec test.npz/validation.npz ou relance la cellule de configuration.'
                )

            dataset_input_dim = int(built.get('input_dim') or 0)
            eval_npz_path = _resolve_eval_npz(built)
            if eval_npz_path is None:
                raise ValueError(
                    f'Dataset selectionne sans split eval exploitable (dataset_id={dataset_id}, output_dir={built.get("output_dir", "")}).'
                )
            if dataset_input_dim <= 0:
                with np.load(eval_npz_path, allow_pickle=True) as eval_npz:
                    dataset_input_dim = int(eval_npz['x'].shape[1])

            selected = None
            locked_model_id_value = str(locked_model_id).strip()
            locked_checkpoint_value = str(locked_checkpoint_path).strip()
            if locked_model_id_value or locked_checkpoint_value:
                candidate_checkpoint = None
                if locked_checkpoint_value:
                    checkpoint_path = Path(locked_checkpoint_value)
                    if not checkpoint_path.exists():
                        raise FileNotFoundError(
                            f'Checkpoint verrouille introuvable pour evaluation: {checkpoint_path}'
                        )
                    candidate_checkpoint = checkpoint_path
                if candidate_checkpoint is None and locked_model_id_value:
                    registry_row = _load_model_registry_record(locked_model_id_value)
                    registry_checkpoint = Path(str(registry_row.get('checkpoint_path', '')).strip())
                    if registry_checkpoint.exists():
                        candidate_checkpoint = registry_checkpoint
                    else:
                        fallback_checkpoint = Path(DRIVE_ROOT) / 'models' / 'final' / f'{locked_model_id_value}.pt'
                        if fallback_checkpoint.exists():
                            candidate_checkpoint = fallback_checkpoint
                if candidate_checkpoint is None:
                    raise ValueError(
                        f'Impossible de resoudre le checkpoint verrouille pour model_id={locked_model_id_value or "<empty>"}'
                    )

                try:
                    checkpoint = torch.load(candidate_checkpoint, map_location='cpu')
                except Exception as exc:
                    raise RuntimeError(
                        f'Lecture checkpoint verrouille impossible: {candidate_checkpoint} | cause={type(exc).__name__}: {exc}'
                    ) from exc
                model_config = checkpoint.get('model_config', {})
                model_input_dim = int(model_config.get('input_dim', 0) or 0)
                if model_input_dim != dataset_input_dim:
                    raise ValueError(
                        f'Checkpoint verrouille incompatible avec le dataset eval '
                        f'(model_input_dim={model_input_dim}, dataset_input_dim={dataset_input_dim}, '
                        f'checkpoint={candidate_checkpoint}, dataset_id={dataset_id}).'
                    )
                selected = {
                    'model_id': locked_model_id_value or str(candidate_checkpoint.stem),
                    'checkpoint_path': str(candidate_checkpoint),
                    'input_dim': model_input_dim,
                }
            else:
                models_root = Path(DRIVE_ROOT) / 'models'
                model_registry_path = models_root / 'model_registry.json'
                model_registry = json.loads(model_registry_path.read_text(encoding='utf-8')) if model_registry_path.exists() else {'models': []}
                candidates = sorted(model_registry.get('models', []), key=lambda item: float(item.get('sort_ts', 0.0)), reverse=True)

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
            eval_block['test_dataset_path'] = str(eval_npz_path)
            eval_block['model_id'] = selected['model_id']
            eval_block['checkpoint_path'] = selected['checkpoint_path']
            eval_cfg['evaluation'] = eval_block
            runtime_eval_cfg_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_eval_cfg_path.write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding='utf-8')

            print('Evaluation runtime config prete')
            print('  dataset_id          =', dataset_id)
            print('  dataset_input_dim   =', dataset_input_dim)
            print('  eval_dataset_path   =', eval_npz_path)
            print('  selected_model_id   =', selected['model_id'])
            print('  selected_checkpoint =', selected['checkpoint_path'])
            print('  runtime config      =', runtime_eval_cfg_path)
            return runtime_eval_cfg_path, selected['model_id']

        def _load_json_dict(path: Path, default=None):
            fallback = {} if default is None else default
            if not path.exists():
                return fallback
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                return fallback
            return payload if isinstance(payload, dict) else fallback

        def _job_prefix_for_rollover(job_id: str) -> str:
            text = str(job_id).strip()
            if not text:
                return text
            parts = text.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
            return text

        def _latest_job_dir_for_job_id(job_id: str) -> Path | None:
            jobs_root = Path(DRIVE_ROOT) / 'jobs'
            if not jobs_root.exists():
                return None
            requested = str(job_id).strip()
            if not requested:
                return None
            prefix = _job_prefix_for_rollover(requested)
            candidates = []
            for path in jobs_root.iterdir():
                if not path.is_dir():
                    continue
                name = path.name
                if name == requested or name == prefix or (prefix and name.startswith(prefix + '_')):
                    candidates.append(path)
            if not candidates:
                return None
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

        def _read_job_summary(job_id: str, filename: str) -> tuple[dict, Path | None]:
            job_dir = _latest_job_dir_for_job_id(job_id)
            if job_dir is None:
                return {}, None
            payload = _load_json_dict(job_dir / filename, default={})
            return payload, job_dir

        def _resolve_train_artifacts(job_id: str) -> tuple[str, str, str]:
            train_summary, train_job_dir = _read_job_summary(job_id, 'training_summary.json')
            dataset_id = str(train_summary.get('dataset_id', '')).strip()
            model_id = str(train_summary.get('model_id', '')).strip()
            checkpoint_path = str(train_summary.get('final_model_path', '')).strip()
            checkpoint = Path(checkpoint_path) if checkpoint_path else None
            if (checkpoint is None or not checkpoint.exists()) and model_id:
                fallback = Path(DRIVE_ROOT) / 'models' / 'final' / f'{model_id}.pt'
                if fallback.exists():
                    checkpoint = fallback
            if not dataset_id:
                raise ValueError(
                    f'dataset_id absent dans training_summary pour job_id={job_id} (job_dir={train_job_dir})'
                )
            if not model_id:
                raise ValueError(
                    f'model_id absent dans training_summary pour job_id={job_id} (job_dir={train_job_dir})'
                )
            if checkpoint is None or not checkpoint.exists():
                raise FileNotFoundError(
                    f'checkpoint introuvable pour job_id={job_id} | model_id={model_id} | '
                    f'checkpoint(train_summary)={checkpoint_path}'
                )
            return dataset_id, model_id, str(checkpoint)

        def _load_model_registry_record(model_id: str) -> dict:
            registry_path = Path(DRIVE_ROOT) / 'models' / 'model_registry.json'
            registry = _load_json_dict(registry_path, default={'models': []})
            models = registry.get('models', []) if isinstance(registry, dict) else []
            for item in models:
                if isinstance(item, dict) and str(item.get('model_id', '')).strip() == str(model_id).strip():
                    return item
            return {}

        def _load_promoted_best_metadata() -> dict:
            metadata_path = Path(DRIVE_ROOT) / 'models' / 'promoted' / 'best' / 'metadata.json'
            return _load_json_dict(metadata_path, default={})

        def _load_latest_benchmark_for_model(model_id: str) -> dict:
            history_path = Path(DRIVE_ROOT) / 'models' / 'history' / 'benchmark_history.jsonl'
            if not history_path.exists():
                return {}
            try:
                lines = history_path.read_text(encoding='utf-8').splitlines()
            except Exception:
                return {}
            target = str(model_id).strip()
            for line in reversed(lines):
                text = str(line).strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                model_value = str(row.get('model_id', row.get('engine', ''))).strip()
                if model_value == target:
                    return row
            return {}

        def _model_signal(eval_summary: dict, benchmark_summary: dict) -> str:
            top1 = float(eval_summary.get('policy_accuracy_top1', -1.0) or -1.0)
            bscore = float(benchmark_summary.get('benchmark_score_weighted', benchmark_summary.get('benchmark_score', -1.0)) or -1.0)
            if top1 >= 0.92 or bscore >= 0.35:
                return 'fort (signal positif)'
            if top1 >= 0.86 or bscore >= 0.15:
                return 'moyen (progression en cours)'
            if top1 >= 0.0:
                return 'faible (a renforcer)'
            return 'inconnu (donnees insuffisantes)'

        def _print_post_train_eval_report(*, mode: str, train_job_id: str, eval_job_id: str, model_id: str) -> None:
            train_summary, train_job_dir = _read_job_summary(train_job_id, 'training_summary.json')
            eval_summary, eval_job_dir = _read_job_summary(eval_job_id, 'evaluation_summary.json')
            registry_row = _load_model_registry_record(model_id)
            promoted = _load_promoted_best_metadata()
            benchmark_row = _load_latest_benchmark_for_model(model_id)
            promoted_model_id = str(promoted.get('model_id', '')).strip()
            promoted_status = 'oui' if promoted_model_id and promoted_model_id == str(model_id).strip() else 'non'
            history = train_summary.get('history', []) if isinstance(train_summary, dict) else []
            last_epoch = history[-1] if history else {}
            last_val_loss = float(last_epoch.get('validation_loss_total', 0.0) or 0.0) if isinstance(last_epoch, dict) else 0.0
            last_val_acc = float(last_epoch.get('validation_policy_accuracy', 0.0) or 0.0) if isinstance(last_epoch, dict) else 0.0

            print('')
            print('=== Post-Run Report ===')
            print('mode                   =', mode)
            print('model_id               =', model_id)
            print('train_job_id(request)  =', train_job_id)
            print('train_job_id(resolved) =', train_summary.get('job_id', '<none>'))
            print('eval_job_id(request)   =', eval_job_id)
            print('eval_job_id(resolved)  =', eval_summary.get('job_id', '<none>'))
            print('train_job_dir          =', train_job_dir if train_job_dir is not None else '<none>')
            print('eval_job_dir           =', eval_job_dir if eval_job_dir is not None else '<none>')
            print('dataset_id(train)      =', train_summary.get('dataset_id', '<none>'))
            print('dataset_mode(train)    =', train_summary.get('dataset_selection_mode', '<none>'))
            print('dataset_id(eval)       =', eval_summary.get('dataset_id', '<none>'))
            print('epochs                 =', train_summary.get('completed_epochs', 0), '/', train_summary.get('epochs', 0))
            print('best_epoch             =', train_summary.get('best_epoch', 0))
            print('best_val_metric        =', float(train_summary.get('best_validation_metric', 0.0) or 0.0))
            print('last_val_loss          =', last_val_loss)
            print('last_val_acc           =', last_val_acc)
            print('eval_examples          =', eval_summary.get('examples', 0))
            print('eval_top1              =', float(eval_summary.get('policy_accuracy_top1', 0.0) or 0.0))
            print('eval_top3              =', float(eval_summary.get('policy_accuracy_top3', 0.0) or 0.0))
            print('eval_value_mae         =', float(eval_summary.get('value_mae', 0.0) or 0.0))
            print('eval_loss_total        =', float(eval_summary.get('loss_total', 0.0) or 0.0))
            print('registry_rank          =', registry_row.get('rank', '<none>'))
            print('registry_eval_top1     =', registry_row.get('evaluation_top1', '<none>'))
            print('registry_benchmark     =', registry_row.get('benchmark_score', '<none>'))
            print('promoted_best          =', promoted_status, '| promoted_model_id =', promoted_model_id or '<none>')
            print('benchmark_latest_score =', benchmark_row.get('benchmark_score', '<none>'))
            print('benchmark_weighted     =', benchmark_row.get('benchmark_score_weighted', '<none>'))
            print('model_signal           =', _model_signal(eval_summary, benchmark_row))
            print('final_model_path       =', train_summary.get('final_model_path', '<none>'))
            print('best_checkpoint_path   =', train_summary.get('best_checkpoint_path', '<none>'))
            print('evaluation_summary     =', eval_summary.get('evaluation_summary_path', '<none>'))
        """
    ),
    code(
        """
        # Cellule A: train continue (depuis best promoted) puis evaluation automatique
        print('Cellule A: train continue + evaluate')
        print('train config =', TRAIN_CONTINUE_CONFIG_ACTIVE)

        train_cmd = (
            f'cd {shlex.quote(WORKTREE)} && '
            f'PYTHONPATH={shlex.quote(f"{WORKTREE}/src")} '
            f'{shlex.quote(PYTHON_BIN)} -m songo_model_stockfish.cli.main '
            f'train --config {shlex.quote(str(TRAIN_CONTINUE_CONFIG_ACTIVE))} '
            f'--job-id {shlex.quote(TRAIN_CONTINUE_JOB_ID)}'
        )
        subprocess.run(['/bin/bash', '-lc', train_cmd], check=True)

        train_dataset_id, train_model_id, train_checkpoint_path = _resolve_train_artifacts(TRAIN_CONTINUE_JOB_ID)
        print('Evaluation lock        = dataset_id', train_dataset_id, '| model_id', train_model_id)
        runtime_eval_cfg_path, selected_model_id = _prepare_eval_runtime_config(
            locked_dataset_id=train_dataset_id,
            locked_model_id=train_model_id,
            locked_checkpoint_path=train_checkpoint_path,
        )
        eval_job_id = f'{EVALUATION_JOB_ID}_continue'
        eval_cmd = (
            f'cd {shlex.quote(WORKTREE)} && '
            f'PYTHONPATH={shlex.quote(f"{WORKTREE}/src")} '
            f'{shlex.quote(PYTHON_BIN)} -m songo_model_stockfish.cli.main '
            f'evaluate --config {shlex.quote(str(runtime_eval_cfg_path))} '
            f'--job-id {shlex.quote(eval_job_id)}'
        )
        subprocess.run(['/bin/bash', '-lc', eval_cmd], check=True)
        print('Train + Eval termines | mode=continue | model_id=', selected_model_id, '| eval_job_id=', eval_job_id)
        _print_post_train_eval_report(
            mode='continue',
            train_job_id=TRAIN_CONTINUE_JOB_ID,
            eval_job_id=eval_job_id,
            model_id=selected_model_id,
        )
        """
    ),
    code(
        """
        # Cellule B: train from scratch (modele de 0) puis evaluation automatique
        print('Cellule B: train scratch + evaluate')
        print('train config =', TRAIN_SCRATCH_CONFIG_ACTIVE)

        train_cmd = (
            f'cd {shlex.quote(WORKTREE)} && '
            f'PYTHONPATH={shlex.quote(f"{WORKTREE}/src")} '
            f'{shlex.quote(PYTHON_BIN)} -m songo_model_stockfish.cli.main '
            f'train --config {shlex.quote(str(TRAIN_SCRATCH_CONFIG_ACTIVE))} '
            f'--job-id {shlex.quote(TRAIN_SCRATCH_JOB_ID)}'
        )
        subprocess.run(['/bin/bash', '-lc', train_cmd], check=True)

        train_dataset_id, train_model_id, train_checkpoint_path = _resolve_train_artifacts(TRAIN_SCRATCH_JOB_ID)
        print('Evaluation lock        = dataset_id', train_dataset_id, '| model_id', train_model_id)
        runtime_eval_cfg_path, selected_model_id = _prepare_eval_runtime_config(
            locked_dataset_id=train_dataset_id,
            locked_model_id=train_model_id,
            locked_checkpoint_path=train_checkpoint_path,
        )
        eval_job_id = f'{EVALUATION_JOB_ID}_scratch'
        eval_cmd = (
            f'cd {shlex.quote(WORKTREE)} && '
            f'PYTHONPATH={shlex.quote(f"{WORKTREE}/src")} '
            f'{shlex.quote(PYTHON_BIN)} -m songo_model_stockfish.cli.main '
            f'evaluate --config {shlex.quote(str(runtime_eval_cfg_path))} '
            f'--job-id {shlex.quote(eval_job_id)}'
        )
        subprocess.run(['/bin/bash', '-lc', eval_cmd], check=True)
        print('Train + Eval termines | mode=scratch | model_id=', selected_model_id, '| eval_job_id=', eval_job_id)
        _print_post_train_eval_report(
            mode='scratch',
            train_job_id=TRAIN_SCRATCH_JOB_ID,
            eval_job_id=eval_job_id,
            model_id=selected_model_id,
        )
        """
    ),
    md("## 8. Benchmark"),
    code(
        """
        !bash -lc "cd $WORKTREE && PYTHONPATH=$WORKTREE/src $PYTHON_BIN -m songo_model_stockfish.cli.main benchmark --config $BENCHMARK_CONFIG_ACTIVE --job-id $BENCHMARK_JOB_ID"
        """
    ),
    md("## 9. Tournoi Inter-Modeles (3/1/0)"),
    code(
        """
        # Tournoi round-robin entre modeles uniquement (sans minimax/mcts)
        import json
        import itertools
        import os
        import platform
        import shutil
        import socket
        import sys
        import time
        import uuid
        from contextlib import contextmanager
        from pathlib import Path
        from datetime import UTC, datetime

        sys.path.insert(0, f"{WORKTREE}/src")
        from songo_model_stockfish.ops.model_registry import load_registry, save_registry, promote_best_model

        TOURNAMENT_GAMES_PER_PAIR = 8   # total games par paire
        TOURNAMENT_MAX_MOVES = 300
        TOURNAMENT_DEVICE = 'cpu'
        TOURNAMENT_MIN_MODELS = 2
        TOURNAMENT_WRITE_REPORT = True
        TOURNAMENT_MODEL_SEARCH_ENABLED = True
        TOURNAMENT_MODEL_SEARCH_TOP_K = 4
        TOURNAMENT_MODEL_SEARCH_POLICY_WEIGHT = 0.35
        TOURNAMENT_MODEL_SEARCH_VALUE_WEIGHT = 1.0
        TOURNAMENT_LOG_EACH_GAME = False
        TOURNAMENT_INCLUDE_GAME_LOGS_IN_REPORT = False
        TOURNAMENT_MAX_GAME_LOGS_PER_PAIR = 0  # 0 = illimite
        TOURNAMENT_PARALLEL_ENABLED = True
        TOURNAMENT_PARALLEL_BACKEND = 'process'  # 'process', 'thread', 'sequential'
        TOURNAMENT_MAX_PARALLEL_PAIRS = 4
        TOURNAMENT_PARALLEL_FALLBACK_SEQUENTIAL = True
        TOURNAMENT_PARALLEL_SECONDARY_BACKEND = 'thread'
        TOURNAMENT_PROCESS_START_METHOD = 'auto'  # 'auto', 'spawn', 'forkserver', 'fork'
        TOURNAMENT_CPU_THREADS_PER_WORKER = 1
        TOURNAMENT_AUTO_CAP_PARALLEL_BY_CPU = True
        TOURNAMENT_MAX_PARALLEL_PAIRS_HARD_CAP = 16
        TOURNAMENT_RETRY_FAILED_PAIRS = 1
        TOURNAMENT_RETRY_FAILED_PAIRS_BACKEND = 'thread'
        TOURNAMENT_AUTO_ACTIONS_MIN_GAMES_PER_PAIR = 20
        TOURNAMENT_DISABLE_AUTO_ACTIONS_WHEN_LOW_GAMES = True
        TOURNAMENT_GLOBAL_LOCK_ENABLED = True
        TOURNAMENT_GLOBAL_LOCK_BACKEND = 'firestore'  # 'firestore' ou 'drive'
        TOURNAMENT_GLOBAL_LOCK_COLLECTION = 'tournament_locks'
        TOURNAMENT_GLOBAL_LOCK_ID = 'inter_models_rankings'
        TOURNAMENT_GLOBAL_LOCK_TTL_SECONDS = 1800
        TOURNAMENT_GLOBAL_LOCK_WAIT_SECONDS = 120
        TOURNAMENT_GLOBAL_LOCK_POLL_SECONDS = 2.0
        TOURNAMENT_ABORT_AUTO_ACTIONS_IF_MODEL_SET_CHANGED = True
        TOURNAMENT_AUTO_PRUNE_ENABLED = True
        TOURNAMENT_AUTO_PRUNE_COUNT = 3
        TOURNAMENT_MIN_MODELS_TO_KEEP = 3
        TOURNAMENT_AUTO_PROMOTE_WINNER = True

        models_root = Path(DRIVE_ROOT) / 'models'
        registry_path = models_root / 'model_registry.json'
        final_dir = models_root / 'final'

        registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {'models': []}
        records = registry.get('models', []) if isinstance(registry, dict) else []

        # Dedup par model_id, garde l'entree la plus recente (source registre)
        ordered = sorted(records, key=lambda item: float(item.get('sort_ts', 0.0)), reverse=True)
        seen_ids = set()
        models = []
        for item in ordered:
            model_id = str(item.get('model_id', '')).strip()
            if not model_id or model_id in seen_ids:
                continue
            checkpoint = Path(str(item.get('checkpoint_path', '')).strip())
            if not checkpoint.exists():
                fallback = final_dir / f'{model_id}.pt'
                checkpoint = fallback if fallback.exists() else checkpoint
            if checkpoint.exists():
                models.append({'model_id': model_id, 'checkpoint_path': str(checkpoint)})
                seen_ids.add(model_id)

        # Fallback robuste: scan direct models/final/*.pt meme si registre vide/incomplet
        if final_dir.exists():
            for pt in sorted(final_dir.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True):
                model_id = pt.stem.strip()
                if not model_id or model_id in seen_ids:
                    continue
                models.append({'model_id': model_id, 'checkpoint_path': str(pt)})
                seen_ids.add(model_id)

        if len(models) < TOURNAMENT_MIN_MODELS:
            raise ValueError(f'Pas assez de modeles pour tournoi: {len(models)} (min={TOURNAMENT_MIN_MODELS})')

        print('Modeles retenus pour tournoi =', len(models))
        for m in models:
            print(' -', m['model_id'])

        model_ids = [m['model_id'] for m in models]
        model_checkpoint_by_id = {m['model_id']: str(m['checkpoint_path']) for m in models}
        fixtures = [{'left': left, 'right': right} for left, right in itertools.combinations(model_ids, 2)]
        print('Paires a jouer =', len(fixtures))
        if len(fixtures) == 0:
            raise ValueError('Tournoi impossible: aucune paire generee.')

        auto_actions_allowed = True
        min_games_for_auto_actions = max(1, int(TOURNAMENT_AUTO_ACTIONS_MIN_GAMES_PER_PAIR))
        if bool(TOURNAMENT_DISABLE_AUTO_ACTIONS_WHEN_LOW_GAMES) and int(TOURNAMENT_GAMES_PER_PAIR) < min_games_for_auto_actions:
            auto_actions_allowed = False
            print(
                'auto_actions_guard     = disabled (games_per_pair insuffisant) | '
                f'games_per_pair={int(TOURNAMENT_GAMES_PER_PAIR)} < min_required={min_games_for_auto_actions}'
            )
        else:
            print(
                'auto_actions_guard     = enabled | '
                f'games_per_pair={int(TOURNAMENT_GAMES_PER_PAIR)}'
            )

        cpu_count = max(1, int(os.cpu_count() or 1))
        requested_parallel_workers = max(1, int(TOURNAMENT_MAX_PARALLEL_PAIRS))
        hard_cap = max(1, int(TOURNAMENT_MAX_PARALLEL_PAIRS_HARD_CAP))
        requested_parallel_workers = min(requested_parallel_workers, hard_cap)
        if bool(TOURNAMENT_AUTO_CAP_PARALLEL_BY_CPU):
            requested_parallel_workers = min(requested_parallel_workers, cpu_count)
        effective_parallel_workers = min(requested_parallel_workers, max(1, len(fixtures)))
        print(
            'Parallelisme tournoi =',
            (
                f"{effective_parallel_workers} pair(s) simultanees"
                if bool(TOURNAMENT_PARALLEL_ENABLED)
                and str(TOURNAMENT_PARALLEL_BACKEND).strip().lower() != 'sequential'
                and effective_parallel_workers > 1
                else '1 paire simultanee (sequentiel)'
            ),
            '| cpu_count =', cpu_count,
            '| max_pairs_requested =', TOURNAMENT_MAX_PARALLEL_PAIRS,
            '| hard_cap =', TOURNAMENT_MAX_PARALLEL_PAIRS_HARD_CAP,
        )

        def _apply_runtime_thread_limits(threads: int) -> None:
            n_threads = max(1, int(threads))
            os.environ['OMP_NUM_THREADS'] = str(n_threads)
            os.environ['MKL_NUM_THREADS'] = str(n_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
            os.environ['BLIS_NUM_THREADS'] = str(n_threads)
            try:
                import torch
                torch.set_num_threads(n_threads)
                if hasattr(torch, 'set_num_interop_threads'):
                    torch.set_num_interop_threads(1)
            except Exception:
                pass

        _apply_runtime_thread_limits(int(TOURNAMENT_CPU_THREADS_PER_WORKER))
        print('CPU threads/worker    =', int(TOURNAMENT_CPU_THREADS_PER_WORKER))

        def _winner_label_for_pair(winner: int | None, left: str, right: str) -> str:
            if winner == 0:
                return left
            if winner == 1:
                return right
            return 'draw'

        def _resolve_process_start_method(preferred: str) -> str:
            import multiprocessing as mp

            preferred_norm = str(preferred).strip().lower()
            methods = [str(m).strip().lower() for m in mp.get_all_start_methods()]
            if preferred_norm and preferred_norm != 'auto':
                if preferred_norm in methods:
                    return preferred_norm
                raise ValueError(f'Process start method non supporte: {preferred_norm} (available={methods})')
            # auto: priorite portabilite/stabilite, pas fork en premier.
            for candidate in ('forkserver', 'spawn', 'fork'):
                if candidate in methods:
                    return candidate
            return mp.get_start_method(allow_none=True) or 'spawn'

        def _run_pair_series(task: dict) -> dict:
            from songo_model_stockfish.benchmark.model_agent import ModelAgent as _ModelAgent
            from songo_model_stockfish.benchmark.play_match import play_match as _play_match

            left = str(task['left'])
            right = str(task['right'])
            left_checkpoint_path = str(task['left_checkpoint_path'])
            right_checkpoint_path = str(task['right_checkpoint_path'])
            games = int(task['games'])
            max_moves = int(task['max_moves'])
            device = str(task['device'])
            search_enabled = bool(task['search_enabled'])
            search_top_k = int(task['search_top_k'])
            search_policy_weight = float(task['search_policy_weight'])
            search_value_weight = float(task['search_value_weight'])
            capture_game_logs = bool(task.get('capture_game_logs', False))
            max_game_logs = int(task.get('max_game_logs', 0))
            thread_limit = int(task.get('thread_limit', 1))

            _apply_runtime_thread_limits(thread_limit)

            left_agent = _ModelAgent(
                left_checkpoint_path,
                display_name=left,
                device=device,
                search_enabled=search_enabled,
                search_top_k=search_top_k,
                search_policy_weight=search_policy_weight,
                search_value_weight=search_value_weight,
            )
            right_agent = _ModelAgent(
                right_checkpoint_path,
                display_name=right,
                device=device,
                search_enabled=search_enabled,
                search_top_k=search_top_k,
                search_policy_weight=search_policy_weight,
                search_value_weight=search_value_weight,
            )

            wins_left = 0
            wins_right = 0
            draws = 0
            total_moves_pair = 0
            game_logs: list[dict] = []
            game_logs_truncated = False
            for game_idx in range(games):
                starter = game_idx % 2
                result = _play_match(left_agent, right_agent, max_moves=max_moves, starter=starter)
                if result.winner == 0:
                    wins_left += 1
                elif result.winner == 1:
                    wins_right += 1
                else:
                    draws += 1
                moves = int(result.moves)
                total_moves_pair += moves
                if capture_game_logs:
                    if max_game_logs <= 0 or len(game_logs) < max_game_logs:
                        game_logs.append(
                            {
                                'game_index': game_idx + 1,
                                'starter': 'left' if starter == 0 else 'right',
                                'winner_label': _winner_label_for_pair(result.winner, left, right),
                                'moves': moves,
                                'reason': str(result.reason),
                            }
                        )
                    else:
                        game_logs_truncated = True
            return {
                'model_a': left,
                'model_b': right,
                'games': games,
                'wins_a': wins_left,
                'wins_b': wins_right,
                'draws': draws,
                'points_a': (wins_left * 3) + draws,
                'points_b': (wins_right * 3) + draws,
                'total_moves': int(total_moves_pair),
                'avg_moves': (float(total_moves_pair) / float(games)) if games > 0 else 0.0,
                'game_logs': game_logs if capture_game_logs else [],
                'game_logs_truncated': bool(game_logs_truncated),
            }

        capture_game_logs = bool(TOURNAMENT_LOG_EACH_GAME) or bool(TOURNAMENT_INCLUDE_GAME_LOGS_IN_REPORT)
        task_payloads = [
            {
                'left': task['left'],
                'right': task['right'],
                'left_checkpoint_path': model_checkpoint_by_id[task['left']],
                'right_checkpoint_path': model_checkpoint_by_id[task['right']],
                'games': int(TOURNAMENT_GAMES_PER_PAIR),
                'max_moves': int(TOURNAMENT_MAX_MOVES),
                'device': str(TOURNAMENT_DEVICE),
                'search_enabled': bool(TOURNAMENT_MODEL_SEARCH_ENABLED),
                'search_top_k': int(TOURNAMENT_MODEL_SEARCH_TOP_K),
                'search_policy_weight': float(TOURNAMENT_MODEL_SEARCH_POLICY_WEIGHT),
                'search_value_weight': float(TOURNAMENT_MODEL_SEARCH_VALUE_WEIGHT),
                'capture_game_logs': bool(capture_game_logs),
                'max_game_logs': int(TOURNAMENT_MAX_GAME_LOGS_PER_PAIR),
                'thread_limit': int(TOURNAMENT_CPU_THREADS_PER_WORKER),
            }
            for task in fixtures
        ]

        def _execute_with_backend(payloads: list[dict], backend_name: str, max_workers: int) -> tuple[list[dict], list[dict], str]:
            backend = str(backend_name).strip().lower()
            workers = max(1, min(int(max_workers), len(payloads)))
            results: list[dict] = []
            failed: list[dict] = []
            backend_error = ''
            try:
                if backend == 'thread':
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        future_map = {pool.submit(_run_pair_series, payload): payload for payload in payloads}
                        for future in as_completed(future_map):
                            payload = future_map[future]
                            try:
                                results.append(future.result())
                            except Exception as exc:
                                failed.append({'payload': payload, 'error': f'{type(exc).__name__}: {exc}'})
                elif backend == 'process':
                    import multiprocessing as mp
                    from concurrent.futures import ProcessPoolExecutor, as_completed

                    method = _resolve_process_start_method(str(TOURNAMENT_PROCESS_START_METHOD))
                    print(f'process start_method     = {method}')
                    mp_ctx = mp.get_context(method)
                    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
                        future_map = {pool.submit(_run_pair_series, payload): payload for payload in payloads}
                        for future in as_completed(future_map):
                            payload = future_map[future]
                            try:
                                results.append(future.result())
                            except Exception as exc:
                                failed.append({'payload': payload, 'error': f'{type(exc).__name__}: {exc}'})
                else:
                    raise ValueError(f'Backend parallel inconnu: {backend}')
            except Exception as exc:
                backend_error = f'{type(exc).__name__}: {exc}'
            return results, failed, backend_error

        def _execute_sequential(payloads: list[dict]) -> tuple[list[dict], list[dict]]:
            results: list[dict] = []
            failed: list[dict] = []
            for payload in payloads:
                try:
                    results.append(_run_pair_series(payload))
                except Exception as exc:
                    failed.append({'payload': payload, 'error': f'{type(exc).__name__}: {exc}'})
            return results, failed

        pair_results: list[dict] = []
        pending_payloads = list(task_payloads)
        parallel_mode = (
            bool(TOURNAMENT_PARALLEL_ENABLED)
            and str(TOURNAMENT_PARALLEL_BACKEND).strip().lower() != 'sequential'
            and effective_parallel_workers > 1
        )
        if parallel_mode:
            primary_backend = str(TOURNAMENT_PARALLEL_BACKEND).strip().lower()
            print(f'Execution parallelisee: backend={primary_backend} | workers={effective_parallel_workers}')
            current_payloads = list(pending_payloads)
            results, failed, backend_error = _execute_with_backend(current_payloads, primary_backend, effective_parallel_workers)
            pair_results.extend(results)
            pending_payloads = list(current_payloads) if (backend_error and not results and not failed) else [item['payload'] for item in failed]
            for item in failed[:5]:
                print('pair_failed(primary)   =', item['error'])
            if len(failed) > 5:
                print(f'pair_failed(primary)   = +{len(failed) - 5} erreurs')
            if backend_error:
                print('parallel_backend_error =', backend_error)

            secondary_backend = str(TOURNAMENT_PARALLEL_SECONDARY_BACKEND).strip().lower()
            if pending_payloads and secondary_backend and secondary_backend not in ('', primary_backend, 'sequential'):
                print(f'Retry backend secondaire: backend={secondary_backend} | pairs={len(pending_payloads)}')
                secondary_workers = min(effective_parallel_workers, max(1, len(pending_payloads)))
                current_payloads = list(pending_payloads)
                sec_results, sec_failed, sec_backend_error = _execute_with_backend(current_payloads, secondary_backend, secondary_workers)
                pair_results.extend(sec_results)
                pending_payloads = list(current_payloads) if (sec_backend_error and not sec_results and not sec_failed) else [item['payload'] for item in sec_failed]
                for item in sec_failed[:5]:
                    print('pair_failed(secondary) =', item['error'])
                if len(sec_failed) > 5:
                    print(f'pair_failed(secondary) = +{len(sec_failed) - 5} erreurs')
                if sec_backend_error:
                    print('secondary_backend_error =', sec_backend_error)

            retry_rounds = max(0, int(TOURNAMENT_RETRY_FAILED_PAIRS))
            retry_backend = str(TOURNAMENT_RETRY_FAILED_PAIRS_BACKEND).strip().lower() or 'thread'
            for retry_idx in range(retry_rounds):
                if not pending_payloads:
                    break
                print(f'Retry pairs {retry_idx + 1}/{retry_rounds} | backend={retry_backend} | pairs={len(pending_payloads)}')
                retry_workers = min(effective_parallel_workers, max(1, len(pending_payloads)))
                current_payloads = list(pending_payloads)
                retry_results, retry_failed, retry_backend_error = _execute_with_backend(current_payloads, retry_backend, retry_workers)
                pair_results.extend(retry_results)
                pending_payloads = list(current_payloads) if (retry_backend_error and not retry_results and not retry_failed) else [item['payload'] for item in retry_failed]
                for item in retry_failed[:5]:
                    print(f'pair_failed(retry#{retry_idx + 1}) =', item['error'])
                if len(retry_failed) > 5:
                    print(f'pair_failed(retry#{retry_idx + 1}) = +{len(retry_failed) - 5} erreurs')
                if retry_backend_error:
                    print(f'retry_backend_error#{retry_idx + 1} =', retry_backend_error)

        if not parallel_mode:
            print('Execution sequentielle: parallel_mode=off')

        if pending_payloads:
            if bool(TOURNAMENT_PARALLEL_FALLBACK_SEQUENTIAL):
                print(f'Fallback sequentiel cible | pairs restantes={len(pending_payloads)}')
                seq_results, seq_failed = _execute_sequential(pending_payloads)
                pair_results.extend(seq_results)
                pending_payloads = [item['payload'] for item in seq_failed]
                for item in seq_failed[:5]:
                    print('pair_failed(sequential) =', item['error'])
                if len(seq_failed) > 5:
                    print(f'pair_failed(sequential) = +{len(seq_failed) - 5} erreurs')
            else:
                raise RuntimeError(
                    f'{len(pending_payloads)} paires en echec et fallback sequentiel desactive.'
                )

        if pending_payloads:
            failed_pairs = [f"{item['left']} vs {item['right']}" for item in pending_payloads]
            raise RuntimeError(
                'Tournoi incomplet: certaines paires ont echoue apres retries. '
                f'failed_pairs={failed_pairs}'
            )

        table = {
            m['model_id']: {
                'model_id': m['model_id'],
                'points': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'played': 0,
                'max_points': 0,
                'score_rate': 0.0,
            }
            for m in models
        }

        pair_summaries = []
        pair_results_by_key = {
            (str(item.get('model_a', '')).strip(), str(item.get('model_b', '')).strip()): item
            for item in pair_results
            if isinstance(item, dict)
        }
        if len(pair_results_by_key) != len(pair_results):
            raise RuntimeError(
                'Resultats tournoi invalides: paires dupliquees detectees '
                f'(results={len(pair_results)}, unique_pairs={len(pair_results_by_key)}).'
            )
        if len(pair_results_by_key) != len(fixtures):
            raise RuntimeError(
                'Resultats tournoi incomplets: nombre de paires inattendu '
                f'(expected={len(fixtures)}, received={len(pair_results_by_key)}).'
            )
        for task in fixtures:
            left = str(task['left'])
            right = str(task['right'])
            series = pair_results_by_key.get((left, right))
            if series is None:
                raise RuntimeError(f'Resultat de paire introuvable: {left} vs {right}')

            wins_left = int(series.get('wins_a', 0))
            wins_right = int(series.get('wins_b', 0))
            draws = int(series.get('draws', 0))
            total_moves_pair = int(series.get('total_moves', 0))
            games_for_pair = int(series.get('games', TOURNAMENT_GAMES_PER_PAIR))
            pair_game_logs = list(series.get('game_logs', []))
            pair_logs_truncated = bool(series.get('game_logs_truncated', False))
            if (wins_left + wins_right + draws) != games_for_pair:
                raise RuntimeError(
                    f'Incoherence score paire {left} vs {right}: '
                    f'wins_left+wins_right+draws={wins_left + wins_right + draws} '
                    f'!= games={games_for_pair}'
                )

            if TOURNAMENT_LOG_EACH_GAME:
                for game_row in pair_game_logs:
                    print(
                        f"game {int(game_row.get('game_index', 0)):>3}/{games_for_pair:<3} | "
                        f"{left} vs {right} | starter={str(game_row.get('starter', '<none>'))} | "
                        f"winner={str(game_row.get('winner_label', '<none>'))} | "
                        f"moves={int(game_row.get('moves', 0))} | reason={str(game_row.get('reason', '<none>'))}"
                    )
                if pair_logs_truncated:
                    print(
                        f'game_logs_truncated   = yes | pair={left} vs {right} | '
                        f'max_game_logs_per_pair={int(TOURNAMENT_MAX_GAME_LOGS_PER_PAIR)}'
                    )
            if not TOURNAMENT_INCLUDE_GAME_LOGS_IN_REPORT:
                series['game_logs'] = []

            table[left]['played'] += games_for_pair
            table[right]['played'] += games_for_pair
            table[left]['wins'] += wins_left
            table[left]['draws'] += draws
            table[left]['losses'] += wins_right
            table[right]['wins'] += wins_right
            table[right]['draws'] += draws
            table[right]['losses'] += wins_left

            table[left]['points'] += (wins_left * 3) + draws
            table[right]['points'] += (wins_right * 3) + draws
            table[left]['max_points'] += games_for_pair * 3
            table[right]['max_points'] += games_for_pair * 3

            pair_summary = {
                'model_a': left,
                'model_b': right,
                'games': games_for_pair,
                'wins_a': wins_left,
                'wins_b': wins_right,
                'draws': draws,
                'points_a': (wins_left * 3) + draws,
                'points_b': (wins_right * 3) + draws,
                'total_moves': total_moves_pair,
                'avg_moves': (float(total_moves_pair) / float(games_for_pair)) if games_for_pair > 0 else 0.0,
                'game_logs_truncated': pair_logs_truncated,
            }
            if TOURNAMENT_INCLUDE_GAME_LOGS_IN_REPORT:
                pair_summary['game_logs'] = pair_game_logs
            pair_summaries.append(pair_summary)
            print(
                f"pair done: {left} vs {right} | "
                f"W={wins_left}-{wins_right} D={draws} | "
                f"score={((wins_left * 3) + draws)}-{((wins_right * 3) + draws)} | "
                f"avg_moves={pair_summary['avg_moves']:.2f}"
            )

        total_games_played = sum(int(item.get('games', 0)) for item in pair_summaries)
        total_draws_played = sum(int(item.get('draws', 0)) for item in pair_summaries)
        total_points_expected = (3 * total_games_played) - total_draws_played
        total_points_actual = sum(int(row.get('points', 0)) for row in table.values())
        if total_points_actual != total_points_expected:
            raise RuntimeError(
                'Incoherence points tournoi: '
                f'actual={total_points_actual} vs expected={total_points_expected} '
                f'(games={total_games_played}, draws={total_draws_played}).'
            )
        print(
            'integrity_check        = ok | '
            f'total_games={total_games_played} | total_draws={total_draws_played} | '
            f'total_points={total_points_actual}'
        )

        for row in table.values():
            max_pts = int(row['max_points'])
            row['score_rate'] = (float(row['points']) / float(max_pts)) if max_pts > 0 else 0.0

        ranking = sorted(
            table.values(),
            key=lambda row: (
                int(row['points']),
                int(row['wins']),
                int(row['draws']),
                -int(row['losses']),
                row['model_id'],
            ),
            reverse=True,
        )

        print('\\n=== Classement Tournoi (models uniquement) ===')
        print(f"{'Rk':>2} | {'Model':<40} | {'Pts':>4} | {'Score':>9} | {'W':>4} | {'D':>4} | {'L':>4} | {'G':>4}")
        print('-' * 92)
        for idx, row in enumerate(ranking, start=1):
            print(
                f\"{idx:>2} | {row['model_id']:<40} | {row['points']:>4} | {row['points']:>4}/{row['max_points']:<4} | {row['wins']:>4} | {row['draws']:>4} | {row['losses']:>4} | {row['played']:>4}\"
            )

        winner_model_id = str(ranking[0]['model_id']).strip()
        print('\\n=== Actions Automatiques Tournoi ===')
        print('winner_model_id =', winner_model_id)

        def _safe_unlink(path: Path) -> bool:
            try:
                if path.exists():
                    path.unlink()
                    return True
            except Exception:
                return False
            return False

        def _remove_model_artifacts(model_id: str) -> dict:
            removed = []
            checkpoints_dir = models_root / 'checkpoints'
            lineage_dir = models_root / 'lineage'
            candidates = [
                final_dir / f'{model_id}.pt',
                final_dir / f'{model_id}.model_card.json',
                lineage_dir / f'{model_id}_parent_snapshot.pt',
            ]
            if checkpoints_dir.exists():
                candidates.extend(sorted(checkpoints_dir.glob(f'{model_id}*.pt')))
            for path in candidates:
                if _safe_unlink(path):
                    removed.append(str(path))
            return {'model_id': model_id, 'removed_paths': removed, 'removed_count': len(removed)}
        def _acquire_firestore_lock(owner_id: str) -> dict:
            from google.cloud import firestore

            client = _get_firestore_client()
            collection = str(TOURNAMENT_GLOBAL_LOCK_COLLECTION).strip() or 'tournament_locks'
            doc_ref = client.collection(collection).document(str(TOURNAMENT_GLOBAL_LOCK_ID).strip() or 'inter_models_rankings')
            ttl_seconds = max(30.0, float(TOURNAMENT_GLOBAL_LOCK_TTL_SECONDS))
            wait_seconds = max(1.0, float(TOURNAMENT_GLOBAL_LOCK_WAIT_SECONDS))
            poll_seconds = max(0.25, float(TOURNAMENT_GLOBAL_LOCK_POLL_SECONDS))
            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                tx = client.transaction()

                @firestore.transactional
                def _try_acquire(transaction):
                    snap = doc_ref.get(transaction=transaction)
                    payload = snap.to_dict() if snap.exists else {}
                    if not isinstance(payload, dict):
                        payload = {}
                    now = time.time()
                    lock_owner = str(payload.get('owner', '')).strip()
                    expires_at_epoch = float(payload.get('expires_at_epoch', 0.0) or 0.0)
                    if lock_owner and expires_at_epoch > now and lock_owner != owner_id:
                        return {'acquired': False, 'owner': lock_owner, 'expires_at_epoch': expires_at_epoch}
                    new_payload = {
                        'owner': owner_id,
                        'acquired_at_epoch': now,
                        'expires_at_epoch': now + ttl_seconds,
                        'updated_at': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                        'hostname': socket.gethostname(),
                        'platform': platform.platform(),
                    }
                    transaction.set(doc_ref, new_payload)
                    return {'acquired': True, 'owner': owner_id, 'expires_at_epoch': now + ttl_seconds}

                result = _try_acquire(tx)
                if bool(result.get('acquired', False)):
                    result.update({'backend': 'firestore', 'client': client, 'doc_ref': doc_ref})
                    return result
                time.sleep(poll_seconds)
            raise TimeoutError(
                f'Impossible d obtenir le lock Firestore {collection}/{TOURNAMENT_GLOBAL_LOCK_ID} '
                f'apres {wait_seconds}s.'
            )

        def _release_firestore_lock(lock_meta: dict, owner_id: str) -> None:
            from google.cloud import firestore

            client = lock_meta.get('client')
            doc_ref = lock_meta.get('doc_ref')
            if client is None or doc_ref is None:
                return
            tx = client.transaction()

            @firestore.transactional
            def _release(transaction):
                snap = doc_ref.get(transaction=transaction)
                if not snap.exists:
                    return False
                payload = snap.to_dict() or {}
                lock_owner = str(payload.get('owner', '')).strip()
                if lock_owner == owner_id:
                    transaction.delete(doc_ref)
                    return True
                return False

            _release(tx)

        def _acquire_drive_lock(owner_id: str) -> dict:
            lock_dir = Path(DRIVE_ROOT) / 'locks' / f"{str(TOURNAMENT_GLOBAL_LOCK_ID).strip() or 'inter_models_rankings'}.lock"
            ttl_seconds = max(30.0, float(TOURNAMENT_GLOBAL_LOCK_TTL_SECONDS))
            wait_seconds = max(1.0, float(TOURNAMENT_GLOBAL_LOCK_WAIT_SECONDS))
            poll_seconds = max(0.25, float(TOURNAMENT_GLOBAL_LOCK_POLL_SECONDS))
            lock_dir.parent.mkdir(parents=True, exist_ok=True)
            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                try:
                    lock_dir.mkdir(parents=False, exist_ok=False)
                    (lock_dir / 'owner.txt').write_text(owner_id, encoding='utf-8')
                    return {'acquired': True, 'backend': 'drive', 'owner': owner_id, 'lock_dir': lock_dir}
                except FileExistsError:
                    try:
                        stale = (time.time() - float(lock_dir.stat().st_mtime)) > ttl_seconds
                    except Exception:
                        stale = False
                    if stale:
                        shutil.rmtree(lock_dir, ignore_errors=True)
                        continue
                    time.sleep(poll_seconds)
            raise TimeoutError(f'Impossible d obtenir le lock Drive {lock_dir} apres {wait_seconds}s.')

        def _release_drive_lock(lock_meta: dict, owner_id: str) -> None:
            lock_dir = lock_meta.get('lock_dir')
            if not isinstance(lock_dir, Path):
                return
            owner_file = lock_dir / 'owner.txt'
            try:
                current_owner = owner_file.read_text(encoding='utf-8').strip() if owner_file.exists() else ''
            except Exception:
                current_owner = ''
            if current_owner and current_owner != owner_id:
                return
            try:
                shutil.rmtree(lock_dir, ignore_errors=True)
            except Exception:
                pass

        @contextmanager
        def _tournament_global_lock():
            lock_enabled = bool(TOURNAMENT_GLOBAL_LOCK_ENABLED)
            owner_id = (
                f"{str(WORKER_TAG).strip() or 'worker'}:"
                f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
            )
            if not lock_enabled:
                yield {'acquired': False, 'backend': 'disabled', 'owner': owner_id}
                return
            backend = str(TOURNAMENT_GLOBAL_LOCK_BACKEND).strip().lower()
            lock_meta = {}
            if backend == 'firestore':
                try:
                    lock_meta = _acquire_firestore_lock(owner_id)
                except Exception as exc:
                    print('global_lock_firestore_error =', f'{type(exc).__name__}: {exc}')
                    print('global_lock_fallback        = drive')
                    lock_meta = _acquire_drive_lock(owner_id)
            elif backend == 'drive':
                lock_meta = _acquire_drive_lock(owner_id)
            else:
                raise ValueError(f'TOURNAMENT_GLOBAL_LOCK_BACKEND non supporte: {backend}')
            try:
                yield lock_meta
            finally:
                try:
                    if str(lock_meta.get('backend', '')).strip() == 'firestore':
                        _release_firestore_lock(lock_meta, owner_id)
                    elif str(lock_meta.get('backend', '')).strip() == 'drive':
                        _release_drive_lock(lock_meta, owner_id)
                except Exception:
                    pass

        prune_ids = []
        prune_details = []
        promoted_meta = None
        new_models = []
        lock_metadata = {'acquired': False, 'backend': 'disabled'}
        auto_actions_applied = bool(auto_actions_allowed)
        auto_actions_reason = 'ok'

        expected_model_set = sorted(str(mid).strip() for mid in model_ids if str(mid).strip())

        with _tournament_global_lock() as lock_metadata:
            print(
                'global_lock            =',
                str(lock_metadata.get('backend', '<none>')),
                '| acquired =',
                bool(lock_metadata.get('acquired', True)),
            )
            if bool(TOURNAMENT_ABORT_AUTO_ACTIONS_IF_MODEL_SET_CHANGED):
                live_model_set = sorted(
                    p.stem.strip()
                    for p in final_dir.glob('*.pt')
                    if p.is_file() and p.stem.strip()
                ) if final_dir.exists() else []
                if live_model_set != expected_model_set:
                    auto_actions_applied = False
                    auto_actions_reason = 'model_set_changed_during_tournament'
                    print('auto_actions_guard     = disabled | reason=model_set_changed_during_tournament')

            if not auto_actions_applied:
                print('auto_prune            = skipped | reason=', auto_actions_reason)
                print('registry_sync         = skipped | reason=', auto_actions_reason)
                print('auto_promote_winner   = skipped | reason=', auto_actions_reason)
            else:
                if TOURNAMENT_AUTO_PRUNE_ENABLED and len(ranking) > TOURNAMENT_MIN_MODELS_TO_KEEP:
                    prune_count = min(
                        int(TOURNAMENT_AUTO_PRUNE_COUNT),
                        max(0, len(ranking) - int(TOURNAMENT_MIN_MODELS_TO_KEEP)),
                    )
                    if prune_count > 0:
                        prune_ids = [str(row.get('model_id', '')).strip() for row in ranking[-prune_count:]]
                        print('auto_prune            = enabled')
                        print('prune_count           =', prune_count)
                        print('pruned_models         =', prune_ids)
                        for model_id in prune_ids:
                            details = _remove_model_artifacts(model_id)
                            prune_details.append(details)
                            print(f" - removed {model_id}: files={details['removed_count']}")
                    else:
                        print('auto_prune            = enabled but nothing to prune')
                else:
                    print(
                        f'auto_prune            = skipped '
                        f'(models={len(ranking)} <= keep_threshold={TOURNAMENT_MIN_MODELS_TO_KEEP})'
                    )

                # Synchroniser model_registry.json avec le resultat du tournoi:
                # winner en tete, puis le reste, sans les modeles prunes.
                live_registry = load_registry(models_root)
                live_records = list(live_registry.get('models', [])) if isinstance(live_registry, dict) else []
                record_map = {}
                for item in live_records:
                    if not isinstance(item, dict):
                        continue
                    model_id = str(item.get('model_id', '')).strip()
                    if model_id:
                        record_map[model_id] = dict(item)

                # Completer entries manquantes avec les checkpoints detectes en tournoi.
                for model in models:
                    model_id = str(model.get('model_id', '')).strip()
                    if not model_id or model_id in prune_ids:
                        continue
                    checkpoint_path = Path(str(model.get('checkpoint_path', '')).strip())
                    if model_id not in record_map:
                        sort_ts = float(checkpoint_path.stat().st_mtime) if checkpoint_path.exists() else time.time()
                        record_map[model_id] = {
                            'model_id': model_id,
                            'checkpoint_path': str(checkpoint_path),
                            'sort_ts': sort_ts,
                            'best_validation_metric': -1.0,
                            'evaluation_top1': -1.0,
                            'benchmark_score': -1.0,
                        }
                    else:
                        rec = record_map[model_id]
                        if checkpoint_path.exists():
                            rec['checkpoint_path'] = str(checkpoint_path)
                        rec.setdefault('sort_ts', time.time())
                        record_map[model_id] = rec

                # Supprimer pruned du registre.
                for model_id in prune_ids:
                    record_map.pop(model_id, None)

                if winner_model_id not in record_map:
                    winner_ckpt = final_dir / f'{winner_model_id}.pt'
                    if not winner_ckpt.exists():
                        raise FileNotFoundError(
                            f'Winner checkpoint introuvable pour promotion: {winner_ckpt}'
                        )
                    record_map[winner_model_id] = {
                        'model_id': winner_model_id,
                        'checkpoint_path': str(winner_ckpt),
                        'sort_ts': float(winner_ckpt.stat().st_mtime),
                        'best_validation_metric': -1.0,
                        'evaluation_top1': -1.0,
                        'benchmark_score': -1.0,
                    }

                # Ordonnancement: winner d'abord, puis le reste du classement tournoi.
                ranked_ids = [str(row.get('model_id', '')).strip() for row in ranking]
                ordered_ids = [winner_model_id]
                for model_id in ranked_ids:
                    if not model_id or model_id == winner_model_id or model_id in prune_ids:
                        continue
                    if model_id in record_map:
                        ordered_ids.append(model_id)
                # Ajouter d'eventuels modeles hors tournoi encore presents dans le registre.
                for model_id in sorted(record_map.keys()):
                    if model_id not in ordered_ids:
                        ordered_ids.append(model_id)

                for idx, model_id in enumerate(ordered_ids, start=1):
                    rec = dict(record_map.get(model_id, {}))
                    if not rec:
                        continue
                    rec['rank'] = idx
                    new_models.append(rec)

                save_registry(models_root, {'models': new_models})
                print('registry_sync         = ok | models_kept =', len(new_models))

                if TOURNAMENT_AUTO_PROMOTE_WINNER:
                    promoted_meta = promote_best_model(models_root)
                    print('auto_promote_winner   = enabled')
                    print('promoted_model_id     =', promoted_meta.get('model_id', '<none>') if isinstance(promoted_meta, dict) else '<none>')
                    print('promoted_checkpoint   =', promoted_meta.get('promoted_checkpoint_path', '<none>') if isinstance(promoted_meta, dict) else '<none>')
                else:
                    print('auto_promote_winner   = disabled')

        if TOURNAMENT_WRITE_REPORT:
            out_dir = Path(DRIVE_ROOT) / 'reports' / 'benchmarks' / 'model_tournaments'
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                'created_at': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                'games_per_pair': TOURNAMENT_GAMES_PER_PAIR,
                'max_moves': TOURNAMENT_MAX_MOVES,
                'device': TOURNAMENT_DEVICE,
                'model_search_enabled': TOURNAMENT_MODEL_SEARCH_ENABLED,
                'model_search_top_k': TOURNAMENT_MODEL_SEARCH_TOP_K,
                'model_search_policy_weight': TOURNAMENT_MODEL_SEARCH_POLICY_WEIGHT,
                'model_search_value_weight': TOURNAMENT_MODEL_SEARCH_VALUE_WEIGHT,
                'models': [m['model_id'] for m in models],
                'pairs': pair_summaries,
                'ranking': ranking,
                'auto_actions': {
                    'winner_model_id': winner_model_id,
                    'auto_actions_applied': bool(auto_actions_applied),
                    'auto_actions_reason': str(auto_actions_reason),
                    'auto_actions_min_games_per_pair': int(TOURNAMENT_AUTO_ACTIONS_MIN_GAMES_PER_PAIR),
                    'games_per_pair': int(TOURNAMENT_GAMES_PER_PAIR),
                    'global_lock': {
                        'enabled': bool(TOURNAMENT_GLOBAL_LOCK_ENABLED),
                        'backend_configured': str(TOURNAMENT_GLOBAL_LOCK_BACKEND),
                        'backend_effective': str(lock_metadata.get('backend', '<none>')),
                        'acquired': bool(lock_metadata.get('acquired', True)),
                    },
                    'auto_prune_enabled': bool(TOURNAMENT_AUTO_PRUNE_ENABLED),
                    'auto_prune_count': int(TOURNAMENT_AUTO_PRUNE_COUNT),
                    'min_models_to_keep': int(TOURNAMENT_MIN_MODELS_TO_KEEP),
                    'pruned_model_ids': prune_ids,
                    'prune_details': prune_details,
                    'auto_promote_winner': bool(TOURNAMENT_AUTO_PROMOTE_WINNER),
                    'promoted_metadata': promoted_meta if isinstance(promoted_meta, dict) else {},
                    'registry_models_kept': len(new_models),
                },
            }
            stamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
            out_path = out_dir / f'model_tournament_{stamp}.json'
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')
            print('\\nReport saved:', out_path)
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
