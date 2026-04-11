# Multi-Colab Architecture 20M (Drive + Firestore + Redis)

## 1. Objectif

Definir une architecture multi-Colab claire et scalable pour atteindre `20_000_000` samples, en reduisant les quotas Firestore sans perdre la reprise fiable.

## 2. Architecture cible

### 2.1 Data plane

- Google Drive = artefacts lourds:
  - raw/sampled/labeled JSONL
  - datasets NPZ
  - checkpoints modeles
  - logs et rapports

### 2.2 Control plane durable

- Firestore = source de verite durable:
  - `global_generation_progress/{global_target_id}`
  - `dataset_registry/primary`
  - `worker_checkpoints/{job_id}`
  - `worker_leases/{global_target_id}`
  - `pipeline_manifests/{worker_tag}`

### 2.3 Real-time plane

- Redis = couche temps reel:
  - compteurs frequents (`samples`, `games`, heartbeat)
  - leases/locks rapides
  - cache de lecture pour monitoring
  - pub/sub pour dashboard live

Regle:

- Redis ne remplace pas Firestore comme source durable.
- Firestore ne doit pas etre pollue par des ecritures ultra frequentes.

## 3. Repartition des ecritures

### 3.1 Ecritures frequentes -> Redis

- increment par game terminee
- heartbeat worker
- etat court terme de progression

### 3.2 Ecritures consolidees -> Firestore

- flush periodique depuis Redis vers Firestore
- snapshots registry/checkpoints/throughput
- transitions critiques (`running`, `failed`, `completed`)

Cadence recommandee:

- flush Redis -> Firestore toutes `60` a `120` secondes
- ou tous `N` games/fichiers (batch)

## 3.3 Mode d'execution par blocs de matchs

Strategie recommande pour multi-Colab (5 -> 20 workers):

1. Un worker prend un bloc de matchup (ex: `minimax:medium vs mcts:insane`) avec un quota de games (ex: `500`).
2. Il joue ces games localement avec son parallellisme interne (ex: `16` parties en parallele).
3. Pendant ce bloc, il ecrit uniquement ses artefacts dans ses dossiers worker sur Drive (pas de melange direct avec les autres).
4. Un autre worker prend un autre bloc matchup libre (ex: `minimax:medium vs minimax:beginner`).
5. En fin de bloc (`500/500`), le worker publie son mini-dataset et lance la fusion vers le dataset principal.
6. Une fois fusion terminee et committee, le worker prend un nouveau bloc matchup.

Regle de gouvernance:

- un bloc matchup actif ne doit pas etre pris par deux workers simultanement
- allocation/lease du bloc geree via `worker_leases`
- completion du bloc marquee explicitement dans l'etat global

## 4. Règles notebook anti-energie

Pour eviter les cellules energivores:

1. Un seul notebook "monitor live" actif.
2. Les notebooks workers ne font pas de polling Firestore en boucle.
3. Monitoring lit Redis en priorite, Firestore seulement pour verification/snapshot.
4. Pas d'ecriture Firestore depuis des cellules bouclees sauf cellule de consolidation dediee.

## 4.1 Frequence de mise a jour globale (recommandation)

Ne pas choisir "uniquement 30 min" ni "uniquement fin de bloc".  
Le modele robuste est hybride:

- micro-batch periodique pendant execution (toutes `N` games finies ou `60..120s`)
- update finale obligatoire a la fin de chaque bloc (`500/500`)
- update supplementaire sur evenements critiques (`failed`, `recovered`, `merged`)

Ce modele donne:

- visibilite quasi temps reel
- quota controle
- etat global coherent meme si un worker tombe avant fin de bloc

## 5. Quota model (ordre de grandeur)

Variables:

- `g`: flush global progress (games)
- `b`: partial export build (files)
- `f`: flush Redis -> Firestore (secondes)
- `m`: refresh monitor (secondes)
- `W`: nombre de workers

Approx:

- `writes_firestore_global ~= total_games / g`
- `writes_firestore_build ~= total_files / b`
- `writes_firestore_consolidation ~= 86400 / f`
- `reads_firestore_monitor ~= monitors * docs_per_loop * (86400 / m)`

Redis absorbe la frequence elevee qui sinon saturerait Firestore.

## 6. Parametres cibles (20M)

- `GLOBAL_BUDGET_ENFORCEMENT_MODE='batched'`
- `GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES=200`
- `DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES=200..500`
- `MONITOR_REFRESH_SECONDS=90..120`
- `SOURCE_POLL_INTERVAL_SECONDS=45..60`
- `PIPELINE_MANIFEST_FIRESTORE_WRITE_ENABLED=false`
- `REDIS_SYNC_FLUSH_SECONDS=60..120`
- `MATCHUP_BLOCK_GAMES=200..500` (recommande `500`)
- `WORKER_PARALLEL_GAMES=8..16` selon CPU

## 7. Plan d'implementation doc-first

### P0

- figer l'architecture Drive + Firestore + Redis dans la doc
- definir la gouvernance des ecritures (qui ecrit quoi, ou, et quand)

### P1

- introduire client Redis + schema de cles
- router compteurs frequents vers Redis
- garder Firestore pour snapshots consolides

### P2

- throttler `worker_checkpoints`
- monitor unique Redis-first
- alertes stale worker / lag consolidation

### P3 (optimisations restantes prioritaires)

1. Throttle strict des `worker_checkpoints`
- ne pas ecrire Firestore a chaque fichier
- flush checkpoint toutes `60..120s` ou tous `N` fichiers
- flush force sur transitions critiques (`phase`, `failed`, `completed`)

2. Reduction des writes `dataset_registry`
- update uniquement en micro-batch + fin de bloc
- eviter les updates "cosmetiques" intermediaires

3. Consolidateur unique Redis -> Firestore
- un seul writer de consolidation actif a la fois
- lock distribue pour eviter les double writes concurrentes

4. Merge idempotent des mini-datasets
- chaque bloc a un `block_id` unique
- une fusion deja marquee `merged` ne doit pas etre rejouee

5. Durcissement des leases matchup
- lease TTL + heartbeat
- reprise automatique des blocs stale
- prevention explicite des collisions de bloc actif

6. Monitoring notebook optimise
- cache local (memo courte) pour eviter relectures inutiles
- une seule session monitor live en continu

7. Alerting operationnel
- alerte worker stale
- alerte lag de consolidation
- alerte quota approaching (read/write)

## 8. Schema de cles Redis (propose)

- `songo:global:{target_id}:samples` (counter)
- `songo:global:{target_id}:games` (counter)
- `songo:worker:{worker_tag}:heartbeat` (ttl key)
- `songo:worker:{worker_tag}:samples` (counter)
- `songo:worker:{worker_tag}:games` (counter)
- `songo:sync:{target_id}:last_flush_ts` (timestamp)
- `songo:block:{target_id}:{block_id}:status` (`running|completed|merged|failed`)
- `songo:block:{target_id}:{block_id}:progress` (`games_done`, `samples_done`)
- `songo:matchup:{target_id}:{matchup_id}:lease` (ttl lease)
- `songo:lock:{target_id}:consolidation` (distributed lock)
- `songo:alert:{target_id}:stale_workers` (set/list)
- `songo:alert:{target_id}:consolidation_lag` (value)

## 9. Source de verite et reprise

- Reprise durable: Firestore + fichiers Drive.
- Redis peut etre perdu sans perte de verite durable, car la consolidation periodique persiste l'etat dans Firestore.

## 10. Decision

Pour la cible 20M et 5 -> 20 Colabs:

- architecture officielle = Drive + Firestore + Redis
- Firestore reste le registre durable
- Redis devient la couche temps reel par defaut

## 11. Logging et erreurs (runtime)

### 11.1 Contrat de logs Firestore checkpoint

Chaque job logge au demarrage:

- `firestore checkpoint sync config`
- champs: `enabled`, `strict`, `project_id`, `collection`, `auth_mode`, `credentials_path_exists`, `api_key_set`, `checkpoint_min_interval_seconds`, `checkpoint_state_only_on_change`

En cas d'echec d'ecriture checkpoint:

- log `warning` avec contexte complet
- event `firestore_worker_checkpoint_sync_failed`
- `hint` explicite (credentials manquants, quota depasse, timeout, permission, auth invalide)
- si `strict=true`: l'exception est re-raise
- si `strict=false`: le job continue avec trace claire de degradation

En fin de job (`completed|failed|cancelled`):

- log `firestore checkpoint sync summary`
- metric `firestore_checkpoint_sync_summary` avec compteurs:
  - `attempted`
  - `written`
  - `skipped_unchanged`
  - `skipped_min_interval`
  - `failed`

### 11.2 Recommandations operatoires

- garder `strict=true` pour les jobs critiques de coordination
- utiliser `strict=false` seulement pour les runs exploratoires
- surveiller la derive via `failed > 0` dans `metrics.jsonl`
- si `quota exceeded`, augmenter batching/throttling avant d'augmenter le nombre de workers
