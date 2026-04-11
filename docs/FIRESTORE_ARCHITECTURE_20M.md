# Firestore Architecture 20M

## 1. Objectif

Definir une architecture multi-Colab claire, robuste et economique en quota pour atteindre `20_000_000` samples sans divergence d'etat entre workers.

Ce document fixe:

- la separation des responsabilites (data plane vs control plane)
- ce qui doit etre ecrit dans Firestore, et a quelle frequence
- le mode economique quota par defaut
- le plan d'implementation P0/P1/P2

## 2. Constats Actuels

### 2.1 Ce qui marche deja

- Firestore est deja utilise comme source de verite pour:
  - `global_generation_progress`
  - `dataset_registry`
  - `worker_leases`
  - `pipeline_manifests`
  - `worker_checkpoints`
- Le mode `global_budget_enforcement_mode=batched` existe et reduit fortement les ecritures globales.
- Le notebook compact expose deja un `LOW_QUOTA_PROFILE`.

### 2.2 Ce qui cree encore des tensions de quota

- Trop d'ecritures checkpoint si `write_state()` est appele tres souvent (notamment pendant `dataset-build`).
- Trop de lectures quand plusieurs Colabs monitorent en parallele avec polling court.
- Des writes utiles mais non critiques (ex: manifest Firestore tres frequent) peuvent etre desactives ou reduites.

### 2.3 Pourquoi les valeurs divergent entre Colabs

- Chaque Colab lit a un instant different.
- Si un worker est stale/inactif, il reste visible dans l'etat global.
- Si des cellules de monitoring tournent en parallele, elles peuvent afficher des snapshots differents de quelques secondes.

Ce comportement est normal tant que:

- les compteurs globaux sont monotones
- l'etat converge en quelques polls
- un seul backend vivant fait foi (Firestore)

## 3. Architecture Cible

## 3.1 Principe directeur

- Google Drive = `data plane` (fichiers lourds: raw/sampled/labeled/checkpoints/modeles/logs).
- Firestore = `control plane` (etat vivant, coordination, reprise, monitoring).

Firestore ne stocke jamais les donnees lourdes de dataset, uniquement des metadonnees et des compteurs.

## 3.2 Collections Firestore (source de verite)

1. `global_generation_progress/{GLOBAL_TARGET_ID}`
- role: compteur global cross-workers
- champs: `target_samples`, `total_samples`, `total_games`, `workers`, `updated_at`
- ecrit par: workers `dataset-generate`
- cadence: batch par `g` games, pas par game

2. `worker_leases/{GLOBAL_TARGET_ID}`
- role: attribution de `WORKER_INDEX` et protection anti-collision
- champs: `leases.{worker_tag}.index`, `updated_at`
- ecrit par: bootstrap notebook (auto-assign)
- cadence: faible (startup/restart)

3. `dataset_registry/primary`
- role: registre unique des `dataset_sources` et `built_datasets`
- ecrit par: `dataset-generate`, `dataset-build`, `dataset-merge-final`
- cadence: snapshots partiels + completions

4. `worker_checkpoints/{job_id}`
- role: reprise fine d'un job (status/state)
- ecrit par: `JobContext` (`write_status`, `write_state`, `set_phase`)
- cadence cible: throttlee (voir section 6)

5. `pipeline_manifests/{WORKER_TAG}`
- role: pid/log paths du pipeline lance
- ecrit par: notebook launch
- cadence: lancement uniquement (ou changement majeur)

## 3.3 Regles de coherence

- Firestore doit etre le seul backend vivant pour l'etat runtime.
- Les compteurs doivent etre monotones (jamais de decrement visible hors rollback explicite).
- Les updates doivent etre idempotentes (rejouables sans corruption).
- Les metadonnees doivent pointer vers des chemins Drive stables.

## 4. Budget Quota (modele simple)

## 4.1 Glossaire des variables

- `g`: `GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES`
- `b`: `DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES`
- `c`: checkpoint flush interval (a introduire) en nb de fichiers ou secondes
- `m`: intervalle de monitoring en secondes
- `W`: nombre de workers Colab actifs

## 4.2 Formules d'ordre de grandeur

1. Ecritures progression globale:
- `writes_global ~= total_games / g`

2. Ecritures snapshots build partiels:
- `writes_build_partial ~= total_labeled_files / b`

3. Ecritures checkpoints:
- actuel: proche de `O(total_files)` si write per file
- cible: `writes_checkpoints ~= total_labeled_files / c` (ou `runtime_seconds / c_seconds`)

4. Lectures monitoring:
- `reads_monitor ~= monitors * docs_per_loop * (86400 / m)`

## 4.3 Exemple 20M (ordre de grandeur)

Hypothese de travail:

- `20_000_000` samples
- ~`300` samples/game en moyenne
- donc ~`66_667` games total

Avec `g=200`:

- `writes_global ~= 66_667 / 200 ~= 334` writes (ordre de grandeur, cluster total)

Si `b=500` et ~`66_000` fichiers labels:

- `writes_build_partial ~= 66_000 / 500 ~= 132`

Si `c=100`:

- `writes_checkpoints ~= 66_000 / 100 ~= 660`

Conclusion: le vrai levier quota est la reduction des writes checkpoint et du polling monitor.

## 5. Parametres Recommandes (mode economique)

Profil recommande pour multi-Colab:

- `GLOBAL_BUDGET_ENFORCEMENT_MODE='batched'`
- `GLOBAL_PROGRESS_FLUSH_EVERY_N_GAMES=200`
- `GLOBAL_TARGET_POLL_INTERVAL_SECONDS=60`
- `SOURCE_POLL_INTERVAL_SECONDS=45` (ou `60`)
- `DATASET_BUILD_EXPORT_PARTIAL_EVERY_N_FILES=200` a `500`
- `MONITOR_REFRESH_SECONDS=90` a `120`
- `PIPELINE_MANIFEST_FIRESTORE_WRITE_ENABLED=False` (sauf besoin debug)

Et ajout recommande (P1):

- `WORKER_CHECKPOINT_FLUSH_EVERY_N_FILES=100`
- `WORKER_CHECKPOINT_FLUSH_INTERVAL_SECONDS=60`
- flush force sur `start`, `phase_change`, `completed`, `failed`

## 6. Plan D'Implementation

## P0 (priorite immediate)

Objectif: supprimer les causes majeures de quota sans changer le comportement metier.

Actions:

1. Conserver Firestore comme backend runtime unique.
2. Garder `global progress` en mode batched (`g>=200`).
3. Garder `manifest Firestore` optionnel/desactive par defaut.
4. Limiter monitoring live a un seul Colab "observateur".

Definition of done P0:

- plus de regressions de compteur dues a multi-sources
- chute nette des writes/reads Firestore journaliers

## P1 (checkpointing robuste et economique)

Objectif: garder une reprise exacte sans write Firestore par fichier.

Actions:

1. Ajouter un buffer checkpoint dans `JobContext`:
  - accumuler updates locales
  - flush sur timer ou palier (fichiers/games)
2. Flush force sur evenements critiques:
  - `write_status(running/completed/failed)`
  - changement de phase
  - exception non geree
3. Garder la reprise exacte via state local Drive + checkpoint Firestore throttle.

Definition of done P1:

- aucune perte de reprise apres interruption
- writes checkpoint reduites d'un facteur important (x10 a x100 selon runs)

## P2 (monitoring unifie Firestore)

Objectif: arreter le polling redondant et clarifier la lecture de l'etat global.

Actions:

1. Un seul notebook monitor lit Firestore en boucle.
2. Les notebooks workers n'executent pas les cellules de monitoring continue.
3. Ajouter un resume compact:
  - workers actifs/inactifs
  - trend 5 min
  - alertes quota/latence
4. Documenter runbook operator:
  - qui lance
  - qui monitor
  - qui relance un worker stale

Definition of done P2:

- lecture globale stable, sans "bruit" multi-monitors
- baisse nette des reads Firestore

## 7. Runbook Multi-Colab Recommande

1. Mettre les credentials Firestore dans Drive (`/content/drive/MyDrive/songo-stockfish/secrets/...json`).
2. Exporter ce chemin dans `FIRESTORE_CREDENTIALS_PATH` sur chaque Colab.
3. Lancer workers avec `LOW_QUOTA_PROFILE=True`.
4. Garder une seule session avec cellules monitor live.
5. Si un worker devient stale:
  - verifier logs pipeline
  - relancer seulement ce worker
  - ne pas redemarrer tous les workers inutilement.

## 8. Limites et Evolution Future

- Firestore suffit pour ce niveau (5 a 10 Colabs) si writes/reads sont throttlees.
- Redis/Kafka ne deviennent utiles que si:
  - tres forte frequence evenementielle
  - besoin de streaming sub-second
  - beaucoup plus de workers concurrents

Pour la cible actuelle 20M, la priorite est l'architecture economique ci-dessus, pas un changement de stack.
