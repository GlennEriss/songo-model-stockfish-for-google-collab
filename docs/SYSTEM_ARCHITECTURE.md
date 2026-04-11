# System Architecture

## 1. Objectif

Definir l'architecture globale du projet pour avoir un systeme solide, exploitable dans Google Colab, et capable de resister aux coupures de session.

Cette architecture doit couvrir:

- le code source
- les artefacts
- les logs
- la reprise des jobs
- la gestion des erreurs
- les metriques

## 2. Principes structurants

Le projet suit ces principes:

- code versionne sur GitHub
- artefacts persistants sur Google Drive
- etat vivant multi-Colab centralise dans Firestore
- jobs toujours resumables
- logs lisibles en direct et enregistrés
- erreurs detaillees et exploitables
- separation stricte entre code et donnees produites
- mode quota-first pour les executions multi-workers

## 3. Separation Code / Artefacts

### 3.1 GitHub

GitHub contient uniquement:

- code source
- notebooks
- scripts
- configurations
- documentation
- manifests statiques du projet

### 3.2 Google Drive

Google Drive contient uniquement:

- datasets
- checkpoints
- logs d'execution
- rapports d'evaluation
- rapports de benchmark
- etats de reprise des jobs
- exports de modeles

### 3.3 Firestore

Firestore contient uniquement l'etat de coordination runtime:

- `global_generation_progress/{global_target_id}`
  - compteur global cross-workers (`total_samples`, `total_games`, `workers`, `updated_at`)
- `dataset_registry/primary`
  - registre des `dataset_sources` et `built_datasets`
- `worker_leases/{global_target_id}`
  - attribution des workers pour eviter les collisions (`worker_index`, lease)
- `worker_checkpoints/{job_id}`
  - miroir resumable de `run_status` et `state`
- `pipeline_manifests/{worker_tag}`
  - metadata du lancement de pipeline (pid/logs)

Les donnees lourdes ne doivent pas etre stockees dans Firestore.

### 3.4 Regle importante

Une mise a jour du code ne doit jamais supprimer ni ecraser les artefacts de Drive.

## 4. Architecture logique

Le projet doit converger vers les modules suivants:

- `src/songo_model_stockfish/adapters/`
- `src/songo_model_stockfish/engine/`
- `src/songo_model_stockfish/evaluation/`
- `src/songo_model_stockfish/data/`
- `src/songo_model_stockfish/training/`
- `src/songo_model_stockfish/benchmark/`
- `src/songo_model_stockfish/ops/`
- `src/songo_model_stockfish/cli/`

## 5. Role des modules

### 5.1 `adapters/`

Responsabilite:

- encapsuler l'acces a `songo-ai`
- centraliser la logique de compatibilite
- isoler les imports externes au projet

### 5.2 `engine/`

Responsabilite:

- recherche
- etat interne
- choix du coup
- statistiques de recherche

### 5.3 `evaluation/`

Responsabilite:

- evaluation heuristique
- evaluation neuronale
- conversion features -> score / policy / value

### 5.4 `data/`

Responsabilite:

- generation de benchmatchs
- sampling de positions
- construction des datasets
- deduplication
- versionnage des datasets

### 5.5 `training/`

Responsabilite:

- entrainement
- checkpoints
- reprise d'entrainement
- calcul des metriques offline

### 5.6 `benchmark/`

Responsabilite:

- matchs contre `minimax`
- matchs contre `mcts`
- matchs entre variantes internes
- rapports comparatifs

### 5.7 `ops/`

Responsabilite:

- logging
- manifests de jobs
- persistance d'etat
- gestion des erreurs
- sauvegardes
- reprise apres coupure

### 5.8 `cli/`

Responsabilite:

- points d'entree scripts
- commandes standardisees
- validation des arguments

## 6. Types de jobs du projet

Le projet doit considerer ces jobs comme objets de premiere classe:

- `dataset_generation`
- `dataset_build`
- `train`
- `evaluation`
- `benchmark`
- `export_model`

Chaque job doit pouvoir:

- demarrer proprement
- logger sa progression
- etre interrompu
- etre repris
- produire un resume final

## 7. Modele de persistance des jobs

Chaque job doit avoir un dossier dedie dans Drive:

```text
jobs/<job_id>/
  config.yaml
  state.json
  run_status.json
  events.jsonl
  metrics.jsonl
  checkpoints/
  artifacts/
```

### 7.1 `config.yaml`

- config effective du job

### 7.2 `state.json`

- etat courant minimal pour reprise

### 7.3 `run_status.json`

- statut global du job
- `pending`, `running`, `failed`, `completed`, `interrupted`

### 7.4 `events.jsonl`

- journal detaille des evenements

### 7.5 `metrics.jsonl`

- metriques progressives lisibles par machine

### 7.6 Sync runtime Firestore

En mode multi-Colab, l'etat vivant doit etre sync aussi dans Firestore:

- `run_status.json` et `state.json` sont conserves en local Drive
- `worker_checkpoints/{job_id}` est mis a jour pour reprise cross-session
- `global_generation_progress/{global_target_id}` est mis a jour en mode batched pour limiter les quotas
- `dataset_registry/primary` est mis a jour transactionnellement pour eviter les races

Le backend Firestore est la source de verite runtime recommandee pour l'etat global.

## 8. Logging

Le logging doit etre double:

- console lisible en direct
- JSONL persistant pour machine

### 8.1 Objectifs du logging

- suivre la progression en temps reel
- comprendre rapidement les echecs
- permettre la reprise
- produire des rapports automatiques

### 8.2 Champs communs de logs

Chaque evenement de log doit pouvoir contenir:

- `timestamp`
- `job_id`
- `run_type`
- `level`
- `message`
- `progress`
- `elapsed_seconds`
- `eta_seconds`
- `phase`
- `artifact_path`

### 8.3 Logs attendus par type de job

Pour `train`:

- epoch
- step
- train loss
- validation loss
- checkpoint saved

Pour `benchmark`:

- matchup
- game index
- score cumule
- temps moyen par coup

Pour `dataset_generation`:

- partie en cours
- nombre de positions collecte
- shard ecrit

## 9. Gestion des erreurs

Chaque job doit attraper les erreurs principales et ecrire:

- le message d'erreur
- la stack trace
- le contexte de phase
- l'etat de reprise disponible

Les erreurs doivent etre classees idealement en:

- erreur de configuration
- erreur d'I/O
- erreur de compatibilite code
- erreur de donnees
- interruption utilisateur ou session

## 10. Strategie de reprise

### 10.1 Entrainement

Reprendre depuis:

- dernier checkpoint
- dernier epoch/step valide
- dernier etat optimiseur sauve

### 10.2 Benchmark

Reprendre depuis:

- dernier matchup non termine
- dernier index de partie termine

### 10.3 Generation de dataset

Reprendre depuis:

- dernier shard ecrit
- dernier `game_id` valide

### 10.4 Evaluation

Reprendre depuis:

- derniere evaluation de split ou de matchup terminee

## 11. Metriques et traçabilite

Chaque modele et chaque run doivent etre lies a:

- un `job_id`
- un `model_id`
- un commit git
- une config exacte
- une version de dataset

Le projet doit produire des fichiers du type:

- `model_card.json`
- `benchmark_summary.json`
- `training_summary.json`

## 12. Gestion des packages

Le projet doit gerer proprement ses dependances:

- `requirements/requirements.in`
- `requirements/requirements.txt`
- `requirements/dev.in`
- `requirements/dev.txt`

Objectif:

- figer les versions runtime
- separer runtime et dev
- rendre le setup Colab reproductible

## 13. Critere de robustesse

L'architecture sera consideree solide si:

- une mise a jour du code ne touche pas aux artefacts Drive
- un job interrompu peut reprendre proprement
- les logs permettent de comprendre ce qui s'est passe
- les metriques et checkpoints sont tracables
- les scripts CLI produisent des sorties coherentes
