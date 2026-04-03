# Decisions Register

## 1. Objectif

Centraliser les decisions deja prises pour le projet et identifier les points encore ouverts.

Ce document sert de reference rapide pour eviter:

- les contradictions entre documents
- les retours en arriere inutiles
- les ambiguïtés avant implementation

## 2. Decisions deja prises

### D-001 - Positionnement du projet

Decision:

- `songo-model-stockfish-for-google-collab` est un nouveau projet
- ce n'est pas une extension directe du modele precedent

Impact:

- nouveau code
- nouvelles conventions
- nouvelles pipelines

### D-002 - Role de `songo-ai`

Decision:

- `songo-ai` est reutilise seulement pour:
  - le moteur du jeu
  - les benchmatchs
  - `minimax`
  - `mcts`

Impact:

- pas de reprise directe de l'ancien modele
- pas de reprise directe des anciens scripts de train

### D-003 - Statut de l'ancien stack `songo-ai`

Decision:

- l'ancien modele neuronal de `songo-ai` est deprecated pour ce projet
- les anciens pipelines de train de `songo-ai` sont deprecated pour ce projet
- les anciens pipelines RL / teacher loop sont deprecated pour ce projet

### D-004 - Strategie moteur

Decision:

- la trajectoire cible est de type "Stockfish du Songo"
- recherche d'abord, evaluation forte ensuite, puis integration hybride

### D-005 - Baseline initiale

Decision:

- baseline V1 moteur avec recherche + evaluation heuristique

Pourquoi:

- stabiliser la recherche
- avoir une baseline claire
- benchmarker avant le neuronal

### D-006 - Strategie modele neuronal

Decision:

- le modele neuronal sera entraine from scratch
- pas de fine-tuning d'un modele generaliste comme strategie principale

### D-007 - Type de modele V1

Decision:

- le premier modele neuronal officiel sera `policy + value`

### D-008 - Backbone modele V1

Decision:

- le backbone recommande pour la V1 est un MLP compact

### D-009 - Representation d'etat V1

Decision:

- un `raw_state` simple et serialisable
- plateau 14 cases
- joueur au trait
- scores
- `legal_mask` derive standard

### D-010 - Representation de coup

Decision:

- la representation publique des coups reste compatible avec `songo-ai`

### D-011 - Dataset V1

Decision:

- pipeline dataset separe en:
  - raw match logs
  - sampled positions
  - labeled positions
  - dataset final

### D-012 - Sources dataset V1

Decision:

- sources initiales:
  - `minimax vs minimax`
  - `mcts vs mcts`
  - `minimax vs mcts`

### D-013 - Teachers dataset V1

Decision:

- les labels proviennent de `minimax`, `mcts` ou d'un mix controle

### D-014 - Format dataset

Decision:

- formats intermediaires: `jsonl`
- format final d'entrainement: `npz`

### D-015 - Split dataset

Decision:

- split par parties et non par positions

### D-016 - Pipeline benchmark

Decision:

- benchmark officiel contre:
  - `minimax:medium`
  - `minimax:hard`
  - `mcts:medium`
  - `mcts:hard`

### D-017 - Robustesse benchmark

Decision:

- benchmark resumable
- alternance premier / second joueur
- logs et resume obligatoires

### D-018 - Separation code / artefacts

Decision:

- code et docs sur GitHub
- artefacts sur Google Drive

### D-019 - Reprise des jobs

Decision:

- tous les jobs importants doivent etre resumables

Concerne:

- train
- benchmark
- dataset generation
- evaluation

### D-020 - Logging

Decision:

- logs console temps reel
- logs persistants JSONL

### D-021 - Fichiers standards de job

Decision:

- chaque job doit avoir:
  - `config.yaml`
  - `run_status.json`
  - `state.json`
  - `events.jsonl`
  - `metrics.jsonl`

### D-022 - Artefacts de modele

Decision:

- chaque modele produit doit avoir un `model_card.json`

### D-023 - Config jobs

Decision:

- chaque job du projet sera pilote par un YAML unique

### D-024 - CLI projet

Decision:

- la CLI cible comporte:
  - `dataset-generate`
  - `dataset-build`
  - `train`
  - `benchmark`
  - `evaluate`
  - `resume`
  - `status`

### D-025 - Colab operations

Decision:

- workflows Colab bases sur:
  - montage Drive
  - clone / update GitHub
  - artefacts uniquement sur Drive
  - reprise apres coupure

### D-026 - Gestion des dependances

Decision:

- organisation cible avec:
  - `requirements/requirements.in`
  - `requirements/requirements.txt`
  - `requirements/dev.in`
  - `requirements/dev.txt`

## 3. Points encore ouverts

### O-001 - Convention exacte des indices de coup

Question:

- interne en `0..6` ou `1..7` ?

Statut:

- a trancher avant implementation moteur

### O-002 - Structure exacte du `state` brut

Question:

- quels champs additionnels garder en plus du minimum ?

Exemples:

- tour courant
- compteur de demi-coups
- drapeaux de fin

Statut:

- a figer au debut de l'implementation `adapters/` et `data/`

### O-003 - Forme exacte de `policy_target`

Question:

- meilleur coup seul
- distribution soft
- ou les deux systematiquement

Statut:

- a trancher avant generation dataset V1

### O-004 - Definition exacte de `value_target`

Question:

- score heuristique
- score de recherche
- resultat final retro-propage
- ou combinaison de plusieurs signaux

Statut:

- a trancher avant labeling V1

### O-005 - Backbone final V1 exact

Question:

- MLP simple
- MLP avec normalisation
- profondeur exacte

Statut:

- a figer avant implementation `training/`

### O-006 - Heuristiques V1 exactes

Question:

- poids initiaux exacts
- formule detaillee

Statut:

- a calibrer pendant implementation moteur

### O-007 - Strategie d'echantillonnage des positions

Question:

- toutes les N plies
- sampling uniforme
- sampling cible sur positions critiques

Statut:

- a trancher avant premier job dataset generation

### O-008 - Niveau de detail des logs de parties

Question:

- journaliser tous les coups completement
- ou resume plus compact selon le type de job

Statut:

- a trancher avant implementation benchmark

## 4. Decisions recommandees a prendre ensuite

Ordre recommande:

1. trancher la convention exacte des coups
2. figer le `raw_state` V1 exact
3. trancher `policy_target` et `value_target`
4. trancher l'echantillonnage dataset V1
5. trancher l'architecture exacte du MLP V1
6. trancher le niveau de detail des logs benchmark

## 5. Documents de reference lies

Les principales references du projet sont:

- `docs/PROJECT_OVERVIEW.md`
- `docs/STOCKFISH_PLAN.md`
- `docs/ENGINE_V1_DESIGN.md`
- `docs/SYSTEM_ARCHITECTURE.md`
- `docs/MODEL_STRATEGY.md`
- `docs/DATASET_V1_SPEC.md`
- `docs/JOB_CONFIG_SPEC.md`
- `docs/CLI_SPEC.md`
- `docs/STATE_REPRESENTATION_SPEC.md`
- `docs/MODEL_V1_SPEC.md`
- `docs/BENCHMARK_V1_SPEC.md`

## 6. Conclusion

Le projet a deja fixe la majorite des decisions structurantes.

Les points encore ouverts sont maintenant suffisamment reduits pour pouvoir:

- faire un dernier passage de clarification
- puis commencer une implementation propre et coherente
