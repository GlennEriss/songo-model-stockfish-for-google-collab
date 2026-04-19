# Docs Index

## 1. Objectif

Fournir une porte d'entree simple dans la documentation du projet.

## 2. Par quoi commencer

Si tu veux comprendre rapidement le projet, lis dans cet ordre:

1. `PROJECT_OVERVIEW.md`
2. `STOCKFISH_PLAN.md`
3. `ENGINE_V1_DESIGN.md`
4. `DECISIONS_REGISTER.md`
5. `OPEN_DECISIONS_V1.md`

## 3. Documents de cadrage

- `PROJECT_OVERVIEW.md`
  - vision generale du projet
  - role exact de `songo-ai`
  - principes directeurs

- `ROADMAP.md`
  - roadmap generale du projet
  - grands blocs de travail

- `IMPROVEMENT_PLAN.md`
  - etat actuel du pipeline
  - ameliorations deja faites
  - prochaines evolutions priorisees

- `EXPERT_ITERATION_ALPHAZERO_ARCHITECTURE.md`
  - architecture cible d'amelioration continue du modele
  - boucle `policy + value + PUCT + self-play + retrain`
  - plan d'integration multi-Colab

- `STOCKFISH_PLAN.md`
  - trajectoire vers un moteur de type Stockfish
  - versions successives

## 4. Documents de design moteur

- `ENGINE_V1_DESIGN.md`
  - API moteur V1
  - modules Python
  - heuristiques V1
  - benchmark V1 cote moteur

- `STATE_REPRESENTATION_SPEC.md`
  - format de l'etat V1
  - legal mask
  - conventions de representation

- `MODEL_V1_SPEC.md`
  - premier modele neuronal officiel
  - backbone MLP V1
  - entrees / sorties

- `BENCHMARK_V1_SPEC.md`
  - protocole benchmark officiel V1
  - matchups, tailles de campagne, metriques

## 5. Documents dataset et benchmark

- `DATASET_AND_BENCHMARK_ARCHITECTURE.md`
  - architecture generale benchmatch + dataset

- `DATASET_V1_SPEC.md`
  - format exact du premier dataset
  - raw games, sampled positions, labeled samples

- `DATASET_MODEL_TRAINING_PLAYBOOK.md`
  - construction dataset actuelle (generate/build)
  - labels utilises (`policy_target`, `value_target`, hard examples, tactique)
  - architecture du modele `PolicyValueMLP`
  - leviers pour rendre le modele plus fort
  - metriques prioritaires train/eval/benchmark

## 6. Documents architecture et operations

- `SYSTEM_ARCHITECTURE.md`
  - architecture systeme globale
  - jobs resumables
  - separation code / artefacts

- `COLAB_PIPELINE.md`
  - pipeline Colab general

- `COLAB_OPERATIONS.md`
  - operations Colab
  - Drive, GitHub, reprise

- `FIRESTORE_ARCHITECTURE_20M.md`
  - architecture multi-Colab Drive + Firestore + Redis
  - Firestore durable + Redis temps reel
  - plan P0/P1/P2 pour viser 20M sans exploser les quotas

- `OPERATIONS_SPEC.md`
  - schemas de `run_status.json`, `state.json`, `events.jsonl`, `metrics.jsonl`
  - `model_card.json`
  - `benchmark_summary.json`

- `ARTIFACTS_AND_LOGGING_SPEC.md`
  - conventions d'artefacts
  - conventions de logging

- `REPO_STRUCTURE_SPEC.md`
  - structure cible du repository

## 7. Documents configuration et CLI

- `JOB_CONFIG_SPEC.md`
  - format YAML des jobs

- `CLI_SPEC.md`
  - commandes cible du projet
  - `train`, `benchmark`, `dataset-generate`, `resume`, etc.

## 8. Documents de gouvernance technique

- `MODEL_STRATEGY.md`
  - pourquoi from scratch
  - pourquoi `policy + value`
  - pourquoi pas de fine-tuning generaliste

- `DECISIONS_REGISTER.md`
  - decisions prises
  - points encore ouverts ou refermes

- `OPEN_DECISIONS_V1.md`
  - recommandations officielles sur les derniers points ouverts V1

## 9. Parcours recommande selon le besoin

### Si tu veux comprendre la vision globale

Lis:

- `PROJECT_OVERVIEW.md`
- `ROADMAP.md`
- `IMPROVEMENT_PLAN.md`
- `STOCKFISH_PLAN.md`

### Si tu veux comprendre le moteur V1

Lis:

- `ENGINE_V1_DESIGN.md`
- `STATE_REPRESENTATION_SPEC.md`
- `BENCHMARK_V1_SPEC.md`

### Si tu veux comprendre le modele neuronal V1

Lis:

- `MODEL_STRATEGY.md`
- `EXPERT_ITERATION_ALPHAZERO_ARCHITECTURE.md`
- `MODEL_V1_SPEC.md`
- `DATASET_V1_SPEC.md`
- `DATASET_MODEL_TRAINING_PLAYBOOK.md`

### Si tu veux comprendre l'exploitation Colab

Lis:

- `COLAB_PIPELINE.md`
- `COLAB_OPERATIONS.md`
- `FIRESTORE_ARCHITECTURE_20M.md`
- `OPERATIONS_SPEC.md`
- `ARTIFACTS_AND_LOGGING_SPEC.md`

### Si tu veux commencer l'implementation

Lis:

- `DECISIONS_REGISTER.md`
- `OPEN_DECISIONS_V1.md`
- `REPO_STRUCTURE_SPEC.md`
- `JOB_CONFIG_SPEC.md`
- `CLI_SPEC.md`

## 10. Conclusion

Le projet a maintenant une documentation organisee en couches:

- vision
- architecture
- specs fonctionnelles
- exploitation
- decisions

Ce document est le point d'entree recommande pour naviguer dans l'ensemble.
