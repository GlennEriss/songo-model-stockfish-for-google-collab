# Roadmap

## Phase 0 - Reframing

- acter que le modele de `songo-ai` est deprecated pour ce projet
- acter que les anciens pipelines d'entrainement de `songo-ai` sont deprecated
- conserver uniquement `songo-ai` comme source pour:
  - le moteur du jeu
  - les benchmatchs
  - `minimax`
  - `mcts`

## Phase 1 - Fondations du nouveau projet

- definir l'architecture du nouveau moteur IA
- formaliser le plan moteur dans `docs/STOCKFISH_PLAN.md`
- fixer les choix d'implementation V1 dans `docs/ENGINE_V1_DESIGN.md`
- formaliser l'architecture systeme dans `docs/SYSTEM_ARCHITECTURE.md`
- formaliser la strategie modele dans `docs/MODEL_STRATEGY.md`
- formaliser l'architecture dataset et benchmark dans `docs/DATASET_AND_BENCHMARK_ARCHITECTURE.md`
- formaliser les operations Colab dans `docs/COLAB_OPERATIONS.md`
- formaliser les schemas de jobs dans `docs/OPERATIONS_SPEC.md`
- formaliser la structure du repo dans `docs/REPO_STRUCTURE_SPEC.md`
- formaliser les conventions d'artefacts et de logs dans `docs/ARTIFACTS_AND_LOGGING_SPEC.md`
- formaliser le dataset V1 dans `docs/DATASET_V1_SPEC.md`
- formaliser les configs YAML dans `docs/JOB_CONFIG_SPEC.md`
- formaliser la CLI dans `docs/CLI_SPEC.md`
- formaliser la representation d'etat dans `docs/STATE_REPRESENTATION_SPEC.md`
- formaliser le premier modele dans `docs/MODEL_V1_SPEC.md`
- formaliser le benchmark officiel V1 dans `docs/BENCHMARK_V1_SPEC.md`
- centraliser les decisions dans `docs/DECISIONS_REGISTER.md`
- fermer les derniers points ouverts V1 dans `docs/OPEN_DECISIONS_V1.md`
- definir les representations d'etat propres a ce projet
- definir le protocole d'evaluation
- definir les interfaces de benchmark avec `songo-ai`

## Phase 2 - Data Generation

- generer des positions et parties de reference via `minimax` et `mcts`
- construire un format de donnees propre a ce projet
- separer clairement generation, preparation et evaluation
- versionner les datasets
- journaliser les benchmatchs et les positions echantillonnees
- rendre la generation dataset resumable

## Phase 3 - Nouveau Training Stack

- implementer un pipeline d'entrainement neuf dans ce repo
- entrainer un premier modele de base
- sauvegarder checkpoints, logs et rapports
- verifier que le modele joue legalement et de maniere stable
- rendre l'entrainement resumable
- stocker les metriques et model cards

## Phase 4 - Competitive Benchmarking

- mesurer le winrate contre `minimax`
- mesurer le winrate contre `mcts`
- comparer les variantes de modeles de ce repo
- identifier les points faibles strategiques
- rendre les benchmarks resumables
- produire des rapports benchmark versionnes

## Phase 5 - Search Integration

- integrer le nouveau modele dans une recherche guidee
- tester un mode evaluateur rapide
- tester un mode policy-value avec recherche
- viser progressivement un moteur "Stockfish du Songo"
