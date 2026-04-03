# Colab Pipeline

## Objectif

Definir une organisation solide pour entrainer le nouveau modele dans Google Colab sans reutiliser l'ancien stack de `songo-ai`.

## Schema cible

1. cloner le repo GitHub
2. installer `requirements.txt`
3. charger ou generer le dataset du nouveau projet
4. lancer le train
5. sauvegarder les checkpoints dans Drive
6. lancer l'evaluation
7. reprendre automatiquement si une session est interrompue

## Organisation recommandee

- GitHub:
  - code source
  - configs
  - notebooks
  - scripts
  - documentation
- Google Drive:
  - datasets volumineux
  - checkpoints
  - logs
  - rapports
  - etats de jobs

## Premier notebook cible

Le premier notebook devra faire seulement:

- installation
- verification GPU
- chargement d'un mini dataset
- entrainement smoke test
- sauvegarde d'un checkpoint
- reprise depuis un checkpoint existant

## Point de cadrage important

Le notebook Colab de ce projet ne doit pas reutiliser les anciens notebooks ou anciens scripts de train de `songo-ai`.

Il peut seulement s'appuyer sur `songo-ai` pour:

- la logique du jeu
- `minimax`
- `mcts`
- les benchmatchs

## Contraintes a garder en tete

- temps limite des sessions Colab
- GPU non garanti selon l'offre
- necessite de reprendre facilement un run interrompu
- importance de sauvegarder regulièrement les artefacts
- separation stricte entre code mis a jour et artefacts existants

## Documents complementaires

Voir aussi:

- `docs/COLAB_OPERATIONS.md`
- `docs/SYSTEM_ARCHITECTURE.md`
- `docs/MODEL_STRATEGY.md`
- `docs/DATASET_AND_BENCHMARK_ARCHITECTURE.md`
