# Repo Structure Spec

## 1. Objectif

Definir la structure finale recommandee du repository avant implementation.

## 2. Structure cible

```text
songo-model-stockfish-for-google-collab/
  README.md
  .gitignore
  docs/
  notebooks/
  config/
  requirements/
    requirements.in
    requirements.txt
    dev.in
    dev.txt
  scripts/
    setup/
    colab/
    dataset/
    benchmark/
    training/
  src/
    songo_model_stockfish/
      __init__.py
      adapters/
      engine/
      evaluation/
      data/
      training/
      benchmark/
      ops/
      cli/
```

## 3. Role des dossiers racine

### `docs/`

- architecture
- decisions
- specs

### `notebooks/`

- notebooks Colab stables et documentes

### `config/`

- configurations YAML de jobs

### `requirements/`

- dependances runtime et dev

### `scripts/`

- scripts shell ou utilitaires de support

### `src/`

- code Python principal

## 4. Sous-dossiers `scripts/`

### `scripts/setup/`

- scripts de preparation environnement

### `scripts/colab/`

- helpers Colab
- montage Drive
- sync code

### `scripts/dataset/`

- generation de donnees
- build dataset

### `scripts/benchmark/`

- lancement benchmark
- reprise benchmark

### `scripts/training/`

- lancement train
- reprise train

## 5. Sous-dossiers `src/songo_model_stockfish/`

### `adapters/`

- interface vers `songo-ai`

### `engine/`

- recherche
- types moteur

### `evaluation/`

- heuristique
- reseaux neuronaux

### `data/`

- parsing
- sampling
- datasets

### `training/`

- boucle d'entrainement
- checkpoints

### `benchmark/`

- matchs
- scores
- rapports

### `ops/`

- logging
- status
- manifests
- reprise

### `cli/`

- points d'entree `python -m ...`

## 6. Exigence

La structure finale doit:

- rester simple
- permettre la croissance
- separer clairement les responsabilites
