# Colab Operations

## 1. Objectif

Definir les operations standard pour executer le projet dans Google Colab de maniere robuste.

## 2. Regle centrale

Le code vit sur GitHub.

Les artefacts vivent sur Google Drive.

Cette separation ne doit jamais etre rompue.

## 3. Arborescence Drive recommandee

```text
MyDrive/songo-stockfish/
  code/
  data/
    raw/
    processed/
    datasets/
  jobs/
  models/
  logs/
  reports/
  exports/
```

## 4. Ce qui doit etre stocke dans Drive

- datasets
- checkpoints
- logs
- manifests de jobs
- rapports benchmark
- rapports evaluation
- model cards

## 5. Ce qui doit rester dans GitHub

- `src/`
- `docs/`
- `notebooks/`
- `scripts/`
- `config/`
- `requirements/`

## 6. Workflow Colab recommande

1. monter Google Drive
2. cloner ou mettre a jour le repo GitHub
3. installer les dependances
4. definir les chemins Drive
5. lancer un job avec `job_id`
6. sauvegarder regulierement l'etat du job
7. reprendre si la session tombe

## 7. Strategie de mise a jour du code

Quand le code change:

- on met a jour le repo clone
- on ne touche pas aux dossiers d'artefacts
- on conserve les `job_id`, checkpoints, datasets et rapports

Le repo clone ne doit pas etre l'endroit ou les artefacts critiques sont ecrits.

## 8. Reprise apres perte de session

Chaque notebook et chaque script doit pouvoir:

- retrouver le dernier `job_id`
- relire `run_status.json`
- relire `state.json`
- reprendre depuis le dernier checkpoint ou la derniere unite terminee

## 9. Notebooks a preparer

Le projet devra avoir au minimum:

- `notebooks/colab_setup.ipynb`
- `notebooks/colab_train.ipynb`
- `notebooks/colab_benchmark.ipynb`
- `notebooks/colab_evaluate.ipynb`

## 10. Commandes Git a standardiser

Le projet devra documenter clairement:

- clone initial
- pull de mise a jour
- checkout de branche si besoin

L'objectif est:

- mettre a jour le code sans impacter les artefacts Drive

## 11. Sauvegardes regulieres

Les scripts devront sauvegarder:

- checkpoints frequents pour le train
- etat de progression pour benchmark et dataset
- metriques progressives
- resume final de job

## 12. Exigence de robustesse

Une execution Colab sera consideree robuste si:

- une session interrompue peut reprendre
- les artefacts precedents ne sont pas perdus
- une mise a jour de code reste propre
- les logs permettent de comprendre l'etat du run
