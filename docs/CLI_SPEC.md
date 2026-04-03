# CLI Spec

## 1. Objectif

Definir les futures commandes CLI du projet.

## 2. Principe

Chaque commande doit:

- accepter un `--config`
- gerer `--job-id`
- gerer `--resume`
- logger vers console et fichiers

## 3. Commandes principales

La V1 doit viser ces commandes:

- `dataset-generate`
- `dataset-build`
- `train`
- `benchmark`
- `evaluate`
- `resume`
- `status`

## 4. Commande `dataset-generate`

Usage cible:

```bash
python -m songo_model_stockfish.cli dataset-generate --config config/dataset_generation.yaml
```

Responsabilite:

- lancer des parties
- echantillonner des positions
- ecrire les sorties brutes

## 5. Commande `dataset-build`

Usage cible:

```bash
python -m songo_model_stockfish.cli dataset-build --config config/dataset_build.yaml
```

Responsabilite:

- lire les positions labelisees
- construire les splits
- ecrire le dataset final versionne

## 6. Commande `train`

Usage cible:

```bash
python -m songo_model_stockfish.cli train --config config/train.yaml
```

Responsabilite:

- charger dataset et config
- lancer l'entrainement
- sauvegarder checkpoints et metriques

## 7. Commande `benchmark`

Usage cible:

```bash
python -m songo_model_stockfish.cli benchmark --config config/benchmark.yaml
```

Responsabilite:

- lancer les matchups
- produire logs et resume benchmark

## 8. Commande `evaluate`

Usage cible:

```bash
python -m songo_model_stockfish.cli evaluate --config config/evaluation.yaml
```

Responsabilite:

- evaluation offline sur dataset
- generation de rapports

## 9. Commande `resume`

Usage cible:

```bash
python -m songo_model_stockfish.cli resume --job-id train_20260402_153000_a1b2
```

Responsabilite:

- retrouver la config du job
- relire l'etat de reprise
- relancer le job depuis le bon point

## 10. Commande `status`

Usage cible:

```bash
python -m songo_model_stockfish.cli status --job-id benchmark_20260402_181500_c8f1
```

Responsabilite:

- lire `run_status.json`
- afficher un resume lisible

## 11. Options communes

Options communes recommandees:

- `--config`
- `--job-id`
- `--resume`
- `--output-root`
- `--dry-run`
- `--verbose`

## 12. Exigence V1

La CLI du projet doit rester:

- simple
- coherente
- compatible Colab
- facile a reprendre
