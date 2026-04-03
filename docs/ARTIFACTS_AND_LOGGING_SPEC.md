# Artifacts And Logging Spec

## 1. Objectif

Definir les artefacts persistants et les conventions de logging du projet.

## 2. Artefacts majeurs

Le projet produira principalement:

- checkpoints
- datasets
- logs d'evenements
- logs de metriques
- resumes benchmark
- resumes evaluation
- model cards

## 3. Niveaux de logs

Niveaux recommandes:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`

## 4. Sorties de logs

Chaque job doit produire:

- log console temps reel
- `events.jsonl`
- `metrics.jsonl`

## 5. Exigences de lisibilite

Les logs console doivent etre:

- lisibles
- synthetiques
- orientes progression

Les logs persistants doivent etre:

- structures
- complets
- faciles a parser

## 6. Evenements critiques a logger

### Pour tous les jobs

- start job
- load config
- load state
- resume detected
- checkpoint saved
- artifact written
- warning notable
- failure
- completion

### Pour `benchmark`

- start matchup
- game completed
- matchup completed

### Pour `dataset_generation`

- game sampled
- shard written
- labeling completed

### Pour `train`

- epoch started
- validation completed
- best checkpoint updated

## 7. Artefacts minimaux par type de job

### `train`

- `run_status.json`
- `state.json`
- `events.jsonl`
- `metrics.jsonl`
- checkpoints
- `training_summary.json`

### `benchmark`

- `run_status.json`
- `state.json`
- `events.jsonl`
- `metrics.jsonl`
- `benchmark_summary.json`

### `dataset_generation`

- `run_status.json`
- `state.json`
- `events.jsonl`
- `metrics.jsonl`
- shards

## 8. Resume final standard

Chaque job doit produire un resume final:

- identite du job
- config utilisee
- statut final
- artefacts crees
- metriques principales

## 9. Conservation

Les artefacts existants ne doivent pas etre supprimes automatiquement lors d'une mise a jour du code.

Toute politique de nettoyage devra etre explicite et manuelle.
