# Dataset And Benchmark Architecture

## 1. Objectif

Definir comment produire:

- les benchmatchs
- les logs de benchmatch
- les positions candidates au dataset
- le dataset final d'entrainement

## 2. Principe general

Le benchmatch a deux usages differents:

- mesurer la performance d'un moteur ou d'un modele
- produire des parties et positions exploitables pour les datasets

Il faut donc separer:

- benchmark d'evaluation
- benchmark de generation de donnees

## 3. Sources de parties

Les parties de reference viendront de:

- `minimax vs minimax`
- `mcts vs mcts`
- `minimax vs mcts`
- plus tard `engine_v1 vs minimax`
- plus tard `engine_v1 vs mcts`

## 4. Pipeline dataset recommande

### Etape 1 - Match logs bruts

Pour chaque partie, on enregistre:

- `game_id`
- `matchup_id`
- seed
- joueurs
- niveaux
- ordre des coups
- resultat final
- temps par coup

Sortie:

- `raw_match_logs/*.jsonl`

### Etape 2 - Sampling de positions

Pendant ou apres les parties, on extrait des positions candidates:

- etat du plateau
- joueur au trait
- coups legaux
- numero de demi-coup
- meta-info de provenance

Sortie:

- `sampled_positions/*.jsonl`

### Etape 3 - Labeling

Chaque position est ensuite labelisee avec un enseignant defini:

- `minimax`
- `mcts`
- ou mix de plusieurs enseignants

Sortie:

- `labeled_positions/*.jsonl`

### Etape 4 - Build dataset

On transforme les positions labelisees en dataset versionne:

- `train`
- `validation`
- `test`

Sortie:

- `datasets/<dataset_version>/`

## 5. Structure d'un exemple de dataset

Un exemple doit contenir au minimum:

- `sample_id`
- `state`
- `player_to_move`
- `legal_moves`
- `policy_target`
- `value_target`
- `source_engine`
- `source_level`
- `game_id`
- `ply`
- `seed`

## 6. Logs de benchmatch

Le benchmark doit avoir:

- logs temps reel console
- logs persistants JSONL
- resume final machine-readable

## 7. Fichiers de sortie benchmark

Pour chaque job benchmark:

```text
jobs/<job_id>/
  config.yaml
  events.jsonl
  metrics.jsonl
  run_status.json
  benchmark_summary.json
  games/
    game_000001.json
    game_000002.json
```

## 8. Resume benchmark

Le `benchmark_summary.json` doit inclure:

- nombre de parties
- victoires / defaites / nulles
- winrate
- temps moyen par coup
- profondeur moyenne si disponible
- infos sur les agents testes
- configuration de recherche du `ModelAgent` si active (`model_search_enabled`, `model_search_top_k`, poids policy/value)

## 9. Reprise benchmark

Un benchmark doit pouvoir reprendre:

- sans rejouer les parties deja terminees
- sans perdre les logs deja ecrits
- en conservant le meme `job_id`

La reprise se base sur:

- la liste des parties terminees
- l'etat courant du matchup

## 10. Reprise generation dataset

La generation dataset doit pouvoir reprendre:

- a partir du dernier `game_id`
- a partir du dernier shard ecrit
- a partir de la derniere position valide persistée

## 11. Versionnage dataset

Chaque dataset doit etre versionne explicitement:

- `dataset_id`
- date
- commit git
- config generation
- enseignant(s)
- sampling strategy

## 12. Critere de qualite dataset

Un dataset est exploitable si:

- les positions sont valides
- les labels sont coherents
- les meta-informations sont presentes
- le split train/val/test est propre
- la provenance est tracable

## 13. Conclusion

Le projet ne doit pas construire le dataset de maniere informelle.

La chaine doit etre:

- benchmatchs traces
- positions samplees
- labels produits proprement
- dataset versionne et reproductible
