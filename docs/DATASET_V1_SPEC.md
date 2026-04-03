# Dataset V1 Spec

## 1. Objectif

Definir le format exact du premier dataset exploitable par le projet.

La V1 doit permettre:

- generation de positions depuis des benchmatchs
- labeling propre
- construction d'un dataset versionne
- entrainement d'un modele `policy + value`

## 2. Pipeline V1

La chaine V1 est:

1. generation de parties de reference
2. extraction de positions candidates
3. labeling des positions
4. construction du dataset final
5. split train / validation / test

## 3. Sources V1

Les sources autorisees pour le dataset V1 sont:

- `minimax vs minimax`
- `mcts vs mcts`
- `minimax vs mcts`

Plus tard, on pourra ajouter:

- `engine_v1 vs minimax`
- `engine_v1 vs mcts`

## 4. Niveaux recommandes en V1

Pour limiter le bruit initial, les premiers niveaux recommandes sont:

- `minimax:medium`
- `minimax:hard`
- `mcts:medium`
- `mcts:hard`

## 5. Unites de donnees

La V1 manipule trois niveaux:

- `game`
- `position sample`
- `dataset sample`

### 5.1 `game`

Une partie complete.

### 5.2 `position sample`

Une position extraite d'une partie.

### 5.3 `dataset sample`

Une position labelisee et transformee dans le format d'entrainement.

## 6. Schema `raw game log`

Chaque partie doit stocker au minimum:

```json
{
  "game_id": "game_000001",
  "matchup_id": "minimax_medium_vs_mcts_medium",
  "seed": 12345,
  "player_a": "minimax:medium",
  "player_b": "mcts:medium",
  "winner": "player_a",
  "moves": [3, 6, 2, 7, 1],
  "ply_count": 85,
  "started_at": "2026-04-02T10:00:00Z",
  "completed_at": "2026-04-02T10:01:20Z"
}
```

## 7. Schema `position sample`

Chaque position echantillonnee doit contenir:

```json
{
  "sample_id": "sample_000001",
  "game_id": "game_000001",
  "matchup_id": "minimax_medium_vs_mcts_medium",
  "ply": 18,
  "seed": 12345,
  "player_to_move": "south",
  "state": {},
  "legal_moves": [1, 3, 5, 6],
  "source_engine": "match_replay",
  "source_level": "mixed"
}
```

## 8. Schema `labeled sample`

Une position labelisee doit contenir:

```json
{
  "sample_id": "sample_000001",
  "game_id": "game_000001",
  "ply": 18,
  "state": {},
  "player_to_move": "south",
  "legal_moves": [1, 3, 5, 6],
  "teacher_engine": "minimax",
  "teacher_level": "hard",
  "policy_target": {
    "best_move": 5,
    "distribution": {
      "1": 0.05,
      "3": 0.10,
      "5": 0.75,
      "6": 0.10
    }
  },
  "value_target": 0.62
}
```

## 9. Champs obligatoires du dataset final

Chaque exemple du dataset final doit avoir:

- `sample_id`
- `game_id`
- `ply`
- `state`
- `player_to_move`
- `legal_moves`
- `policy_target`
- `value_target`
- `teacher_engine`
- `teacher_level`
- `seed`

## 10. Representation `state`

La V1 ne fige pas encore la representation tensorielle finale, mais le dataset doit conserver une representation brute versionnee.

Options possibles:

- representation plateau brute
- representation features derivees

Decision V1:

- conserver un `state` brut suffisamment riche
- permettre ensuite une transformation reproductible vers les features du modele

## 11. `policy_target`

La V1 autorise deux formes:

- `best_move` seul
- distribution cible sur coups legaux

Recommendation:

- stocker `best_move`
- stocker aussi une `distribution` quand elle est disponible

## 12. `value_target`

Le `value_target` doit etre normalise.

Recommendation V1:

- intervalle `[-1, 1]`

Convention:

- `+1` = position tres favorable au joueur au trait
- `0` = position neutre
- `-1` = position tres defavorable au joueur au trait

## 13. Split train / validation / test

Le split doit etre fait par parties et non par simple position.

Pourquoi:

- eviter les fuites entre positions proches

Recommendation V1:

- train: 80%
- validation: 10%
- test: 10%

## 14. Format de stockage final

La V1 pourra utiliser:

- `jsonl` pour les etapes intermediaires
- `npz` ou format tensoriel pour le dataset final d'entrainement

Recommendation:

- intermediaire: `jsonl`
- final training dataset: `npz`

## 15. Versionnage dataset

Chaque dataset doit avoir:

- `dataset_id`
- date de creation
- config generation
- commit git
- teachers utilises
- sampling strategy

## 16. Qualite minimale

Un dataset V1 est valide si:

- toutes les positions sont jouables
- tous les `legal_moves` sont coherents
- `policy_target.best_move` appartient aux coups legaux
- `value_target` est dans `[-1, 1]`
- le split est propre
- la provenance est tracee
