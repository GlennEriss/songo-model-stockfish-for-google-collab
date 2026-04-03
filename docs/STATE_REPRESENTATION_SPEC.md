# State Representation Spec

## 1. Objectif

Definir la representation d'etat officielle pour la V1 du projet.

Cette specification doit servir a:

- la recherche
- le dataset
- l'evaluation heuristique
- le modele neuronal plus tard

## 2. Principe directeur

La V1 doit conserver une representation:

- simple
- stable
- versionnable
- compatible avec `songo-ai`

## 3. Source de verite

La source de verite des regles reste `songo-ai`.

Mais le projet doit definir sa propre forme de representation serialisable.

## 4. Representation brute V1

Chaque etat V1 doit au minimum contenir:

- les graines sur les 14 cases
- le joueur au trait
- les scores des deux joueurs
- un indicateur terminal si utile

## 5. Schema recommande

Exemple logique:

```json
{
  "board": [4, 4, 0, 6, 1, 3, 5, 2, 4, 4, 1, 0, 7, 3],
  "player_to_move": "south",
  "scores": {
    "south": 18,
    "north": 16
  },
  "turn_index": 42,
  "is_terminal": false
}
```

## 6. Convention d'ordre des cases

Le projet doit fixer une convention d'ordre unique pour les 14 cases.

Recommendation:

- cases `south` de gauche a droite
- puis cases `north` de gauche a droite

Cette convention devra etre documentee une seule fois et respectee partout.

## 7. Champs obligatoires de l'etat brut

Champs obligatoires:

- `board`
- `player_to_move`
- `scores`

Champs utiles recommandes:

- `turn_index`
- `is_terminal`

## 8. Validation minimale

Un etat brut est valide si:

- `board` contient exactement 14 entiers
- aucun nombre de graines n'est negatif
- `player_to_move` est valide
- les scores sont coherents

## 9. Representation des coups

La representation du coup reste compatible avec `songo-ai`:

- entier correspondant a la case jouee

Le projet devra documenter clairement:

- base 1..7 ou 0..6 en interne
- convention de conversion vers l'exterieur

Recommendation V1:

- interface publique compatible avec `songo-ai`
- representation interne eventuellement normalisee si besoin

## 10. Features derivees

La V1 separe:

- `raw_state`
- `model_features`

Le dataset doit pouvoir stocker le `raw_state`, puis une transformation deterministe produit les features.

## 11. Features candidates pour le modele

Les premieres features possibles sont:

- les 14 valeurs du plateau
- joueur au trait
- scores des deux joueurs
- legal mask

## 12. Legal mask

Le `legal_mask` doit faire partie des donnees derivees standard.

Format recommande:

- vecteur binaire de taille 7

## 13. Versionnage

Chaque representation d'etat doit avoir une version explicite.

Recommendation:

- `state_format_version: "v1"`

## 14. Conclusion

La V1 adopte:

- un `raw_state` simple et serialisable
- une convention unique d'ordre des cases
- un `legal_mask` standard
- une transformation reproductible vers les features modele
