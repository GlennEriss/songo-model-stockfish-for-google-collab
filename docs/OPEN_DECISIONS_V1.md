# Open Decisions V1

## 1. Objectif

Fermer les derniers points ouverts avant implementation.

Ce document ne se contente pas de lister les questions. Il donne la recommandation officielle V1 du projet.

## 2. Decision O-001 - Convention exacte des indices de coup

### Question

Faut-il utiliser:

- `1..7`
- ou `0..6`

### Recommendation officielle

- interface publique: `1..7`
- representation interne modele et tensors: `0..6`

### Pourquoi

- `1..7` reste compatible avec `songo-ai`
- `0..6` est plus naturel pour les tableaux, logits et masks

### Regle V1

- tout ce qui touche l'API benchmark / moteur externe reste compatible `1..7`
- tout ce qui touche `policy logits`, `legal_mask`, tensors et dataset peut etre converti en `0..6`

## 3. Decision O-002 - Structure exacte du `raw_state`

### Question

Quels champs garder dans la representation brute ?

### Recommendation officielle

Le `raw_state` V1 doit contenir:

- `state_format_version`
- `board`
- `player_to_move`
- `scores`
- `turn_index`
- `is_terminal`

### Schema recommande

```json
{
  "state_format_version": "v1",
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

### Pourquoi

- assez riche pour dataset et benchmark
- assez simple pour etre stable
- pas trop couple a l'implementation interne de `songo-ai`

## 4. Decision O-003 - Forme exacte de `policy_target`

### Question

Faut-il stocker:

- seulement le meilleur coup
- seulement une distribution
- ou les deux

### Recommendation officielle

- stocker les deux

### Format V1

```json
{
  "best_move": 5,
  "distribution": {
    "1": 0.05,
    "3": 0.10,
    "5": 0.75,
    "6": 0.10
  }
}
```

### Regle

- `best_move` est obligatoire
- `distribution` est recommandee quand disponible

### Pourquoi

- `best_move` simplifie la V1
- `distribution` laisse de la marge pour un apprentissage plus riche

## 5. Decision O-004 - Definition exacte de `value_target`

### Question

Quelle cible utiliser pour la tete value ?

### Recommendation officielle

Le `value_target` V1 doit etre un score normalise dans `[-1, 1]`, vu du joueur au trait.

### Priorite de construction

Ordre recommande:

1. valeur issue d'un enseignant de recherche quand disponible
2. sinon valeur derivee du resultat final de la partie

### Convention

- `+1` = position tres favorable au joueur au trait
- `0` = position neutre
- `-1` = position tres defavorable au joueur au trait

### Pourquoi

- coherent avec une tete `value`
- compatible avec recherche plus tard
- compatible avec supervision simple

## 6. Decision O-005 - Backbone final V1 exact

### Question

Quel MLP exact choisir pour la V1 ?

### Recommendation officielle

Le backbone V1 recommande est:

- MLP `input -> 256 -> 256 -> 128`
- activations simples type ReLU ou GELU
- head `policy`
- head `value`

### Pourquoi

- assez petit pour Colab
- assez simple pour etre stable
- assez expressif pour une premiere iteration

### Regle V1

Avant de tester des variantes plus complexes, il faut benchmarker ce backbone de reference.

## 7. Decision O-006 - Heuristiques V1 exactes

### Question

Quelles heuristiques garder dans la baseline moteur ?

### Recommendation officielle

Heuristiques V1 a garder:

- ecart de score
- mobilite
- potentiel de capture immediate
- risque de capture immediate
- repartition simple des graines

### Regle

- baseline simple
- pas de formule lourde en V1

## 8. Decision O-007 - Strategie d'echantillonnage des positions

### Question

Comment choisir les positions pour le dataset ?

### Recommendation officielle

V1 doit commencer par:

- echantillonnage regulier toutes les `N` plies

Recommendation initiale:

- `sample_every_n_plies = 2`

### Pourquoi

- simple
- reproductible
- peu de biais arbitraire

### Evolution plus tard

On pourra ensuite tester:

- sampling plus dense en fin de partie
- sampling cible sur positions tactiques

## 9. Decision O-008 - Niveau de detail des logs benchmark

### Question

Faut-il logger toutes les parties en detail complet ?

### Recommendation officielle

V1 doit avoir deux niveaux:

- niveau standard: resume de partie + metriques
- niveau detaille optionnel: coups complets par partie

### Pourquoi

- le standard reste leger
- le mode detaille sert au debug

## 10. Decisions officielles V1 en resume

Les recommandations V1 retenues sont:

- coups publics `1..7`, interne modele `0..6`
- `raw_state` avec `board`, `player_to_move`, `scores`, `turn_index`, `is_terminal`
- `policy_target` = `best_move` obligatoire + `distribution` si disponible
- `value_target` dans `[-1, 1]`
- backbone MLP `256 -> 256 -> 128`
- sampling positions toutes les 2 plies
- logs benchmark standard + mode detaille optionnel

## 11. Conclusion

Avec ces choix, la V1 a maintenant assez de decisions fermees pour passer a une implementation propre sans devoir improviser les bases.
