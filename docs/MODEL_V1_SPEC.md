# Model V1 Spec

## 1. Objectif

Definir le premier modele neuronal officiel du projet.

## 2. Positionnement

Le modele V1 n'est pas le point de depart absolu du projet.

Le point de depart absolu reste:

- moteur de recherche
- evaluation heuristique

Le modele V1 arrive ensuite comme premiere evaluation apprise exploitable.

## 3. Decision officielle

Le modele V1 sera:

- un modele `policy + value`
- compact
- entraine from scratch
- concu pour etre integre a un moteur

## 4. Type de backbone

Recommendation officielle V1:

- MLP compact

Pourquoi:

- simple a implementer
- rapide a entrainer
- bon pour une premiere iteration
- bon point de comparaison avant un modele plus sophistique

## 5. Entrees du modele

Le modele V1 consomme des features derivees du `raw_state`.

Features minimales recommandees:

- 14 valeurs du plateau
- joueur au trait
- score joueur courant
- score adversaire
- `legal_mask`

## 6. Sorties du modele

Le modele produit:

- une tete `policy`
- une tete `value`

### 6.1 Tete `policy`

Sortie:

- logits sur les coups possibles

Format recommande:

- taille 7

### 6.2 Tete `value`

Sortie:

- scalaire normalise dans `[-1, 1]`

## 7. Architecture recommande

Exemple cible:

- input layer
- hidden layer 256
- hidden layer 256
- hidden layer 128
- head `policy`
- head `value`

Ce n'est pas une obligation stricte, mais c'est le point de depart recommande.

## 8. Fonctions de perte

Le modele V1 doit utiliser:

- loss policy
- loss value

Loss globale:

```text
loss_total = loss_policy + alpha * loss_value
```

## 9. Labels

Le modele apprend sur:

- `policy_target`
- `value_target`

La `policy` doit respecter le `legal_mask`.

## 10. Critere de qualite V1

Le modele V1 est acceptable si:

- il respecte les coups legaux
- il apprend une policy coherente
- il apprend une value stable
- il peut etre benchmarke contre les references

## 11. Exports

Le modele V1 doit etre exportable sous forme:

- checkpoint PyTorch
- `model_card.json`

## 12. Evolutions prevues

Apres V1, on pourra tester:

- MLP plus large
- petit reseau residuel
- meilleure featurisation
- meilleure policy target

## 13. Conclusion

Le modele officiel V1 est:

- `policy + value`
- from scratch
- backbone MLP compact
- integre a un pipeline de benchmark solide
