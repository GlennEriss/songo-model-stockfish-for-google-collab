# Model Strategy

## 1. Question centrale

Faut-il partir sur:

- un modele existant a fine-tuner
- ou un reseau neuronal propre au projet

## 2. Decision retenue

La recommandation officielle du projet est:

- ne pas fine-tuner un modele generaliste existant
- construire un reseau neuronal custom from scratch

## 3. Pourquoi ne pas faire du fine-tuning

Le Songo est un jeu de strategie a structure compacte.

Un modele generaliste de type LLM ou grand transformer n'est pas bien adapte car:

- il n'est pas specialise pour les etats de jeu
- il sera trop lourd pour l'inference dans un moteur
- il compliquera l'integration avec la recherche
- il ne nous donnera pas un bon controle sur les sorties

## 4. Pourquoi un modele custom est mieux adapte

Un modele custom apporte:

- une inference rapide
- un controle fort sur l'architecture
- une integration simple dans un moteur de recherche
- un cout d'entrainement plus faible
- une meilleure adequation a la structure du plateau et des coups

## 5. Recommandation d'architecture

Le projet doit viser un modele:

- `policy + value`
- compact
- entrainable from scratch
- facile a exporter et recharger

## 6. Options de modeles raisonnables

### Option A - MLP

Avantages:

- simple
- rapide
- bon point de depart

Limites:

- moins expressif si les features sont pauvres

### Option B - Petit reseau residuel

Avantages:

- plus adapte a des representations plateau riches
- meilleure marge de progression

Limites:

- un peu plus complexe

## 7. Recommandation par phase

### Phase V1

- moteur de recherche + heuristique

### Phase V2

- petit modele `policy + value` from scratch

### Phase V3

- integration de ce modele dans la recherche

### Phase V4

- optimisation architecture / dataset / benchmark

## 8. Type de labels cibles

Le modele devra apprendre a partir de:

- `policy_target`
- `value_target`

Le `policy_target` represente:

- le meilleur coup ou une distribution cible sur les coups

Le `value_target` represente:

- la valeur estimee de la position

## 9. Ce que le projet n'adopte pas

Le projet n'adopte pas comme strategie principale:

- fine-tuning de LLM
- gros transformers autoregressifs
- transfert depuis un modele non specialise jeu

## 10. Conclusion

La ligne officielle du projet est:

- heuristique V1 pour baseline moteur
- reseau neuronal custom V2
- architecture `policy + value`
- pas de fine-tuning externe comme base principale

## 11. Evolution cible

La trajectoire d'amelioration continue retenue est documentee dans:

- `EXPERT_ITERATION_ALPHAZERO_ARCHITECTURE.md`

Cette trajectoire formalise l'integration de:

- `PUCT/MCTS` guide par reseau
- `self-play`
- reentrainement iteratif sur cibles de recherche
