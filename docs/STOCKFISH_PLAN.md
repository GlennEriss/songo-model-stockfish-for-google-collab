# Stockfish Plan

## 1. Objectif

Construire progressivement un vrai moteur "Stockfish du Songo".

Dans ce projet, cela veut dire:

- un moteur de recherche fort
- une evaluation de position forte
- une integration propre entre recherche et evaluation
- un protocole de benchmark strict contre `minimax` et `mcts`

Le but n'est pas simplement d'entrainer un modele qui joue des coups. Le but est de construire un moteur competitif.

## 2. Philosophie generale

Un moteur de type Stockfish repose d'abord sur la recherche, puis sur la qualite de l'evaluation.

Pour notre projet, la trajectoire recommande est:

1. construire un nouveau moteur de recherche dans ce repo
2. commencer avec une evaluation heuristique simple mais solide
3. benchmarker tres tot contre `minimax` et `mcts`
4. generer des positions utiles pour apprendre une meilleure evaluation
5. remplacer progressivement l'evaluation heuristique par une evaluation apprise

## 3. Decision de depart

La V1 ne doit pas etre "un modele qui joue seul sans recherche".

La V1 doit etre:

- un moteur alpha-beta / negamax
- une evaluation heuristique ecrite dans ce repo
- une boucle de benchmark automatique

Ensuite seulement, on passe a:

- une evaluation neuronale
- une recherche guidee par le modele
- une version plus proche d'un moteur moderne optimise

## 4. Ce qu'on reutilise depuis `songo-ai`

On reutilise seulement:

- le moteur du jeu
- les benchmatchs
- `minimax`
- `mcts`

On ne reutilise pas comme base de travail:

- l'ancien modele neuronal de `songo-ai`
- les anciens scripts de train de `songo-ai`
- les anciens pipelines RL de `songo-ai`

## 5. Architecture cible

Le repo devrait converger vers cette organisation:

- `src/songo_model_stockfish/engine/`
  - representation interne
  - generation des coups
  - recherche alpha-beta / negamax
  - iterative deepening
  - move ordering
  - table de transposition
- `src/songo_model_stockfish/evaluation/`
  - evaluation heuristique
  - evaluation neuronale
  - conversion features -> score
- `src/songo_model_stockfish/data/`
  - generation de positions
  - serialisation dataset
  - nettoyage et deduplication
- `src/songo_model_stockfish/training/`
  - entrainement de l'evaluateur neuronal
  - logs
  - checkpoints
- `src/songo_model_stockfish/benchmark/`
  - matchs contre `minimax`
  - matchs contre `mcts`
  - rapports de performance
- `notebooks/`
  - notebooks Colab

## 6. Plan technique par versions

### V1 - Search Engine de base

Objectif:

avoir un premier moteur de recherche natif a ce repo.

Fonctionnalites:

- negamax ou alpha-beta
- iterative deepening
- limite de profondeur
- limite de temps
- move ordering de base
- statistiques de recherche

Livrable:

un moteur capable de jouer legalement et de finir une partie complete.

### V2 - Evaluation heuristique

Objectif:

donner au moteur une vraie fonction d'evaluation de position.

Contenu possible:

- score de graines capturees
- potentiel de capture
- vulnerabilite des cases
- mobilite
- controle tactique
- penalites sur les positions dangereuses

Livrable:

une evaluation rapide et stable qui rend la recherche utile.

### V3 - Benchmark competitif

Objectif:

mesurer si le moteur progresse reellement.

Benchmark minimal:

- contre `minimax:medium`
- contre `minimax:hard`
- contre `mcts:medium`
- contre `mcts:hard`

Metriques:

- winrate
- draws
- temps moyen par coup
- profondeur atteinte
- vitesse d'inference / evaluation

### V4 - Data Generation

Objectif:

utiliser le moteur et les benchmarks pour produire des positions d'apprentissage.

Sources:

- parties du moteur natif
- parties `minimax` vs moteur natif
- parties `mcts` vs moteur natif
- analyses de positions importantes

Chaque exemple pourra contenir:

- etat
- coups legaux
- meilleur coup
- score de position
- meta-info sur la source

### V5 - Evaluation neuronale

Objectif:

entrainer une fonction d'evaluation apprise dans ce repo.

Approches possibles:

- value only
- policy + value
- evaluateur compact inspire NNUE-like

Recommandation:

commencer par une petite evaluation `value` ou `policy + value`, puis benchmarker son utilite dans la recherche.

### V6 - Moteur hybride

Objectif:

combiner:

- recherche alpha-beta
- evaluation apprise
- optimisations de recherche

Cette etape correspond au vrai coeur d'un moteur de type Stockfish.

## 7. Priorites de developpement

Ordre recommande:

1. nouveau moteur de recherche
2. evaluation heuristique
3. benchmark natif
4. generation de dataset
5. evaluation neuronale
6. moteur hybride

## 8. Fonctionnalites de recherche a viser

Fonctionnalites initiales:

- negamax
- alpha-beta pruning
- iterative deepening
- move ordering simple
- time management simple

Fonctionnalites ensuite:

- table de transposition
- aspiration windows si utile
- killer moves / history heuristic si utile
- quiescence search si le Songo le justifie
- pruning et reductions si elles apportent un gain reel

On n'a pas besoin d'implementer toutes les techniques avancees au debut. Il faut d'abord prouver qu'elles aident sur le Songo.

## 9. Fonction d'evaluation heuristique V1

La premiere evaluation heuristique devra etre:

- simple
- interpretable
- rapide
- facile a ajuster

Familles de features candidates:

- score courant
- ecart de graines capturees
- nombre de coups legaux
- opportunites de capture immediates
- risque de capture adverse
- distribution des graines
- stabilite de la position
- potentiel de fin de partie

Le but n'est pas d'etre parfaite. Le but est d'avoir une base de comparaison pour la suite.

## 10. Fonction d'evaluation apprise V2

Quand la V1 heuristique sera stable, on pourra entrainer une meilleure evaluation.

Deux options serieuses:

### Option A - Value Network

Le modele predit un score de position.

Usage:

- remplace ou corrige l'evaluation heuristique dans la recherche

### Option B - Policy + Value Network

Le modele predit:

- les bons coups probables
- la valeur de la position

Usage:

- aide au move ordering
- aide a l'evaluation
- preparation a une recherche guidee plus moderne

Recommandation:

la meilleure option pour le projet semble etre `policy + value`, mais uniquement apres avoir valide la recherche et l'evaluation heuristique.

## 11. Protocole de benchmark

Le benchmark doit etre central dans le projet.

Chaque version du moteur doit etre testee contre:

- `minimax`
- `mcts`

Et si possible sur plusieurs budgets:

- rapide
- moyen
- fort

Le benchmark doit produire:

- CSV ou JSON
- winrate
- temps moyen par coup
- informations sur la profondeur
- version du moteur teste

## 12. Criteres de succes

Le projet progresse si:

- le moteur joue legalement
- le moteur termine les parties correctement
- le moteur devient competitif contre les niveaux intermediaires
- l'evaluation ameliore vraiment le winrate
- les nouvelles optimisations apportent un gain mesurable

## 13. Risques principaux

- partir trop tot sur le neuronal
- ne pas avoir de baseline heuristique solide
- creer une recherche trop lente
- melanger benchmark et entrainement sans protocole clair
- chercher a copier Stockfish sans tenir compte des specificites du Songo

## 14. Ligne directrice finale

Le projet doit suivre cette logique:

- d'abord un moteur de recherche fort
- ensuite une evaluation forte
- ensuite une integration recherche + evaluation
- enfin une optimisation progressive vers un vrai moteur competitif

## 15. Prochaine etape concrete

La prochaine etape concrete a ecrire dans le projet est un document `ENGINE_V1_DESIGN` qui tranche:

- l'API du moteur
- les modules a creer
- la representation d'etat
- le type de recherche de V1
- les heuristiques de V1
- le protocole de benchmark de V1
