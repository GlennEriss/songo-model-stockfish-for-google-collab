# Project Overview

## 1. Contexte

Le projet `songo-ai` contient plusieurs briques:

- un moteur de jeu Songo
- une IA `minimax`
- une IA `mcts`
- un ancien modele neuronal
- plusieurs anciens scripts d'entrainement

Pour `songo-model-stockfish-for-google-collab`, la decision de cadrage est la suivante:

- on reutilise `songo-ai` uniquement pour le moteur du jeu
- on reutilise `songo-ai` uniquement pour les benchmatchs
- on reutilise `songo-ai` uniquement pour `minimax`
- on reutilise `songo-ai` uniquement pour `mcts`

Tout ce qui concerne:

- l'ancien modele de `songo-ai`
- les anciens pipelines de train
- les approches RL, teacher loop et variantes associees dans `songo-ai`

doit etre considere comme deprecated pour ce nouveau projet.

## 2. Ce que ce projet est

`songo-model-stockfish-for-google-collab` est un nouveau projet d'IA, pas une simple extension du modele precedent.

L'objectif est de construire un systeme neuf, mieux cadre, plus propre, et pense des le depart pour:

- Google Colab
- l'experimentation rapide
- l'evaluation competitive
- l'evolution vers un moteur fort

## 3. Ce que ce projet n'est pas

Ce projet n'est pas:

- une continuation directe du modele de `songo-ai`
- une simple migration des anciens scripts
- une copie du pipeline precedent

Le nouveau repo doit avoir ses propres:

- representations
- jeux de donnees
- scripts de train
- checkpoints
- benchmarks internes
- logs et manifests d'execution

## 4. Role exact de `songo-ai`

Dans ce cadre, `songo-ai` joue seulement 4 roles:

### 4.1 Moteur du jeu

Il fournit:

- les regles
- la simulation des coups
- la validation des etats
- la terminaison des parties

### 4.2 Adversaire `minimax`

Il fournit une IA de reference forte, utile pour:

- generer des parties
- produire des coups de reference
- benchmarker le nouveau modele

### 4.3 Adversaire `mcts`

Il fournit une autre IA de reference, utile pour:

- varier les styles d'opposition
- tester la robustesse du nouveau modele
- completer le benchmark

### 4.4 Environnement de benchmark

Il fournit le cadre de comparaison pour mesurer si le nouveau modele progresse reellement.

## 5. Vision produit

Le but final est de construire une IA capable de jouer au Songo a haut niveau avec une philosophie proche d'un moteur moderne:

- evaluation forte
- bonne vitesse d'inference
- capacite a etre combinee avec une recherche
- progression mesurable contre des moteurs enseignants

Ici, "Stockfish du Songo" veut dire:

- un modele d'evaluation ou de decision fort
- une boucle d'amelioration continue
- une comparaison stricte contre des references solides
- une integration future dans un moteur avec recherche

## 6. Principes directeurs

Le projet doit suivre ces principes:

- nouveau code d'entrainement
- nouvelles conventions de dataset
- nouvelles experiences versionnees
- benchmark systematique contre `minimax` et `mcts`
- pas de dependance conceptuelle forte a l'ancien modele de `songo-ai`
- jobs resumables
- logs detailles et persistants
- artefacts separes du code

## 7. Questions fondatrices

Avant d'implementer en profondeur, il faut fixer:

### 7.1 Quel type de modele construire ?

Possibilites:

- modele `policy`
- modele `policy + value`
- evaluateur specialise type NNUE-like
- modele hybride integre a une recherche

### 7.2 Quelle cible prioritaire ?

Possibilites:

- un modele rapide et compact
- un modele plus fort mais plus lourd
- un modele fait pour jouer seul
- un modele fait pour assister une recherche

### 7.3 Quel protocole d'apprentissage ?

Possibilites:

- imitation a partir de `minimax`
- imitation a partir de `mcts`
- dataset mixte `minimax + mcts`
- self-play ensuite
- distillation ensuite

### 7.4 Faut-il fine-tuner un modele existant ?

Decision recommandee:

- non
- construire un reseau neuronal custom from scratch
- viser une architecture compacte `policy + value`

## 8. Recommandation de depart

La meilleure approche pour commencer reste:

- un nouveau modele `policy + value`
- un pipeline de donnees propre a ce repo
- une generation de labels via `minimax` et `mcts`
- un benchmark reel contre ces memes adversaires
- une separation stricte GitHub pour le code / Drive pour les artefacts
- un systeme de reprise pour train, benchmark, dataset et evaluation

Cette recommandation ne depend pas de l'ancien modele de `songo-ai`.

## 9. Pipeline cible du nouveau projet

Le pipeline doit etre pense comme suit:

1. generation de positions via `minimax` et `mcts`
2. preparation des donnees dans un format versionne
3. entrainement dans Colab avec scripts propres a ce repo
4. sauvegarde reguliere de checkpoints et etats de jobs
5. benchmark contre `minimax` et `mcts`
6. export de rapports, model cards et metriques
7. iteration sur architecture, donnees et recherche

## 10. Metriques de succes

Le projet devra mesurer:

- legalite des coups proposes
- stabilite du modele en partie complete
- accuracy sur les labels de reference
- winrate contre `minimax`
- winrate contre `mcts`
- vitesse d'inference
- profondeur atteinte et temps moyen par coup
- traceabilite dataset / config / commit git

Le vrai critere de succes reste la performance en jeu reel.

## 11. Risques principaux

- recreer sans le vouloir les erreurs de l'ancien pipeline
- construire un dataset trop proche d'un seul enseignant
- confondre bonne loss et vraie force de jeu
- produire un modele trop lent pour etre utile

## 12. Garde-fous

Pour eviter cela:

- garder `songo-ai` au role de reference et non de socle d'entrainement
- benchmarker tres tot
- versionner les datasets
- versionner les configs
- comparer plusieurs variantes de modeles du nouveau repo
- separer strictement code et artefacts
- rendre tous les jobs resumables

## 13. Premier objectif concret

Le premier objectif concret de ce projet est:

- definir le premier format de dataset natif
- definir le premier modele natif
- definir le premier notebook Colab natif
- definir le premier protocole de benchmark natif
- definir l'architecture logs / erreurs / reprise

## 14. Decision de cadrage actuelle

Decision retenue a ce stade:

- `songo-ai` sert seulement de moteur, benchmark, `minimax`, `mcts`
- l'ancien modele de `songo-ai` est deprecated ici
- les anciennes methodes de train de `songo-ai` sont deprecated ici
- le nouveau modele doit etre pense et code dans ce repo

## 15. Prochaine etape logique

La prochaine etape logique est de produire un document de design qui tranche:

- l'architecture du modele V1
- le format de dataset V1
- le protocole de benchmark V1
- le pipeline Colab V1
- la strategie de persistance et de reprise
