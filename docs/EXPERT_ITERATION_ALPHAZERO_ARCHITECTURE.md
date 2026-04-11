# Expert Iteration AlphaZero Architecture

## 1. Objectif

Definir la strategie cible pour augmenter fortement la puissance de decision:

- reseau `policy + value`
- recherche `PUCT/MCTS` guidee par le reseau
- generation de donnees par `self-play`
- reentrainement sur cibles de recherche

Cette boucle est la base d'une progression continue du niveau de jeu.

## 2. Pourquoi cette technique

Le modele policy brut (argmax) apprend des patterns utiles mais reste limite face a des moteurs de recherche.

Expert Iteration de type AlphaZero apporte:

- un "expert" dynamique: la recherche guidee par le reseau
- des cibles de train de meilleure qualite (`visit counts` plutot qu'un simple coup unique)
- une boucle d'amelioration cumulative:
  - meilleur reseau -> meilleure recherche -> meilleures donnees -> meilleur reseau

## 3. Principe de boucle

Cycle global:

1. jouer des parties `self-play` avec `PUCT` (pas policy argmax brut)
2. enregistrer pour chaque position:
   - etat encode
   - masque legal
   - distribution policy cible issue des visites MCTS
   - valeur finale de la partie (ou reanalyse value)
3. construire dataset train/validation/test
4. entrainer un nouveau modele `policy + value`
5. evaluer + benchmarker + tournoi interne
6. promouvoir seulement si le gate de qualite est valide
7. recommencer

## 4. Architecture cible

## 4.1 Inference / Decision

Pendant une partie:

- reseau donne `prior policy` + `value`
- `PUCT` selectionne/explore les coups
- decision finale basee sur les visites MCTS

Config de base recommandee:

- `model_search_enabled=true`
- `model_search_top_k` (phase intermediaire) puis vrai budget simulations PUCT
- temperature de decision:
  - debut de partie: exploration legere
  - milieu/fin: plus deterministe

## 4.2 Data generation

Nouveau flux dataset (self-play):

- collection de blocs de parties par worker
- chaque bloc produit un mini-dataset local worker
- merge verrouille vers dataset global (lock distribue deja en place)

Formats cibles par sample:

- `x`, `legal_mask`
- `policy_target_full` = distribution normalisee des visites PUCT
- `value_target` = resultat final du point de vue joueur a jouer
- metadata:
  - `game_id`, `ply`, `model_id`, `search_budget`, `temperature`

## 4.3 Training

Loss cible:

- policy loss sur distribution MCTS (`cross-entropy` soft)
- value loss (`MSE` ou `Huber`)
- regularisations existantes du projet conservables

Selection dataset:

- priorite dataset global fusionne (pas seulement shard worker)
- fallback sur plus gros dataset de la meme famille

## 4.4 Promotion gate

Promotion d'un modele seulement si:

- benchmark externe (`minimax/mcts`) en hausse
- tournoi interne en hausse (ou non-regression stricte)
- pas de regression critique sur evaluation offline

Exemple de gate:

- `benchmark_score_weighted` > best + marge
- `benchmark_elo_estimate` non regressif
- top-N tournoi interne stable ou meilleur

## 5. Integration avec l'architecture actuelle

## 5.1 Stockage et coordination

- Drive: artefacts lourds (games, datasets, checkpoints, reports)
- Redis: heartbeat/cache temps reel
- Firestore: etat durable (registry, progress, checkpoints, leases)

Le mode quota-first reste compatible:

- writes Firestore en micro-batch, pas par partie
- snapshots de progression a intervalle
- merge final protege par lock

## 5.2 Multi-Colab

Chaque worker:

- prend un bloc de travail (parties self-play)
- produit un mini-dataset local
- sync periodic des compteurs
- merge sequentiel vers global via lock

Comportement attendu:

- pas de collision de merge
- reprise possible apres interruption
- progression globale monotone

## 5.3 Notebook compact

Le notebook doit piloter:

- generation config active self-play + search
- lancement dataset generate/build
- suivi progression (Drive/Firestore/Redis)
- train + eval auto
- benchmark + tournoi interne

Etats a afficher:

- dataset effectivement utilise en train/eval
- config search benchmark active
- progression bloc worker et progression globale

## 6. Plan d'implementation recommande

## Phase A - Search practical (court terme)

- etendre `ModelAgent` vers vrai PUCT budgete
- garder compatibilite backward avec mode simple
- ajouter params YAML search explicites

Livrable:

- gain benchmark mesurable sans changer tout le pipeline

## Phase B - Self-play dataset (moyen terme)

- nouveau mode `dataset_generation: self_play_puct`
- enregistrement `visit counts` + winner final
- build dataset compatible train actuel

Livrable:

- premier dataset Expert Iteration exploitable

## Phase C - Reanalyse (moyen/long terme)

- reanalyser positions historiques avec meilleur modele
- corriger policy/value targets

Livrable:

- meilleure stabilite et moins de bruit de labels

## Phase D - Full loop automation (long terme)

- scheduler de cycles:
  - self-play -> train -> eval -> benchmark -> promotion
- guardrails de non-regression automatiques

Livrable:

- pipeline d'amelioration continue autonome

## 7. Metriques de succes

Le changement est valide si:

- tendance benchmark externe positive
- elo estimate en hausse sur plusieurs cycles
- tournoi interne non regressant
- baisse des "blunders" tactiques sur matchups cibles

Metriques a historiser:

- score benchmark par matchup
- score tournoi interne (points, wins, draws)
- perf par role (first/second player)
- vitesse d'inference + cout compute

## 8. Risques et mitigations

Risques:

- cout compute plus eleve (recherche)
- surapprentissage sur adversaires internes
- explosion du volume de donnees

Mitigations:

- budgets search progressifs
- benchmark externe obligatoire pour promotion
- quotas stricts et micro-batch writes
- retention et dedupe des jeux/samples

## 9. Decision projet

La strategie cible retenue est:

- Expert Iteration AlphaZero
- `policy + value` + `PUCT/MCTS` + `self-play` + `retrain`

Le pipeline actuel (Drive + Firestore + Redis + multi-Colab) sert de socle d'execution.
Le plan ci-dessus definit la trajectoire d'integration sans rupture de production.

## 10. Etat implementation (code)

Les briques suivantes sont maintenant implementees:

- `dataset_generation.source_mode=self_play_puct`
  - self-play modele vs modele
  - recherche PUCT avec budget (`self_play_num_simulations`, `self_play_c_puct`)
  - exploration controlee (`self_play_temperature*`, bruit Dirichlet racine)
  - cibles en sortie par sample:
    - `policy_target` (distribution de visites)
    - `value_target` (resultat final du point de vue joueur a jouer)
- `dataset_build.build_mode`
  - `teacher_label` (mode historique)
  - `source_prelabeled` (consomme directement policy/value deja presents)
  - `auto` (bascule automatique vers `source_prelabeled` pour les sources `self_play_puct`)
- notebook compact (generation de config active)
  - selection du mode de generation (`DATASET_GENERATE_SOURCE_MODE`)
  - passage des params self-play/PUCT
  - `dataset_build.build_mode` coherent automatiquement avec le mode generation
