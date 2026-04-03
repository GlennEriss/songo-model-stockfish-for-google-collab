# Engine V1 Design

## 1. Objectif du document

Ce document fixe les decisions concretes pour la V1 du moteur du projet `songo-model-stockfish-for-google-collab`.

La V1 doit permettre de:

- construire un moteur de recherche jouable
- avoir une evaluation heuristique simple mais utile
- brancher un benchmark propre contre `minimax` et `mcts`
- preparer l'arrivee d'une evaluation neuronale plus tard

## 2. Perimetre de la V1

La V1 couvre:

- un moteur de recherche natif a ce repo
- une representation d'etat exploitable par la recherche
- une evaluation heuristique V1
- une interface claire pour jouer un coup
- un protocole de benchmark V1

La V1 ne couvre pas encore:

- une evaluation neuronale
- une architecture NNUE-like
- des optimisations avancees type late move reductions
- un moteur final optimise

## 3. Decision d'architecture

La V1 sera basee sur:

- un moteur `negamax` avec `alpha-beta pruning`
- `iterative deepening`
- une limite de profondeur
- une limite de temps
- un `move ordering` simple
- une evaluation heuristique rapide

Pourquoi `negamax`:

- plus simple a implementer proprement
- plus lisible qu'un minimax separe max/min
- standard pour evoluer ensuite vers un moteur plus fort

## 4. API cible du moteur

Le moteur devra exposer une API simple.

### 4.1 Objet de configuration

Le moteur doit avoir une configuration du type:

```python
from dataclasses import dataclass


@dataclass
class EngineConfig:
    max_depth: int = 6
    time_ms: int | None = None
    use_iterative_deepening: bool = True
    use_transposition_table: bool = False
    use_move_ordering: bool = True
```

### 4.2 Coup choisi

Le point d'entree principal doit ressembler a:

```python
def choose_move(state: "GameState", config: EngineConfig) -> tuple[int, "SearchInfo"]:
    ...
```

Le coup retourne reste compatible avec le format deja utilise dans `songo-ai`:

- un entier representant la case a jouer

### 4.3 Informations de recherche

Le moteur doit retourner des informations de debug et de benchmark:

```python
from dataclasses import dataclass


@dataclass
class SearchInfo:
    best_score: float
    depth_reached: int
    nodes_searched: int
    elapsed_ms: float
    pv: list[int]
```

Le champ `pv` correspond a la variation principale estimee.

## 5. Modules Python a creer

La structure V1 recommandee est:

- `src/songo_model_stockfish/engine/__init__.py`
- `src/songo_model_stockfish/engine/config.py`
- `src/songo_model_stockfish/engine/types.py`
- `src/songo_model_stockfish/engine/search.py`
- `src/songo_model_stockfish/engine/order.py`
- `src/songo_model_stockfish/evaluation/__init__.py`
- `src/songo_model_stockfish/evaluation/heuristic_v1.py`
- `src/songo_model_stockfish/benchmark/__init__.py`
- `src/songo_model_stockfish/benchmark/play_match.py`
- `src/songo_model_stockfish/benchmark/run_benchmark.py`

### 5.1 Role de chaque module

`engine/config.py`

- configuration du moteur
- presets simples pour tests rapides

`engine/types.py`

- dataclasses de base
- `SearchInfo`
- types internes du moteur

`engine/search.py`

- `choose_move`
- `negamax`
- `alpha-beta pruning`
- iterative deepening
- gestion du temps

`engine/order.py`

- tri des coups
- heuristiques simples de priorisation

`evaluation/heuristic_v1.py`

- score heuristique d'une position

`benchmark/play_match.py`

- execution d'une partie entre deux agents

`benchmark/run_benchmark.py`

- execution de plusieurs parties
- export des resultats

## 6. Representation d'etat V1

La V1 doit garder une representation simple, compatible avec le moteur de `songo-ai`.

### 6.1 Principe

Au debut, on ne reinvente pas les regles du jeu. On s'appuie sur le state du moteur de `songo-ai`.

La representation interne V1 peut donc etre:

- soit le state natif du moteur de `songo-ai`
- soit un petit wrapper local autour de ce state

Recommendation:

utiliser un wrapper local pour garder une frontiere claire.

### 6.2 Type recommande

```python
from dataclasses import dataclass


@dataclass
class EngineState:
    raw_state: object
```

Ensuite, les fonctions internes du moteur consomment ce wrapper.

### 6.3 Fonctions minimales necessaires

Le moteur V1 devra disposer de wrappers pour:

- lister les coups legaux
- simuler un coup
- savoir si la partie est terminee
- obtenir le joueur courant
- obtenir le gagnant ou l'utilite terminale

Exemple d'interface:

```python
def legal_moves(state: EngineState) -> list[int]:
    ...


def play_move(state: EngineState, move: int) -> EngineState:
    ...


def is_terminal(state: EngineState) -> bool:
    ...
```

## 7. Strategie d'integration avec `songo-ai`

La V1 doit integrer `songo-ai` comme backend regles/benchmark, sans dependre de ses anciens pipelines de modeles.

On doit creer un petit adaptateur local, par exemple:

- `src/songo_model_stockfish/adapters/songo_ai_game.py`

Ce module devra centraliser:

- import du moteur du jeu
- import des helpers de simulation
- conversion eventuelle des etats

Cela evitera de disperser des imports `songo-ai` dans tout le repo.

## 8. Search Algorithm V1

### 8.1 Algorithme retenu

Le moteur V1 utilisera:

- `negamax`
- `alpha-beta pruning`
- iterative deepening optionnelle

### 8.2 Boucle de haut niveau

Pseudo-code:

```python
def choose_move(state, config):
    best_move = None
    best_info = None
    for depth in range(1, config.max_depth + 1):
        move, info = search_depth(state, depth, config)
        best_move = move
        best_info = info
        if timeout:
            break
    return best_move, best_info
```

### 8.3 Choix de score

Convention recommandee:

- score positif = bon pour le joueur au trait
- score negatif = mauvais pour le joueur au trait

Les scores terminaux doivent etre clairement plus grands que les scores heuristiques.

Exemple:

- victoire terminale: `+1_000_000 - ply`
- defaite terminale: `-1_000_000 + ply`

Cela aide a preferer:

- les victoires rapides
- les defaites tardives

## 9. Move Ordering V1

La V1 doit utiliser un `move ordering` simple et peu couteux.

Ordre recommande:

1. coups menant a une capture immediate probable
2. coups deja meilleurs a profondeur precedente si iterative deepening
3. autres coups

Au debut, il ne faut pas chercher des heuristiques trop complexes.

Le but est juste d'ameliorer l'efficacite de `alpha-beta`.

## 10. Evaluation heuristique V1

### 10.1 Objectif

L'evaluation heuristique V1 doit etre:

- simple
- rapide
- interpretable
- facilement ajustable

### 10.2 Forme generale

L'evaluation doit combiner quelques signaux stables:

```text
eval = 
    w_score * score_diff
  + w_mobility * mobility_diff
  + w_capture * capture_potential_diff
  + w_risk * capture_risk_diff
  + w_seeds * board_seed_balance
```

### 10.3 Features V1 retenues

Les heuristiques V1 proposees sont:

#### A. Score diff

- difference de graines capturees entre joueur courant et adversaire

Pourquoi:

- c'est le signal le plus important

#### B. Mobility diff

- difference du nombre de coups legaux

Pourquoi:

- evite les positions bloquees
- favorise la flexibilite

#### C. Immediate capture potential

- estimation simple du potentiel de capture au prochain coup

Pourquoi:

- utile tactiquement

#### D. Immediate opponent risk

- estimation simple du risque de capture adverse

Pourquoi:

- evite des coups naifs

#### E. Seed distribution balance

- mesure grossiere de la repartition des graines sur les cases

Pourquoi:

- certaines distributions sont plus dangereuses ou steriles que d'autres

### 10.4 Heuristiques non retenues en V1

On ne met pas en V1:

- formule tres compliquee
- heuristiques lourdes en calcul
- evaluation dependante d'une simulation trop profonde

La V1 doit rester rapide.

## 11. Calibration initiale des poids

Les poids V1 doivent etre definis manuellement, puis ajustes au benchmark.

Exemple de depart:

```text
w_score   = 100.0
w_mobility = 5.0
w_capture = 20.0
w_risk    = -18.0
w_seeds   = 2.0
```

Ce ne sont pas des valeurs finales. Ce sont des valeurs de depart a tester.

## 12. Gestion du temps V1

Le moteur V1 doit pouvoir fonctionner:

- avec une profondeur fixe
- ou avec un budget temps simple

Regle recommandee:

- si `time_ms` est defini, iterative deepening s'arrete au timeout
- sinon le moteur cherche jusqu'a `max_depth`

## 13. Table de transposition V1

Decision:

- pas obligatoire en toute premiere sous-version
- a ajouter tres vite en V1.1 si la base marche bien

Pourquoi:

- la priorite initiale est la simplicite
- mais la transposition sera probablement tres utile ensuite

## 14. Protocole de benchmark V1

### 14.1 Objectif

Verifier rapidement si le nouveau moteur est:

- legal
- stable
- competitif

### 14.2 Adversaires de reference

Le benchmark V1 doit au minimum inclure:

- `minimax:medium`
- `minimax:hard`
- `mcts:medium`
- `mcts:hard`

### 14.3 Series minimales

Chaque benchmark V1 doit lancer:

- au moins 20 parties par matchup en smoke test
- puis 50 a 100 parties pour une evaluation un peu plus fiable

### 14.4 Alternance des couleurs

Le moteur doit jouer:

- une moitie des parties en premier joueur
- une moitie des parties en second joueur

### 14.5 Metriques a exporter

Le benchmark doit exporter:

- nombre de victoires
- nombre de defaites
- nombre de nulles si applicable
- winrate
- temps moyen par coup
- temps total par partie
- profondeur moyenne atteinte
- nombre moyen de noeuds explores

### 14.6 Formats de sortie

Formats recommandes:

- `csv`
- `json`

## 15. Criteres d'acceptation de la V1

La V1 sera consideree comme valide si:

- le moteur joue toujours un coup legal
- le moteur termine des parties completes sans erreur
- le moteur retourne des infos de recherche coherentes
- le benchmark s'execute contre `minimax` et `mcts`
- l'evaluation heuristique peut etre ajustee facilement

## 16. Ordre concret d'implementation

Ordre recommande:

1. creer l'adaptateur `songo-ai`
2. creer les types et la config moteur
3. implementer `legal_moves`, `play_move`, `is_terminal`
4. implementer `negamax` simple
5. brancher l'evaluation heuristique
6. ajouter iterative deepening
7. ajouter move ordering
8. creer le benchmark V1
9. mesurer et ajuster les poids heuristiques

## 17. Suite apres la V1

Une fois la V1 stable:

- ajouter une table de transposition
- generer un dataset de positions
- entrainer une evaluation neuronale
- tester un moteur hybride recherche + evaluation apprise
