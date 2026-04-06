# Improvement Plan

## 1. Objectif

Ce document sert de feuille de route courte et reutilisable pour:

- comprendre l'etat actuel du pipeline
- voir ce qui a deja ete ameliore
- identifier les prochaines evolutions a plus fort levier
- savoir quelles options sont deja disponibles dans le code

## 2. Etat Actuel Du Projet

Le projet dispose maintenant d'un pipeline Colab complet et resumable:

- `dataset-generate` pour produire des parties et positions echantillonnees
- `dataset-build` pour labelliser les positions avec un teacher fort
- `train` pour entrainer un modele `policy + value`
- `evaluation` pour mesurer les performances hors entrainement
- `benchmark` pour verifier la force pratique contre des adversaires de reference

Le dataset de travail principal vise actuellement:

- `dataset_v2_full_matrix_colab_pro_insane_1m`

Le teacher principal pour la construction du dataset est:

- `minimax:insane`

## 3. Ce Qui A Deja Ete Fait

### Dataset et build

- generation incrementale vers `1_000_000` positions
- rollover automatique des `job_id` de dataset generation et dataset build
- cache de labels partage sur Drive
- reconstruction de dataset final avec un teacher `insane`
- parallelisation du `dataset-build` par fichiers
- logs plus propres pour le build:
  - scan resume
  - distinction `reused` / `pending`
  - mode `parallel`, `sequential`, `sequential_fallback`
  - `files_per_sec`, `samples_per_sec` et `eta`

### Train, evaluation et benchmark

- configs Colab Pro dediees au full matrix
- `train` pointe vers le dataset `insane_1m`
- `evaluation` full matrix pointe aussi vers le dataset `insane_1m`
- logs benchmark plus detailles par partie et par matchup

## 4. Ameliorations Ajoutees Dans Ce Lot

### 4.1 Train plus robuste

Le pipeline d'entrainement supporte maintenant:

- `gradient_clip_norm`
- `early_stopping_patience`
- un scheduler de learning rate de type `cosine`
- l'export du meilleur checkpoint, pas seulement du dernier etat du modele

Effet attendu:

- moins d'instabilite sur de gros batchs
- moins de surentrainement inutile
- meilleure reutilisation GPU/temps Colab
- modele final plus coherent avec la meilleure validation observee

### 4.2 Architecture MLP preparée pour la suite

Le modele supporte maintenant, via config:

- `use_layer_norm`
- `dropout`
- `residual_connections`

Important:

- ces options existent dans le code
- elles restent desactivees par defaut dans la config full matrix actuelle
- cela preserve la compatibilite avec les checkpoints deja produits

### 4.3 Evaluation plus informative

L'evaluation calcule maintenant aussi:

- `value_mae`

En plus de:

- `loss_total`
- `loss_policy`
- `loss_value`
- `policy_accuracy_top1`
- `policy_accuracy_top3`

## 5. Config Full Matrix Recommandee Actuelle

La config de train full matrix active maintenant:

- `gradient_clip_norm: 1.0`
- `early_stopping_patience: 6`
- `scheduler.type: cosine`
- `scheduler.min_lr: 0.00005`

Les options d'architecture sont presentes mais gardees a:

- `use_layer_norm: false`
- `dropout: 0.0`
- `residual_connections: false`

Cela permet de:

- beneficier tout de suite des ameliorations de training
- sans casser la compatibilite avec les checkpoints existants

## 5.1 Variante Experimentale MLP Residuel

Une config de test dediee existe maintenant:

- `config/train.full_matrix.colab_pro.residual.yaml`

Cette variante active:

- `use_layer_norm: true`
- `dropout: 0.05`
- `residual_connections: true`
- `hidden_sizes: [1024, 1024, 1024, 512, 512, 256]`
- `model_id_prefix: songo_policy_value_colab_pro_residual`
- `init_from_promoted_best: false`

Le choix `init_from_promoted_best: false` est volontaire:

- les checkpoints entraines avec l'ancienne architecture simple ne sont pas compatibles avec cette nouvelle variante
- le run residual doit donc demarrer from scratch

Utilisation dans Colab:

- config stable:
  - `TRAIN_CONFIG = 'config/train.full_matrix.colab_pro.yaml'`
- config experimentale residual:
  - `TRAIN_CONFIG = 'config/train.full_matrix.colab_pro.residual.yaml'`

Recommandation:

- garder la config stable pour la production courante
- utiliser la config residual pour un run de comparaison dedie
- comparer ensuite evaluation et benchmark entre les deux familles de modeles

## 5.2 Protocole De Comparaison Stable vs Residual

Le but est de comparer deux familles de modeles sur:

- le meme dataset
- le meme protocole d'evaluation
- le meme protocole de benchmark

### Variante stable

- `TRAIN_CONFIG = 'config/train.full_matrix.colab_pro.yaml'`
- `TRAIN_JOB_ID = 'train_colab_pro_stable_001'`
- `EVALUATION_JOB_ID = 'eval_colab_pro_stable_001'`
- `BENCHMARK_JOB_ID = 'benchmark_colab_pro_stable_001'`

### Variante residual

- `TRAIN_CONFIG = 'config/train.full_matrix.colab_pro.residual.yaml'`
- `TRAIN_JOB_ID = 'train_colab_pro_residual_001'`
- `EVALUATION_JOB_ID = 'eval_colab_pro_residual_001'`
- `BENCHMARK_JOB_ID = 'benchmark_colab_pro_residual_001'`

### Sequence recommandee

1. verifier que le dataset `dataset_v2_full_matrix_colab_pro_insane_1m` est bien termine
2. lancer le training `stable`
3. lancer `evaluation` puis `benchmark` du modele stable
4. lancer le training `residual`
5. lancer `evaluation` puis `benchmark` du modele residual
6. comparer ensuite:
   - `best_validation_metric`
   - `policy_accuracy_top1`
   - `policy_accuracy_top3`
   - `value_mae`
   - `benchmark_score`
   - les winrates benchmark par matchup

### Regles pratiques

- ne pas reutiliser le meme `TRAIN_JOB_ID` entre `stable` et `residual`
- ne pas melanger les familles de modele dans les noms de jobs
- conserver la meme config d'evaluation et de benchmark pour les deux runs
- changer une seule variable experimentale a la fois quand c'est possible

### Conclusion attendue

Si `residual` est meilleur, il doit montrer un gain coherent sur:

- l'evaluation hors train
- et surtout le benchmark pratique

Si `residual` n'est meilleur qu'en validation mais pas en benchmark, la conclusion doit rester prudente.

## 5.3 Grille De Lecture Des Resultats

Utilise la grille suivante apres les runs `stable` et `residual`.

### A regarder en premier

- `benchmark_score`
- winrate total benchmark
- winrates benchmark par matchup

Pourquoi:

- le benchmark mesure la force pratique reelle
- c'est la metrique la plus importante pour choisir le meilleur modele

### A regarder ensuite

- `policy_accuracy_top1`
- `policy_accuracy_top3`
- `value_mae`
- `best_validation_metric`

Pourquoi:

- ces metriques aident a comprendre pourquoi un modele est meilleur ou moins bon
- elles ne remplacent pas le benchmark

### Regle simple de decision

- si `residual` est meilleur en benchmark et pas moins bon en evaluation, il gagne
- si `residual` est meilleur en evaluation mais pas en benchmark, conserver `stable`
- si `residual` est legerement moins bon en evaluation mais clairement meilleur en benchmark, il peut gagner
- si les resultats sont contradictoires ou trop proches, refaire un cycle de comparaison

### Signaux positifs pour residual

- `benchmark_score` au-dessus de `stable`
- meilleure robustesse sur plusieurs matchups, pas juste un seul
- `value_mae` en baisse ou stable
- `policy_accuracy_top1` qui ne se degrade pas fortement

### Signaux de prudence

- meilleure validation mais benchmark en baisse
- gain benchmark uniquement sur un matchup marginal
- `value_mae` qui se degrade fortement
- gros ecart entre train et validation

### Format pratique de comparaison

Quand les deux runs sont termines, compare:

- `jobs/<eval_job_id>/evaluation_summary.json`
- `jobs/<benchmark_job_id>/benchmark_summary.json`
- `jobs/<train_job_id>/training_summary.json`

Tableau minimal recommande:

- modele
- best_validation_metric
- policy_accuracy_top1
- policy_accuracy_top3
- value_mae
- benchmark_score
- winrate total
- meilleur matchup
- pire matchup
- decision finale

## 6. Plan D'Amelioration Priorise

### Priorite 1 - Qualite du dataset

- remplacer la policy cible "best move + lissage simple" par une vraie distribution issue des evaluations teacher
- sur-echantillonner les positions tactiques, retournements et fins de partie
- ajouter des stats de composition du dataset par matchup et par phase de jeu

### Priorite 2 - Architecture du modele

- activer puis tester `LayerNorm + residual_connections`
- comparer plusieurs tailles de backbone MLP
- mesurer l'impact de `dropout` sur generalisation et benchmark

Etat:

- une premiere config experimentale `residual` existe deja
- la prochaine etape sera d'executer un run complet et de mesurer son gain reel

### Priorite 3 - Evaluation plus fine

- ajouter des rapports par phase de jeu
- ajouter des rapports par famille de matchup
- ajouter une calibration plus fine de la value head

### Priorite 4 - Benchmark plus fort

- ajouter un score de type Elo ou pseudo-Elo
- suivre la progression par version de modele
- figer certains seeds et openings pour des comparaisons plus propres

### Priorite 5 - Boucle d'amelioration iterative

- faire jouer les modeles entraines
- relabelliser leurs positions avec le teacher
- reinjecter ces positions dans les cycles suivants

## 7. Recommandation Pratique

Pour la prochaine iteration utile:

1. laisser tourner le pipeline `insane_1m` avec les nouveaux controles de training
2. comparer les resultats benchmark avant/apres ce lot
3. si le gain est stable, ouvrir un deuxieme lot oriente `dataset quality`
4. ensuite tester une variante de modele avec `LayerNorm + residual_connections`

Cette variante est maintenant prete a etre lancee via:

1. `TRAIN_CONFIG = 'config/train.full_matrix.colab_pro.residual.yaml'`
2. la commande Colab de `train` habituelle

Pour une vraie comparaison, utiliser le protocole `stable vs residual` decrit plus haut plutot qu'un simple changement improvise de config.

## 8. Fichiers Concernes

- `src/songo_model_stockfish/training/model.py`
- `src/songo_model_stockfish/training/jobs.py`
- `src/songo_model_stockfish/evaluation/jobs.py`
- `config/train.full_matrix.colab_pro.yaml`
- `config/train.full_matrix.colab_pro.residual.yaml`

Ce document doit rester la reference courte pour comprendre:

- ce qui est deploye aujourd'hui
- ce qui est deja fait
- ce qui doit venir ensuite
