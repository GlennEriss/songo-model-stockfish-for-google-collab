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
- debut de gestion explicite du versionning dataset:
  - registre central `data/dataset_registry.json`
  - sources de datasets enregistrees separement des datasets construits
  - selection explicite de la source pour `dataset-build`

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
- la selection automatique du plus grand dataset final construit quand `dataset_selection_mode: largest_built`

Effet attendu:

- moins d'instabilite sur de gros batchs
- moins de surentrainement inutile
- meilleure reutilisation GPU/temps Colab
- modele final plus coherent avec la meilleure validation observee
- training stable lance par defaut sur le dataset final le plus grand disponible dans le registre

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

Elle supporte maintenant aussi:

- `dataset_selection_mode: largest_built`

Effet attendu:

- evaluation automatique sur le plus grand dataset final disponible
- moins de risque d'evaluer un dataset obsolete par erreur
- logs plus clairs sur le checkpoint et le dataset reellement resolus
- notebook avec previsualisation explicite du dataset et du modele resolus avant `evaluation`

### 4.4 Benchmark plus lisible dans le notebook

Le notebook affiche maintenant aussi, juste avant `benchmark`:

- la cible benchmark resolue selon la config (`auto_latest`, `auto_best`, `engine_v1` ou checkpoint explicite)
- le dernier modele disponible dans `model_registry.json` quand le mode est `auto_latest`
- le modele promu courant pour comparaison rapide
- les adversaires benchmark (`matchups`)
- `games_per_matchup`
- `alternate_first_player`
- `max_moves`

Effet attendu:

- moins d'erreurs de lancement benchmark sur un mauvais checkpoint
- verification visuelle immediate du modele oppose aux adversaires de reference
- meilleure coherence entre notebook, config et execution reelle

## 5. Config Full Matrix Recommandee Actuelle

La config de train full matrix active maintenant:

- `dataset_selection_mode: largest_built`
- `gradient_clip_norm: 1.0`
- `early_stopping_patience: 6`
- `scheduler.type: cosine`
- `scheduler.min_lr: 0.00005`

Les options d'architecture sont presentes mais gardees a:

- `use_layer_norm: false`
- `dropout: 0.0`
- `residual_connections: false`

Cela permet de:

- toujours viser automatiquement le plus grand dataset final disponible
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

### Priorite 1 bis - Versionning dataset

- garder `benchmatch` comme source longue et sure
- permettre aussi la creation d'une nouvelle source a partir d'un corpus existant
- versionner les sources et les datasets construits separement
- conserver les originaux intacts lors des enrichissements

Etat:

- `dataset-generate` supporte maintenant:
  - `--generation-mode benchmatch`
  - `--generation-mode clone_existing`
  - `--generation-mode derive_existing`
  - `--generation-mode merge_existing`
  - `--dataset-source-id`
  - `--source-dataset-id`
  - `--source-dataset-ids`
  - `--derivation-strategy`
  - `--target-samples`
- `dataset-build` supporte maintenant:
  - `--source-dataset-id`
  - `--dataset-id-override`
  - `--target-labeled-samples`
- `dataset-merge-final` supporte maintenant:
  - `--dataset-id`
  - `--source-dataset-ids`
  - `--include-all-built`
  - `--dedupe-sample-ids`
- une commande de listing est disponible:
  - `dataset-list`
  - pour voir rapidement les sources et datasets deja enregistres
- les metadonnees sont enregistrees dans:
  - `data/dataset_registry.json`

Strategies de derivation deja disponibles:

- `unique_positions`
  - deduplication globale des positions equivalentes
- `endgame_focus`
  - conserve surtout les positions de fin de partie
- `high_branching`
  - conserve surtout les positions avec beaucoup de coups legaux

Recommandation simple:

- `benchmatch` pour produire un nouveau corpus source fiable
- `clone_existing` pour versionner une copie de travail
- `derive_existing` pour fabriquer rapidement une variante utile sans relancer les matchs
- `merge_existing` pour construire un `giant dataset source` a partir de plusieurs sources deja versionnees

### Gestion dataset en place

Commandes utiles des maintenant:

- lister toutes les sources et tous les datasets:
  - `python -m songo_model_stockfish.cli.main dataset-list --config config/dataset_generation.full_matrix.colab_pro.yaml`
- lister uniquement les sources:
  - `python -m songo_model_stockfish.cli.main dataset-list --config config/dataset_generation.full_matrix.colab_pro.yaml --kind sources`
- lister uniquement les datasets finaux:
  - `python -m songo_model_stockfish.cli.main dataset-list --config config/dataset_generation.full_matrix.colab_pro.yaml --kind built`
- sortie JSON exploitable dans un notebook ou un script:
  - `python -m songo_model_stockfish.cli.main dataset-list --config config/dataset_generation.full_matrix.colab_pro.yaml --json`
- generer une source cible a `2_000_000` positions:
  - `python -m songo_model_stockfish.cli.main dataset-generate --config config/dataset_generation.full_matrix.colab_pro.yaml --target-samples 2000000`
- fusionner plusieurs sources deja versionnees en une nouvelle grande source:
  - `python -m songo_model_stockfish.cli.main dataset-generate --config config/dataset_generation.full_matrix.colab_pro.yaml --generation-mode merge_existing --dataset-source-id sampled_full_matrix_colab_pro_giant --source-dataset-ids sampled_full_matrix_colab_pro sampled_full_matrix_colab_pro_unique`
- construire un dataset final cible a `2_000_000` labels:
  - `python -m songo_model_stockfish.cli.main dataset-build --config config/dataset_build.full_matrix.colab_pro.yaml --target-labeled-samples 2000000`
- fusionner tous les datasets finaux existants en un nouveau dataset versionne:
  - `python -m songo_model_stockfish.cli.main dataset-merge-final --config config/dataset_merge_final.colab_pro.yaml --dataset-id dataset_merged_final_colab_pro_insane_1m --include-all-built`

Cette commande permet de verifier rapidement:

- les sources disponibles
- leur mode de creation
- leur parent eventuel
- les variantes deja construites
- le teacher utilise pour les datasets finaux

Important:

- `target_samples` pilote la taille de la source dataset produite par `dataset-generate`
- `target_labeled_samples` pilote la taille du dataset final teacher-labeled produit par `dataset-build`
- pour viser un dataset final `2M`, il est en pratique recommande d'aligner les deux cibles a `2_000_000`
- `merge_existing` deduplique par `sample_id` par defaut
- `merge_existing` range les fichiers fusionnes par sous-dossier `dataset_source_id` pour eviter les collisions de chemins entre sources
- `merge_existing` produit maintenant aussi un `source_breakdown` par source:
  - fichiers scannes
  - positions scannees
  - fichiers retenus
  - positions retenues
  - doublons retires
  - fichiers raw recopies

### Ameliorations recommandees sur la gestion dataset

- ajouter un vrai champ `dataset_version` lisible humainement:
  - exemple: `2026-04-06__insane_1m__variant_a`
- distinguer clairement:
  - `dataset_source_id` pour les corpus sources
  - `dataset_id` pour les datasets finaux teacher-labeled
- stocker dans le registre:
  - parent direct
  - date de creation
  - mode de creation
  - taille cible
  - taille effective
  - teacher
  - tags libres comme `dedup`, `balanced`, `rare_positions`

Conventions recommandees des maintenant:

- pour les sources:
  - `sampled_full_matrix_colab_pro`
  - `sampled_full_matrix_colab_pro_unique`
  - `sampled_full_matrix_colab_pro_endgame`
- pour les datasets finaux:
  - `dataset_v3_full_matrix_colab_pro_unique_insane_1m`
  - `dataset_v3_full_matrix_colab_pro_endgame_insane_1m`

Regles utiles:

- le nom doit indiquer la famille source
- le nom doit indiquer la variante
- le nom du dataset final doit indiquer le teacher cible si important
- ne pas reutiliser un identifiant pour un contenu different

### Ameliorations recommandees sur la validation dataset

- ajouter un mode `dry-run` specifique aux transformations dataset pour afficher:
  - combien de fichiers seraient scannes
  - combien de positions seraient gardees
  - quelle source serait utilisee
- ajouter un rapport de legalite:
  - nombre de positions terminales
  - nombre de positions sans coup legal
  - nombre de doublons retires
- ajouter un rapport de distribution:
  - score gap
  - nombre de coups legaux
  - total de graines sur le plateau
  - repartition des matchups

### Ameliorations recommandees sur les modes de generation dataset

- garder `benchmatch` pour produire ou etendre le corpus de reference
- garder `clone_existing` pour dupliquer rapidement une source existante sans la modifier
- ajouter ensuite un mode `derive_existing` pour:
  - dedupliquer
  - filtrer
  - reequilibrer
  - reechantillonner
  - produire des variantes sans relancer de matchs

Ce mode `derive_existing` serait ideal pour:

- enrichir les positions rares
- reduire les doublons
- mieux couvrir certaines distributions de graines par case
- construire plusieurs variantes rapidement a partir du corpus `1m`

### Ameliorations recommandees sur les logs dataset

- ajouter dans `dataset-generate` un log de mode explicite:
  - `source_mode=benchmatch`
  - `source_mode=clone_existing`
  - `source_mode=derive_existing`
- ajouter un log de source cible:
  - `dataset_source_id=...`
- ajouter un log de parent si present:
  - `source_dataset_id=...`
- ajouter un log de strategie si mode derive:
  - `derivation_strategy=...`
- ajouter a la fin un recap lisible:
  - nombre de fichiers
  - nombre de positions
  - dossier de sortie
  - source parente

### Ameliorations recommandees sur les logs dataset-build

- logger explicitement:
  - `source_dataset_id`
  - `dataset_id`
  - teacher utilise
  - taille finale reelle des splits
- logger aussi:
  - `build_mode=teacher_label`
  - `output_dir`
  - pour une fusion finale: nombre de datasets sources, doublons retires, taille finale par split
- ajouter plus tard un mini resume des top-level metrics:
  - samples train
  - samples validation
  - samples test
  - skipped ratios

### Ameliorations deja ajoutees sur les datasets finaux

- les datasets finaux enregistres dans le registre portent maintenant aussi:
  - `build_mode`
  - `source_dataset_ids`
  - `parent_dataset_ids`
  - `dataset_version`
- `dataset-build` logge plus clairement:
  - `source_dataset_id`
  - `output_dir`
  - `build_mode`
- `dataset-merge-final` permet maintenant de:
  - fusionner plusieurs datasets finaux teachers identiques
  - ou fusionner directement tous les datasets finaux du registre
  - dedupliquer les `sample_ids` pendant la fusion
  - produire un nouveau dataset final versionne et re-enregistre dans le registre
  - produire un `source_breakdown` par split et par dataset final source:
    - `input_samples`
    - `kept_samples`
    - `duplicate_samples`
    - `unique_games`

### Ameliorations recommandees sur le notebook

- afficher le registre dataset directement dans Colab
- afficher les variables dataset effectives avant lancement
- proposer des presets nommes:
  - `stable_source`
  - `unique_positions`
  - `endgame_focus`
  - `high_branching`
- ajouter plus tard une cellule de comparaison rapide entre deux `dataset_metadata.json`

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
- `src/songo_model_stockfish/data/jobs.py`
- `src/songo_model_stockfish/cli/main.py`
- `config/train.full_matrix.colab_pro.yaml`
- `config/train.full_matrix.colab_pro.residual.yaml`
- `config/dataset_generation.full_matrix.colab_pro.yaml`
- `config/dataset_build.full_matrix.colab_pro.yaml`

Ce document doit rester la reference courte pour comprendre:

- ce qui est deploye aujourd'hui
- ce qui est deja fait
- ce qui doit venir ensuite
