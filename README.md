# songo-model-stockfish-for-google-collab

Nouveau projet dedie a la construction d'un vrai moteur d'IA pour le Songo dans Google Colab.

## Positionnement

Ce projet repart sur une base neuve.

Dans `songo-ai`, on conserve uniquement:

- le moteur du jeu
- les benchmatchs
- l'IA `minimax`
- l'IA `mcts`

Pour ce nouveau projet, il faut considerer comme deprecated:

- l'ancien modele de `songo-ai`
- les anciennes methodes d'entrainement de `songo-ai`
- les anciens pipelines de RL / teacher loop / imitation de `songo-ai`

Autrement dit, `songo-ai` devient ici un projet de reference pour:

- simuler le jeu
- produire des adversaires de benchmark
- verifier la force du nouveau modele

Le coeur du nouveau modele doit etre pense et implemente dans ce repo.

## Vision

Construire progressivement un moteur "Stockfish du Songo":

- un modele d'IA propre a ce projet
- un pipeline d'entrainement propre a ce projet
- une evaluation forte contre `minimax` et `mcts`
- une trajectoire vers un moteur avec recherche guidee

La ligne technique retenue est:

- baseline moteur heuristique pour stabiliser la recherche
- reseau neuronal custom from scratch
- architecture cible `policy + value`
- pas de fine-tuning d'un modele generaliste comme base principale
- jobs resumables en runtime local + artefacts persistants sur Drive
- coordination multi-Colab via Redis (temps reel) + Firestore (durable)

## Contenu initial

- `docs/DOCS_INDEX.md` : point d'entree central dans toute la documentation
- `docs/PROJECT_OVERVIEW.md` : cadrage du nouveau projet
- `docs/ROADMAP.md` : roadmap alignee sur le nouveau cadre
- `docs/IMPROVEMENT_PLAN.md` : etat actuel, ameliorations deja faites et plan d'evolution priorise
- `docs/COLAB_PIPELINE.md` : organisation cible pour l'execution Colab
- `docs/STOCKFISH_PLAN.md` : plan de construction du moteur type Stockfish
- `docs/ENGINE_V1_DESIGN.md` : design concret de la premiere version du moteur
- `docs/SYSTEM_ARCHITECTURE.md` : architecture globale, logs, erreurs, reprise
- `docs/MODEL_STRATEGY.md` : strategie modele et decision from scratch
- `docs/EXPERT_ITERATION_ALPHAZERO_ARCHITECTURE.md` : architecture cible d'amelioration continue (policy+value+PUCT+self-play+retrain)
- `docs/DATASET_AND_BENCHMARK_ARCHITECTURE.md` : pipeline dataset et benchmatch
- `docs/COLAB_OPERATIONS.md` : operations Colab, Drive, GitHub, reprise
- `docs/FIRESTORE_ARCHITECTURE_20M.md` : architecture multi-Colab quota-first et plan P0/P1/P2
  - Drive + Firestore + Redis
- `docs/OPERATIONS_SPEC.md` : schemas des jobs, fichiers de reprise et model cards
- `docs/REPO_STRUCTURE_SPEC.md` : structure cible du repository
- `docs/ARTIFACTS_AND_LOGGING_SPEC.md` : conventions d'artefacts et de logs
- `docs/DATASET_V1_SPEC.md` : specification du premier dataset
- `docs/JOB_CONFIG_SPEC.md` : specification des fichiers YAML de jobs
- `docs/CLI_SPEC.md` : specification des commandes CLI disponibles
- `docs/STATE_REPRESENTATION_SPEC.md` : specification de la representation d'etat V1
- `docs/MODEL_V1_SPEC.md` : specification du premier modele neuronal
- `docs/BENCHMARK_V1_SPEC.md` : specification du protocole benchmark V1
- `docs/DECISIONS_REGISTER.md` : registre central des decisions prises et points ouverts
- `docs/OPEN_DECISIONS_V1.md` : recommandations tranchees sur les derniers points ouverts V1
- `src/songo_model_stockfish/` : package Python principal du projet
- `config/` : configurations YAML du pipeline dataset/train/evaluation/benchmark
- `notebooks/colab_compact.ipynb` : notebook principal multi-Colab
- `scripts/` : scripts utilitaires (generation notebook, bootstrap, operations)

## Regle de lecture importante

Si une idee, un modele ou un script vient de `songo-ai`, on ne le reprend pas automatiquement.

On ne reprend que:

- les regles du jeu
- les outils de benchmark
- `minimax`
- `mcts`

Tout le reste doit etre re-evalue et, si besoin, re-implemente proprement dans ce projet.

## Priorites

1. definir l'architecture du nouveau modele
2. definir le nouveau pipeline de donnees
3. definir le nouveau pipeline d'entrainement et de reprise
4. definir le benchmark contre `minimax` et `mcts`
5. definir les operations Colab, Drive et GitHub
6. definir les schemas de logs, jobs et artefacts

## Etape suivante recommandee

La suite logique est de documenter puis implementer:

- le type exact de modele cible
- le format des donnees de ce nouveau projet
- les scripts de generation de donnees via `minimax` et `mcts`
- le premier pipeline d'entrainement vraiment propre a ce repo
- le systeme de logs, checkpoints et reprise

## Configs Colab Pro

Le repo contient maintenant des configurations dediees a une machine Google Colab Pro avec forte RAM et GPU:

- `config/train.colab_pro.yaml`
- `config/evaluation.colab_pro.yaml`
- `config/dataset_generation.colab_pro.yaml`
- `config/dataset_build.colab_pro.yaml`
- `config/dataset_build.full_matrix.colab_pro.yaml`
- `config/dataset_merge_final.colab_pro.yaml`
- `config/train.full_matrix.colab_pro.yaml`
- `config/train.full_matrix.colab_pro.residual.yaml`
- `config/evaluation.full_matrix.colab_pro.yaml`
- `config/benchmark.colab_pro.yaml`

L'intention est la suivante:

- `train` et `evaluate` exploitent le GPU avec `device: cuda`, `mixed_precision`, `pin_memory`, `persistent_workers` et `prefetch_factor`
- `dataset_generation` reste surtout CPU-bound a cause de `minimax` et `mcts`, mais il supporte maintenant l'execution parallele par parties via `num_workers` et `max_pending_futures`
- `dataset_build` produit un dataset Colab dedie pour ne pas melanger les artefacts smoke locaux et les artefacts de production

Important:

- `dataset_generation` via `minimax` / `mcts` n'utilise pas le GPU comme levier principal, mais exploite mieux CPU + RAM en parallele
- l'optimisation GPU est aujourd'hui principalement effective sur `train`, `evaluate` et `benchmark` quand la cible est un checkpoint neuronal
- les configs full matrix de `train` et `evaluation` resolvent maintenant automatiquement le plus grand dataset final disponible dans le registre

## Pipeline Incremental Vers 1M

Le pipeline dataset full matrix est maintenant pense pour l'accumulation incrementale:

- `dataset-generate` n'est plus traite comme une production "une fois pour toutes"
- a chaque relance, il ajoute de nouvelles parties dans `data/raw_full_matrix_colab_pro` et `data/sampled_full_matrix_colab_pro`
- la config full matrix fixe maintenant un objectif `target_samples: 1000000`
- les jobs `dataset_generation` et `dataset_build` font eux aussi un rollover automatique de `job_id` quand un cycle precedent est deja termine

Pour la construction du dataset final:

- `dataset-build` utilise maintenant un cache de labels partage sur Drive
- ce cache est separe par teacher, donc passer de `hard` a `insane` ne reutilise pas les anciens labels faibles
- la config full matrix vise maintenant un dataset final `insane` dedie:
  - `dataset_v2_full_matrix_colab_pro_insane_1m`

La gestion dataset supporte maintenant plusieurs modes:

- `dataset-generate --generation-mode benchmatch`
  - produit ou etend un corpus source fiable a partir de matchs
- `dataset-generate --generation-mode self_play_puct`
  - produit un corpus Expert Iteration via auto-jeu modele+PUCT avec `policy_target` (visites) et `value_target` (issue finale)
- `dataset-generate --generation-mode clone_existing`
  - duplique une source existante pour creer une nouvelle base versionnee
- `dataset-generate --generation-mode derive_existing`
  - cree une variante rapide a partir d'un corpus existant sans relancer les matchs
- `dataset-generate --generation-mode augment_existing`
  - cree une nouvelle grande source en rejouant des coups legaux a partir d'un corpus existant, avec deduplication par etat
- `dataset-generate --generation-mode merge_existing`
  - fusionne plusieurs sources existantes en une nouvelle grande source dedupliquee
- `dataset-build --source-dataset-id ...`
  - permet de choisir explicitement la source a enrichir avec le teacher
- `dataset-build --build-mode source_prelabeled`
  - consomme directement des samples deja labels (`policy_target` + `value_target`) sans relabel teacher
- `dataset-build --build-mode auto`
  - bascule automatiquement sur `source_prelabeled` pour une source `self_play_puct`
- `dataset-build --dataset-id-override ...`
  - permet de versionner une nouvelle sortie sans modifier le YAML
- `dataset-generate --target-samples ...`
  - permet de choisir une taille cible comme `2000000` ou `3000000` au lancement
- `dataset-build --target-labeled-samples ...`
  - permet de choisir la taille cible du dataset final labelise au lancement
- `dataset-merge-final --dataset-id ... --include-all-built`
  - permet de fusionner tous les datasets finaux existants en un nouveau dataset final versionne
- `dataset-merge-final --dataset-id ... --source-dataset-ids ...`
  - permet de fusionner seulement une selection de datasets finaux
- `dataset-list`
  - affiche les sources et datasets deja enregistres dans le registre

Strategies de derivation deja disponibles:

- `unique_positions`
- `endgame_focus`
- `high_branching`

Pour les datasets finaux teacher-labeled:

- `dataset-build` produit les fichiers:
  - `train.npz`
  - `validation.npz`
  - `test.npz`
- `dataset-merge-final` peut maintenant fusionner ces datasets finaux entre eux
- la fusion conserve un nouveau dataset final versionne
- les `sample_ids` peuvent etre dedupliques automatiquement pour eviter un gros dataset artificiellement duplique

Donc:

- on conserve les positions deja generees
- on ne repart pas de zero sur le corpus brut
- on peut maintenant composer une grande source a partir de plusieurs sources deja versionnees
- la fusion de sources fournit aussi un detail par source sur la contribution reelle et les doublons retires
- mais on reconstruit les labels finaux avec un teacher plus fort et un objectif de plus grande echelle

## Workflow Colab

Le projet contient maintenant une couche Colab dediee pour mettre a jour le code sans toucher aux artefacts persistants:

- `notebooks/colab_compact.ipynb`
  Notebook principal pour monter Drive, generer les configs actives, lancer `dataset-generate` + `dataset-build` en parallele, monitorer la progression globale, puis enchainer train/evaluation/benchmark.

Important pour Colab:

- le notebook installe les dependances directement dans le runtime Colab
- il ne cree pas de `.venv`
- toutes les commandes du notebook s'executent donc directement avec `python`
- il n'a plus besoin de cloner `songo-ai`
- le moteur de reference, `minimax` et `mcts` necessaires au dataset et au benchmark sont maintenant embarques dans ce repo
- le runtime multi-Colab utilise Redis (temps reel) + Firestore (source durable de coordination)
- en pratique, il faut renseigner `FIRESTORE_CREDENTIALS_PATH` avec un service account JSON
- en multi-workers, activer `LOW_QUOTA_PROFILE=True` pour limiter les reads/writes Firestore
- les configs actives train/eval privilegient le dataset global fusionne, puis fallback sur le plus gros shard de la famille
- le benchmark modele utilise une recherche legere configurable (`model_search_enabled`, `model_search_top_k`, etc.)

Scripts utiles:

- `scripts/colab/init_drive_layout.sh`
- `scripts/colab/update_repo_from_github.sh`
- `scripts/colab/bootstrap_from_github.sh`
- `scripts/colab/snapshot_code_to_drive.sh`
- `scripts/colab/status_watch.sh`

Principe recommande:

- code de travail dans `/content/songo-model-stockfish-for-google-collab`
- artefacts persistants dans `/content/drive/MyDrive/songo-stockfish`
- etat runtime volatil dans `/content/songo-stockfish-runtime` (recommande)
- backup hybride des etats essentiels jobs vers `/content/drive/MyDrive/songo-stockfish/runtime_backup/jobs`
- mise a jour Git sur le worktree uniquement
- datasets, checkpoints et rapports sur Drive
- `jobs/`, `logs/pipeline/*`, manifests live et snapshots monitoring en local runtime (sync Firestore/Redis pour coordination)

Commande utile pour voir rapidement ce qui existe deja:

- `python -m songo_model_stockfish.cli.main dataset-list --config config/dataset_generation.full_matrix.colab_pro.yaml`

## Continuer Un Modele

Le pipeline distingue trois cas:

- meme `job_id`
  Reprise du meme entrainement depuis son dernier checkpoint

- nouveau `job_id` + `init_checkpoint_path: ""` + `init_from_promoted_best: true`
  Nouveau modele initialise par defaut depuis le meilleur modele promu courant

- nouveau `job_id` + `init_checkpoint_path` renseigne
  Nouveau run initialise a partir d'un modele existant, pour continuer a l'ameliorer sans ecraser l'historique precedent

- nouveau `job_id` + `init_checkpoint_path: ""` + `init_from_promoted_best: false`
  Nouveau modele entraine from scratch

Dans ce troisieme cas:

- le modele parent est charge au demarrage
- une copie du checkpoint parent est stockee dans `models/lineage/`
- le nouveau run ecrit son propre `model_id`, ses propres checkpoints et son propre modele final

Par defaut, le training:

- attribue automatiquement le prochain `model_id` versionne a partir de `model_id_prefix`
- repart depuis le meilleur modele promu si disponible

Il pointe donc vers:

- `models/promoted/best/model.pt`

Ce dossier est mis a jour automatiquement a chaque changement du classement des modeles.

Il contient:

- `model.pt`
- `model_card.json`
- `metadata.json`

Donc en pratique, si tu ne touches pas la config, le prochain nouvel entrainement repartira du meilleur modele promu disponible.

Pour garder un controle manuel si besoin, tu peux encore utiliser:

- `scripts/training/prepare_next_version.py`

Ce script genere automatiquement:

- `config/generated/train.colab_pro.vN.yaml`
- `config/generated/evaluation.colab_pro.vN.yaml`
- `config/generated/benchmark.colab_pro.vN.yaml`

et affiche les commandes associees, mais ce n'est plus obligatoire.

## Classement Des Modeles

Le projet maintient un registre persistant:

- `models/model_registry.json`

Ce registre agrege progressivement:

- `best_validation_metric`
- `evaluation_top1`
- `evaluation_top3`
- `benchmark_score`
- `checkpoint_path`
- `model_card_path`
- la filiation eventuelle avec un modele parent

Le classement courant est ordonne en priorite par:

1. `benchmark_score`
2. `evaluation_top1`
3. ordre de creation recent

## Etat Actuel Et Ameliorations Recentes

Pour une vue concise et reutilisable de l'etat du projet, lire:

- `docs/IMPROVEMENT_PLAN.md`

Ce document centralise:

- ce qui est deja en place dans le pipeline
- ce qui a ete ameliore recemment
- les nouvelles options de training deja disponibles
- les prochaines evolutions recommandees

Sur le lot actuel, les evolutions concretes ajoutees sont:

- `dataset-build` plus lisible avec scan resume, debit et ETA
- `dataset-generate` peut maintenant fonctionner en:
  - `benchmatch`
  - `clone_existing`
  - `derive_existing`
- `dataset-build` peut maintenant choisir explicitement quelle source de dataset il enrichit
- versionning dataset via `dataset_registry/primary` (Firestore) avec compatibilite locale `data/dataset_registry.json`
- `train` avec `gradient clipping`, scheduler `cosine` et `early stopping`
- export du meilleur checkpoint comme modele final
- `evaluation` enrichie avec `value_mae`
- support dans le code pour un MLP plus robuste avec:
  - `use_layer_norm`
  - `dropout`
  - `residual_connections`

Une config experimentale separee est maintenant disponible pour tester cette variante sans toucher a la config stable:

- `config/train.full_matrix.colab_pro.residual.yaml`

Le protocole recommande pour comparer proprement les variantes `stable` et `residual` est documente dans:

- `docs/IMPROVEMENT_PLAN.md`
