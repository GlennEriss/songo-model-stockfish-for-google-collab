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
- jobs resumables et artefacts persistants sur Drive

## Contenu initial

- `docs/DOCS_INDEX.md` : point d'entree central dans toute la documentation
- `docs/PROJECT_OVERVIEW.md` : cadrage du nouveau projet
- `docs/ROADMAP.md` : roadmap alignee sur le nouveau cadre
- `docs/COLAB_PIPELINE.md` : organisation cible pour l'execution Colab
- `docs/STOCKFISH_PLAN.md` : plan de construction du moteur type Stockfish
- `docs/ENGINE_V1_DESIGN.md` : design concret de la premiere version du moteur
- `docs/SYSTEM_ARCHITECTURE.md` : architecture globale, logs, erreurs, reprise
- `docs/MODEL_STRATEGY.md` : strategie modele et decision from scratch
- `docs/DATASET_AND_BENCHMARK_ARCHITECTURE.md` : pipeline dataset et benchmatch
- `docs/COLAB_OPERATIONS.md` : operations Colab, Drive, GitHub, reprise
- `docs/OPERATIONS_SPEC.md` : schemas des jobs, fichiers de reprise et model cards
- `docs/REPO_STRUCTURE_SPEC.md` : structure cible du repository
- `docs/ARTIFACTS_AND_LOGGING_SPEC.md` : conventions d'artefacts et de logs
- `docs/DATASET_V1_SPEC.md` : specification du premier dataset
- `docs/JOB_CONFIG_SPEC.md` : specification des fichiers YAML de jobs
- `docs/CLI_SPEC.md` : specification des futures commandes CLI
- `docs/STATE_REPRESENTATION_SPEC.md` : specification de la representation d'etat V1
- `docs/MODEL_V1_SPEC.md` : specification du premier modele neuronal
- `docs/BENCHMARK_V1_SPEC.md` : specification du protocole benchmark V1
- `docs/DECISIONS_REGISTER.md` : registre central des decisions prises et points ouverts
- `docs/OPEN_DECISIONS_V1.md` : recommandations tranchees sur les derniers points ouverts V1
- `src/songo_model_stockfish/` : futur package Python du projet
- `config/` : futures configurations du nouveau pipeline
- `notebooks/` : futurs notebooks Colab
- `scripts/` : futurs scripts utilitaires

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
- `config/benchmark.colab_pro.yaml`

L'intention est la suivante:

- `train` et `evaluate` exploitent le GPU avec `device: cuda`, `mixed_precision`, `pin_memory`, `persistent_workers` et `prefetch_factor`
- `dataset_generation` reste surtout CPU-bound a cause de `minimax` et `mcts`, mais il supporte maintenant l'execution parallele par parties via `num_workers` et `max_pending_futures`
- `dataset_build` produit un dataset Colab dedie pour ne pas melanger les artefacts smoke locaux et les artefacts de production

Important:

- `dataset_generation` via `minimax` / `mcts` n'utilise pas le GPU comme levier principal, mais exploite mieux CPU + RAM en parallele
- l'optimisation GPU est aujourd'hui principalement effective sur `train`, `evaluate` et `benchmark` quand la cible est un checkpoint neuronal

## Workflow Colab

Le projet contient maintenant une couche Colab dediee pour mettre a jour le code sans toucher aux artefacts persistants:

- `notebooks/colab_setup_and_update.ipynb`
  Monte Drive, prepare le layout persistant, clone ou met a jour le repo GitHub et installe les dependances.

- `notebooks/colab_dataset_and_build.ipynb`
  Lance la generation des parties puis construit le dataset final.

- `notebooks/colab_train_evaluate_benchmark.ipynb`
  Lance l'entrainement, l'evaluation et le benchmark avec des `job_id` resumables.

Scripts utiles:

- `scripts/colab/init_drive_layout.sh`
- `scripts/colab/update_repo_from_github.sh`
- `scripts/colab/bootstrap_from_github.sh`
- `scripts/colab/snapshot_code_to_drive.sh`
- `scripts/colab/status_watch.sh`

Principe recommande:

- code de travail dans `/content/songo-model-stockfish-for-google-collab`
- artefacts persistants dans `/content/drive/MyDrive/songo-stockfish`
- mise a jour Git sur le worktree uniquement
- datasets, checkpoints, jobs, rapports et logs uniquement sur Drive

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
