# Dataset Model Training Playbook

## 1. Objectif

Ce document explique, de facon operationnelle:

- comment le dataset est construit
- quels labels sont utilises pour l'entrainement
- comment le modele `policy + value` est construit
- quelles metriques optimiser en priorite
- comment rendre le modele plus fort sans perdre la stabilite

Les points ci-dessous sont alignes avec le code actuel (`data/jobs.py`, `training/model.py`, `training/jobs.py`, `evaluation/jobs.py`, `benchmark/model_agent.py`).

## 2. Construction du dataset

Le pipeline dataset se fait en deux grandes etapes:

1. `dataset-generate`
- produit les donnees source (`raw` + `sampled`) depuis benchmatch, self-play PUCT, clone/derive/merge
- ecrit un `dataset_generation_summary.json`

2. `dataset-build`
- lit les `sampled` et produit un dataset final entraineable
- ecrit `train.npz`, `validation.npz`, `test.npz`
- ecrit un `dataset_build_summary.json`
- met a jour `data/dataset_registry.json`

Le split final est configure en YAML (souvent 80/10/10).

## 3. Labels utilises pour l'entrainement

### 3.1 `policy_target`

Chaque sample doit contenir:

- `policy_target.best_move` (entier 1..7)
- `policy_target.distribution` (probas par coup legal)

Le pipeline reconstruit aussi un vecteur dense `policy_target_full` de taille 7 (normalise) pour la loss soft.

### 3.2 `value_target`

Le `value_target` final est dans `[-1, 1]` et est calcule comme un mix:

- `value_target_teacher` (valeur du teacher)
- `value_target_outcome` (issue de partie quand disponible)
- `value_target_mix_teacher_weight` / `value_target_mix_outcome_weight`

Formule logique:

- `value_target = teacher_mix * teacher_value + (1 - teacher_mix) * outcome_value`

### 3.3 Enrichissements tactiques

Si active (`include_tactical_analysis: true`), chaque sample inclut:

- `tactical_analysis.summary`
- `tactical_analysis.per_move`

Ces signaux sont utilises pour:

- les features d'entree
- une regularisation auxiliaire tactique pendant l'entrainement

### 3.4 Hard examples

Le build calcule aussi:

- `hard_example_score`
- `hard_example_weight`
- `hard_example_margin`, `hard_example_margin_hardness`, `hard_example_outcome_hardness`

Ces poids servent a sur-echantillonner les positions difficiles pendant le train.

### 3.5 Modes de build

- `teacher_label`: le build relabelise avec teacher (`minimax`/`mcts`)
- `source_prelabeled`: consomme des samples deja labels
- `auto`: choisit automatiquement le mode selon la source

## 4. Features d'entree du modele

Le modele consomme des features numeriques:

1. Features de base (`encode_raw_state`)
- plateau aplati
- joueur au trait
- scores sud/nord

2. Features tactiques (`encode_tactical_analysis`)
- resume tactique global
- details par coup (7 coups)

L'entree est adaptee a `input_dim` du modele via `adapt_feature_dim` (truncate/pad) pour compatibilite checkpoints.

## 5. Architecture du modele

Le modele principal est `PolicyValueMLP`:

- backbone MLP (liste `hidden_sizes`)
- options: `use_layer_norm`, `dropout`, `residual_connections`
- tete policy: 7 logits (un par coup)
- tete value: 1 scalaire avec `tanh` (borne `[-1, 1]`)

Par defaut technique:

- famille: `policy_value`
- backbone: `mlp`
- dimensions configurees dans `train.*.yaml`

## 6. Fonction de cout (train)

La loss combine 3 composantes:

1. Policy hard (`cross_entropy` sur meilleur coup)
2. Policy soft (distillation sur `policy_target_full`)
3. Value loss (`MSE` sur `value_target`)
4. Tactique auxiliaire (regularisation sur masques capture/safe/risky)

Forme utilisee:

- `loss_policy = (1-soft_w)*loss_policy_hard + soft_w*loss_policy_soft`
- `loss_total = loss_policy + value_w*loss_value + tactical_w*loss_tactical_aux`

Le modele choisit le meilleur checkpoint sur:

- `validation_policy_accuracy` (metrique de reference pour `best_metric`)

## 7. Evaluation et benchmark

### 7.1 Evaluation hors train

Le job `evaluation` suit en priorite:

- `policy_accuracy_top1`
- `policy_accuracy_top3`
- `value_mae`
- `loss_total`

### 7.2 Benchmark force de jeu

Le benchmark joue contre `minimax`/`mcts` et produit:

- `benchmark_score`
- `benchmark_score_weighted`
- `benchmark_elo_estimate`
- details first player / second player

Inference benchmark actuelle cote modele:

- profile `fort_plusplus` (seul profile supporte)
- top-k racine + top-k enfant
- negamax multi-ply + alpha-beta optionnel

## 8. Comment rendre le modele plus fort

### 8.1 Levier data (priorite 1)

- augmenter la couverture de positions utiles (ouvertures + milieu + finales)
- augmenter la qualite teacher (`insane` quand possible)
- bien regler `value_target_mix_teacher_weight`
- conserver `dedupe_sample_ids`
- monitorer `skipped_invalid`, `skipped_no_legal`, `skipped_terminal`

### 8.2 Levier entrainement (priorite 2)

- augmenter progressivement la capacite (`hidden_sizes`) selon VRAM/temps
- stabiliser avec scheduler cosine + clipping + early stopping
- garder oversampling hard examples quand le dataset est heterogene
- surveiller l'ecart train/validation pour eviter overfit

### 8.3 Levier inference/search (priorite 3)

Sans changer les poids:

- augmenter `model_search_depth`
- ajuster `model_search_top_k` / `model_search_top_k_child`
- garder `model_search_alpha_beta: true`

Effet attendu:

- plus fort strategiquement, surtout contre minimax/mcts
- plus lent en inference (cout normal de recherche plus profonde)

## 9. Metriques a optimiser (ordre recommande)

1. `validation_policy_accuracy` (metrique de selection checkpoint)
2. `policy_accuracy_top1` (evaluation)
3. `benchmark_score_weighted` et `benchmark_elo_estimate`
4. `value_mae` (stabilite de la tete value)
5. `loss_total` (train/eval, en lecture secondaire)

Metriques data a suivre en parallele:

- `labeled_samples`, `duplicate_samples_removed`
- `skipped_invalid_samples`, `skipped_no_legal_samples`, `skipped_terminal_samples`

## 10. Artefacts a lire apres chaque run

- `jobs/<train_job>/training_summary.json`
- `jobs/<eval_job>/evaluation_summary.json`
- `jobs/<dataset_build_job>/dataset_build/dataset_build_summary.json`
- `reports/benchmarks/<...>.json`
- `models/final/<model_id>.model_card.json`

## 11. Point de gouvernance important

Les actifs finaux prioritaires du projet sont:

- datasets finaux (`train.npz`, `validation.npz`, `test.npz`)
- checkpoints/modeles finaux (`models/final/*.pt`)

Les artefacts intermediaires doivent rester purgables via retention TTL, mais pas ces actifs finaux.

## 12. Historique des datasets utilises

Le pipeline maintient aussi un historique d'usage des datasets en train.

Commande utile:

- `python -m songo_model_stockfish.cli.main dataset-usage --config <config_train.yaml> --top 20`

Ce rapport aide a:

- identifier les datasets les plus utilises
- trouver les candidats purge (anciens + peu utilises)
- garder un noyau de datasets vraiment utiles avant fusion finale
