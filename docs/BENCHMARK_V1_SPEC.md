# Benchmark V1 Spec

## 1. Objectif

Definir le protocole officiel de benchmark pour la V1.

## 2. Role du benchmark

Le benchmark V1 sert a:

- mesurer la force du moteur ou du modele
- comparer des variantes internes
- suivre les progres dans le temps
- produire des donnees exploitables

## 3. Cibles de benchmark V1

Le benchmark V1 doit au minimum couvrir:

- `minimax:medium`
- `minimax:hard`
- `mcts:medium`
- `mcts:hard`

## 4. Matchups minimaux

Les matchups minimaux recommandes sont:

- `engine_v1` vs `minimax:medium`
- `engine_v1` vs `minimax:hard`
- `engine_v1` vs `mcts:medium`
- `engine_v1` vs `mcts:hard`

Quand le modele neuronal V1 existera:

- `model_v1` vs `minimax:medium`
- `model_v1` vs `mcts:medium`

## 5. Taille des campagnes

### Smoke test

- 20 parties par matchup

### Benchmark standard

- 50 parties par matchup

### Benchmark plus fiable

- 100 parties par matchup

## 6. Equilibrage

Le benchmark doit alterner:

- premier joueur
- second joueur

Pourquoi:

- eviter les biais de couleur / ordre

## 7. Logs obligatoires

Le benchmark doit produire:

- logs console
- `events.jsonl`
- `metrics.jsonl`
- `benchmark_summary.json`
- details de parties si active

## 8. Metriques minimales

Chaque benchmark V1 doit mesurer:

- victoires
- defaites
- nulles si applicable
- winrate
- temps moyen par coup
- temps total de partie

Quand disponible:

- profondeur moyenne
- nombre moyen de noeuds explores

## 9. Resume standard

Le resume standard doit contenir:

- identite de l'agent teste
- identite de l'adversaire
- nombre de parties
- winrate
- temps moyen
- config de benchmark

## 10. Reprise

Le benchmark V1 doit etre resumable.

Il doit pouvoir reprendre:

- au dernier matchup non termine
- a la derniere partie non terminee

## 11. Criteres de validation V1

Le benchmark V1 est valide si:

- il execute tous les matchups prevus
- il alterne bien l'ordre de jeu
- il produit les logs et le summary
- il peut reprendre sans recommencer tout

## 12. Conclusion

Le benchmark V1 doit etre:

- simple
- traçable
- resumable
- assez stable pour servir de reference officielle
