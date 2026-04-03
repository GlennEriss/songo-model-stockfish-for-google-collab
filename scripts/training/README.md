Scripts utilitaires pour piloter les versions d'entrainement.

Fichiers principaux:

- `prepare_next_version.py`
  Genere automatiquement les configs `train`, `evaluation` et `benchmark` de la prochaine version (`vN+1`) a partir des configs Colab Pro de base.

Usage typique:

```bash
cd /content/songo-model-stockfish-for-google-collab
source .venv/bin/activate
python scripts/training/prepare_next_version.py
```

Le script:

- detecte le prochain `model_id`
- ecrit des configs versionnees dans `config/generated/`
- affiche les `job_id` recommandes
- affiche les commandes train / evaluate / benchmark a lancer

Par defaut, le training versionne continue d'utiliser le meilleur modele promu courant.
