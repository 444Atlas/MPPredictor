# MPPredictor - World Cup 2026 AI Predictor

API de prédiction de matchs de football (Coupe du Monde 2026) basée sur un réseau de neurones PyTorch et des simulations de Monte Carlo (Loi de Poisson).

## Fonctionnalités

- **Modèle Neuronal (PyTorch)** : Réseau feedforward pour estimer la probabilité de victoire de l'équipe à domicile.
- **Simulation Monte Carlo** : Modèle de Poisson ajusté par le ratio de force issu du réseau de neurones pour prédire les scores exacts.
- **API FastAPI** : Endpoint interactif `/predict` avec documentation Swagger intégrée (`/docs`).

## Installation

### Avec Conda :
```bash
conda activate mppredictor
pip install -r requirements.txt
```

### Avec un environnement virtuel standard (venv) :
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Linux/macOS
pip install -r requirements.txt
```

## Lancement de l'API

Pour démarrer le serveur FastAPI en local :
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Une fois lancé, accédez à :
- **Documentation interactive Swagger** : [http://localhost:8000/docs](http://localhost:8000/docs)
- **Documentation ReDoc** : [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Exemple d'utilisation

Effectuer une requête POST sur `/predict` :

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "team_home": "France",
       "team_away": "Brazil",
       "is_neutral": 1.0,
       "is_world_cup": 1.0
     }'
```

Exemple de réponse :
```json
{
  "match": "France vs Brazil",
  "proba_victoire_home": 54.21,
  "proba_nul_ou_defaite": 45.79,
  "score_predit": "2 - 1",
  "home_goals": 2,
  "away_goals": 1
}
```

## Structure du projet

- `api.py` : Serveur FastAPI et pipeline d'inférence.
- `model_match_predictor.pth` : Poids pré-entraînés du modèle PyTorch.
- `modele.ipynb` : Notebook d'entraînement et d'analyse du modèle.
- `data/processed/` : Données nettoyées et prêtes pour l'inférence.
- `data/raw/` : Données sources brutes.
