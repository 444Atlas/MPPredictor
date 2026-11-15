from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ==========================================
# 1. DÉFINITION DE L'ARCHITECTURE DU MODÈLE
# ==========================================
class MatchPredictor(nn.Module):
    def __init__(self, input_dim):
        super(MatchPredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.sigmoid(self.output_layer(x))
        return x

# ==========================================
# 2. INITIALISATION DE L'APPLICATION
# ==========================================
app = FastAPI(
    title="World Cup 2026 AI Predictor",
    description="API de prédiction de matchs basée sur un réseau de neurones et Monte Carlo.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemin du template HTML
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

# Variables globales stockant les objets en mémoire
model = None
scaler = None
imputer = None
df_derniere_stats = None
features_cols = None
moyenne_globale_buts = 1.35

# Modèle de données attendu pour la requête POST
class MatchRequest(BaseModel):
    team_home: str
    team_away: str
    is_neutral: float = 1.0
    is_world_cup: float = 1.0

# ==========================================
# 3. CHARGEMENT AU DÉMARRAGE DU SERVEUR
# ==========================================
@app.on_event("startup")
def load_assets():
    global model, scaler, imputer, df_derniere_stats, features_cols, moyenne_globale_buts
    print("Démarrage du moteur d'IA...")

    # 1. Chargement et préparation des données pour le Scaler
    df = pd.read_csv("data/processed/features_complet_16_26_clean.csv")
    df['_date'] = pd.to_datetime(df['_date'])
    
    colonnes_a_exclure = ['_date', '_home_team', '_away_team', '_tournament', 'home_goals', 'away_goals', 'target_win']
    if 'target_win' not in df.columns:
        df['target_win'] = (df['home_goals'] > df['away_goals']).astype(int)
        
    features_cols = [c for c in df.columns if c not in colonnes_a_exclure]
    
    # Entraînement du Scaler et Imputer sur les données historiques
    df_train = df[df['_date'] < '2026-01-01'].copy()
    X_train = df_train[features_cols].values
    
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    
    scaler = StandardScaler()
    scaler.fit(X_train_imputed)
    
    moyenne_globale_buts = (df_train['home_goals'].mean() + df_train['away_goals'].mean()) / 2

    # 2. Extraction du profil actuel des équipes (Dernier match joué)
    df_recent = df.sort_values('_date').groupby('_home_team').last().reset_index()
    df_derniere_stats = df_recent.set_index('_home_team')

    # 3. Chargement du modèle PyTorch
    input_dim = len(features_cols)
    model = MatchPredictor(input_dim)
    try:
        model.load_state_dict(torch.load("model_match_predictor.pth"))
        model.eval()
        print("Modèle PyTorch chargé avec succès.")
    except Exception as e:
        print(f"ERREUR : Impossible de charger model_match_predictor.pth : {e}")

# ==========================================
# 4. FONCTION MONTE CARLO (POISSON)
# ==========================================
def simuler_score(p_win, att_home, def_away, att_away, def_home):
    lambda_home = moyenne_globale_buts * att_home * def_away
    lambda_away = moyenne_globale_buts * att_away * def_home
    
    ratio_force = p_win / (1.0 - p_win + 1e-5)
    facteur_ajustement = np.sqrt(ratio_force / (lambda_home / (lambda_away + 1e-5) + 1e-5))
    
    lh_ajuste = max(0.1, min(5.0, lambda_home * facteur_ajustement))
    la_ajuste = max(0.1, min(5.0, lambda_away / (facteur_ajustement + 1e-5)))

    buts_home = np.random.poisson(lh_ajuste, 10000)
    buts_away = np.random.poisson(la_ajuste, 10000)
    
    scores = list(zip(buts_home, buts_away))
    return max(set(scores), key=scores.count)

# ==========================================
# 5. ENDPOINTS
# ==========================================

@app.get("/", response_class=HTMLResponse)
def home():
    """Interface éditoriale pour tester le modèle directement depuis le navigateur."""
    if TEMPLATE_PATH.exists():
        return TEMPLATE_PATH.read_text(encoding="utf-8")
    return "<h1>MPPredictor API active. Rendez-vous sur /docs</h1>"

@app.get("/teams")
def get_teams():
    """Retourne la liste alphabétique des équipes disponibles dans la base de données."""
    if df_derniere_stats is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore initialisé.")
    return {"teams": sorted(list(df_derniere_stats.index))}

@app.get("/health")
def health_check():
    """Vérification de l'état de l'API."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "total_teams": len(df_derniere_stats.index) if df_derniere_stats is not None else 0
    }

@app.post("/predict")
def predict_match(request: MatchRequest):
    t_home = request.team_home
    t_away = request.team_away
    
    if df_derniere_stats is None or model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé.")

    if t_home not in df_derniere_stats.index or t_away not in df_derniere_stats.index:
        raise HTTPException(status_code=404, detail="Une ou les deux équipes sont introuvables dans la base de données historique.")

    # 1. Récupération des stats des deux équipes
    stats_h = df_derniere_stats.loc[t_home]
    stats_a = df_derniere_stats.loc[t_away]

    # 2. Construction du vecteur de caractéristiques à la volée
    feature_dict = {}
    for col in features_cols:
        if col.startswith('home_'):
            feature_dict[col] = stats_h.get(col, 0)
        elif col.startswith('away_'):
            equiv_col = col.replace('away_', 'home_')
            feature_dict[col] = stats_a.get(equiv_col, 0)
        elif col == 'elo_diff':
            feature_dict[col] = stats_h.get('home_elo', 1500) - stats_a.get('home_elo', 1500)
        elif col == 'overall_diff':
            feature_dict[col] = stats_h.get('home_avg_overall', 70) - stats_a.get('home_avg_overall', 70)
        elif col == 'attack_diff':
            feature_dict[col] = stats_h.get('home_avg_attack', 70) - stats_a.get('home_avg_attack', 70)
        elif col == 'defense_diff':
            feature_dict[col] = stats_h.get('home_avg_defense', 70) - stats_a.get('home_avg_defense', 70)
        elif col == 'is_neutral':
            feature_dict[col] = request.is_neutral
        elif col == 'is_world_cup':
            feature_dict[col] = request.is_world_cup
        else:
            feature_dict[col] = 0

    # Conversion en tableau 2D
    X_input = np.array([list(feature_dict.values())])
    
    # 3. Traitement mathématique (Imputation -> Scaling -> PyTorch)
    X_imputed = imputer.transform(X_input)
    X_scaled = scaler.transform(X_imputed)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # 4. Inférence PyTorch
    with torch.no_grad():
        proba_victoire_home = model(X_tensor).item()

    # 5. Calcul du score Monte Carlo
    att_h = stats_h.get('home_form_scored', moyenne_globale_buts) / moyenne_globale_buts
    def_h = stats_h.get('home_form_conceded', moyenne_globale_buts) / moyenne_globale_buts
    att_a = stats_a.get('home_form_scored', moyenne_globale_buts) / moyenne_globale_buts
    def_a = stats_a.get('home_form_conceded', moyenne_globale_buts) / moyenne_globale_buts

    score_h, score_a = simuler_score(proba_victoire_home, att_h, def_a, att_a, def_h)

    # 6. Réponse de l'API
    return {
        "match": f"{t_home} vs {t_away}",
        "proba_victoire_home": round(proba_victoire_home * 100, 2),
        "proba_nul_ou_defaite": round((1 - proba_victoire_home) * 100, 2),
        "score_predit": f"{int(score_h)} - {int(score_a)}",
        "home_goals": int(score_h),
        "away_goals": int(score_a)
    }
