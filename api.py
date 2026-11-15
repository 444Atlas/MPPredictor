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
    allow_origins=["*"], # Autorise toutes les interfaces web à se connecter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # On isole uniquement les stats brutes (sans les différentiels)
    df_derniere_stats = df_recent.set_index('_home_team')

    # 3. Chargement du modèle PyTorch
    input_dim = len(features_cols)
    model = MatchPredictor(input_dim)
    try:
        model.load_state_dict(torch.load("model_match_predictor.pth"))
        model.eval() # Mode inférence
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
    """Interface visuelle interactive pour tester le modèle directement depuis le navigateur."""
    return """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>World Cup 2026 AI Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col justify-between">
    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-5xl mx-auto px-4 py-4 flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center space-x-3">
                <span class="text-3xl">⚽</span>
                <div>
                    <h1 class="text-xl font-extrabold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
                        World Cup 2026 AI Predictor
                    </h1>
                    <p class="text-xs text-slate-400">Réseau de neurones PyTorch & Simulations Monte Carlo</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <a href="/docs" target="_blank" class="text-xs bg-slate-800 hover:bg-slate-700 text-blue-400 px-3 py-1.5 rounded-lg border border-slate-700 transition">
                    📖 Swagger Docs (/docs)
                </a>
                <a href="/redoc" target="_blank" class="text-xs bg-slate-800 hover:bg-slate-700 text-emerald-400 px-3 py-1.5 rounded-lg border border-slate-700 transition">
                    📑 ReDoc (/redoc)
                </a>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-3xl mx-auto px-4 py-8 flex-1 w-full">
        <div class="bg-slate-800/60 border border-slate-700/60 rounded-2xl p-6 shadow-xl backdrop-blur">
            <h2 class="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
                <span>🎯</span> Simulation d'un Match
            </h2>

            <form id="predict-form" class="space-y-5">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                            Équipe Domicile (Home)
                        </label>
                        <select id="team_home" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-blue-500 transition">
                            <option value="">Chargement des équipes...</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                            Équipe Extérieur (Away)
                        </label>
                        <select id="team_away" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-blue-500 transition">
                            <option value="">Chargement des équipes...</option>
                        </select>
                    </div>
                </div>

                <div class="flex flex-wrap gap-6 pt-2">
                    <label class="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input type="checkbox" id="is_neutral" checked class="w-4 h-4 rounded text-blue-600 bg-slate-900 border-slate-700 focus:ring-blue-500">
                        <span>Terrain Neutre (Coupe du Monde)</span>
                    </label>
                    <label class="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input type="checkbox" id="is_world_cup" checked class="w-4 h-4 rounded text-blue-600 bg-slate-900 border-slate-700 focus:ring-blue-500">
                        <span>Contexte Coupe du Monde</span>
                    </label>
                </div>

                <button type="submit" id="submit-btn" class="w-full bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 text-white font-bold py-3.5 px-6 rounded-xl shadow-lg transition transform active:scale-[0.99] flex items-center justify-center gap-2">
                    <span>⚡</span> <span>Calculer la prédiction</span>
                </button>
            </form>

            <!-- Result Card -->
            <div id="result-container" class="mt-8 hidden">
                <div class="border-t border-slate-700 pt-6">
                    <div class="text-center mb-6">
                        <span class="text-xs uppercase tracking-wider text-slate-400 font-bold" id="match-title">Match</span>
                        <div class="text-4xl font-black text-white mt-2" id="score-predit">0 - 0</div>
                        <p class="text-xs text-slate-400 mt-1">Score exact le plus probable (Monte Carlo Poisson)</p>
                    </div>

                    <!-- Probabilities Gauge -->
                    <div class="space-y-4 bg-slate-900/80 p-4 rounded-xl border border-slate-700/50">
                        <div>
                            <div class="flex justify-between text-sm font-semibold mb-1">
                                <span class="text-blue-400">Victoire <span id="home-name">Domicile</span></span>
                                <span id="proba-home" class="text-blue-400">50%</span>
                            </div>
                            <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                                <div id="bar-home" class="bg-blue-500 h-3 rounded-full transition-all duration-500" style="width: 50%"></div>
                            </div>
                        </div>

                        <div>
                            <div class="flex justify-between text-sm font-semibold mb-1">
                                <span class="text-slate-400">Nul ou Défaite Domicile</span>
                                <span id="proba-away" class="text-slate-400">50%</span>
                            </div>
                            <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                                <div id="bar-away" class="bg-emerald-500 h-3 rounded-full transition-all duration-500" style="width: 50%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Error container -->
            <div id="error-container" class="mt-6 hidden bg-red-900/30 border border-red-700/50 text-red-300 p-4 rounded-xl text-sm">
            </div>
        </div>
    </main>

    <footer class="border-t border-slate-800 py-4 text-center text-xs text-slate-500">
        World Cup 2026 Match Predictor • FastAPI & PyTorch
    </footer>

    <script>
        async function loadTeams() {
            try {
                const res = await fetch('/teams');
                const data = await res.json();
                const teams = data.teams || [];
                const homeSelect = document.getElementById('team_home');
                const awaySelect = document.getElementById('team_away');
                homeSelect.innerHTML = '';
                awaySelect.innerHTML = '';
                teams.forEach(team => {
                    const opt1 = document.createElement('option');
                    opt1.value = team;
                    opt1.textContent = team;
                    if (team === 'France') opt1.selected = true;
                    homeSelect.appendChild(opt1);

                    const opt2 = document.createElement('option');
                    opt2.value = team;
                    opt2.textContent = team;
                    if (team === 'Brazil') opt2.selected = true;
                    awaySelect.appendChild(opt2);
                });
            } catch (err) {
                console.error('Erreur chargement équipes:', err);
            }
        }

        document.getElementById('predict-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submit-btn');
            const resContainer = document.getElementById('result-container');
            const errContainer = document.getElementById('error-container');
            errContainer.classList.add('hidden');
            resContainer.classList.add('hidden');
            btn.disabled = true;
            btn.innerHTML = '<span>⏳</span> <span>Calcul en cours...</span>';

            const payload = {
                team_home: document.getElementById('team_home').value,
                team_away: document.getElementById('team_away').value,
                is_neutral: document.getElementById('is_neutral').checked ? 1.0 : 0.0,
                is_world_cup: document.getElementById('is_world_cup').checked ? 1.0 : 0.0
            };

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok) {
                    throw new Error(data.detail || 'Erreur lors de la prédiction');
                }

                document.getElementById('match-title').textContent = data.match;
                document.getElementById('score-predit').textContent = data.score_predit;
                document.getElementById('home-name').textContent = payload.team_home;
                document.getElementById('proba-home').textContent = data.proba_victoire_home + '%';
                document.getElementById('proba-away').textContent = data.proba_nul_ou_defaite + '%';
                document.getElementById('bar-home').style.width = data.proba_victoire_home + '%';
                document.getElementById('bar-away').style.width = data.proba_nul_ou_defaite + '%';
                resContainer.classList.remove('hidden');
            } catch (err) {
                errContainer.textContent = err.message;
                errContainer.classList.remove('hidden');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span>⚡</span> <span>Calculer la prédiction</span>';
            }
        });

        loadTeams();
    </script>
</body>
</html>
"""

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
            # On cherche l'équivalent 'home_' dans l'historique de l'équipe B (car on a stocké les stats sous l'angle 'home_team')
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
            feature_dict[col] = 0 # Valeur de secours

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
    # Extraction des notes offensives/défensives pour le ratio
    att_h = stats_h.get('home_form_scored', moyenne_globale_buts) / moyenne_globale_buts
    def_h = stats_h.get('home_form_conceded', moyenne_globale_buts) / moyenne_globale_buts
    att_a = stats_a.get('home_form_scored', moyenne_globale_buts) / moyenne_globale_buts # pris depuis les stats de A
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
