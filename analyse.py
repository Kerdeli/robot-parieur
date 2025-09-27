# analyse.py
import os
import math
import warnings
import logging
import argparse
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
from tqdm import tqdm
from colorama import init, Fore, Style
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize

# --- CONFIGURATION ---
init(autoreset=True)
warnings.filterwarnings('ignore')
EWMA_SPAN = 10
CONVICTION_THRESHOLD = 0.55
MAX_KELLY = 0.10  # fraction maximale suggérée
MODEL_PATH = "model_be.joblib"
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- UTILITAIRES ---
def validate_df_columns(df, required):
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Colonnes manquantes: {miss}")

def save_model(model, features, path=MODEL_PATH):
    try:
        joblib.dump({'model': model, 'features': features}, path)
        logging.info(f"Modèle sauvegardé: {path}")
    except Exception as e:
        logging.warning(f"Échec sauvegarde modèle: {e}")

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            data = joblib.load(path)
            return data.get('model'), data.get('features')
        except Exception as e:
            logging.warning(f"Échec chargement modèle: {e}")
    return None, None

# --- DATA ---
def charger_database(chemin_db):
    logging.info("Démarrage - chargement base...")
    if not os.path.exists(chemin_db):
        raise FileNotFoundError(f"'{chemin_db}' introuvable.")
    df = pd.read_csv(chemin_db)
    logging.info(f"Base chargée : {len(df)} lignes.")
    return df

# --- CERVEAU 1 : EWMA + XGBoost ---
def create_ewma_features(df):
    validate_df_columns(df, ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG'])
    df = df.copy()
    df['Date'] = pd.to_datetime(df.get('Date'), dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')
    df['Result_coded'] = df['FTR'].map({'A': 0, 'D': 1, 'H': 2})
    df['HomeGoals'] = df['FTHG']; df['AwayGoals'] = df['FTAG']
    df['HomeConceded'] = df['FTAG']; df['AwayConceded'] = df['FTHG']
    df['HomePoints'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0}); df['AwayPoints'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    featured_df = df.copy()
    for col in tqdm(['Goals', 'Conceded', 'Points'], desc="Calcul de la forme"):
        home_col = 'Home' + col if col != 'Points' else 'HomePoints'
        away_col = 'Away' + col if col != 'Points' else 'AwayPoints'
        if home_col in featured_df.columns:
            featured_df[f'H_EWMA_{col}'] = featured_df.groupby('HomeTeam')[home_col].transform(lambda x: x.ewm(span=EWMA_SPAN, adjust=False).mean().shift(1))
        else:
            featured_df[f'H_EWMA_{col}'] = np.nan
        if away_col in featured_df.columns:
            featured_df[f'A_EWMA_{col}'] = featured_df.groupby('AwayTeam')[away_col].transform(lambda x: x.ewm(span=EWMA_SPAN, adjust=False).mean().shift(1))
        else:
            featured_df[f'A_EWMA_{col}'] = np.nan
    return featured_df

def train_model_be(df, search_params=True):
    logging.info("Entraînement modèle principal...")
    ewma_cols = [c for c in df.columns if c.startswith(('H_EWMA_', 'A_EWMA_'))]
    training_df = df.dropna(subset=ewma_cols + ['Result_coded'])
    if training_df.empty:
        raise ValueError("Données d'entraînement insuffisantes après features EWMA.")
    features = ewma_cols
    X = training_df[features]; y = training_df['Result_coded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    base = xgb.XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, random_state=RANDOM_STATE)
    model = None
    if search_params:
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        try:
            rs = RandomizedSearchCV(base, param_dist, n_iter=8, cv=3, scoring='neg_log_loss', random_state=RANDOM_STATE, n_jobs=-1)
            rs.fit(X_train, y_train)
            best = rs.best_estimator_
            logging.info(f"Recherche hyperparam OK, meilleur params: {rs.best_params_}")
            try:
                clf = CalibratedClassifierCV(best, cv=3, method='isotonic')
                clf.fit(X_train, y_train)
                model = clf
            except Exception:
                best.fit(X_train, y_train)
                model = best
        except Exception as e:
            logging.warning(f"Recherche hyperparam failed: {e}")
    if model is None:
        try:
            clf = CalibratedClassifierCV(base, cv=3, method='isotonic')
            clf.fit(X_train, y_train)
            model = clf
        except Exception:
            base.fit(X_train, y_train)
            model = base
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    save_model(model, features)
    logging.info(f"Acc (holdout): {acc*100:.2f}%")
    return model, features, acc

def evaluate_model_be(model, features, df):
    ewma_cols = features
    df_clean = df.dropna(subset=ewma_cols + ['Result_coded'])
    if df_clean.empty:
        return {}
    X = df_clean[ewma_cols]; y = df_clean['Result_coded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    y_bin = label_binarize(y_test, classes=[0,1,2])
    brier_multi = np.mean((y_bin - probs)**2)
    logging.info(f"Éval — Acc: {acc*100:.2f}% | LogLoss: {ll:.4f} | BrierMulti: {brier_multi:.4f}")
    return {'accuracy': acc, 'log_loss': ll, 'brier_multi': brier_multi}

def predict_be(model, features, df, home, away):
    if 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
        return None
    teams_home = set(df['HomeTeam'].unique())
    teams_away = set(df['AwayTeam'].unique())
    # league means for fallback
    league_means = {}
    for f in features:
        league_means[f] = df[f].dropna().mean() if f in df.columns else 0.0
    live_features = {}
    for feature in features:
        if feature.startswith('H_EWMA_'):
            col_root = feature.replace('H_EWMA_', '')
            col_name = 'Home' + col_root if col_root != 'Points' else 'HomePoints'
            if home in teams_home and col_name in df.columns:
                team_data = df[df['HomeTeam'] == home][col_name].dropna()
            else:
                team_data = pd.Series(dtype=float)
        elif feature.startswith('A_EWMA_'):
            col_root = feature.replace('A_EWMA_', '')
            col_name = 'Away' + col_root if col_root != 'Points' else 'AwayPoints'
            if away in teams_away and col_name in df.columns:
                team_data = df[df['AwayTeam'] == away][col_name].dropna()
            else:
                team_data = pd.Series(dtype=float)
        else:
            team_data = pd.Series(dtype=float)
        if not team_data.empty:
            val = team_data.ewm(span=EWMA_SPAN, adjust=False).mean().iloc[-1]
        else:
            val = league_means.get(feature, 0.0)
        live_features[feature] = val
    X_live = pd.DataFrame([live_features], columns=features)
    proba = model.predict_proba(X_live)[0]
    return proba

# --- CERVEAU 2 : POISSON (SCORES & BTTS) ---
def preparer_modele_poisson(df):
    validate_df_columns(df, ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
    df_p = df.rename(columns={'HomeTeam': 'Domicile', 'AwayTeam': 'Exterieur', 'FTHG': 'Buts_Domicile', 'FTAG': 'Buts_Exterieur'})
    avg_d = df_p['Buts_Domicile'].mean(); avg_e = df_p['Buts_Exterieur'].mean()
    stats_d = df_p.groupby('Domicile').agg(FA_D=('Buts_Domicile', 'mean'), FD_D=('Buts_Exterieur', 'mean'))
    stats_e = df_p.groupby('Exterieur').agg(FA_E=('Buts_Exterieur', 'mean'), FD_E=('Buts_Domicile', 'mean'))
    stats_eq = pd.concat([stats_d, stats_e], axis=1).fillna(1)
    stats_eq['FA_Domicile'] = stats_eq['FA_D'] / (avg_d if avg_d else 1)
    stats_eq['FD_Domicile'] = stats_eq['FD_D'] / (avg_e if avg_e else 1)
    stats_eq['FA_Exterieur'] = stats_eq['FA_E'] / (avg_e if avg_e else 1)
    stats_eq['FD_Exterieur'] = stats_eq['FD_E'] / (avg_d if avg_d else 1)
    return stats_eq, avg_d, avg_e

def predire_poisson(stats, dom, ext, avg_d, avg_e, max_goals=5):
    if stats is None or dom not in stats.index or ext not in stats.index:
        return None
    buts_d = stats.at[dom, 'FA_Domicile'] * stats.at[ext, 'FD_Exterieur'] * (avg_d if avg_d else 1)
    buts_e = stats.at[ext, 'FA_Exterieur'] * stats.at[dom, 'FD_Domicile'] * (avg_e if avg_e else 1)
    prob = []
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob.append({'Score': f"{i}-{j}", 'Prob': poisson.pmf(i, buts_d) * poisson.pmf(j, buts_e)})
    return pd.DataFrame(prob)

# --- JUGE / ANALYSE VALEUR ---
def _implied_prob(odd):
    return 1.0 / odd if odd and odd > 0 else 0.0

def _remove_vig_and_normalize(odds_dict):
    implied = {k: _implied_prob(v) for k, v in odds_dict.items()}
    total = sum(implied.values())
    if total <= 0:
        n = len(implied) or 1
        return {k: 1.0 / n for k in implied}
    return {k: implied[k] / total for k in implied}

def _kelly_fraction(p, odd):
    if odd <= 1 or p <= 0:
        return 0.0
    b = odd - 1.0
    q = 1.0 - p
    k = (b * p - q) / b
    if k <= 0:
        return 0.0
    return min(k, MAX_KELLY)

def juge_final(prob_be, df_scores_or_dict, home, away):
    if isinstance(df_scores_or_dict, dict):
        prob_p_vnd = {k: float(df_scores_or_dict.get(k, 0.0)) for k in ('H', 'D', 'A')}
        total = sum(prob_p_vnd.values())
        prob_p_vnd = {k: (v / total) for k, v in prob_p_vnd.items()} if total > 0 else {k: 0.0 for k in prob_p_vnd}
    else:
        df_scores = df_scores_or_dict
        if df_scores is None or df_scores.empty:
            prob_p_vnd = {'H': 0.0, 'D': 0.0, 'A': 0.0}
        else:
            H = df_scores[df_scores['Score'].apply(lambda x: int(x.split('-')[0]) > int(x.split('-')[1]))]['Prob'].sum()
            D = df_scores[df_scores['Score'].apply(lambda x: int(x.split('-')[0]) == int(x.split('-')[1]))]['Prob'].sum()
            A = max(0.0, 1.0 - H - D)
            prob_p_vnd = {'H': H, 'D': D, 'A': A}
    winner_be_idx = int(np.argmax(prob_be))
    winner_be = {2: 'H', 1: 'D', 0: 'A'}.get(winner_be_idx, 'D')
    conf_be = float(prob_be[winner_be_idx])
    winner_p = max(prob_p_vnd, key=prob_p_vnd.get)
    if winner_be == winner_p and conf_be > CONVICTION_THRESHOLD:
        w_name = home if winner_be == 'H' else away if winner_be == 'A' else "Match nul"
        return f"{Fore.GREEN}✅ Accord : {w_name}.", "HAUTE_CONFIANCE", winner_be
    elif conf_be < 0.45:
        return f"{Fore.YELLOW}⚠️ Faible confiance du modèle principal.", "INCERTITUDE", None
    else:
        w_be_name = home if winner_be == 'H' else away if winner_be == 'A' else "Match nul"
        w_p_name = home if winner_p == 'H' else away if winner_p == 'A' else "Match nul"
        return f"{Fore.RED}🚨 Conflit : modèle fav {w_be_name} vs historique fav {w_p_name}.", "CONFLIT", None

def analyze_value_and_traps(prob_be, prob_p_vnd, odds_vnd):
    model_map = {'Domicile': float(prob_be[2]), 'Nul': float(prob_be[1]), 'Exterieur': float(prob_be[0])}
    market_norm = _remove_vig_and_normalize(odds_vnd)
    singles = []
    for key in ('Domicile', 'Nul', 'Exterieur'):
        mprob = model_map.get(key, 0.0)
        market_prob = market_norm.get(key, 0.0)
        market_odd = odds_vnd.get(key, float('nan'))
        fair_odd_model = (1.0 / mprob) if mprob > 0 else float('inf')
        ev = (mprob * market_odd - 1.0) if (market_odd and mprob > 0) else -1.0
        ev_percent = ev * 100
        diff = mprob - market_prob
        trap_strength = 0.0
        trap_flag = False
        if fair_odd_model not in (0, float('inf')) and market_odd and market_odd > 0:
            trap_strength = market_odd / fair_odd_model
            if (trap_strength >= 1.25 and mprob < 0.25) or (market_prob < mprob * 0.5 and diff > 0.12):
                trap_flag = True
        kelly = _kelly_fraction(mprob, market_odd) if not math.isnan(market_odd) else 0.0
        singles.append({
            'outcome': key,
            'model_prob': mprob,
            'market_prob': market_prob,
            'market_odd': market_odd,
            'fair_odd_model': fair_odd_model,
            'ev_pct': ev_percent,
            'diff_pct': diff * 100,
            'trap': trap_flag,
            'trap_strength': trap_strength,
            'kelly_frac': kelly
        })
    dc_home = model_map['Domicile'] + model_map['Nul']
    dc_away = model_map['Exterieur'] + model_map['Nul']
    double_chance = {'Home_or_Draw_prob': dc_home, 'Away_or_Draw_prob': dc_away}
    best_single = max(singles, key=lambda x: x['ev_pct'])
    best_reco = {'type': 'single', 'detail': best_single, 'reason': f"Valeur {best_single['ev_pct']:.2f}%"}
    if double_chance['Home_or_Draw_prob'] > best_single['model_prob'] + 0.05:
        best_reco = {'type': 'double', 'detail': ('Home_or_Draw', double_chance['Home_or_Draw_prob']), 'reason': 'Sécurité: domicile ou nul'}
    if double_chance['Away_or_Draw_prob'] > best_single['model_prob'] + 0.05 and double_chance['Away_or_Draw_prob'] > double_chance['Home_or_Draw_prob']:
        best_reco = {'type': 'double', 'detail': ('Away_or_Draw', double_chance['Away_or_Draw_prob']), 'reason': 'Sécurité: extérieur ou nul'}
    if best_single.get('trap') and best_reco['type'] == 'single':
        if max(double_chance.values()) > 0.40:
            pick = 'Home_or_Draw' if double_chance['Home_or_Draw_prob'] > double_chance['Away_or_Draw_prob'] else 'Away_or_Draw'
            best_reco = {'type': 'double', 'detail': (pick, max(double_chance.values())), 'reason': 'Single marqué comme piège'}
    return {'singles': singles, 'double_chance': double_chance, 'best_reco': best_reco}

# --- AFFICHAGE / CLI (léger) ---
def _friendly_label(outcome, home, away):
    if outcome == 'Domicile':
        return f"{home} (à domicile)"
    if outcome == 'Exterieur':
        return f"{away} (à l'extérieur)"
    return "Match nul"

def display_gauge(label, probability):
    p = max(0.0, min(1.0, probability))
    bar = '█' * int(25 * p) + '─' * (25 - int(25 * p))
    color = Fore.GREEN if p > 0.5 else Fore.YELLOW if p > 0.4 else Fore.RED
    return f"  ║ {label:<18} {color}│{bar}│ {p*100:6.2f}% {Style.RESET_ALL}║"

# --- UTILITAIRE POUR TESTS / API SIMPLE ---
def analyse_function(input_data, train_if_needed=True):
    """
    Wrapper utilitaire pour tests / usages rapides.
    - mode déterministe si input_data dict contient 'force_simple': True.
    - accepte DataFrame, chemin CSV (str) ou dict{'df': list/dict, 'home', 'away'}.
    Retourne dict {'home','away','prob_be':[p_A,p_D,p_H],'recommended'} ou {'error':...}
    """
    try:
        # mode déterministe (tests) sans ML
        if isinstance(input_data, dict) and input_data.get('force_simple') is True:
            df = pd.DataFrame(input_data.get('df', [])) if 'df' in input_data else None
            home = input_data.get('home')
            away = input_data.get('away')
            if df is not None and home is None and away is None and not df.empty:
                last = df.iloc[-1]
                home = last.get('HomeTeam'); away = last.get('AwayTeam'); ftr = last.get('FTR')
            else:
                ftr = input_data.get('last_result', None)
            if ftr == 'H':
                proba = [0.05, 0.10, 0.85]
                recommended = 'Domicile'
            elif ftr == 'A':
                proba = [0.85, 0.10, 0.05]
                recommended = 'Exterieur'
            else:
                proba = [0.10, 0.80, 0.10]
                recommended = 'Nul'
            return {'home': home, 'away': away, 'prob_be': proba, 'recommended': recommended}

        # normal path: build df
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif isinstance(input_data, str) and os.path.exists(input_data):
            df = pd.read_csv(input_data)
        elif isinstance(input_data, dict) and 'df' in input_data:
            df = pd.DataFrame(input_data['df'])
        else:
            return {'error': 'input not understood (expect DataFrame / csv path / dict with key \"df\")'}

        featured = create_ewma_features(df.copy())

        model, features = load_model()
        if model is None and train_if_needed:
            try:
                model, features, _ = train_model_be(featured, search_params=False)
            except Exception as e:
                logging.warning(f"Entraînement fallback failed: {e}")
                model = None

        if model is None or not features:
            return {'error': 'no model available'}

        if isinstance(input_data, dict) and 'home' in input_data and 'away' in input_data:
            home = input_data['home']; away = input_data['away']
        else:
            if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns and len(df) > 0:
                last = df.iloc[-1]
                home = last['HomeTeam']; away = last['AwayTeam']
            else:
                return {'error': 'teams not found in input data'}

        proba = predict_be(model, features, featured, home, away)
        if proba is None:
            return {'error': 'prediction failed (teams unknown or insufficient data)'}
        mapping = {2: 'Domicile', 1: 'Nul', 0: 'Exterieur'}
        recommended = mapping[int(np.argmax(proba))]
        return {'home': home, 'away': away, 'prob_be': [float(x) for x in proba], 'recommended': recommended}
    except Exception as e:
        logging.exception("analyse_function failed")
        return {'error': str(e)}

# --- BOUCLE INTERACTIVE LÉGÈRE ---
def main_loop(full_data, model, features, stats_poisson, avg_dom, avg_ext):
    equipes_connues = sorted(full_data['HomeTeam'].unique()) if 'HomeTeam' in full_data.columns else []
    print(Fore.YELLOW + f"Équipes connues ({len(equipes_connues)}): " + ", ".join(equipes_connues[:20]) + ("..." if len(equipes_connues)>20 else ""))
    while True:
        home = input("Équipe à DOMICILE (ou 'quitter') : ").strip()
        if home.lower() == 'quitter': break
        away = input("Équipe à L'EXTÉRIEUR (ou 'quitter') : ").strip()
        if away.lower() == 'quitter': break
        prob_be = predict_be(model, features, full_data, home, away)
        df_scores = predire_poisson(stats_poisson, home, away, avg_dom, avg_ext)
        if prob_be is None or df_scores is None:
            print(Fore.RED + "Équipe inconnue ou données manquantes.")
            continue
        print(display_gauge(home, prob_be[2])); print(display_gauge("Match nul", prob_be[1])); print(display_gauge(away, prob_be[0]))
        try:
            odds_vnd = {
                'Domicile': float(input(f" Cote {home} : ").replace(',', '.')),
                'Nul': float(input(" Cote Nul : ").replace(',', '.')),
                'Exterieur': float(input(f" Cote {away} : ").replace(',', '.'))
            }
        except Exception:
            print(Fore.RED + "Cotes invalides.")
            continue
        analysis = analyze_value_and_traps(prob_be, {
            'H': df_scores[df_scores['Score'].apply(lambda x: int(x.split('-')[0]) > int(x.split('-')[1]))]['Prob'].sum(),
            'D': df_scores[df_scores['Score'].apply(lambda x: int(x.split('-')[0]) == int(x.split('-')[1]))]['Prob'].sum(),
            'A': 0.0
        }, odds_vnd)
        best = analysis['best_reco']
        if best['type'] == 'single' and best['detail']['ev_pct'] > 0 and not best['detail']['trap']:
            print(Fore.GREEN + f"Recommendation SINGLE: {_friendly_label(best['detail']['outcome'], home, away)} | EV {best['detail']['ev_pct']:.2f}%")
        else:
            name, prob = best['detail'] if isinstance(best['detail'], tuple) else (best['detail'], None)
            pretty = "Domicile / Nul" if name == 'Home_or_Draw' else "Extérieur / Nul"
            print(Fore.YELLOW + f"Recommendation SECURITE: {pretty} | Prob {prob*100:.2f}%")
    print("Fin.")

# --- ENTRYPOINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot parieur - analyse")
    parser.add_argument("--db", default="database_consolidated.csv", help="Chemin CSV")
    parser.add_argument("--no-train", action="store_true", help="Ne pas réentraîner si modèle existant")
    args = parser.parse_args()

    try:
        print(Fore.CYAN + "Démarrage...")
        df = charger_database(args.db)
        featured = create_ewma_features(df.copy())
        model, features = load_model()
        if model is None and not args.no_train:
            model, features, acc = train_model_be(featured, search_params=True)
        else:
            logging.info("Modèle chargé depuis disque." if model is not None else "Aucun modèle chargé, mode lecture seule.")
        if model is None:
            logging.warning("Aucun modèle disponible après tentative. Le programme continuera en mode lecture seule.")
        evals = evaluate_model_be(model, features, featured) if model is not None else {}
        stats_poisson, avg_dom, avg_ext = preparer_modele_poisson(df.copy())
        logging.info("Prêt.")
        if model is not None and features:
            main_loop(df, model, features, stats_poisson, avg_dom, avg_ext)
    except Exception as e:
        logging.exception(f"Erreur: {e}")