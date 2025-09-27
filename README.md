# Robot-Parieur

Projet de prédiction de résultats de matchs (EWMA + XGBoost + Poisson).  
Tests et CI: pytest + GitHub Actions.

[![CI](https://github.com/kerdeli/robot-parieur/actions/workflows/pytest.yml/badge.svg)](https://github.com/kerdeli/robot-parieur/actions)

## Installation rapide

1. Créez et activez un venv:
   - Windows PowerShell:
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

2. Installez dépendances:
   ```
   python -m pip install --upgrade pip
   pip install pandas numpy xgboost scikit-learn scipy tqdm colorama joblib pytest
   ```

3. Lancer les tests:
   ```
   python -m pytest -q
   ```

## Usage

- Entraîner / lancer l'outil en local:
  ```
  python analyse.py --db data/database_consolidated.csv
  ```

## Contribuer

- Utilisez git: init, add, commit, push. Voir `.github/workflows/pytest.yml` pour CI.
