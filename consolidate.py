# consolidate.py
import pandas as pd
import os
from colorama import init, Fore, Style

init(autoreset=True)

def consolidate_all_csvs(dossier_data, fichier_sortie):
    """
    Lit tous les fichiers CSV, les nettoie, les harmonise
    et les fusionne en une seule base de données parfaite.
    """
    print(Fore.CYAN + Style.BRIGHT + "--- DÉBUT DE LA CONSOLIDATION DE LA BASE DE DONNÉES ---")
    fichiers_csv = [f for f in os.listdir(dossier_data) if f.endswith('.csv')]
    if not fichiers_csv:
        print(Fore.RED + "Aucun fichier CSV trouvé dans le dossier 'data'.")
        return

    print(f"✅ {len(fichiers_csv)} fichiers détectés. Harmonisation en cours...")
    
    liste_df_propres = []
    for fichier in fichiers_csv:
        try:
            df = pd.read_csv(os.path.join(dossier_data, fichier), encoding='ISO-8859-1', on_bad_lines='warn')
            
            # Dictionnaire de traduction universel
            rename_map = {
                'Home': 'HomeTeam', 'Away': 'AwayTeam',
                'HG': 'FTHG', 'AG': 'FTAG',
                'Res': 'FTR'
            }
            df.rename(columns=rename_map, inplace=True)

            # Vérification des colonnes essentielles
            colonnes_requises = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
            if set(colonnes_requises).issubset(df.columns):
                df_filtre = df[colonnes_requises].copy()
                liste_df_propres.append(df_filtre)
                print(f"  - {Fore.GREEN}Succès : Le fichier '{fichier}' a été lu et ajouté.")
            else:
                print(f"  - {Fore.YELLOW}Échec : Le fichier '{fichier}' ne contient pas les colonnes requises et a été ignoré.")
        
        except Exception as e:
            print(f"  - {Fore.RED}Erreur critique lors de la lecture de '{fichier}': {e}")

    if not liste_df_propres:
        print(Fore.RED + "Aucun fichier n'a pu être traité. La consolidation a échoué.")
        return

    # Fusion de tous les dataframes propres
    database_complete = pd.concat(liste_df_propres, ignore_index=True)
    
    # Nettoyage final et complet
    database_complete.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], inplace=True)
    database_complete['FTHG'] = pd.to_numeric(database_complete['FTHG'], errors='coerce')
    database_complete['FTAG'] = pd.to_numeric(database_complete['FTAG'], errors='coerce')
    database_complete.dropna(subset=['FTHG', 'FTAG'], inplace=True)
    database_complete['FTHG'] = database_complete['FTHG'].astype(int)
    database_complete['FTAG'] = database_complete['FTAG'].astype(int)
    
    # Sauvegarde du fichier final à la racine du projet
    database_complete.to_csv(fichier_sortie, index=False)
    print(Fore.GREEN + Style.BRIGHT + f"\n--- CONSOLIDATION TERMINÉE ---")
    print(f"La base de données finale a été créée : '{fichier_sortie}'")
    print(f"Nombre total de matchs traités : {len(database_complete)}")

if __name__ == "__main__":
    consolidate_all_csvs('./data', 'database_consolidated.csv')