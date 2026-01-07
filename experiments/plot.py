# experiments/plot_doc_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# --- CONFIGURATION ---
CSV_FILE = Path(__file__).parent / "table_comparative_real_docs.csv"
OUT_DIR = Path(__file__).parent / "plots" / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_results():
    if not CSV_FILE.exists():
        print(f"Erreur: Le fichier {CSV_FILE} n'existe pas.")
        return

    # Chargement des données
    df = pd.read_csv(CSV_FILE)
    
    # Tri par taille de dataset (n) pour la lisibilité
    df = df.sort_values('n')
    
    datasets = df['dataset'].str.replace('.mat', '', regex=False)
    x = np.arange(len(datasets))
    width = 0.25

    # ==========================================
    # PLOT 1: ACCURACY COMPARISON
    # ==========================================
    plt.figure(figsize=(14, 7))
    
    # Barres
    plt.bar(x - width, df['acc_L1'], width, label='L1-ONMF', color='#d62728', alpha=0.8)
    plt.bar(x, df['acc_Fro'], width, label='Fro-ONMF', color='#1f77b4', alpha=0.8)
    plt.bar(x + width, df['acc_KL'], width, label='KL-ONMF', color='#2ca02c', alpha=0.8)
    
    # Esthétique
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Comparaison de la Précision de Clustering sur Documents Réels', fontsize=14)
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.ylim(0, 100)
    
    # Annoter les victoires de L1 sur Fro
    for i in range(len(df)):
        if df['acc_L1'].iloc[i] > df['acc_Fro'].iloc[i]:
            plt.text(x[i] - width, df['acc_L1'].iloc[i] + 1, "★", 
                     ha='center', color='red', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "docs_accuracy_comparison.png", dpi=150)
    print(f"Graphique sauvegardé : {OUT_DIR / 'docs_accuracy_comparison.png'}")

    # ==========================================
    # PLOT 2: TIME COMPARISON (Log Scale)
    # ==========================================
    plt.figure(figsize=(14, 6))
    
    plt.plot(datasets, df['time_L1'], 'o-', label='L1-ONMF', color='#d62728')
    plt.plot(datasets, df['time_Fro'], 's-', label='Fro-ONMF', color='#1f77b4')
    plt.plot(datasets, df['time_KL'], '^-', label='KL-ONMF', color='#2ca02c')
    
    plt.yscale('log') # Log scale car L1 est beaucoup plus lent
    plt.ylabel('Time (seconds) - Log Scale', fontsize=12)
    plt.title('Comparaison des Temps de Calcul', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "docs_time_comparison.png", dpi=150)
    print(f"Graphique sauvegardé : {OUT_DIR / 'docs_time_comparison.png'}")

if __name__ == "__main__":
    plot_results()