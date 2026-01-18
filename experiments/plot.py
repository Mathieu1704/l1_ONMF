# experiments/plot_doc_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
CSV_FILE = Path(__file__).parent / "../docs_full_comparison.csv"
# Si tu utilises le nouveau CSV généré: remplace par
# CSV_FILE = Path(__file__).parent / "../docs_full_from_latex_in_sample_schema.csv"

OUT_DIR = Path(__file__).parent / "plots" / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_results():
    if not CSV_FILE.exists():
        print(f"Erreur: Le fichier {CSV_FILE} n'existe pas.")
        return

    df = pd.read_csv(CSV_FILE)

    # -------------------------
    # NETTOYAGE (IMPORTANT)
    # -------------------------
    # Enlever lignes sans dataset (ligne vide finale) + enlever "Averages" pour les plots dataset-par-dataset
    df["dataset"] = df["dataset"].astype("string")
    df = df[df["dataset"].notna() & (df["dataset"].str.strip() != "")]
    df = df[df["dataset"].str.lower() != "averages"]

    # Convertir colonnes numériques proprement (au cas où des champs vides existent)
    num_cols = ["n","m","r","acc_L1","acc_Fro","acc_KL","time_L1","time_Fro","time_KL","it_L1","it_Fro","it_KL"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Garder uniquement les lignes complètes pour les plots
    df = df.dropna(subset=["n","acc_L1","acc_Fro","acc_KL","time_L1","time_Fro","time_KL"])

    # Tri par taille (n)
    df = df.sort_values("n").reset_index(drop=True)

    datasets = df["dataset"].str.replace(".mat", "", regex=False).tolist()
    x = np.arange(len(datasets))
    width = 0.25

    # ==========================================
    # PLOT 1: ACCURACY COMPARISON
    # ==========================================
    plt.figure(figsize=(14, 7))

    plt.bar(x - width, df["acc_L1"],  width, label="L1-ONMF", alpha=0.8)
    plt.bar(x,         df["acc_Fro"], width, label="Fro-ONMF", alpha=0.8)
    plt.bar(x + width, df["acc_KL"],  width, label="KL-ONMF", alpha=0.8)

    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Comparaison de la Précision de Clustering sur Documents Réels", fontsize=14)
    plt.xticks(x, datasets, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.ylim(0, 100)

    # Annoter seulement quand L1 bat Fro ET KL
    for i in range(len(df)):
        l1  = df["acc_L1"].iloc[i]
        fro = df["acc_Fro"].iloc[i]
        kl  = df["acc_KL"].iloc[i]
        if (l1 > fro) and (l1 > kl):
            plt.text(x[i] - width, l1 + 1, "★", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out1 = OUT_DIR / "docs_accuracy_comparison.png"
    plt.savefig(out1, dpi=150)
    print(f"Graphique sauvegardé : {out1}")

    # ==========================================
    # PLOT 2: TIME COMPARISON (Log Scale)
    # ==========================================
    plt.figure(figsize=(14, 6))

    # IMPORTANT: on utilise x (indices) => plus de problème de catégories/NaN
    plt.plot(x, df["time_L1"],  "o-", label="L1-ONMF")
    plt.plot(x, df["time_Fro"], "s-", label="Fro-ONMF")
    plt.plot(x, df["time_KL"],  "^-", label="KL-ONMF")

    plt.yscale("log")
    plt.ylabel("Time (seconds) - Log Scale", fontsize=12)
    plt.title("Comparaison des Temps de Calcul", fontsize=14)
    plt.xticks(x, datasets, rotation=45, ha="right")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    out2 = OUT_DIR / "docs_time_comparison.png"
    plt.savefig(out2, dpi=150)
    print(f"Graphique sauvegardé : {out2}")

if __name__ == "__main__":
    plot_results()
