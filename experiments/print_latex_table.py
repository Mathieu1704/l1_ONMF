# examples/print_latex_table.py
import csv, sys
from statistics import mean

def esc(s: str) -> str:
    # Échapper _ pour LaTeX
    return s.replace("_", r"\_")

def main(csv_path, label="tab:l1_onmf_docs"):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["m"] = int(r["m"]); r["n"] = int(r["n"]); r["r"] = int(r["r"])
            for k in ["acc","ari","nmi","time_s"]:
                r[k] = float(r[k])
            r["iters"] = int(r["iters"]) if r["iters"] not in (None, "", "None") else None
            rows.append(r)

    # Moyennes pondérées par n (nb de documents), comme dans le papier
    total_n = sum(r["n"] for r in rows)
    wmean_acc = sum(r["acc"] * r["n"] for r in rows) / total_n if total_n > 0 else mean([r["acc"] for r in rows])
    wmean_ari = sum(r["ari"] * r["n"] for r in rows) / total_n if total_n > 0 else mean([r["ari"] for r in rows])
    wmean_nmi = sum(r["nmi"] * r["n"] for r in rows) / total_n if total_n > 0 else mean([r["nmi"] for r in rows])

    print("\\begin{center}")
    print("\\begin{table}[h!]")
    print("\\begin{center}")
    print("\\begin{tabular}{|c||c|c|c||c|c|c||c|c|}")
    print("\\hline")
    print("& $m$ & $n$ & $r$ & acc & ARI & NMI & time(s) & iters \\\\ \\hline")
    print("\\hline")
    for r in rows:
        name = esc(r['dataset'])
        iters = r['iters'] if r['iters'] is not None else '-'
        print(f"{name} & {r['m']} & {r['n']} & {r['r']} "
              f"& {100*r['acc']:.1f} & {r['ari']:.3f} & {r['nmi']:.3f} "
              f"& {r['time_s']:.2f} & {iters} \\\\ \\hline")
    print("\\hline")
    iters_mean = [x['iters'] for x in rows if x['iters'] is not None]
    iters_mean_str = f"{int(mean(iters_mean))}" if iters_mean else "-"
    print("Weighted avg & & & "
          f"& {100*wmean_acc:.1f} "
          f"& {wmean_ari:.3f} "
          f"& {wmean_nmi:.3f} "
          f"& {mean([x['time_s'] for x in rows]):.2f} "
          f"& {iters_mean_str} \\\\ \\hline")
    print("\\end{tabular}")
    print("\\caption{Résultats L1-ONMF sur 15 jeux de documents.}")
    print(f"\\label{{{label}}}")
    print("\\end{center}")
    print("\\end{table}")
    print("\\end{center}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m examples.print_latex_table results_docs.csv")
        sys.exit(1)
    main(sys.argv[1])
