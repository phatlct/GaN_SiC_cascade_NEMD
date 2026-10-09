#!/usr/bin/env python3
# Kiem tra co che tren 165 run cu: nguon nhiet S (dt=1 fs) co tang theo so cap Ga-Ga/Ga-N trong vung chuyen tiep cutoff?
# Input : recompute_G.csv (165 run cu), cutoff_pairs.py, GaN.tersoff
# Output: cutoff_vs_S.csv + FigD4_S_vs_cutoffpairs.png + Spearman (permutation)
# Usage : python3 cutoff_vs_S.py recompute_G.csv cutoff_pairs.py GaN.tersoff
import sys, os, re, glob, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ns = {"__name__": "lib", "sys": sys}; exec(open(sys.argv[2]).read().split("RD = tersoff_RD")[0], ns)
RD = ns["tersoff_RD"](sys.argv[3])

def datafile(log):
    m = re.search(r"_(\d+)K_(Ga|N)_([\d.]+)keV", log); T, sp, E = m.group(1), m.group(2), float(m.group(3))
    d = os.path.dirname(log)
    c = glob.glob(os.path.join(d, "data.postanneal_*.data"))
    if not c:                                            # Seed1: Logs/<T>K/ -> Raw_Data/<T>K/
        root = d.split("/Logs/")[0]
        c = [f for f in glob.glob(f"{root}/Raw_Data/*/data.postanneal_{T}K_{sp}_*keV.data")
             if abs(float(re.search(r"_([\d.]+)keV\.data$", f).group(1)) - E) < 1e-6]
    return c[0] if c else None

def count(f, a, b):
    t, X, box = ns["read_data"](f); R, D = RD[({1: "Ga", 2: "N"}[a], {1: "Ga", 2: "N"}[b])]
    r = ns["pairs"](X[t == a], X[t == b], box, R + D, a == b); return int(np.sum(r > R - D))

rows = []
R0 = list(csv.DictReader(open(sys.argv[1])))
for k, r in enumerate(R0):
    f = datafile(r["path"])
    if not f: print("THIEU data:", r["path"]); continue
    rows.append(dict(path=r["path"], T=r["T"], species=r["species"], E=r["E"], S=float(r["S"]),
                     nGaGa=count(f, 1, 1), nGaN=count(f, 1, 2)))
    print(f"\r{k + 1}/{len(R0)}", end="", flush=True)
print()
with open("cutoff_vs_S.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def spearman(x, y, n=20000, rng=np.random.default_rng(0)):
    rk = lambda a: np.argsort(np.argsort(a)).astype(float); rx, ry = rk(x), rk(y); rho = np.corrcoef(rx, ry)[0, 1]
    return rho, np.mean([abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(rho) for _ in range(n)])
S = np.array([r["S"] for r in rows]); E = np.array([float(r["E"]) for r in rows])
for key in ("nGaGa", "nGaN"):
    x = np.array([r[key] for r in rows]); rho, p = spearman(x, S)
    # trong tung muc E: tach anh huong cua E (S va so cap deu tang theo E)
    within = [spearman(x[E == e], S[E == e], 5000)[0] for e in sorted(set(E)) if np.sum(E == e) > 5]
    print(f"S vs {key}: rho={rho:+.3f} p={p:.4f} n={len(x)} | rho trong tung muc E: {np.round(within, 2)}")
fig, ax = plt.subplots(figsize=(3.6, 2.9)); col = {"300": "#1f5fa8", "350": "#7a7a7a", "400": "#c4501b"}
for T in ("300", "350", "400"):
    m = np.array([r["T"] == T for r in rows])
    ax.scatter(np.array([r["nGaGa"] for r in rows])[m], S[m], s=10, color=col[T], alpha=0.75, label=f"{T} K")
ax.set(xlabel="Ga–Ga pairs in cutoff transition zone", ylabel="$S = J_h + J_c$ (eV ps$^{-1}$), $\\Delta t$ = 1 fs")
ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("FigD4_S_vs_cutoffpairs.png", dpi=300); print("Da luu cutoff_vs_S.csv, FigD4_S_vs_cutoffpairs.png")
