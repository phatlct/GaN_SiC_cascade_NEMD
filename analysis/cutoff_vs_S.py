#!/usr/bin/env python3
"""
Test whether the spurious heat-source imbalance S = J_h + J_c (measured at dt = 1 fs)
correlates with the number of Ga-Ga and Ga-N atom pairs that lie within the Tersoff
cutoff transition zone (R-D < r < R+D) in the post-anneal structure.

A Spearman rank correlation with a permutation p-value is reported for each pair type
both globally and within each cascade-energy level (to remove the confounding effect
of energy, which raises both S and the pair counts).

Input
-----
    recompute_G.csv   : summary CSV produced by recompute_G.py (165 original runs)
    cutoff_pairs.py   : imported to reuse read_data() and pairs()
    GaN.tersoff       : Tersoff potential file (for R and D values)

Output
------
    cutoff_vs_S.csv               : one row per run with nGaGa, nGaN, S
    FigD4_S_vs_cutoffpairs.png    : scatter plot of S vs Ga-Ga pairs, coloured by temperature

Usage
-----
    python3 cutoff_vs_S.py recompute_G.csv cutoff_pairs.py GaN.tersoff
"""

import sys, os, re, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import helper functions from cutoff_pairs.py without executing its main block
ns = {"__name__": "lib", "sys": sys}
exec(open(sys.argv[2]).read().split("RD = tersoff_RD")[0], ns)
RD = ns["tersoff_RD"](sys.argv[3])


def datafile(log):
    """
    Locate the post-anneal LAMMPS data file corresponding to a given log path.

    Seed 1 structures are stored under Raw_Data/*/ (the temperature subdirectory
    may not match the nominal temperature in the filename); a glob over all
    subdirectories is therefore used instead of a direct path lookup.
    """
    m    = re.search(r"_(\d+)K_(Ga|N)_([\d.]+)keV", log)
    T, sp, E = m.group(1), m.group(2), float(m.group(3))
    d    = os.path.dirname(log)

    # Preferred location: same directory as the log
    c = glob.glob(os.path.join(d, "data.postanneal_*.data"))
    if not c:
        # Fallback for Seed 1: Logs/<T>K/ -> Raw_Data/*/
        root = d.split("/Logs/")[0]
        c = [
            f for f in glob.glob(f"{root}/Raw_Data/*/data.postanneal_{T}K_{sp}_*keV.data")
            if abs(float(re.search(r"_([\d.]+)keV\.data$", f).group(1)) - E) < 1e-6
        ]
    return c[0] if c else None


def count(f, a, b):
    """Count pairs of types a and b in the cutoff transition zone of structure f."""
    t, X, box = ns["read_data"](f)
    R, D = RD[({1: "Ga", 2: "N"}[a], {1: "Ga", 2: "N"}[b])]
    r = ns["pairs"](X[t == a], X[t == b], box, R + D, a == b)
    return int(np.sum(r > R - D))


# ---------------------------------------------------------------------------
# Main loop: count pairs for each run
# ---------------------------------------------------------------------------

rows = []
R0   = list(csv.DictReader(open(sys.argv[1])))

for k, r in enumerate(R0):
    f = datafile(r["path"])
    if not f:
        print("MISSING data file:", r["path"])
        continue
    rows.append(dict(
        path    = r["path"],
        T       = r["T"],
        species = r["species"],
        E       = r["E"],
        S       = float(r["S"]),
        nGaGa   = count(f, 1, 1),
        nGaN    = count(f, 1, 2),
    ))
    print(f"\r{k + 1}/{len(R0)}", end="", flush=True)

print()

with open("cutoff_vs_S.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)


# ---------------------------------------------------------------------------
# Spearman correlation with permutation p-value
# ---------------------------------------------------------------------------

def spearman(x, y, n=20000, rng=np.random.default_rng(0)):
    """
    Spearman rank correlation between x and y with a two-sided permutation p-value.
    The p-value is the fraction of n permutations that yield |rho| >= observed |rho|.
    """
    rk  = lambda a: np.argsort(np.argsort(a)).astype(float)
    rx, ry = rk(x), rk(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    p   = np.mean(
        [abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(rho) for _ in range(n)]
    )
    return rho, p


S = np.array([r["S"] for r in rows])
E = np.array([float(r["E"]) for r in rows])

for key in ("nGaGa", "nGaN"):
    x         = np.array([r[key] for r in rows])
    rho, p    = spearman(x, S)
    # Within-energy-level Spearman removes the confound of E driving both S and pair counts
    within    = [
        spearman(x[E == e], S[E == e], 5000)[0]
        for e in sorted(set(E))
        if np.sum(E == e) > 5
    ]
    print(
        f"S vs {key}: rho={rho:+.3f}  p={p:.4f}  n={len(x)}  |  "
        f"within-E rho: {np.round(within, 2)}"
    )


# ---------------------------------------------------------------------------
# Figure: S vs Ga-Ga pairs in the cutoff transition zone
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(3.6, 2.9))

col = {"300": "#1f5fa8", "350": "#7a7a7a", "400": "#c4501b"}
for T in ("300", "350", "400"):
    m = np.array([r["T"] == T for r in rows])
    ax.scatter(
        np.array([r["nGaGa"] for r in rows])[m],
        S[m],
        s=10, color=col[T], alpha=0.75, label=f"{T} K"
    )

ax.set_xlabel("Ga–Ga pairs in cutoff transition zone")
ax.set_ylabel(r"$S = J_h + J_c$ (eV ps$^{-1}$), $\Delta t$ = 1 fs")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7, frameon=False)

fig.tight_layout()
fig.savefig("FigD4_S_vs_cutoffpairs.png", dpi=300)
print("Saved: cutoff_vs_S.csv, FigD4_S_vs_cutoffpairs.png")
