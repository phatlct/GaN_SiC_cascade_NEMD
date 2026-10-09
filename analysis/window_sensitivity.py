#!/usr/bin/env python3
"""
Assess the sensitivity of G_corr to the choice of fitting-window parameters
(exclusion half-width EX and fitting half-width FH).

For each (EX, FH) combination the script reports:
  - whether the fitting window on the GaN side overlaps the hot thermostat (z < zlo + 25.2 A)
  - median G_corr at each cascade energy level
  - Spearman rank correlation between E and G_corr for E <= 5 keV and all energies

Only QC-passed runs (QC = 1, using the original window EX = 3, FH = 15) are included.

Input
-----
    recompute_G.csv   : summary CSV produced by recompute_G.py
    recompute_G.py    : imported to reuse deltaT() and related functions

Output
------
    Printed table: one row per (EX, FH) combination.

Usage
-----
    python3 window_sensitivity.py recompute_G.csv /path/to/recompute_G.py
"""

import sys, csv
import numpy as np

src = open(sys.argv[2]).read()
ns  = {"__name__": "lib"}
exec(src.split("TSTART = None")[0], ns)  # import function definitions only
ns["TSTART"] = None

CONV = 1.602176634e-7  # eV ps^-1 -> W


def spearman(x, y, n=5000, rng=np.random.default_rng(0)):
    """
    Spearman rank correlation with a two-sided permutation p-value.
    """
    rk      = lambda a: np.argsort(np.argsort(a)).astype(float)
    rx, ry  = rk(x), rk(y)
    r       = np.corrcoef(rx, ry)[0, 1]
    p       = np.mean(
        [abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(r) for _ in range(n)]
    )
    return r, p


# Read QC-passed runs and cache the temperature profiles
R     = [r for r in csv.DictReader(open(sys.argv[1])) if r["QC"] == "1"]
cache = {}
for r in R:
    o        = ns["parse_log"](r["path"])
    Lz, A, zlo = o[2], o[3], o[6]
    bl       = ns["parse_profile"](r["profile"], Lz, zlo)[0]
    cache[r["path"]] = (bl[0][:, 0], np.mean([b[:, 1] for b in bl], axis=0), A)

# Hot thermostat ends at approximately zlo + 25.2 A; FH > ~12.9 A causes overlap
print(
    "EX  FH  hot_overlap  "
    + "  ".join(f"medG(E={e:g})" for e in (1, 3, 5, 7, 10))
    + "   rho/p(E<=5)      rho/p(all)"
)

for EX in (2, 3, 4, 5):
    for FH in (10, 12, 15):
        ns["EX"], ns["FH"] = EX, FH
        E_list, G_list = [], []
        for r in R:
            z, T, A = cache[r["path"]]
            dT = ns["deltaT"](z, T)
            E_list.append(float(r["E"]))
            G_list.append(float(r["Jc"]) * CONV / (A * 1e-20) / dT / 1e6)
        E_arr, G_arr = np.array(E_list), np.array(G_list)
        med  = "  ".join(f"{np.median(G_arr[E_arr == e]):10.1f}" for e in (1, 3, 5, 7, 10))
        r5,  p5  = spearman(E_arr[E_arr <= 5], G_arr[E_arr <= 5])
        r10, p10 = spearman(E_arr, G_arr)
        overlap  = "yes" if 38.126 - FH < 25.2 else "no"
        print(
            f"{EX:2d}  {FH:2d}  {overlap:11s}  {med}   "
            f"{r5:+.2f}/{p5:.3f}     {r10:+.2f}/{p10:.3f}"
        )

print(f"\nn = {len(R)} QC-passed runs (QC flag set with original window EX=3, FH=15).")
print("G_corr = J_c / (A * deltaT); positive Spearman rho indicates G rises with E.")
