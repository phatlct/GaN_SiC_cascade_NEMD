#!/usr/bin/env python3
"""
Cross-validate G_old (recomputed with recompute_G.py) against the original
post-processing pipeline to confirm numerical consistency before further analysis.

Two comparisons are made:
  (1) Seed 1 only: run-level G from the original summary file vs recomputed G_old.
  (2) Ensemble of Seeds 1-5: mean G from the original ensemble file vs the mean of
      recomputed G_old across the same five seeds.

Discrepancies larger than ~1 % indicate a difference in input data or parsing logic
and should be investigated before proceeding.

Input
-----
    recompute_G.csv          : summary CSV from recompute_G.py (all seeds)
    summary_G_sweep.csv      : original per-run results (Seed 1, status = "OK")
    ensemble5_ToHop01_30.csv : original ensemble means for Seeds 1-5

Usage
-----
    python3 validate_G.py recompute_G.csv summary_G_sweep.csv ensemble5_ToHop01_30.csv
"""

import sys, csv
import numpy as np
import collections as C

R   = list(csv.DictReader(open(sys.argv[1])))
key = lambda T, s, E: (int(float(T)), s, round(float(E), 1))

# Load original Seed 1 results (last OK entry per condition)
old1 = {}
for r in csv.DictReader(open(sys.argv[2])):
    if r["status"] == "OK":
        old1[key(r["T_K"], r["pka_species"], r["E_pka_keV"])] = float(r["G_MW_m2K"])

# Build recomputed G_old indexed by (T, species, E, seed)
new = C.defaultdict(dict)
for r in R:
    if int(r["seed"]) <= 5:
        new[key(r["T"], r["species"], r["E"])][int(r["seed"])] = float(r["G_old"])

# (1) Seed 1 comparison
print("(1) Seed 1: original G vs recomputed G_old")
d1 = [
    (k, old1[k], new[k][1])
    for k in sorted(old1)
    if 1 in new.get(k, {})
]
for k, a, b in d1:
    print(f"  {k}  original={a:8.1f}  recomputed={b:8.1f}  diff={100*(b-a)/a:+6.1f}%")

# (2) Ensemble of Seeds 1-5 comparison
print("\n(2) Seeds 1-5 ensemble: original mean G vs recomputed mean")
for r in csv.DictReader(open(sys.argv[3])):
    k = key(r["T_K"], r["pka_species"], r["E_pka_keV"])
    v = new.get(k, {})
    if len(v) == 5:
        m = np.mean(list(v.values()))
        a = float(r["G_mean_MW_m2K"])
        print(f"  {k}  original={a:8.1f}  recomputed={m:8.1f}  diff={100*(m-a)/a:+6.1f}%")
    else:
        print(f"  {k}  available seeds: {sorted(v)} (expected 5)")
