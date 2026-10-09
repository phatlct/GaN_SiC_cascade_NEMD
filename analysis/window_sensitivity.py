#!/usr/bin/env python3
# Do nhay cua Delta T / G_corr theo cua so fit (cam ket o Methods 2.7).
# Input : recompute_G.csv (cot path, profile, Jc, E, species, QC) + recompute_G.py (dung lai ham doc log/profile)
# Output: bang tung to hop (EX, FH): median G_corr theo E (QC goc) + Spearman E<=5 va E<=10
# Usage : python3 window_sensitivity.py recompute_G.csv /duong/dan/recompute_G.py
import sys, csv, numpy as np
src = open(sys.argv[2]).read()
ns = {"__name__": "lib"}; exec(src.split("TSTART = None")[0], ns)     # chi lay phan dinh nghia ham
ns["TSTART"], CONV = None, 1.602176634e-7

def spearman(x, y, n=5000, rng=np.random.default_rng(0)):
    rk = lambda a: np.argsort(np.argsort(a)).astype(float)
    rx, ry = rk(x), rk(y); r = np.corrcoef(rx, ry)[0, 1]
    return r, np.mean([abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(r) for _ in range(n)])

R = [r for r in csv.DictReader(open(sys.argv[1])) if r["QC"] == "1"]
cache = {}
for r in R:                                   # doc profile + hinh hop 1 lan
    o = ns["parse_log"](r["path"]); Lz, A, zlo = o[2], o[3], o[6]
    bl = ns["parse_profile"](r["profile"], Lz, zlo)[0]
    cache[r["path"]] = (bl[0][:, 0], np.mean([b[:, 1] for b in bl], axis=0), A)

# Vung nong ket thuc o z_lo + 25 A (~25.2 A): FH > ~12.9 A thi cua so trai chong len thermostat
print("EX  FH  chong_vung_nong  " + "  ".join(f"medG(E={e:g})" for e in (1, 3, 5, 7, 10)) + "   rho/p(E<=5)      rho/p(E<=10)")
for EX in (2, 3, 4, 5):
    for FH in (10, 12, 15):
        ns["EX"], ns["FH"] = EX, FH
        E, G = [], []
        for r in R:
            z, T, A = cache[r["path"]]
            dT = ns["deltaT"](z, T)
            E.append(float(r["E"])); G.append(float(r["Jc"]) * CONV / (A * 1e-20) / dT / 1e6)
        E, G = np.array(E), np.array(G)
        med = "  ".join(f"{np.median(G[E == e]):10.1f}" for e in (1, 3, 5, 7, 10))
        r5, p5 = spearman(E[E <= 5], G[E <= 5]); r10, p10 = spearman(E, G)
        ov = "co" if 38.126 - FH < 25.2 else "khong"
        print(f"{EX:2d}  {FH:2d}  {ov:15s}  {med}   {r5:+.2f}/{p5:.3f}     {r10:+.2f}/{p10:.3f}")
print(f"\nn = {len(R)} run QC (QC theo cua so goc EX=3, FH=15). G_corr = J_c / (A dT).")
