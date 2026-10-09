#!/usr/bin/env python3
# Tach nhieu thong ke khoi su phu thuoc he thong cua G vao vung loai tru EX (bootstrap theo block profile).
# Input : recompute_G.csv (tao voi --tstart 50) + recompute_G.py ; chi xet run co 'dt025' trong duong dan
# Output: moi run: G(EX=2..6) + CI95, do doc tuong doi dG/dEX (%/A) + CI95, G_reg + CI95 (chi phan profile)
# Usage : python3 bootstrap_EX.py recompute_G.csv recompute_G.py [B=1000]
import sys, csv, numpy as np
ns = {"__name__": "lib"}; exec(open(sys.argv[2]).read().split("TSTART = None")[0], ns); ns["TSTART"] = 50.0
B = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
CONV, ZI, EXS = 1.602176634e-7, 38.126, np.array([2, 3, 4, 5, 6], float)
rng = np.random.default_rng(0)

def metrics(z, T, Jtb, A, zlo):
    g = []
    for e in EXS:
        ns["EX"], ns["FH"] = e, 12.0; g.append(Jtb * CONV / (A * 1e-20) / ns["deltaT"](z, T) / 1e6)
    g = np.array(g); slope = np.polyfit(EXS, g, 1)[0] / g.mean() * 100          # %/A
    T1 = np.interp(zlo + 27.0, z, T); m = (z > ZI + 3) & (z < ZI + 25)
    T2 = np.polyval(np.polyfit(z[m], T[m], 1), ZI + 15)
    return g, slope, (g.max() - g.min()) / g.mean(), Jtb * CONV / (A * 1e-20) / (T1 - T2) / 1e6

ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
for r in csv.DictReader(open(sys.argv[1])):
    if "dt025" not in r["path"]: continue
    o = ns["parse_log"](r["path"]); Lz, A, zlo, dt = o[2], o[3], o[6], o[8]
    bl = ns["parse_profile"](r["profile"], Lz, zlo, round(60 / dt))[0]
    if len(bl) < 5: print(r["path"].split("/")[-2], f"-> chi co {len(bl)} block, bo qua"); continue
    z = bl[0][:, 0]; Tb = np.array([b[:, 1] for b in bl]); Jtb = (float(r["Jc2"]) - float(r["Jh2"])) / 2
    g0, s0, sens0, gr0 = metrics(z, Tb.mean(0), Jtb, A, zlo)
    bs = [metrics(z, Tb[rng.integers(0, len(bl), len(bl))].mean(0), Jtb, A, zlo) for _ in range(B)]
    G = np.array([b[0] for b in bs]); S = np.array([b[1] for b in bs]); SE = np.array([b[2] for b in bs]); GR = np.array([b[3] for b in bs])
    lo, hi = ci(S); sig = "CO (CI khong chua 0)" if lo > 0 or hi < 0 else "khong (CI chua 0)"
    print(f"\n== {r['path'].split('/')[-2]}  (nblk={len(bl)}, B={B})")
    print("   G(EX):   " + "  ".join(f"EX{int(e)}={g:6.1f} [{np.percentile(G[:, i], 2.5):5.1f},{np.percentile(G[:, i], 97.5):5.1f}]" for i, (e, g) in enumerate(zip(EXS, g0))))
    print(f"   dG/dEX = {s0:+5.1f} %/A  CI95 [{lo:+5.1f}, {hi:+5.1f}]  -> phu thuoc EX co he thong: {sig}")
    print(f"   do nhay quan sat = {sens0:4.0%} ; median bootstrap = {np.median(SE):4.0%} [{ci(SE)[0]:4.0%}, {ci(SE)[1]:4.0%}]")
    print(f"   G_reg = {gr0:6.1f}  CI95 (chi profile) [{ci(GR)[0]:5.1f}, {ci(GR)[1]:5.1f}]   (chua gom sai so cua J)")
print("\nGhi chu: bootstrap coi cac block 10 ps doc lap; neu co tuong quan giua block, CI se hep hon thuc te.")
