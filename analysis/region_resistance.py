#!/usr/bin/env python3
# Dien tro nhiet cua ca VUNG chua lop hu hai, khong phu thuoc dinh nghia "mat interface".
# R_reg = [T(z1) - T(z2)] / q, z1 = mep vung nong + 2 A (~27 A), z2 = z_int + 15 A (SiC, ngoai vung hu hai).
# Input : recompute_G.csv (tao voi --tstart) + recompute_G.py ; Output: bang R_reg, G_reg, T(z1), T(z2)
# Usage : python3 region_resistance.py recompute_G.csv recompute_G.py [tstart=50]
import sys, csv, numpy as np
ns = {"__name__": "lib"}; exec(open(sys.argv[2]).read().split("TSTART = None")[0], ns)
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0; ns["TSTART"] = t0
CONV, ZI = 1.602176634e-7, 38.126
print(f"{'run':32s} {'T(z1)':>7s} {'T(z2)':>7s} {'dT_reg':>7s} {'J_tb':>6s} {'G_reg':>7s} {'G_bal':>7s}")
for r in csv.DictReader(open(sys.argv[1])):
    o = ns["parse_log"](r["path"]); Lz, A, zlo, dt = o[2], o[3], o[6], o[8]
    bl = ns["parse_profile"](r["profile"], Lz, zlo, round((t0 + 10) / dt))[0]
    z, T = bl[0][:, 0], np.mean([b[:, 1] for b in bl], axis=0)
    z1 = zlo + 25.0 + 2.0
    T1 = np.interp(z1, z, T)                                          # 2 A ben trong GaN tu do
    m = (z > ZI + 3) & (z < ZI + 25); T2 = np.polyval(np.polyfit(z[m], T[m], 1), ZI + 15)
    J = (float(r["Jc2"]) - float(r["Jh2"])) / 2
    Gr = J * CONV / (A * 1e-20) / (T1 - T2) / 1e6
    print(f"{r['path'].split('/')[-2][:32]:32s} {T1:7.2f} {T2:7.2f} {T1-T2:7.2f} {J:6.3f} {Gr:7.1f} {float(r['G_bal']):7.1f}")
print("G_reg = J_tb/(A*dT_reg): do dan cua ca vung [z1, z_int+15], gom GaN hu hai + interface + 15 A SiC.")
