#!/usr/bin/env python3
# Dem cap Ga-Ga, N-N, Ga-N nam trong vung chuyen tiep cutoff Tersoff (R-D < r < R+D).
# Input : GaN.tersoff + cac file LAMMPS data (type 1=Ga, 2=N; hop triclinic, tuan hoan x,y)
# Output: so cap trong vung chuyen tiep va trong vung tuong tac day du (r < R-D), moi cau truc
# Usage : python3 cutoff_pairs.py GaN.tersoff data1 [data2 ...]
import sys, re, numpy as np

def tersoff_RD(path):
    tok = " ".join(l.split("#")[0] for l in open(path)).split(); out = {}
    for i in range(0, len(tok) - 16, 17):
        e = tok[i:i + 3]; v = list(map(float, tok[i + 3:i + 17]))
        if e[1] == e[2]: out[(e[0], e[1])] = (v[10], v[11])          # R, D cua tuong tac 2 than
    return out

def read_data(path):
    L = open(path).read().splitlines(); box = {}
    for l in L[:40]:
        p = l.split()
        if "xlo" in l: box["x"] = (float(p[0]), float(p[1]))
        if "ylo" in l: box["y"] = (float(p[0]), float(p[1]))
        if "xy xz yz" in l: box["xy"] = float(p[0])
    i = next(k for k, l in enumerate(L) if l.strip().startswith("Atoms")) + 2; rows = []
    while i < len(L) and L[i].strip():
        p = L[i].split(); rows.append((int(p[1]), float(p[2]), float(p[3]), float(p[4]))); i += 1
    a = np.array(rows); return a[:, 0].astype(int), a[:, 1:], box

def pairs(Xa, Xb, box, rmax, same):
    lx, ly, xy = box["x"][1] - box["x"][0], box["y"][1] - box["y"][0], box.get("xy", 0.0)
    out = []
    for s in range(0, len(Xa), 400):
        d = Xb[None, :, :] - Xa[s:s + 400, None, :]
        ny = np.round(d[..., 1] / ly); d[..., 1] -= ny * ly; d[..., 0] -= ny * xy   # anh gan nhat (nghieng xy)
        d[..., 0] -= np.round(d[..., 0] / lx) * lx
        r = np.sqrt((d ** 2).sum(-1))
        if same: r[:, :] = np.where(np.arange(len(Xb))[None, :] > (np.arange(s, s + len(r)))[:, None], r, np.inf)
        out.append(r[r < rmax])
    return np.concatenate(out)

RD = tersoff_RD(sys.argv[1]); names = {1: "Ga", 2: "N"}
print("file".ljust(44), "  ".join(f"{a}-{b}: R-D<r<R+D | r<R-D".ljust(30) for a, b in (("Ga", "Ga"), ("N", "N"), ("Ga", "N"))))
print("R,D:", {f"{k[0]}-{k[1]}": v for k, v in RD.items() if k in (("Ga", "Ga"), ("N", "N"), ("Ga", "N"))})
for f in sys.argv[2:]:
    t, X, box = read_data(f); cols = []
    for a, b in ((1, 1), (2, 2), (1, 2)):
        R, D = RD[(names[a], names[b])]
        r = pairs(X[t == a], X[t == b], box, R + D, a == b)
        cols.append(f"{np.sum(r > R - D):8d} | {np.sum(r <= R - D):8d}".ljust(30))
    print(f.split("/")[-1][:44].ljust(44), "  ".join(cols))
