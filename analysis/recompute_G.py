#!/usr/bin/env python3
# Input : cac thu muc chua log.04_nemd_* + profile_T_*.dat
# Output: recompute_G.csv (1 dong/run) + bang tom tat theo (E, T)
# Usage : python3 recompute_G.py <root1> [<root2> ...]
import sys, re, glob, os, csv, collections
import numpy as np

Z_INT, FH, EX = 38.126, 15.0, 3.0           # giong post_process cu
CONV = 1.602176634e-7                        # eV/ps -> W

def parse_log(path):
    blk, last, kv, dt, prod = 0, None, {}, 0.001, []
    hotre = re.compile(r"^fix\s+fix_hot\s+g_hot\s+langevin\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+\d+")
    tsre = re.compile(r"^\s*timestep\s+([0-9.eE+-]+)\s*$")
    pat = {"Lz_print": r"Lz \(Angstrom[^=]*=\s*([-\d.eE+]+)"}
    boxre = re.compile(r"(?:triclinic|orthogonal) box = \(([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+)\) to \(([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+)\)")
    for line in open(path, errors="ignore"):
        mh = hotre.match(line)
        if mh: kv["Thot"] = float(mh.group(1))          # nhiet do dich thermostat nong (da thay so)
        mt = tsre.match(line)
        if mt: dt = float(mt.group(1))                 # gia tri timestep da thay so (cuoi cung)
        if re.match(r"^\s+Step\s", line): blk += 1; continue
        if blk == 2 and re.match(r"^\s+\d+\s+[\d.]", line):
            p = line.split(); last = (int(p[0]), float(p[4]), float(p[5])); prod.append(last)
        mb = boxre.search(line)
        if mb:                                           # hop mo phong (boundary f theo z -> co dinh)
            x0, y0, z0, x1, y1, z1 = map(float, mb.groups())
            kv["zlo"], kv["Lz"], kv["A"] = z0, z1 - z0, (x1 - x0) * (y1 - y0)
        for k, r in pat.items():
            m = re.search(r, line)
            if m: kv[k] = float(m.group(1))
    s, h, c = last
    t = s * dt                                    # ps
    t0 = TSTART if TSTART is not None else t / 2       # cua so phan tich (khop voi cua so profile)
    a = np.array(prod, float); a = a[a[:, 0] * dt >= t0]; ta = a[:, 0] * dt
    Jh2 = np.polyfit(ta, a[:, 1], 1)[0]; Jc2 = np.polyfit(ta, a[:, 2], 1)[0]
    # sai so J_tb: block 30 ps (block 10 ps bi tuong quan am qua dau mut chung -> uoc luong sai so qua cao)
    edges = np.arange(ta[0], ta[-1] + 1e-9, 30.0); jb = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        i0, i1 = np.argmin(abs(ta - e0)), np.argmin(abs(ta - e1))
        if i1 > i0: jb.append(((a[i1, 2] - a[i0, 2]) - (a[i1, 1] - a[i0, 1])) / 2 / (ta[i1] - ta[i0]))
    sJ = np.std(jb, ddof=1) / np.sqrt(len(jb)) if len(jb) > 1 else np.nan
    return h / t, c / t, kv["Lz"], kv["A"], Jh2, Jc2, kv["zlo"], kv.get("Lz_print", np.nan), dt, sJ, kv.get("Thot")        # quy uoc: + = nang luong bi rut khoi nguyen tu

def parse_profile(path, Lz, zlo=0.0, tmin_step=None):
    blocks, cur, ts = [], [], None
    for line in open(path):
        if line.startswith("#"): continue
        p = line.split()
        if len(p) == 3:
            if cur: blocks.append((ts, cur))
            ts, cur = int(p[0]), []
        elif len(p) == 4: cur.append((zlo + float(p[1]) * Lz, float(p[3])))   # Coord1 reduced = phan so cua HOP
    if cur: blocks.append((ts, cur))
    # file co the chua nhieu lan chay noi tiep (timestep reset) -> chi lay doan cuoi
    start = max([i for i in range(1, len(blocks)) if blocks[i][0] <= blocks[i-1][0]], default=0)
    bl = blocks[start:]
    n = collections.Counter(len(c) for _, c in bl).most_common(1)[0][0]
    bad = sum(len(c) != n for _, c in bl)
    bl = [(ts, np.array(c)) for ts, c in bl if len(c) == n]
    if tmin_step is not None: return [c for ts, c in bl if ts >= tmin_step], start, bad
    return [c for _, c in bl][len(bl) // 2:], start, bad

def deltaT(z, T):
    L = (z > Z_INT - FH) & (z < Z_INT - EX); R = (z < Z_INT + FH) & (z > Z_INT + EX)
    return np.polyval(np.polyfit(z[L], T[L], 1), Z_INT) - np.polyval(np.polyfit(z[R], T[R], 1), Z_INT)

TSTART = None                                       # --tstart X: phan tich tu t_prod >= X ps (J va profile)
if "--tstart" in sys.argv:
    i = sys.argv.index("--tstart"); TSTART = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
LEGACY = "--legacy" in sys.argv                     # tai tao dung quy doi z cua script cu (Lz in o cuoi log)
sys.argv = [a for a in sys.argv if a != "--legacy"]
if len(sys.argv) < 2: sys.exit("Usage: python3 recompute_G.py [--legacy] <root1> [<root2> ...]")
TAG = re.compile(r"_(\d+)K_(Ga|N|P)_([\d.]+)keV")

def find_profile(log, root, T0, sp, E):
    p = glob.glob(os.path.join(os.path.dirname(log), "profile_T_*.dat"))
    if p: return p[0]
    for f in glob.glob(os.path.join(root, "**", f"profile_T_{T0}K_{sp}_*keV.dat"), recursive=True):
        m = re.search(r"_([\d.]+)keV\.dat$", f)
        if m and abs(float(m.group(1)) - E) < 1e-6: return f      # Seed1: Logs/ + Raw_Data/
    return None

rows, skipped, notes = [], [], []
for root in sys.argv[1:]:
    logs = glob.glob(os.path.join(root, "**", "log.04_nemd*"), recursive=True)
    newest = {}                                    # nhieu log cung tag & cung thu muc profile -> lay log moi nhat
    for log in logs:
        m = TAG.search(os.path.basename(log)) or TAG.search(log)
        if not m: skipped.append((log, "no tag")); continue
        sm = re.search(r"Seed(\d+)_", log); seed = int(sm.group(1)) if sm else 1
        key = (seed, m.group(0), os.path.dirname(log))
        if key not in newest or os.path.getmtime(log) > os.path.getmtime(newest[key][0]):
            newest[key] = (log, m, seed)
    for log, m, seed in newest.values():
        T0, sp, E = int(m.group(1)), m.group(2), float(m.group(3))
        prof = find_profile(log, root, T0, sp, E)
        if not prof: skipped.append((log, "no profile")); continue
        try:
            Jh, Jc, Lz, A, Jh2, Jc2, zlo, Lzp, dt, sJ, Thot = parse_log(log)
            bl, seg0, nbad = parse_profile(prof, Lz if not LEGACY else Lzp, zlo if not LEGACY else 0.0,
                                           None if TSTART is None else round((TSTART + 10) / dt))
            if seg0 or nbad: notes.append((log, f"profile: bo {seg0} block cua lan chay truoc, {nbad} block lech so chunk"))
            zm, Tm = bl[0][:, 0], np.mean([b[:, 1] for b in bl], axis=0)
            dT = deltaT(zm, Tm)
            dT_blk = [deltaT(b[:, 0], b[:, 1]) for b in bl]
            sdT = np.std(dT_blk, ddof=1) / np.sqrt(len(dT_blk))
            q = lambda J: J * CONV / (A * 1e-20) / dT / 1e6
            # do nhay theo vung loai tru (FH = 12 A, khong chong vung nong) + do dan ca vung G_reg
            ex0, fh0 = EX, FH; Jtb = (Jc2 - Jh2) / 2; gex = []
            for e in (2, 3, 4, 5, 6):
                EX, FH = e, 12.0; gex.append(Jtb * CONV / (A * 1e-20) / deltaT(zm, Tm) / 1e6)
            EX, FH = ex0, fh0
            T1 = np.interp(zlo + 27.0, zm, Tm); ms = (zm > Z_INT + 3) & (zm < Z_INT + 25)
            T2 = np.polyval(np.polyfit(zm[ms], Tm[ms], 1), Z_INT + 15)
            G_reg = Jtb * CONV / (A * 1e-20) / (T1 - T2) / 1e6
            gan = (zm > 25.2) & (zm < Z_INT)
            sic = (zm > Z_INT + EX) & (zm < Z_INT + 40)
            sic_peak = Tm[sic].max() - Tm[sic][0]          # >0: SiC co diem nong hon sat interface -> nguon lan sang SiC
            qc = (dT > 0) and (dT / sdT >= 3) and (sic_peak < 1.0)
            rows.append(dict(path=log, profile=prof, seed=seed, T=T0, species=sp, E=E, Jh=Jh, Jc=Jc, S=Jh + Jc,
                             dT=dT, sdT=sdT, G_old=abs(q((abs(Jh) + abs(Jc)) / 2)),   # script cu dung |DeltaT|
                             G_corr=q(Jc), Jh2=Jh2, Jc2=Jc2, G_c2=q(Jc2), G_bal=q((Jc2 - Jh2) / 2), sG_bal=abs(q((Jc2 - Jh2) / 2)) * np.hypot(sJ / ((Jc2 - Jh2) / 2), sdT / dT), S_win=Jh2 + Jc2, nblk=len(bl), eta2=abs(Jh2 + Jc2) / (abs(Jc2 - Jh2) / 2), TmaxGaN_minus_Thot=Tm[gan].max() - (Thot if Thot else T0 + 10), dTset=2 * ((Thot if Thot else T0 + 10) - T0), sic_peak=sic_peak, QC=int(qc),
                         **{f'G_EX{e}': g for e, g in zip((2, 3, 4, 5, 6), gex)},
                         EXsens=(max(gex) - min(gex)) / np.mean(gex), dT_reg=T1 - T2, G_reg=G_reg))
        except Exception as e:
            skipped.append((log, repr(e))); continue
for l, why in skipped: print("SKIP", why, l)
for l, why in notes: print("NOTE", why, l)
if not rows: sys.exit("Khong co run nao hop le")
with open("recompute_G.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

g = collections.defaultdict(list)
for r in rows: g[(r["E"], r["T"], r["species"])].append(r)
print(f"{len(rows)} runs, QC pass {sum(r['QC'] for r in rows)}")
print("  E    T  sp   n  nQC  G_old_mean  G_corr_med(all)  G_corr_med(QC) [min-max QC]")
for k in sorted(g):
    a = g[k]; ok = [x["G_corr"] for x in a if x["QC"]]
    rng = f"[{min(ok):6.1f}-{max(ok):6.1f}]" if ok else ""
    print(f"{k[0]:4.1f} {k[1]} {k[2]:2s} {len(a):3d} {len(ok):3d}  {np.mean([x['G_old'] for x in a]):9.1f}  "
          f"{np.median([x['G_corr'] for x in a]):12.1f}  {np.median(ok) if ok else float('nan'):12.1f} {rng}")

new = [r for r in rows if r.get("G_reg") is not None and "dt025" in r["path"]]
if new:
    print("\nRun dt=0.25 fs: G_bal theo EX (FH=12 A), do nhay, G_reg")
    print(f"{'run':34s} " + " ".join(f"EX{e:>5d}" for e in (2, 3, 4, 5, 6)) + "   sens   dT_reg   G_reg   S_win")
    for r in sorted(new, key=lambda x: x["path"]):
        print(f"{r['path'].split('/')[-2][:34]:34s} " + " ".join(f"{r[f'G_EX{e}']:7.1f}" for e in (2, 3, 4, 5, 6))
              + f"  {r['EXsens']:5.0%}  {r['dT_reg']:6.2f}  {r['G_reg']:6.1f}  {r['S_win']:+.2f}")

