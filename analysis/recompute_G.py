#!/usr/bin/env python3
"""
Recompute thermal boundary conductance G from raw NEMD log and temperature-profile files.

For each run directory the script locates the LAMMPS log (log.04_nemd_*) and the
corresponding temperature profile (profile_T_*.dat), fits linear regressions to the
reservoir-energy accumulation curves and to the near-interface temperature profiles,
and writes one row per run to recompute_G.csv together with a summary table grouped
by (E, T, species).

Usage
-----
    python3 recompute_G.py [--tstart X] [--legacy] <root1> [<root2> ...]

Options
-------
    --tstart X   Analyse only the production segment starting at t >= X ps
                 (applied to both energy integrals and temperature profiles).
    --legacy     Reproduce the original coordinate mapping that uses Lz printed
                 at the end of the log rather than the simulation box dimensions.
"""

import sys, re, glob, os, csv, collections
import numpy as np

Z_INT = 38.126   # GaN/SiC interface position (Angstrom)
FH    = 15.0     # half-width of the fitting window on each side of the interface
EX    = 3.0      # exclusion zone around the interface (Angstrom); atoms in [Z_INT-EX, Z_INT+EX] are not fitted
CONV  = 1.602176634e-7  # eV ps^-1 -> W

# ---------------------------------------------------------------------------
# Log-file parser
# ---------------------------------------------------------------------------

def parse_log(path):
    """
    Parse a LAMMPS log file and return the reservoir heat fluxes, box geometry,
    timestep, and related diagnostics.

    Returns
    -------
    tuple : (Jh, Jc, Lz, A, Jh2, Jc2, zlo, Lz_print, dt, sJ, Thot)
        Jh, Jc   : mean energy injection/extraction rates (eV ps^-1), from cumulative sums
        Lz, A    : box length along z (Ang) and cross-sectional area (Ang^2)
        Jh2, Jc2 : linear-regression slopes of the reservoir energy vs time (eV ps^-1)
        zlo      : lower z boundary of the simulation box (Ang)
        Lz_print : Lz as printed in the log (used only with --legacy)
        dt       : timestep (ps)
        sJ       : standard error of J_tb estimated from 30 ps blocks
        Thot     : target temperature of the hot thermostat (K)

    Sign convention: positive Jh means energy is added to the hot reservoir,
    positive Jc means energy is removed from the cold reservoir.
    """
    blk, last, kv, dt, prod = 0, None, {}, 0.001, []

    hotre = re.compile(
        r"^fix\s+fix_hot\s+g_hot\s+langevin\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+\d+"
    )
    tsre  = re.compile(r"^\s*timestep\s+([0-9.eE+-]+)\s*$")
    pat   = {"Lz_print": r"Lz \(Angstrom[^=]*=\s*([-\d.eE+]+)"}
    boxre = re.compile(
        r"(?:triclinic|orthogonal) box = "
        r"\(([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+)\) to "
        r"\(([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+)\)"
    )

    for line in open(path, errors="ignore"):
        mh = hotre.match(line)
        if mh:
            kv["Thot"] = float(mh.group(1))
        mt = tsre.match(line)
        if mt:
            dt = float(mt.group(1))  # last timestep command wins
        if re.match(r"^\s+Step\s", line):
            blk += 1
            continue
        if blk == 2 and re.match(r"^\s+\d+\s+[\d.]", line):
            p = line.split()
            last = (int(p[0]), float(p[4]), float(p[5]))
            prod.append(last)
        mb = boxre.search(line)
        if mb:
            x0, y0, z0, x1, y1, z1 = map(float, mb.groups())
            kv["zlo"] = z0
            kv["Lz"]  = z1 - z0
            kv["A"]   = (x1 - x0) * (y1 - y0)
        for k, r in pat.items():
            m = re.search(r, line)
            if m:
                kv[k] = float(m.group(1))

    s, h, c = last
    t  = s * dt                                   # total simulation time (ps)
    t0 = TSTART if TSTART is not None else t / 2  # start of analysis window

    a   = np.array(prod, float)
    a   = a[a[:, 0] * dt >= t0]
    ta  = a[:, 0] * dt

    Jh2 = np.polyfit(ta, a[:, 1], 1)[0]
    Jc2 = np.polyfit(ta, a[:, 2], 1)[0]

    # Standard error of J_tb from non-overlapping 30 ps blocks.
    # 30 ps blocks reduce the cross-boundary negative autocorrelation that
    # inflates the error estimate when 10 ps blocks are used.
    edges = np.arange(ta[0], ta[-1] + 1e-9, 30.0)
    jb = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        i0 = np.argmin(abs(ta - e0))
        i1 = np.argmin(abs(ta - e1))
        if i1 > i0:
            jb.append(
                ((a[i1, 2] - a[i0, 2]) - (a[i1, 1] - a[i0, 1])) / 2 / (ta[i1] - ta[i0])
            )
    sJ = np.std(jb, ddof=1) / np.sqrt(len(jb)) if len(jb) > 1 else np.nan

    Jh_mean = h / t
    Jc_mean = c / t

    return (
        Jh_mean, Jc_mean,
        kv["Lz"], kv["A"], Jh2, Jc2,
        kv["zlo"], kv.get("Lz_print", np.nan),
        dt, sJ, kv.get("Thot")
    )


# ---------------------------------------------------------------------------
# Temperature-profile parser
# ---------------------------------------------------------------------------

def parse_profile(path, Lz, zlo=0.0, tmin_step=None):
    """
    Read a LAMMPS ave/chunk temperature profile file.

    The file may contain several back-to-back runs (timestep counter resets);
    only the last continuous segment is used.  The function returns chunks
    with absolute z coordinates (Angstrom).

    Parameters
    ----------
    path       : path to the profile file
    Lz         : simulation box length along z (Ang)
    zlo        : lower z boundary (Ang)
    tmin_step  : if given, only blocks with timestep >= tmin_step are kept

    Returns
    -------
    (blocks, seg_start_index, n_bad_blocks)
    """
    blocks, cur, ts = [], [], None
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) == 3:
            if cur:
                blocks.append((ts, cur))
            ts, cur = int(p[0]), []
        elif len(p) == 4:
            # Coord1 is a reduced coordinate; convert to absolute z
            cur.append((zlo + float(p[1]) * Lz, float(p[3])))
    if cur:
        blocks.append((ts, cur))

    # Detect the start of the last continuous segment (timestep resets on restart)
    start = max(
        [i for i in range(1, len(blocks)) if blocks[i][0] <= blocks[i - 1][0]],
        default=0
    )
    bl = blocks[start:]
    n  = collections.Counter(len(c) for _, c in bl).most_common(1)[0][0]
    bad = sum(len(c) != n for _, c in bl)
    bl  = [(ts, np.array(c)) for ts, c in bl if len(c) == n]

    if tmin_step is not None:
        return [c for ts, c in bl if ts >= tmin_step], start, bad
    return [c for _, c in bl][len(bl) // 2:], start, bad


# ---------------------------------------------------------------------------
# Interface temperature drop
# ---------------------------------------------------------------------------

def deltaT(z, T):
    """
    Estimate the temperature discontinuity at z = Z_INT by linear extrapolation
    from the GaN side (left window) and the SiC side (right window).

    Fitting windows: [Z_INT - FH, Z_INT - EX] on the GaN side and
                     [Z_INT + EX, Z_INT + FH] on the SiC side.
    """
    L = (z > Z_INT - FH) & (z < Z_INT - EX)
    R = (z < Z_INT + FH) & (z > Z_INT + EX)
    return (
        np.polyval(np.polyfit(z[L], T[L], 1), Z_INT)
        - np.polyval(np.polyfit(z[R], T[R], 1), Z_INT)
    )


# ---------------------------------------------------------------------------
# Command-line parsing
# ---------------------------------------------------------------------------

TSTART = None  # global analysis start time; set by --tstart
if "--tstart" in sys.argv:
    i = sys.argv.index("--tstart")
    TSTART = float(sys.argv[i + 1])
    del sys.argv[i:i + 2]

LEGACY = "--legacy" in sys.argv  # reproduce original z mapping for validation
sys.argv = [a for a in sys.argv if a != "--legacy"]

if len(sys.argv) < 2:
    sys.exit("Usage: python3 recompute_G.py [--tstart X] [--legacy] <root1> [<root2> ...]")

TAG = re.compile(r"_(\d+)K_(Ga|N|P)_([\d.]+)keV")


def find_profile(log, root, T0, sp, E):
    """Locate the temperature profile file for a given log file."""
    p = glob.glob(os.path.join(os.path.dirname(log), "profile_T_*.dat"))
    if p:
        return p[0]
    for f in glob.glob(
        os.path.join(root, "**", f"profile_T_{T0}K_{sp}_*keV.dat"), recursive=True
    ):
        m = re.search(r"_([\d.]+)keV\.dat$", f)
        if m and abs(float(m.group(1)) - E) < 1e-6:
            return f
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

rows, skipped, notes = [], [], []

for root in sys.argv[1:]:
    logs = glob.glob(os.path.join(root, "**", "log.04_nemd*"), recursive=True)

    # If multiple logs share the same (tag, directory), keep the most recent one.
    newest = {}
    for log in logs:
        m = TAG.search(os.path.basename(log)) or TAG.search(log)
        if not m:
            skipped.append((log, "no tag"))
            continue
        sm   = re.search(r"Seed(\d+)_", log)
        seed = int(sm.group(1)) if sm else 1
        key  = (seed, m.group(0), os.path.dirname(log))
        if key not in newest or os.path.getmtime(log) > os.path.getmtime(newest[key][0]):
            newest[key] = (log, m, seed)

    for log, m, seed in newest.values():
        T0, sp, E = int(m.group(1)), m.group(2), float(m.group(3))
        prof = find_profile(log, root, T0, sp, E)
        if not prof:
            skipped.append((log, "no profile"))
            continue
        try:
            Jh, Jc, Lz, A, Jh2, Jc2, zlo, Lzp, dt, sJ, Thot = parse_log(log)
            bl, seg0, nbad = parse_profile(
                prof,
                Lz if not LEGACY else Lzp,
                zlo if not LEGACY else 0.0,
                None if TSTART is None else round((TSTART + 10) / dt)
            )
            if seg0 or nbad:
                notes.append((
                    log,
                    f"profile: skipped {seg0} blocks from earlier run segment; "
                    f"{nbad} blocks with unexpected chunk count"
                ))

            zm, Tm = bl[0][:, 0], np.mean([b[:, 1] for b in bl], axis=0)
            dT     = deltaT(zm, Tm)
            dT_blk = [deltaT(b[:, 0], b[:, 1]) for b in bl]
            sdT    = np.std(dT_blk, ddof=1) / np.sqrt(len(dT_blk))

            q = lambda J: J * CONV / (A * 1e-20) / dT / 1e6  # MW m^-2 K^-1

            # Sensitivity to exclusion-zone width (FH fixed at 12 A)
            ex0, fh0 = EX, FH
            Jtb = (Jc2 - Jh2) / 2
            gex = []
            for e in (2, 3, 4, 5, 6):
                EX, FH = e, 12.0
                gex.append(Jtb * CONV / (A * 1e-20) / deltaT(zm, Tm) / 1e6)
            EX, FH = ex0, fh0

            # Region-based conductance G_reg
            T1  = np.interp(zlo + 27.0, zm, Tm)
            ms  = (zm > Z_INT + 3) & (zm < Z_INT + 25)
            T2  = np.polyval(np.polyfit(zm[ms], Tm[ms], 1), Z_INT + 15)
            G_reg = Jtb * CONV / (A * 1e-20) / (T1 - T2) / 1e6

            # Quality-control flag: valid run requires positive deltaT,
            # signal-to-noise >= 3, and no spurious heat source leaking into SiC
            gan      = (zm > 25.2) & (zm < Z_INT)
            sic      = (zm > Z_INT + EX) & (zm < Z_INT + 40)
            sic_peak = Tm[sic].max() - Tm[sic][0]
            qc = (dT > 0) and (dT / sdT >= 3) and (sic_peak < 1.0)

            rows.append(dict(
                path=log, profile=prof, seed=seed, T=T0, species=sp, E=E,
                Jh=Jh, Jc=Jc, S=Jh + Jc,
                dT=dT, sdT=sdT,
                G_old=abs(q((abs(Jh) + abs(Jc)) / 2)),  # original script used |deltaT|
                G_corr=q(Jc),
                Jh2=Jh2, Jc2=Jc2,
                G_c2=q(Jc2),
                G_bal=q((Jc2 - Jh2) / 2),
                sG_bal=abs(q((Jc2 - Jh2) / 2)) * np.hypot(
                    sJ / ((Jc2 - Jh2) / 2), sdT / dT
                ),
                S_win=Jh2 + Jc2,
                nblk=len(bl),
                eta2=abs(Jh2 + Jc2) / (abs(Jc2 - Jh2) / 2),
                TmaxGaN_minus_Thot=Tm[gan].max() - (Thot if Thot else T0 + 10),
                dTset=2 * ((Thot if Thot else T0 + 10) - T0),
                sic_peak=sic_peak,
                QC=int(qc),
                **{f"G_EX{e}": g for e, g in zip((2, 3, 4, 5, 6), gex)},
                EXsens=(max(gex) - min(gex)) / np.mean(gex),
                dT_reg=T1 - T2,
                G_reg=G_reg,
            ))
        except Exception as exc:
            skipped.append((log, repr(exc)))
            continue

for l, why in skipped:
    print("SKIP", why, l)
for l, why in notes:
    print("NOTE", why, l)

if not rows:
    sys.exit("No valid runs found.")

with open("recompute_G.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# Summary table grouped by (E, T, species)
g = collections.defaultdict(list)
for r in rows:
    g[(r["E"], r["T"], r["species"])].append(r)

print(f"{len(rows)} runs processed; {sum(r['QC'] for r in rows)} passed QC")
print("  E    T  sp   n  nQC  G_old_mean  G_corr_med(all)  G_corr_med(QC) [min-max QC]")
for k in sorted(g):
    a  = g[k]
    ok = [x["G_corr"] for x in a if x["QC"]]
    rng = f"[{min(ok):6.1f}-{max(ok):6.1f}]" if ok else ""
    print(
        f"{k[0]:4.1f} {k[1]} {k[2]:2s} {len(a):3d} {len(ok):3d}"
        f"  {np.mean([x['G_old'] for x in a]):9.1f}"
        f"  {np.median([x['G_corr'] for x in a]):12.1f}"
        f"  {np.median(ok) if ok else float('nan'):12.1f} {rng}"
    )

# Detailed table for corrected runs (dt = 0.25 fs)
new = [r for r in rows if r.get("G_reg") is not None and "dt025" in r["path"]]
if new:
    print("\ndt = 0.25 fs runs — G_bal vs exclusion-zone width (FH = 12 A), G_reg")
    print(
        f"{'run':34s} "
        + " ".join(f"EX{e:>5d}" for e in (2, 3, 4, 5, 6))
        + "   sens   dT_reg   G_reg   S_win"
    )
    for r in sorted(new, key=lambda x: x["path"]):
        print(
            f"{r['path'].split('/')[-2][:34]:34s} "
            + " ".join(f"{r[f'G_EX{e}']:7.1f}" for e in (2, 3, 4, 5, 6))
            + f"  {r['EXsens']:5.0%}  {r['dT_reg']:6.2f}  {r['G_reg']:6.1f}  {r['S_win']:+.2f}"
        )
