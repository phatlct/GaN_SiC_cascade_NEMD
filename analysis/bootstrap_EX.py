#!/usr/bin/env python3
"""
Block bootstrap to assess whether G depends systematically on the exclusion-zone
width EX used in the plane-extrapolation method.

For each corrected run (dt = 0.25 fs), G is computed at EX = 2, 3, 4, 5, 6 A
and the relative slope dG/dEX (%/A) is reported together with a 95 % bootstrap CI.
The bootstrap resamples the 10 ps profile blocks with replacement, treating each
block as independent.  The reported CI is therefore conservative if consecutive
blocks are positively correlated (which would understate the effective sample size).

Input
-----
    recompute_G.csv   : summary CSV produced by recompute_G.py (--tstart 50)
    recompute_G.py    : imported to reuse parse_log() and parse_profile()
    B                 : number of bootstrap replicates (default 1000)

Output
------
    Printed table per run: G(EX=2..6) with 95 % CI, dG/dEX with 95 % CI, G_reg with 95 % CI.

Usage
-----
    python3 bootstrap_EX.py recompute_G.csv recompute_G.py [B=1000]
"""

import sys, csv
import numpy as np

# Import helper functions from recompute_G.py (stop before the TSTART assignment)
ns = {"__name__": "lib"}
exec(open(sys.argv[2]).read().split("TSTART = None")[0], ns)
ns["TSTART"] = 50.0

B    = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
CONV = 1.602176634e-7        # eV ps^-1 -> W
ZI   = 38.126                # GaN/SiC interface z coordinate (Angstrom)
EXS  = np.array([2, 3, 4, 5, 6], float)

rng = np.random.default_rng(0)


def metrics(z, T, Jtb, A, zlo):
    """
    Compute G at each exclusion-zone width in EXS, the relative slope dG/dEX,
    the observed sensitivity range, and G_reg.

    Parameters
    ----------
    z, T  : z coordinates and mean temperature profile
    Jtb   : time-averaged net heat flux (eV ps^-1)
    A     : cross-sectional area (Angstrom^2)
    zlo   : lower z boundary of the simulation box (Angstrom)

    Returns
    -------
    (g, slope, sensitivity, G_reg)
        g           : G values at EX = 2, 3, 4, 5, 6 (MW m^-2 K^-1)
        slope       : linear slope dG/dEX normalised by mean G (% per Angstrom)
        sensitivity : (G_max - G_min) / G_mean
        G_reg       : region-based conductance (MW m^-2 K^-1)
    """
    g = []
    for e in EXS:
        ns["EX"], ns["FH"] = e, 12.0
        g.append(Jtb * CONV / (A * 1e-20) / ns["deltaT"](z, T) / 1e6)
    g     = np.array(g)
    slope = np.polyfit(EXS, g, 1)[0] / g.mean() * 100  # % per Angstrom

    # Region-based deltaT: T at z1 = zlo+27 A minus linear extrapolation at ZI+15 A
    T1  = np.interp(zlo + 27.0, z, T)
    m   = (z > ZI + 3) & (z < ZI + 25)
    T2  = np.polyval(np.polyfit(z[m], T[m], 1), ZI + 15)
    G_reg = Jtb * CONV / (A * 1e-20) / (T1 - T2) / 1e6

    return g, slope, (g.max() - g.min()) / g.mean(), G_reg


ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for r in csv.DictReader(open(sys.argv[1])):
    if "dt025" not in r["path"]:
        continue

    o       = ns["parse_log"](r["path"])
    Lz, A, zlo, dt = o[2], o[3], o[6], o[8]
    bl      = ns["parse_profile"](r["profile"], Lz, zlo, round(60 / dt))[0]

    if len(bl) < 5:
        print(r["path"].split("/")[-2], f"-> only {len(bl)} blocks, skipped")
        continue

    z   = bl[0][:, 0]
    Tb  = np.array([b[:, 1] for b in bl])
    Jtb = (float(r["Jc2"]) - float(r["Jh2"])) / 2

    g0, s0, sens0, gr0 = metrics(z, Tb.mean(0), Jtb, A, zlo)

    bs  = [
        metrics(z, Tb[rng.integers(0, len(bl), len(bl))].mean(0), Jtb, A, zlo)
        for _ in range(B)
    ]
    G   = np.array([b[0] for b in bs])
    S   = np.array([b[1] for b in bs])
    SE  = np.array([b[2] for b in bs])
    GR  = np.array([b[3] for b in bs])

    lo, hi = ci(S)
    sig = (
        "YES (CI excludes zero)"
        if lo > 0 or hi < 0
        else "NO  (CI includes zero)"
    )

    print(f"\n== {r['path'].split('/')[-2]}  (n_blocks={len(bl)}, B={B})")
    print(
        "   G(EX):   "
        + "  ".join(
            f"EX{int(e)}={g:6.1f} [{np.percentile(G[:, i], 2.5):5.1f},{np.percentile(G[:, i], 97.5):5.1f}]"
            for i, (e, g) in enumerate(zip(EXS, g0))
        )
    )
    print(
        f"   dG/dEX = {s0:+5.1f} %/A  "
        f"CI95 [{lo:+5.1f}, {hi:+5.1f}]  "
        f"-> systematic EX dependence: {sig}"
    )
    print(
        f"   observed sensitivity = {sens0:4.0%};  "
        f"bootstrap median = {np.median(SE):4.0%}  "
        f"[{ci(SE)[0]:4.0%}, {ci(SE)[1]:4.0%}]"
    )
    print(
        f"   G_reg = {gr0:6.1f}  "
        f"CI95 (profile only) [{ci(GR)[0]:5.1f}, {ci(GR)[1]:5.1f}]  "
        f"(flux uncertainty not included)"
    )

print(
    "\nNote: bootstrap treats 10 ps profile blocks as independent. "
    "Positive inter-block autocorrelation would make the CI narrower than the true interval."
)
