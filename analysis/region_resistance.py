#!/usr/bin/env python3
"""
Compute the region-based thermal conductance G_reg for each run.

G_reg is defined as J_tb / (A * deltaT_reg), where deltaT_reg = T(z1) - T(z2).
  z1 = zlo + 27 A  (2 A inside the damage-free GaN, just beyond the hot-reservoir edge)
  z2 = Z_INT + 15 A (SiC side, outside the damaged region, estimated by linear extrapolation)

This measure is insensitive to the precise location of the interface plane and
includes the thermal resistance of the entire damaged layer rather than just
the Kapitza resistance at the nominal interface position.

Input
-----
    recompute_G.csv   : summary CSV produced by recompute_G.py
    recompute_G.py    : imported to reuse parse_log() and parse_profile()
    tstart            : analysis start time in ps (default 50 ps)

Output
------
    Printed table: T(z1), T(z2), deltaT_reg, J_tb, G_reg, G_bal per run.

Usage
-----
    python3 region_resistance.py recompute_G.csv recompute_G.py [tstart=50]
"""

import sys, csv
import numpy as np

ns = {"__name__": "lib"}
exec(open(sys.argv[2]).read().split("TSTART = None")[0], ns)

t0       = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
ns["TSTART"] = t0

CONV = 1.602176634e-7  # eV ps^-1 -> W
ZI   = 38.126          # interface z coordinate (Angstrom)

print(
    f"{'run':32s} {'T(z1)':>7s} {'T(z2)':>7s} "
    f"{'dT_reg':>7s} {'J_tb':>6s} {'G_reg':>7s} {'G_bal':>7s}"
)

for r in csv.DictReader(open(sys.argv[1])):
    o        = ns["parse_log"](r["path"])
    Lz, A, zlo, dt = o[2], o[3], o[6], o[8]
    bl       = ns["parse_profile"](r["profile"], Lz, zlo, round((t0 + 10) / dt))[0]

    z, T = bl[0][:, 0], np.mean([b[:, 1] for b in bl], axis=0)

    # z1: 2 A inside the GaN free region (hot-reservoir right edge is at zlo + 25 A)
    z1 = zlo + 25.0 + 2.0
    T1 = np.interp(z1, z, T)

    # z2: extrapolate the SiC temperature profile to ZI + 15 A
    m  = (z > ZI + 3) & (z < ZI + 25)
    T2 = np.polyval(np.polyfit(z[m], T[m], 1), ZI + 15)

    J  = (float(r["Jc2"]) - float(r["Jh2"])) / 2
    Gr = J * CONV / (A * 1e-20) / (T1 - T2) / 1e6

    print(
        f"{r['path'].split('/')[-2][:32]:32s} "
        f"{T1:7.2f} {T2:7.2f} {T1 - T2:7.2f} "
        f"{J:6.3f} {Gr:7.1f} {float(r['G_bal']):7.1f}"
    )

print(
    "\nG_reg = J_tb / (A * deltaT_reg): conductance of the region [z1, ZI + 15 A], "
    "spanning the GaN damage zone, the interface, and 15 A of SiC."
)
