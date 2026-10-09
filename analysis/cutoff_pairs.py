#!/usr/bin/env python3
"""
Count atom pairs that fall within the Tersoff cutoff transition zone (R-D < r < R+D)
and within the fully-interacting region (r < R-D) for each post-anneal LAMMPS structure.

Pair types reported: Ga-Ga, N-N, Ga-N.
Periodic boundary conditions are applied in x and y; atom type 1 = Ga, type 2 = N.

Usage
-----
    python3 cutoff_pairs.py GaN.tersoff <data_file1> [<data_file2> ...]
"""

import sys, re
import numpy as np


def tersoff_RD(path):
    """
    Extract the two-body cutoff parameters (R, D) for each element pair from
    a Tersoff potential file in LAMMPS format.

    Only i-i same-species entries (where element2 == element3) are read,
    as the two-body cutoffs are defined there.

    Returns
    -------
    dict : {(element_i, element_j): (R, D)}
    """
    tok = " ".join(l.split("#")[0] for l in open(path)).split()
    out = {}
    for i in range(0, len(tok) - 16, 17):
        e = tok[i:i + 3]
        v = list(map(float, tok[i + 3:i + 17]))
        if e[1] == e[2]:
            out[(e[0], e[1])] = (v[10], v[11])  # R, D for the two-body term
    return out


def read_data(path):
    """
    Read a LAMMPS data file (atom style full) and return atom types, positions,
    and box parameters.

    Supports triclinic boxes with xy tilt factor.

    Returns
    -------
    types : int array  (N,)
    X     : float array (N, 3)  [x, y, z in Angstrom]
    box   : dict with keys 'x', 'y', 'xy'
    """
    L   = open(path).read().splitlines()
    box = {}
    for l in L[:40]:
        if "xlo" in l:
            p = l.split(); box["x"] = (float(p[0]), float(p[1]))
        if "ylo" in l:
            p = l.split(); box["y"] = (float(p[0]), float(p[1]))
        if "xy xz yz" in l:
            box["xy"] = float(l.split()[0])

    i = next(k for k, l in enumerate(L) if l.strip().startswith("Atoms")) + 2
    rows = []
    while i < len(L) and L[i].strip():
        p = L[i].split()
        rows.append((int(p[1]), float(p[2]), float(p[3]), float(p[4])))
        i += 1

    a = np.array(rows)
    return a[:, 0].astype(int), a[:, 1:], box


def pairs(Xa, Xb, box, rmax, same):
    """
    Return pairwise distances between atoms in Xa and Xb that are less than rmax,
    applying minimum-image periodic boundary conditions in x and y.

    Parameters
    ----------
    Xa, Xb : coordinate arrays (Na, 3), (Nb, 3)
    box    : dict with 'x', 'y', 'xy'
    rmax   : cutoff distance (Angstrom)
    same   : True if Xa and Xb refer to the same species (avoids double counting)

    Returns
    -------
    r : 1-D array of distances < rmax
    """
    lx = box["x"][1] - box["x"][0]
    ly = box["y"][1] - box["y"][0]
    xy = box.get("xy", 0.0)
    out = []
    for s in range(0, len(Xa), 400):
        d = Xb[None, :, :] - Xa[s:s + 400, None, :]
        # Minimum image in y, then apply xy tilt correction to x
        ny        = np.round(d[..., 1] / ly)
        d[..., 1] -= ny * ly
        d[..., 0] -= ny * xy
        d[..., 0] -= np.round(d[..., 0] / lx) * lx
        r = np.sqrt((d ** 2).sum(-1))
        if same:
            # Exclude self-pairs and count each pair once (j > i in the original array)
            idx = np.arange(len(Xb))[None, :]
            row = np.arange(s, s + len(r))[:, None]
            r   = np.where(idx > row, r, np.inf)
        out.append(r[r < rmax])
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RD    = tersoff_RD(sys.argv[1])
names = {1: "Ga", 2: "N"}

header_pairs = (("Ga", "Ga"), ("N", "N"), ("Ga", "N"))
print(
    "file".ljust(44),
    "  ".join(
        f"{a}-{b}: R-D<r<R+D | r<R-D".ljust(30)
        for a, b in header_pairs
    )
)
print("R, D:", {f"{k[0]}-{k[1]}": v for k, v in RD.items() if k in set(header_pairs)})

for f in sys.argv[2:]:
    t, X, box = read_data(f)
    cols = []
    for a, b in ((1, 1), (2, 2), (1, 2)):
        R, D = RD[(names[a], names[b])]
        r    = pairs(X[t == a], X[t == b], box, R + D, a == b)
        cols.append(
            f"{np.sum(r > R - D):8d} | {np.sum(r <= R - D):8d}".ljust(30)
        )
    print(f.split("/")[-1][:44].ljust(44), "  ".join(cols))
