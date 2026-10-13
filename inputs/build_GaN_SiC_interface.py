#!/usr/bin/env python3
"""
Build the GaN(0001)/3C-SiC(111) interface cell and write a LAMMPS data file.

Wurtzite GaN on zinc-blende 3C-SiC; 3C is the polytype covered by the SiC
Tersoff parameterisation. Lattice constants: GaN a = 3.189, c = 5.185 A;
3C-SiC a = 4.3596 A.

Output: GaN_SiC_interface.data (atom_style atomic, types Ga N Si C)
Usage:  python3 build_GaN_SiC_interface.py
"""

import numpy as np
from ase import Atoms
from ase.build import bulk, surface, make_supercell
from ase.io import write

A_GAN, C_GAN = 3.189, 5.185       # Angstrom, wurtzite GaN
A_SIC_CUBIC = 4.3596               # Angstrom, 3C-SiC

def find_csl_ratio(a1, a2, max_n=20, tol=0.03):
    """Return (m, n, mismatch) minimising |m*a1 - n*a2| / (n*a2) below tol."""
    best = None
    for n in range(1, max_n + 1):
        for m in range(1, max_n + 1):
            mismatch = abs(m * a1 - n * a2) / (n * a2)
            if mismatch < tol:
                if best is None or mismatch < best[2]:
                    best = (m, n, mismatch)
    return best


def build_wurtzite(a, c, symbol_cation, symbol_anion):
    """Wurtzite unit cell in the gamma = 60 deg setting, to match ase.build.surface() for SiC(111).
    Coordinates converted from the gamma = 120 setting via u' = u - v (mod 1); Ga-N = 1.950 A."""
    # u from ase.build.bulk("GaN", "wurtzite", a=3.189, c=5.185)
    u_internal = 0.12390721
    atoms = Atoms(
        symbols=[symbol_cation, symbol_anion, symbol_cation, symbol_anion],
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (2/3, 2/3, u_internal),
            (2/3, 2/3, 0.5),
            (0.0, 0.0, 0.5 + u_internal),
        ],
        cell=[a, a, c, 90, 90, 60],
        pbc=True,
    )
    return atoms


def build_gan_slab(nx, ny, nz_layers, vacuum=0.0, interface_termination="Ga"):
    """GaN slab; interface_termination ("Ga" or "N") is the bottom species facing SiC."""
    gan_unit = build_wurtzite(A_GAN, C_GAN, "Ga", "N")
    gan_slab = gan_unit.repeat((nx, ny, nz_layers))
    if interface_termination not in ("Ga", "N"):
        raise ValueError("interface_termination must be 'Ga' or 'N'")
    if interface_termination == "N":
        pos = gan_slab.get_positions()
        pos[:, 2] = pos[:, 2].max() - pos[:, 2]
        gan_slab.set_positions(pos)
    bottom_species = gan_slab.get_chemical_symbols()[np.argmin(gan_slab.get_positions()[:, 2])]
    print(f"GaN bottom (interface) species: {bottom_species}")
    return gan_slab


def build_3c_sic_111_slab(nx, ny, n_atomic_layers, interface_termination="Si"):
    """3C-SiC(111) slab cut with ase.build.surface(), repeated nx x ny in-plane.
    interface_termination ("Si" or "C") is the top species facing GaN."""
    sic_bulk = bulk("SiC", crystalstructure="zincblende", a=A_SIC_CUBIC, cubic=True)
    sic_slab = surface(sic_bulk, indices=(1, 1, 1), layers=n_atomic_layers, vacuum=None)
    sic_slab = sic_slab.repeat((nx, ny, 1))
    pos = sic_slab.get_positions()
    pos[:, 2] -= pos[:, 2].min()
    sic_slab.set_positions(pos)

    if interface_termination not in ("Si", "C"):
        raise ValueError("interface_termination must be 'Si' or 'C'")
    top_species = sic_slab.get_chemical_symbols()[np.argmax(sic_slab.get_positions()[:, 2])]
    if top_species != interface_termination:
        pos = sic_slab.get_positions()
        pos[:, 2] = pos[:, 2].max() - pos[:, 2]
        sic_slab.set_positions(pos)
        top_species = sic_slab.get_chemical_symbols()[np.argmax(sic_slab.get_positions()[:, 2])]
    print(f"SiC top (interface) species: {top_species}")
    return sic_slab


def stack_interface(gan_slab, sic_slab, gap=2.0, vacuum_top=0.0):
    """Place GaN above SiC along z; GaN takes the SiC in-plane vectors via set_cell(scale_atoms=True)."""
    cell_sic = sic_slab.get_cell()
    cell_gan_orig = gan_slab.get_cell()

    a_gan_inplane = np.linalg.norm(cell_gan_orig[0])
    a_sic_inplane = np.linalg.norm(cell_sic[0])
    strain_pct = (a_sic_inplane - a_gan_inplane) / a_gan_inplane * 100
    print(f"GaN in-plane strain: {strain_pct:.2f}%")
    if abs(strain_pct) > 4.0:
        print("Strain above 4%.")

    new_cell = cell_gan_orig.copy()
    new_cell[0] = cell_sic[0]
    new_cell[1] = cell_sic[1]
    gan_slab.set_cell(new_cell, scale_atoms=True)

    # surface(..., vacuum=None) gives zero cell height, so use the slab thickness from positions
    sic_pos_z = sic_slab.get_positions()[:, 2]
    sic_thickness = sic_pos_z.max() - sic_pos_z.min()
    z_offset = sic_thickness + gap

    gan_pos = gan_slab.get_positions()
    gan_pos[:, 2] += z_offset - gan_slab.get_positions()[:, 2].min()
    gan_slab.set_positions(gan_pos)

    combined = sic_slab + gan_slab
    total_cell = cell_sic.copy()
    total_cell[2, 2] = z_offset + cell_gan_orig[2, 2] + vacuum_top
    combined.set_cell(total_cell, scale_atoms=False)
    combined.pbc = (True, True, True if vacuum_top == 0 else False)

    from scipy.spatial import cKDTree
    tree = cKDTree(combined.get_positions())
    dmin = tree.query(combined.get_positions(), k=2)[0][:, 1].min()
    if dmin < 1.0:
        raise RuntimeError(
            f"[STRUCTURE ERROR] Minimum interatomic distance = "
            f"{dmin:.4f} Angstrom (< 1.0 A, definite overlap). Data file not "
            f"written. Check stack_interface() before continuing."
        )
    print(f"Minimum interatomic distance: {dmin:.4f} A")
    return combined


if __name__ == "__main__":
    import sys

    # optional args: size ("production" for the larger cell), GaN termination, SiC termination
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"

    if mode == "production":
        NX_SIC, NY_SIC = 22, 22
        NZ_SIC, NZ_GAN = 15, 30
    else:
        NX_SIC, NY_SIC = 10, 10
        NZ_SIC, NZ_GAN = 15, 15
        print(f"Cell: {NX_SIC}x{NY_SIC} SiC repeats (default; "
              "pass 'production' for 22x22)")

    # GaN repeats are set from the actual SiC in-plane size (surface() cell is not primitive)
    GAN_TERM = sys.argv[2] if len(sys.argv) > 2 else "Ga"
    SIC_TERM = sys.argv[3] if len(sys.argv) > 3 else "Si"

    sic = build_3c_sic_111_slab(NX_SIC, NY_SIC, n_atomic_layers=NZ_SIC,
                                  interface_termination=SIC_TERM)
    sic_a1_len = np.linalg.norm(sic.get_cell()[0])
    sic_a2_len = np.linalg.norm(sic.get_cell()[1])

    NX_GAN = max(1, round(sic_a1_len / A_GAN))
    NY_GAN = max(1, round(sic_a2_len / A_GAN))
    print(f"SiC in-plane: a1={sic_a1_len:.3f} A, a2={sic_a2_len:.3f} A")
    print(f"GaN repeats: NX_GAN={NX_GAN}, NY_GAN={NY_GAN} "
          f"(a_GaN={A_GAN} A)")

    csl = find_csl_ratio(NX_GAN * A_GAN, sic_a1_len)
    print(f"Block size ratio (m, n, mismatch): {csl}")

    gan = build_gan_slab(NX_GAN, NY_GAN, NZ_GAN, interface_termination=GAN_TERM)

    interface = stack_interface(gan, sic, gap=2.0, vacuum_top=0.0)

    print(f"Total number of atoms: {len(interface)}")
    print(f"Cell: {interface.get_cell()}")

    z_thickness = interface.get_positions()[:, 2].max() - interface.get_positions()[:, 2].min()
    print(f"z thickness: {z_thickness:.1f} A")

    write(
        "GaN_SiC_interface.data",
        interface,
        format="lammps-data",
        atom_style="atomic",
        specorder=["Ga", "N", "Si", "C"],
    )
    print("Written: GaN_SiC_interface.data")
