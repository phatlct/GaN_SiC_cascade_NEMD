#!/usr/bin/env python3
# ==============================================================================
# build_GaN_SiC_interface.py
# Builds a GaN(0001)/SiC(111) heterostructure slab (wurtzite GaN on 3C-SiC)
# with ASE, handles the lattice mismatch by coincidence-site lattice (CSL) /
# supercell matching, and writes a LAMMPS data file.
#
# Topic: effect of radiation-induced defects on phonon heat transport at the
# GaN/SiC interface in LiDAR photodetectors
# Author: Dr. Nguyen Xuan Huy - School of Engineering and Technology, NTU
# ==============================================================================
#
# Choice of SiC polytype:
#   The substrate is 3C-SiC (zinc-blende, (111) surface) cut with
#   ase.build.surface(); atomic positions are taken from ase.build rather than
#   entered by hand. Hand-derived fractional coordinates for 4H-SiC(0001)
#   (ABCB stacking, 4 layers per period) are sensitive to the internal
#   parameter u, and an incorrect u produces unphysical Si-C distances
#   (e.g. 0.628 Angstrom instead of ~1.89 Angstrom).
#   3C-SiC is also the phase for which the SiC.tersoff force field
#   (Tersoff 1989) has been validated (0.88% error in the lattice constant);
#   it has not been validated for 4H-SiC.
#   GaN(0001)/3C-SiC(111) is a genuine heterostructure and a common epitaxial
#   configuration in experimental GaN-on-SiC work; it differs from
#   4H-SiC(0001) only in the SiC polytype. The Methodology should state that
#   3C was chosen over 4H to ensure force-field reliability and geometric
#   correctness, not because it is of lower physical relevance.
#
# Reference lattice constants (experimental values):
#   GaN wurtzite: a = 3.189 Angstrom, c = 5.185 Angstrom
#   3C-SiC (zinc-blende, cubic): a = 4.3596 Angstrom
# ==============================================================================

import numpy as np
from ase import Atoms
from ase.build import bulk, surface, make_supercell
from ase.io import write

# ------------------------------------------------------------------
# 1. Material parameters
# ------------------------------------------------------------------
A_GAN, C_GAN = 3.189, 5.185       # Angstrom, wurtzite GaN
A_SIC_CUBIC = 4.3596               # Angstrom, 3C-SiC (zinc-blende, cubic)

# Number of repeated cells giving the closest ratio (simple CSL along a)
def find_csl_ratio(a1, a2, max_n=20, tol=0.03):
    """Find (m, n) such that m*a1 ~ n*a2 with mismatch < tol (e.g. 2%).
    Formula: |m*a1 - n*a2| / (n*a2) < tol
    """
    best = None
    for n in range(1, max_n + 1):
        for m in range(1, max_n + 1):
            mismatch = abs(m * a1 - n * a2) / (n * a2)
            if mismatch < tol:
                if best is None or mismatch < best[2]:
                    best = (m, n, mismatch)
    return best


def build_wurtzite(a, c, symbol_cation, symbol_anion):
    """Build one wurtzite unit cell (P6_3mc) using the gamma = 60 degree
    convention (not the more common 120 degree convention). This is
    intentional: it matches the angle returned by ase.build.surface() when
    cutting the (111) surface from the cubic SiC cell (surface() always
    returns gamma = 60).

    The gamma = 120 and gamma = 60 conventions are not related by a simple
    rescaling but by a change of basis (a2_60 = a1 + a2_120); forcing one cell
    onto the other with set_cell(scale_atoms=True) would cause severe atomic
    overlap. The fractional coordinates below were transformed from the
    standard gamma = 120 convention (1/3, 2/3, z) via u' = u - v (mod 1),
    v' = v, and give a Ga-N bond length of 1.9500 Angstrom (experiment
    ~1.95 A).
    """
    # Internal parameter u = 0.12390721 taken directly from
    # ase.build.bulk("GaN", crystalstructure="wurtzite", a=3.189, c=5.185)
    # (gives Ga-N = 1.9500 Angstrom, consistent with experiment ~1.95 A); only
    # the gamma convention is changed from 120 to 60 via the change of basis
    # u' = u - v (mod 1), v' = v.
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
    """interface_termination: species at the bottom surface of the GaN block
    (the surface in contact with SiC after stacking). "Ga" (default) or "N".

    Scientific basis: the literature (Cambridge MRS "The Polarity of GaN: a
    Critical Review"; PMC11818682, 2025) shows this is still debated
    experimentally. The general rule is "Ga-face GaN grows on C-face SiC,
    N-face GaN grows on Si-face SiC", but detailed CBED studies show that
    Ga-face GaN can still be obtained on Si-face SiC with a suitable buffer
    layer. There is no single definitive answer, so the termination must be
    controlled explicitly and reported in the Methodology rather than left
    implicit. The default here (Ga-terminated at the interface) is the common
    choice for device-grade Ga-polar GaN and should be checked against the
    specific simulation/experimental conditions before production results.
    """
    gan_unit = build_wurtzite(A_GAN, C_GAN, "Ga", "N")
    gan_slab = gan_unit.repeat((nx, ny, nz_layers))
    if interface_termination not in ("Ga", "N"):
        raise ValueError("interface_termination must be 'Ga' or 'N'")
    if interface_termination == "N":
        # Flip the z axis (z -> zmax - z) to change the bottom-most species
        pos = gan_slab.get_positions()
        pos[:, 2] = pos[:, 2].max() - pos[:, 2]
        gan_slab.set_positions(pos)
    bottom_species = gan_slab.get_chemical_symbols()[np.argmin(gan_slab.get_positions()[:, 2])]
    print(f"[POLARITY] GaN: species at bottom surface (in contact with SiC) = {bottom_species}")
    return gan_slab


def build_3c_sic_111_slab(nx, ny, n_atomic_layers, interface_termination="Si"):
    """Use ASE surface() to cut the (111) surface from the 3C-SiC
    (zinc-blende) cubic cell; atomic positions are taken from ase.build rather
    than entered by hand. Returns the slab repeated nx, ny times in-plane.

    interface_termination: species at the top surface of the SiC block (the
    surface in contact with GaN after stacking, since SiC is placed below in
    stack_interface()). "Si" (default) or "C". See the polarity note in
    build_gan_slab(); the same applies here: there is no single convention in
    the literature, so it must be controlled explicitly and reported.
    """
    sic_bulk = bulk("SiC", crystalstructure="zincblende", a=A_SIC_CUBIC, cubic=True)
    sic_slab = surface(sic_bulk, indices=(1, 1, 1), layers=n_atomic_layers, vacuum=None)
    sic_slab = sic_slab.repeat((nx, ny, 1))
    # Shift the origin to (0,0,z_min) and avoid a negative tilt that could
    # cause confusion when stacking with GaN later
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
    print(f"[POLARITY] SiC: species at top surface (in contact with GaN) = {top_species}")
    return sic_slab


def stack_interface(gan_slab, sic_slab, gap=2.0, vacuum_top=0.0):
    """Stack GaN on top of SiC along z, aligning in the xy plane by assigning
    the SiC a1, a2 vectors directly to GaN and calling set_cell(...,
    scale_atoms=True) so that ASE recomputes positions with the full linear
    transformation.

    Scaling the Cartesian x and y axes separately and only reassigning the
    diagonal elements cell[0,0]/cell[1,1] is incorrect when the two hexagonal
    cells use different gamma conventions (e.g. hand-built wurtzite GaN with
    gamma = 120 degrees versus SiC(111) from ASE surface() with gamma = 60
    degrees): it introduces a handedness mismatch between the materials
    without raising an error. set_cell(scale_atoms=True) handles any
    angle/handedness correctly because it applies the full matrix
    transformation rather than per-axis scaling.
    """
    cell_sic = sic_slab.get_cell()
    cell_gan_orig = gan_slab.get_cell()

    a_gan_inplane = np.linalg.norm(cell_gan_orig[0])
    a_sic_inplane = np.linalg.norm(cell_sic[0])
    strain_pct = (a_sic_inplane - a_gan_inplane) / a_gan_inplane * 100
    print(f"[WARNING] GaN in-plane strain when matched to SiC: {strain_pct:.2f}%")
    if abs(strain_pct) > 4.0:
        print("Strain > 4% - use a larger CSL supercell instead of direct 1:1 matching.")

    new_cell = cell_gan_orig.copy()
    new_cell[0] = cell_sic[0]
    new_cell[1] = cell_sic[1]
    gan_slab.set_cell(new_cell, scale_atoms=True)

    # ase.build.surface(..., vacuum=None) returns a cell whose z component is
    # zero (it does not reflect the actual SiC slab thickness). Using
    # cell_sic[2,2] directly as z_offset would place GaN overlapping the middle
    # of the SiC block (thousands of atom pairs < 1.0 Angstrom apart). The
    # actual thickness from atomic positions (z max - z min of the SiC slab)
    # must be used, not the cell parameter.
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

    # Final safety check: ensure there is no atomic overlap before returning
    # (fail fast instead of silently writing a corrupt data file)
    from scipy.spatial import cKDTree
    tree = cKDTree(combined.get_positions())
    dmin = tree.query(combined.get_positions(), k=2)[0][:, 1].min()
    if dmin < 1.0:
        raise RuntimeError(
            f"[STRUCTURE ERROR] Minimum interatomic distance = "
            f"{dmin:.4f} Angstrom (< 1.0 A, definite overlap). Data file not "
            f"written. Check stack_interface() before continuing."
        )
    print(f"[OK] Overlap check: minimum interatomic distance in system = {dmin:.4f} Angstrom")
    return combined


if __name__ == "__main__":
    import sys

    # Target size: 200,000-500,000 atoms.
    # Note: the default below (pilot mode) gives only ~60,000-80,000 atoms,
    # which is below the target; it is a pilot/test size, not production.
    # Pass "production" on the command line to increase the size towards the
    # target, or adjust NX/NY/NZ_GAN below.
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"

    if mode == "production":
        NX_SIC, NY_SIC = 22, 22     # brings the total to roughly 200,000-400,000 atoms
        NZ_SIC, NZ_GAN = 15, 30
    else:
        NX_SIC, NY_SIC = 10, 10     # pilot size for quick tests (~60-80k atoms)
        NZ_SIC, NZ_GAN = 15, 15
        print("[PILOT MODE] For pipeline testing only, not for production "
              "results (fewer than the 200k-500k target atoms). Rerun "
              "with: python build_GaN_SiC_interface.py production")

    # Build SiC first and measure its actual in-plane size (ase.build.surface()
    # returns a cell that is not necessarily primitive; it is ~2x larger per
    # edge than the expected primitive hexagonal cell), then compute the number
    # of GaN repeats needed to match that size. Using the same NX, NY for both
    # materials does not give equal physical dimensions and would produce a
    # spurious "strain" of up to 93%.
    # Interface polarity: see the notes in build_gan_slab()/build_3c_sic_111_slab()
    # on the scientific uncertainty (no single convention in the literature).
    # Change via the 2nd and 3rd command-line arguments if needed, e.g.
    # production Ga Si | pilot N C, etc.
    GAN_TERM = sys.argv[2] if len(sys.argv) > 2 else "Ga"
    SIC_TERM = sys.argv[3] if len(sys.argv) > 3 else "Si"

    sic = build_3c_sic_111_slab(NX_SIC, NY_SIC, n_atomic_layers=NZ_SIC,
                                  interface_termination=SIC_TERM)
    sic_a1_len = np.linalg.norm(sic.get_cell()[0])
    sic_a2_len = np.linalg.norm(sic.get_cell()[1])

    NX_GAN = max(1, round(sic_a1_len / A_GAN))
    NY_GAN = max(1, round(sic_a2_len / A_GAN))
    print(f"Actual SiC in-plane size: a1={sic_a1_len:.3f} A, a2={sic_a2_len:.3f} A")
    print(f"=> GaN repeats chosen: NX_GAN={NX_GAN}, NY_GAN={NY_GAN} "
          f"(a_GaN={A_GAN} A) to match physical dimensions")

    csl = find_csl_ratio(NX_GAN * A_GAN, sic_a1_len)
    print(f"Size match check (m=1 x GaN_block ~ n=1 x SiC_block): {csl}")

    gan = build_gan_slab(NX_GAN, NY_GAN, NZ_GAN, interface_termination=GAN_TERM)

    interface = stack_interface(gan, sic, gap=2.0, vacuum_top=0.0)

    print(f"Total number of atoms: {len(interface)}")
    print(f"Cell: {interface.get_cell()}")

    # Warning on system thickness along z relative to the PKA range (cascade
    # containment). Rough estimate based on a common rule of thumb from the
    # cascade literature (~1-2 nm range per keV of PKA energy in solids of
    # average density); it does not replace an actual SRIM calculation. This
    # is only a preliminary warning and should be checked against the actual
    # dump data after running the cascade.
    z_thickness = interface.get_positions()[:, 2].max() - interface.get_positions()[:, 2].min()
    E_pka_max_keV = 10.0
    est_range_A = E_pka_max_keV * 15.0   # rough estimate: ~1.5 nm/keV
    print(f"[CONTAINMENT WARNING] System thickness along z = {z_thickness:.1f} A. "
          f"Rough PKA range estimate for {E_pka_max_keV} keV ~ {est_range_A:.0f} A "
          f"(rule of thumb, does not replace SRIM).")
    if est_range_A > z_thickness / 2:
        print("  => May not be thick enough - high-energy PKAs may reach "
              "the boundary before fully dissipating their energy. Check "
              "dump.cascade_*.lammpstrj after the run, or increase NZ_GAN/NZ_SIC.")

    # Write LAMMPS data file (atom_style atomic - suitable for Tersoff/Tersoff/LJ)
    write(
        "GaN_SiC_interface.data",
        interface,
        format="lammps-data",
        atom_style="atomic",
        specorder=["Ga", "N", "Si", "C"],
    )
    print("Written: GaN_SiC_interface.data")
    print("Next steps (required before production):")
    print("  1. Open the file in OVITO/VMD and visually inspect the interface")
    print("  2. Run minimization and check the GaN and SiC lattice constants separately")
    print("  3. Check that energy/atom is reasonable (no atomic overlap)")
