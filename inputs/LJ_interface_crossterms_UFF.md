# Interatomic potentials for the GaN/SiC interface

The Ga-N-Si-C system is described with three components:

1. GaN bulk (Ga-Ga, Ga-N, N-N): Tersoff potential of Nord et al.
2. SiC bulk (Si-Si, Si-C, C-C): Tersoff potential of Tersoff.
3. Cross-interface pairs (Ga-Si, Ga-C, N-Si, N-C): Lennard-Jones 12-6,
   parameterised with the Universal Force Field (UFF) combining rules.

The two Tersoff parameterisations are combined in LAMMPS as two instances of
`pair_style hybrid/overlay tersoff tersoff`, and the cross-interface pairs are
added with `lj/cut` (cutoff 10 Angstrom). Same-material LJ pairs are switched
off (epsilon = 0).

## Bulk potentials

| File | Reference | Source |
|---|---|---|
| `GaN.tersoff` | J. Nord, K. Albe, P. Erhart, K. Nordlund, *J. Phys.: Condens. Matter* 15, 5649 (2003) | LAMMPS distribution, `potentials/GaN.tersoff` |
| `SiC.tersoff` | J. Tersoff, *Phys. Rev. B* 39, 5566 (1989); erratum *Phys. Rev. B* 41, 3248 (1990) | LAMMPS distribution, `potentials/SiC.tersoff` |

## Cross-interface Lennard-Jones parameters

UFF element parameters (A. K. Rappé, C. J. Casewit, K. S. Colwell,
W. A. Goddard III, W. M. Skiff, *J. Am. Chem. Soc.* 114, 10024 (1992)):

| Element | x_I (Angstrom) | D_I (kcal/mol) |
|---|---|---|
| C  | 3.851 | 0.105 |
| N  | 3.660 | 0.069 |
| Si | 4.295 | 0.402 |
| Ga | 4.383 | 0.415 |

Combining rules (geometric mean) and conversion to LAMMPS `lj/cut`
(units metal):

    x_IJ = sqrt(x_I * x_J)
    D_IJ = sqrt(D_I * D_J)
    sigma   = x_IJ / 2^(1/6)
    epsilon = D_IJ * 0.0433641   (kcal/mol -> eV)

| Pair | x_IJ (Angstrom) | D_IJ (kcal/mol) | sigma (Angstrom) | epsilon (eV) |
|---|---|---|---|---|
| Ga-Si | 4.3388 | 0.4084 | 3.8654 | 0.01771 |
| Ga-C  | 4.1084 | 0.2087 | 3.6602 | 0.00905 |
| N-Si  | 3.9648 | 0.1665 | 3.5322 | 0.00722 |
| N-C   | 3.7543 | 0.0851 | 3.3447 | 0.00369 |

These values are used in all LAMMPS inputs (`pair_coeff 1 3`, `1 4`, `2 3`,
`2 4`).

The LAMMPS inputs multiply all cross-interface epsilon values by `lj_scale`
(default 1.0).
