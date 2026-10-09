# GaN/SiC cascade-NEMD: code and data

Code and data for the manuscript "Spurious enhancement of thermal boundary
conductance in cascade-NEMD simulations of GaN/SiC caused by damage-amplified
integration error" (X.-H. Nguyen, T.-T. Nguyen, C.-T.-P. Le; submitted to
Computational Materials Science, COMMAT-D-26-04311).

Code is released under the MIT License (LICENSE). Data are released under
CC BY 4.0. Some code comments are in Vietnamese.

## Contents

inputs/      LAMMPS inputs and potentials for the full pipeline
  build_GaN_SiC_interface.py     builds the GaN(0001)/3C-SiC(111) cell (ASE)
  GaN.tersoff, SiC.tersoff       Tersoff potentials (Nord et al. 2003; Tersoff 1989)
  LJ_interface_crossterms_UFF.md cross-interface LJ parameters (UFF)
  in.01_minimize_equilibrate.lammps
  in.02_pka_cascade.lammps, elstop_GaNSiC.txt   cascade + electronic stopping
  in.03_anneal_cooldown.lammps
  in.04_nemd_thermal.lammps      original NEMD protocol (dt = 1 fs)
  in.04_nemd_dt.lammps           corrected NEMD protocol (dt = 0.25 fs)
  run_pairs.sh                   driver for the corrected runs

analysis/    post-processing and plotting
  recompute_G.py        reservoir rates, plane-based and region conductance
  bootstrap_EX.py       block bootstrap of plane-based G and G_reg
  region_resistance.py  region temperature drop and conductance
  window_sensitivity.py fit-window sensitivity (Table S1)
  cutoff_pairs.py       atom pairs in the Tersoff cutoff transition zone
  cutoff_vs_S.py        correlation of source strength with cutoff-zone pairs
  plot_paper_figs.py    Figs. 2-7 and S1
  plot_fig1.py          Fig. 1 (schematic, no data needed)

data/
  summary/recompute_G_original165.csv   per-run results, 165 original runs
  summary/recompute_G_corrected.csv     per-run results, 6 corrected runs
  summary/cutoff_vs_S.csv               cutoff-zone pair counts and S, 155 runs
  corrected_runs/   logs, temperature profiles and structures of the
                    6 corrected runs (3 pristine, 3 x 10 keV N-PKA)
  structures_1keV/  post-anneal structures used for cutoff-zone pair counts
  nve_drift/        thermostat-free NVE tests (inputs, structures, logs)

## Requirements

LAMMPS with the MANYBODY and EXTRA-FIX packages (fix electron/stopping).
Python 3 with numpy and matplotlib; ASE for the structure builder.

## Reproducing the results

Corrected-run summary (Table 3):
  cd data/corrected_runs
  python3 ../../analysis/recompute_G.py --tstart 50 .
This writes recompute_G.csv with paths relative to this folder; run the
commands below from the same folder.

Block bootstrap (Section 3.6):
  python3 ../../analysis/bootstrap_EX.py recompute_G.csv ../../analysis/recompute_G.py 1000

Figures (run from data/corrected_runs):
  Fig. 1:  python3 ../../analysis/plot_fig1.py
  Fig. 2:  python3 ../../analysis/plot_paper_figs.py fig2 ../summary/recompute_G_original165.csv recompute_G.csv
  Fig. 5:  python3 ../../analysis/plot_paper_figs.py fig4 ../nve_drift ../summary/cutoff_vs_S.csv
  Fig. 6:  python3 ../../analysis/plot_paper_figs.py fig5 ../summary/recompute_G_original165.csv recompute_G.csv
  Fig. 7:  python3 ../../analysis/plot_paper_figs.py fig7 recompute_G.csv ../../analysis/recompute_G.py
  Fig. S1: python3 ../../analysis/plot_paper_figs.py si3 ../summary/recompute_G_original165.csv
Figs. 3 and 4 take explicit log and profile paths; see the usage notes at the
top of plot_paper_figs.py.

Cutoff-zone pair counts (Table 2, Fig. 5c):
  python3 ../../analysis/cutoff_pairs.py ../../inputs/GaN.tersoff \
    Seed1_pristine_300K_P_0.0keV/data.equil_300K.data \
    ../structures_1keV/Seed*/data.postanneal_*.data \
    Seed*_300K_N_10.0keV/data.postanneal_*.data

NVE drift test: see data/nve_drift/run_nve.sh and drift.sh.

## Notes

- The path columns in recompute_G_original165.csv refer to the authors' local
  storage and are kept for provenance only.
- Electronic stopping uses the SRIM-derived Si-in-Si table for all species
  (see the manuscript, Section 2.2); PKA energies are nominal.

## Original 165 runs (raw data, archived separately)

Logs, temperature profiles and post-anneal structures of the 165 original
runs are archived at Zenodo: <Zenodo DOI>
(GaN_SiC_original165_raw.tar.gz, 495 files).

From the repository root:
  tar -xzf GaN_SiC_original165_raw.tar.gz
  python3 analysis/recompute_G.py 02_WSL2_KOKKOS_CUDA 03_KOKKOS_Ensemble_Seeds
      -> recompute_G.csv (Fig. 2, Fig. S1, Table 2)
  python3 analysis/window_sensitivity.py recompute_G.csv analysis/recompute_G.py
      -> Table S1
  python3 analysis/cutoff_vs_S.py recompute_G.csv analysis/cutoff_pairs.py inputs/GaN.tersoff
      -> cutoff_vs_S.csv (Fig. 5b, n = 165); the seed 1 structures at 400 K
         are stored under Raw_Data/350K/ (file names carry the correct 400K label)
Adding --legacy to the recompute_G.py call reproduces the original z
conversion (atomic extent instead of box length) used for Table S2.
