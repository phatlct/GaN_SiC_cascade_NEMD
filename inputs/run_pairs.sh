#!/bin/bash
# Cap bo sung pristine + N-10keV (Seed2, Seed3), dung protocol Seed1: dt 0.25 fs, dT 20 K, 50+200 ps.
NP=${NP:-4}
cd "$(dirname "$0")"
for s in 2 3; do
  for spec in "pristine_300K_P_0.0keV|P|0.0|data.equil_300K.data|$((100*s))" \
              "300K_N_10.0keV|N|10.0|data.postanneal_300K_N_10.0keV.data|0"; do
    IFS='|' read -r tag sp E df ls <<< "$spec"
    d="Seed${s}_${tag}"; log="log.04_nemd_300K_${sp}_${E}keV_dt025"
    grep -qs "KET QUA NEMD" "$d/$log" && { echo "skip $d"; continue; }
    echo "$(date '+%F %H:%M') start $d" >> progress.txt
    (cd "$d" && mpirun -np $NP lmp_mpi -in ../in.04_nemd_dt.lammps -var T_equil 300 -var pka_species $sp -var E_pka_keV $E \
       -var datafile $df -var lseed $ls -var t_stab 50 -var t_prod 200 -log $log > /dev/null)
    grep -qs "KET QUA NEMD" "$d/$log" && echo "$(date '+%F %H:%M') end   $d" >> progress.txt || echo "$(date '+%F %H:%M') LOI   $d" >> progress.txt
  done
done
echo "$(date '+%F %H:%M') DONE pairs" >> progress.txt
