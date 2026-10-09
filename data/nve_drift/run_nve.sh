#!/bin/bash
# Chay trong thu muc 08_NVE_Drift. 2 ps NVE moi run.
cd "$(dirname "$0")"
for f in data.equil_300K.data data.postanneal_300K_Ga_1.0keV.data data.postanneal_300K_N_10.0keV.data; do
  for pair in "0.001 2000" "0.0005 4000"; do
    set -- $pair
    echo "$(date +%H:%M) start $f dt=$1"
    lmp_serial -in in.05_nve_drift.lammps -var datafile "$f" -var dt "$1" -var nsteps "$2" \
      -log "log.nve_${f%.data}_dt$1" > /dev/null
  done
done
echo "$(date +%H:%M) start 10keV dt=0.00025"
lmp_serial -in in.05_nve_drift.lammps -var datafile data.postanneal_300K_N_10.0keV.data \
  -var dt 0.00025 -var nsteps 8000 -log log.nve_data.postanneal_300K_N_10.0keV_dt0.00025 > /dev/null
echo "$(date +%H:%M) DONE"
