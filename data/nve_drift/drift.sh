#!/bin/bash
# Print dE_tot/dt (eV/ps) for each NVE log in this directory (linear fit of E_tot vs. time)
cd "$(dirname "$0")"
for L in log.nve_* log.speedtest; do
  [ -f "$L" ] || continue
  awk -v f="$L" '/^ +Step/{b=1;next} b && /^ +[0-9]+ /{n++; t=$2; e=$6; st+=t; se+=e; stt+=t*t; ste+=t*e}
    /^Loop/{b=0} END{if(n>2) printf "%-50s n=%4d  dE/dt = %8.4f eV/ps\n", f, n, (n*ste-st*se)/(n*stt-st*st); else printf "%-50s insufficient data\n", f}' "$L"
done
