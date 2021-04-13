#!/bin/bash

for STATION in "Blood" "DeepTissue" "Urine"; do
  echo
  echo Station: $STATION
  poetry run python collect_results.py -T ../results/resample_per_workstation/lr/*_only_${STATION}.json

  echo
  echo Resampled without "HospitalHygiene"
  poetry run python collect_results.py -T ../results/resample_per_workstation/lr/*_resample_${STATION}.json
done

