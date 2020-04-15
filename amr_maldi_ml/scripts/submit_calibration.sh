#!/usr/bin/env bash
#
# The purpose of this script is to submit classifier calibration jobs to
# an LSF system in order to speed up job processing.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../calibrate_classifier.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 3:59 -o "calibration_%J.out" -R "rusage[mem=32000]"'
fi

# Evaluates its first argument either by submitting a job, or by
# executing the command without parallel processing.
run() {
  if [ -z "$BSUB" ]; then
    eval "$1";
  else
    eval "${BSUB} $1";
  fi
}

# S. aureus jobs
for FILE in $(find ../../results/fig4_curves_per_species_and_antibiotics/ -name "*aureus*Oxacillin*" -or -name "*aureus*Ciprofloxacin*" ); do
  run "${MAIN} ${FILE}"
done

# K. pneumoniae jobs
for FILE in $(find ../../results/fig4_curves_per_species_and_antibiotics/ -name "*pneu*Meropenem*" -or -name "*pneu*Ceftriaxone*" -or -name "*pneu*Tobramycin*" ); do
  run "${MAIN} ${FILE}"
done

# E. coli jobs
for FILE in $(find ../../results/fig4_curves_per_species_and_antibiotics/ -name "*coli*Ceftriaxone*" -or -name "*coli*Cefepime*"); do
  run "${MAIN} ${FILE}"
done
