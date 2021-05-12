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
  BSUB='bsub -W 23:59 -o "calibration_%J.out" -R "rusage[mem=32000]"'
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

for MODEL in "lr" "lightgbm" "mlp"; do 
  for FILE in $(find ../../results/curves_per_species_and_antibiotics_case_based_stratification/$MODEL/ -name "*.json" ); do
    run "${MAIN} ${FILE}"
  done
done
