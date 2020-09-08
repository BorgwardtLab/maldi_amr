#!/usr/bin/env bash
#
# The purpose of this script is to submit feature importance calculation
# jobs to an LSF system in order to speed up job processing.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../single_prediction_scores.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 3:59 -o "feature_importance_values_%J.out" -R "rusage[mem=32000]"'
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
for code in "68025e38-6ed8-4250-8fdd-555c33a35c6d_MALDI1" \
            "c9aac38c-112a-4226-bc34-e3e39d07d046_MALDI1"; do
    for FILE in $(find ../../results/fig4_curves_per_species_and_antibiotics/lr/ -name "*aureus*Oxacillin*164.json"); do
        run "${MAIN} --input $code ${FILE}"
    done
done

## K. pneumoniae jobs
#for FILE in $(find ../../results/fig4_curves_per_species_and_antibiotics/lr/ -name "*pneu*Meropenem*"); do
#  run "${MAIN} ${FILE}"
#done
#
## E. coli jobs
#for FILE in $(find ../../results/fig4_curves_per_species_and_antibiotics/lr/ -name "*coli*Ceftriaxone*"); do
#  run "${MAIN} ${FILE}"
#done
