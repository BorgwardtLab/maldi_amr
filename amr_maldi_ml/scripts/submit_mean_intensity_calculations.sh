#!/usr/bin/env bash
#
# The purpose of this script is to submit mean intensity calculation
# jobs to an LSF system in order to speed up job processing. This is
# the sibling script to the feature importance value calculations.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../mean_intensities.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 3:59 -o "mean_intensities_%J.out" -R "rusage[mem=32000]"'
fi

# Evaluates its first argument either by submitting a job, or by
# executing the command without parallel processing.
run() {
  if [ -z "$BSUB" ]; then
    eval "$1";
  else
    echo "foo $1";
    eval "${BSUB} $1";
  fi
}

# S. aureus jobs
FILES=$(find ../../results/fig4_curves_per_species_and_antibiotics/lr/ -name "*aureus*Oxacillin*") | tr "\n" " "
run "${MAIN} ${FILES}"

# K. pneumoniae jobs
FILES=$(find ../../results/fig4_curves_per_species_and_antibiotics/lr/ -name "*pneu*Meropenem*") | tr "\n" " " 
run "${MAIN} ${FILES}"

# E. coli jobs
FILES=$(find ../../results/fig4_curves_per_species_and_antibiotics/lr/ -name "*coli*Ceftriaxone*") | tr "\n" " "
run "${MAIN} ${FILES}"
