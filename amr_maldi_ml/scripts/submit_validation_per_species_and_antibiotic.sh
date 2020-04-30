#!/usr/bin/env bash
#
# The purpose of this script is to submit validation computation jobs to
# an LSF system in order to speed up job processing. This array of jobs,
# even though meant for a different purpose, is similar to the baseline,
# as the computation is repeated for numerous combinations. In contrast
# to the other validation script, this one uses species and antibiotics
# combinations.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../validation_per_species_and_antibiotic.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "validation_per_species_and_antibiotic_%J.out" -R "rusage[mem=32000]"'
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

for SEED in 344 172 188 270 35 164 545 480 89 409; do
  for TRAIN in "DRIAMS-A" "DRIAMS-B" "DRIAMS-C" "DRIAMS-D"; do
    # Switching the order for the submission around because we are
    # primarily interested in obtaining information about unknown
    # sites quickly.
    for TEST in "DRIAMS-B" "DRIAMS-C" "DRIAMS-D" "DRIAMS-A"; do

      # E. coli and K. pneu
      for ANTIBIOTIC in "Ceftriaxone" "Ciprofloxacin"; do
        for SPECIES in "Escherichia coli" "Klebsiella pneumoniae"; do
            CMD="${MAIN} --train-site $TRAIN --test-site $TEST --antibiotic \"$ANTIBIOTIC\" --model $MODEL --seed $SEED"
            run "$CMD";
        done
      done
    done # test
  done # train
done # seed
