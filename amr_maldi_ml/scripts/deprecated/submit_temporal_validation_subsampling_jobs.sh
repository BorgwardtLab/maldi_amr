#!/usr/bin/env bash
#
# The purpose of this script is to submit validation computation jobs to
# an LSF system in order to speed up job processing. This array of jobs,
# even though meant for a different purpose, is similar to the baseline,
# as the computation is repeated for numerous combinations. In contrast
# to the other validation script, this one uses species and antibiotics
# combinations as well as subsampling.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../temporal_validation_subsampling.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "temporal_validation_subsampling_%J.out" -R "rusage[mem=32000]"'
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

SITE="DRIAMS-A"

submit_all_jobs() {
  for SEED in 344 172 188 270 35 164 545 480 89 409; do
    for TRAIN in "2016" "2017" "2018" "2016 2017" "2017 2018" "2016 2017 2018"; do
      # Models are ordered by their 'utility' for the project. We are
      # most interested in logistic regression.
      for MODEL in "lr" "lightgbm"; do
        for SPECIES in "Escherichia coli" "Klebsiella pneumoniae"; do
            CMD="${MAIN} --train-years $TRAIN --test-year 2018 --site $SITE --species \"$SPECIES\" --antibiotic \"Ceftriaxone\" --model $MODEL --seed $SEED $1"
            run "$CMD";
        done # species

        CMD="${MAIN} --train-years $TRAIN --test-year 2018 --site $SITE --species \"Staphylococcus aureus\" --antibiotic \"Oxacillin\" --model $MODEL --seed $SEED $1"
        run "$CMD";

      done # models
    done # train
  done # seed
}

submit_all_jobs
submit_all_jobs "--resample"
