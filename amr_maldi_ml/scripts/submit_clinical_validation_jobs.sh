#!/usr/bin/env bash
#
# Submits sliding window validation jobs to the cluster. These jobs
# assess to what extent performance changes if the training data set
# is moving closer (in time) towards the test data set.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../sliding_window_validation.py --duration 29 "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 47:59 -o "clinical_validation_%J.out" -R "rusage[mem=16000]"'
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
  # Models are ordered by their 'utility' for the project. We are
  # most interested in logistic regression.
  for MODEL in "lr" "lightgbm"; do
    for SPECIES in "Escherichia coli" "Klebsiella pneumoniae"; do
        CMD="${MAIN} --species \"$SPECIES\" --antibiotic \"Ceftriaxone\" --model $MODEL --seed $SEED --log-codes"
        run "$CMD";
    done # species

    CMD="${MAIN} --species \"Staphylococcus aureus\" --antibiotic \"Oxacillin\" --model $MODEL --seed $SEED --log-codes"
    run "$CMD";
  done # models
done # seed
