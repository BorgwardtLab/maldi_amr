#!/usr/bin/env bash
#
# The purpose of this script is to submit validation computation jobs to
# an LSF system in order to speed up job processing. This array of jobs,
# even though meant for a different purpose, is similar to the baseline,
# as the computation is repeated for numerous combinations.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../validation.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "validation_%J.out" -R "rusage[mem=32000]"'
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
      for ANTIBIOTIC in 'Amikacin'\
          'Amoxicillin-Clavulanic acid'\
          'Cefepime'\
          'Ceftriaxone'\
          'Ciprofloxacin'\
          'Clindamycin'\
          'Gentamicin'\
          'Imipenem'\
          'Piperacillin-Tazobactam';
      do
        # Models are ordered by their 'utility' for the project. We are
        # most interested in logistic regression.
        for MODEL in "lr" "svm-rbf" "rf" "lightgbm" "svm-linear"; do
          CMD="${MAIN} --train-site $TRAIN --test-site $TEST --antibiotic \"$ANTIBIOTIC\" --model $MODEL --seed $SEED"
          run "$CMD";
        done # models
      done # antibiotics
    done # test
  done # train
done # seed
