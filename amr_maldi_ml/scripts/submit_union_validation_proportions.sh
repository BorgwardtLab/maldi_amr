#!/usr/bin/env bash
#
# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../union_validation_proportions.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "union_validation_proportions_%J.out" -R "rusage[mem=16000]"'
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

function make_jobs {
  local SEED=${1}
  local TRAIN=${2}
  local PROPS=${3}
  local TEST="DRIAMS-B"

  # S. aureus jobs
  CMD="${MAIN} --antibiotic Oxacillin --species \"Staphylococcus aureus\" --train-site $TRAIN --test-site $TEST --model lightgbm --seed $SEED --train-proportions $PROPS"
  run "$CMD";

  CMD="${MAIN} --antibiotic Ceftriaxone --species \"Escherichia coli\" --train-site $TRAIN --test-site $TEST --model lightgbm --seed $SEED --train-proportions $PROPS"
  run "$CMD";

  CMD="${MAIN} --antibiotic Ceftriaxone --species \"Klebsiella pneumoniae\" --train-site $TRAIN --test-site $TEST --model mlp --seed $SEED --train-proportions $PROPS"
  run "$CMD";

}

# The grid is kept sparse for now. This is *not* an inconsistency.
for SEED in 344 172 188 270 35 164 545 480 89 409; do
    for P1 in 0.0 0.25 0.5 0.75 1.0; do
      for P2 in 0.0 0.25 0.5 0.75 1.0; do
        for P3 in 0.0 0.25 0.75 1.0; do
    #for P1 in 0.5; do
    #  for P2 in 0.5; do
    #    for P3 in 0.5; do
          make_jobs $SEED "DRIAMS-A DRIAMS-B DRIAMS-C" "$P1 $P2 $P3"
        done
      done
    done
done
