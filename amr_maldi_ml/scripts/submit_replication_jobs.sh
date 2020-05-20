#!/usr/bin/env bash
#
# Submits replication jobs to an LSF system or executes them in a simple
# sequential manner.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../replication.py "

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
  for MODEL in "lr" "svm-rbf" "rf" "lightgbm" "svm-linear"; do
    for ANTIBIOTIC in "Ceftriaxone" "Ciprofloxacin" "Cefepime"; do
      CMD="${MAIN} -train-site \"DRIAMS-E\" --test-site \"DRIAMS-F\" --species \"Escherichia coli\" --antibiotic \"$ANTIBIOTIC\" --model $MODEL --seed $SEED"
      run "$CMD";
    done # antibiotic

    for ANTIBIOTIC in "Ciprofloxacin" "Oxacillin" "Fusidic acid"; do
      CMD="${MAIN} -train-site \"DRIAMS-E\" --test-site \"DRIAMS-F\" --species \"Staphyloccocus aureus\" --antibiotic \"$ANTIBIOTIC\" --model $MODEL --seed $SEED"
      run "$CMD";
    done # antibiotic
  done # model
done # seed
