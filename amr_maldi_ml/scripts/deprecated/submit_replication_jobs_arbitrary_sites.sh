#!/usr/bin/env bash
#
# Submits replication jobs to an LSF system or executes them in a simple
# sequential manner. This script executes the arbitrary replication
# scenario.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../replication_arbitrary_sites.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "replication_arbitrary_sites_%J.out" -R "rusage[mem=32000]"'
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

submit_all_jobs() {
  for SEED in 344 172 188 270 35 164 545 480 89 409; do
    for MODEL in "lr" "lightgbm"; do
      for ANTIBIOTIC in "Ceftriaxone" "Ciprofloxacin" "Cefepime"; do
        # This could be solved more elegantly, as the script understands
        # the meaning of the '*' argument for the years.
        if [ -z "$3" ]; then
          CMD="${MAIN} --train-site \"$1\" --test-site \"$2\" --species \"Escherichia coli\" --antibiotic \"$ANTIBIOTIC\" --model $MODEL --seed $SEED"
        else
          CMD="${MAIN} --train-site \"$1\" --test-site \"$2\" --species \"Escherichia coli\" --antibiotic \"$ANTIBIOTIC\" --model $MODEL --seed $SEED --train-years $3"
        fi
        run "$CMD";
      done # antibiotic

      for ANTIBIOTIC in "Ciprofloxacin" "Oxacillin" "Fusidic acid"; do
        # See above for my lamentation re: elegance.
        if [ -z "$3" ]; then
          CMD="${MAIN} --train-site \"$1\" --test-site \"$2\" --species \"Staphylococcus aureus\" --antibiotic \"$ANTIBIOTIC\" --model $MODEL --seed $SEED"
        else
          CMD="${MAIN} --train-site \"$1\" --test-site \"$2\" --species \"Staphylococcus aureus\" --antibiotic \"$ANTIBIOTIC\" --model $MODEL --seed $SEED --train-years $3"
        fi
        run "$CMD";
      done # antibiotic
    done # model
  done # seed
}

submit_all_jobs "DRIAMS-A" "DRIAMS-E"
submit_all_jobs "DRIAMS-A" "DRIAMS-F"

# Now with a restriction to 2018...
submit_all_jobs "DRIAMS-A" "DRIAMS-E" "2018"
submit_all_jobs "DRIAMS-A" "DRIAMS-F" "2018"
