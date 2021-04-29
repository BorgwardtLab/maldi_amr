#!/usr/bin/env bash
#
# Submits domain adaptation jobs for shallow models. The output of these
# models is rather terse, and it should be compared to curves obtained
# from a species--antibiotic combination.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../domain_adaptation_shallow.py --force "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "domain_adaptation_shallow_%J.out" -R "rusage[mem=16000]"'
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
  for TRAIN in "DRIAMS-A"; do
    for TEST in "DRIAMS-B" "DRIAMS-C" "DRIAMS-D"; do
      # FIXME: the model is currently ignored in this configuration
      for MODEL in "lr"; do 

        # E. coli and K. pneu jobs
        for ANTIBIOTIC in "Ceftriaxone"; do
          for SPECIES in "Escherichia coli" "Klebsiella pneumoniae"; do
              CMD="${MAIN} --source-site $TRAIN --target-site $TEST --species \"$SPECIES\" --antibiotic \"$ANTIBIOTIC\" --seed $SEED"
              run "$CMD";
          done # species
        done # E. coli and K. pneu

        # S. aureus jobs
        CMD="${MAIN} --source-site $TRAIN --target-site $TEST --antibiotic Oxacillin --species \"Staphylococcus aureus\" --seed $SEED"
        run "$CMD";
      done # models
    done # test
  done # train
done # seed
