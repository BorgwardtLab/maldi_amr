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
MAIN="poetry run python ../domain_adaptation_shallow_iw.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "domain_adaptation_shallow_iw_%J.out" -R "rusage[mem=16000]"'
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
      for MODEL in "lr"; do

        # E. coli and K. pneu jobs
        for ANTIBIOTIC in "Ceftriaxone"; do
          for SPECIES in "Escherichia coli" "Klebsiella pneumoniae"; do
              CMD="${MAIN} --source-site $TRAIN --target-site $TEST --species \"$SPECIES\" --antibiotic \"$ANTIBIOTIC\" --model $MODEL --seed $SEED"
              run "$CMD";
          done # species
        done # E. coli and K. pneu

        # S. aureus jobs
        CMD="${MAIN} --source-site $TRAIN --target-site $TEST --antibiotic Oxacillin --species \"Staphylococcus aureus\" --model $MODEL --seed $SEED"
        run "$CMD";
      done # models
    done # target
  done # source
done # seed
