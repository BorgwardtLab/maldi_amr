#!/usr/bin/env bash
#
# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../curves_per_species_and_antibiotic.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "curves_per_species_and_antibiotic_%J.out" -R "rusage[mem=16000]"'
fi

# Evaluates its first argument either by submitting a job, or by
# executing the command without parallel processing.
run() {
  echo $1
  if [ -z "$BSUB" ]; then
    eval "$1";
  else
    eval "${BSUB} $1";
  fi
}

function make_jobs {
  local SEED=${1}
  local MODEL=${2}
  local TRAIN=${3}
  local TEST="DRIAMS-B"

  # S. aureus jobs
  CMD="${MAIN} --antibiotic Oxacillin --species \"Staphylococcus aureus\" --train-site $TRAIN --test-site $TEST --model $MODEL --seed $SEED"
  run "$CMD";
  CMD="${MAIN} --antibiotic Penicillin --species \"Staphylococcus aureus\" --train-site $TRAIN --test-site $TEST --model $MODEL --seed $SEED"
  run "$CMD";
  CMD="${MAIN} --antibiotic Ciprofloxacin --species \"Staphylococcus aureus\" --train-site $TRAIN --test-site $TEST --model $MODEL --seed $SEED"
  run "$CMD";
  CMD="${MAIN} --antibiotic \"Fusidic acid\" --species \"Staphylococcus aureus\" --model $MODEL --train-site $TRAIN --test-site $TEST --seed $SEED"
  run "$CMD";

  # E. coli and K. pneumoniae jobs
  for SPECIES in 'Escherichia coli' 'Klebsiella pneumoniae'; do
    CMD="${MAIN} --antibiotic Ceftriaxone --species \"$SPECIES\" --train-site $TRAIN --test-site $TEST --model $MODEL --seed $SEED"
    run "$CMD";
    CMD="${MAIN} --antibiotic Ciprofloxacin --species \"$SPECIES\" --train-site $TRAIN --test-site $TEST --model $MODEL --seed $SEED"
    run "$CMD";
    CMD="${MAIN} --antibiotic Cefepime --species \"$SPECIES\" --train-site $TRAIN --test-site $TEST --model $MODEL --seed $SEED"
    run "$CMD";
    CMD="${MAIN} --antibiotic Tobramycin --species \"$SPECIES\" --train-site $TRAIN --test-site $TEST --model $MODEL --seed $SEED"
    run "$CMD";
  done

  CMD="${MAIN} --antibiotic \"Piperacillin-Tazobactam\" --species \"Escherichia coli\" --train-site $TRAIN --test-site $TEST --model $MODEL --seed $SEED"
  run "$CMD";
  CMD="${MAIN} --antibiotic \"Meropenem\" --species \"Klebsiella pneumoniae\" --train-site $TRAIN --test-site $TEST --model $MODEL --seed $SEED"
  run "$CMD";
}

# The grid is kept sparse for now. This is *not* an inconsistency.
for SEED in 344 172 188 270 35 164 545 480 89 409; do
  for MODEL in "lr" "lightgbm" "mlp"; do
    make_jobs $SEED $MODEL "DRIAMS-A"
    make_jobs $SEED $MODEL "DRIAMS-A DRIAMS-B"
    make_jobs $SEED $MODEL "DRIAMS-A DRIAMS-B DRIAMS-C"
    make_jobs $SEED $MODEL "DRIAMS-A DRIAMS-B DRIAMS-C DRIAMS-D"
  done
done
