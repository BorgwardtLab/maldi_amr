#!/usr/bin/env bash

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../curves_per_species_and_antibiotic_stratified_warped.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "curve_per_species_and_antibiotic_stratified_warped_%J.out" -R "rusage[mem=64000]"'
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
  # S. aureus jobs
  #for ANTIBIOTIC in 'Oxacillin'; do
  for ANTIBIOTIC in 'Clindamycin' 'Ciprofloxacin' 'Fusidic acid' 'Penicillin'; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Staphylococcus aureus\" --seed $SEED --model lightgbm"
    run "$CMD";
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Staphylococcus aureus\" --seed $SEED --model lr"
    run "$CMD";
  done

  # E. coli jobs
  #for ANTIBIOTIC in 'Ceftriaxone'; do
  for ANTIBIOTIC in 'Amoxicillin-Clavulanic acid' 'Cefepime' 'Ciprofloxacin' 'Piperacillin-Tazobactam' 'Tobramycin'; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Escherichia coli\" --seed $SEED --model lightgbm"
    run "$CMD";
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Escherichia coli\" --seed $SEED --model lr"
    run "$CMD";
  done

  # K. pneumoniae jobs
  #for ANTIBIOTIC in 'Ceftriaxone'; do
  for ANTIBIOTIC in 'Amoxicillin-Clavulanic acid' 'Cefepime' 'Ciprofloxacin' 'Meropenem' 'Piperacillin-Tazobactam' 'Tobramycin'; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Klebsiella pneumoniae\" --seed $SEED --model lightgbm"
    run "$CMD";
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Klebsiella pneumoniae\" --seed $SEED --model lr"
    run "$CMD";
  done
done
