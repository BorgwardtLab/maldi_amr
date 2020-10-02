#!/usr/bin/env bash
#
# The purpose of this script is to submit jobs to an LSF system in order
# to speed up job processing. This script handles the label permutations
# experiment, which assesses stability and generalisability.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../label_permutation.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "label_permutation_%J.out" -R "rusage[mem=64000]"'
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
  for CORRUPTION in 0.1 0.2 0.3 0.4 0.5; do
    # S. aureus jobs
    for ANTIBIOTIC in 'Ciprofloxacin' 'Fusidic acid' 'Oxacillin' 'Penicillin'; do
      CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Staphylococcus aureus\" --seed $SEED --corruption $CORRUPTION"
      run "$CMD";
    done

    # E. coli jobs
    for ANTIBIOTIC in 'Cefepime' 'Ceftriaxone' 'Ciprofloxacin' 'Piperacillin-Tazobactam' 'Tobramycin'; do
      CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Escherichia coli\" --seed $SEED --corruption $CORRUPTION"
      run "$CMD";
    done

    # K. pneumoniae jobs
    for ANTIBIOTIC in 'Cefepime' 'Ceftriaxone' 'Ciprofloxacin' 'Meropenem' 'Tobramycin'; do
      CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Klebsiella pneumoniae\" --seed $SEED --corruption $CORRUPTION"
      run "$CMD";
    done
  done
done
