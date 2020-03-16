#!/usr/bin/env bash
#
# The purpose of this script is to submit ensemble classifier
# computation jobs to an LSF system in order to speed up job processing.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../ensemble.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "ensemble_%J.out" -R "rusage[mem=64000]"'
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
  for INDEX in $(seq 0 19); do
    for ANTIBIOTIC in 'Amoxicillin-Clavulanic acid' 'Ciprofloxacin'; do
      for SPECIES in 'Escherichia coli' 'Staphylococcus aureus'; do
        CMD="${MAIN} --index $INDEX --species \"$SPECIES\" --antibiotic \"$ANTIBIOTIC\" --seed $SEED --force"
        run "$CMD";
      done
    done

    for SPECIES in 'Escherichia coli' 'Klebsiella pneumoniae'; do
      CMD="${MAIN} --index $INDEX --species \"$SPECIES\" --antibiotic \"Amikacin\" --seed $SEED --force"
      run "$CMD";
    done

    for ANTIBIOTIC in 'Ceftriaxone' 'Cefepime' 'Imipenem' 'Piperacillin-Tazobactam'; do
      for SPECIES in 'Escherichia coli' 'Staphylococcus aureus' 'Staphylococcus epidermidis' 'Klebsiella pneumoniae'; do
        CMD="${MAIN} --index $INDEX --species \"$SPECIES\" --antibiotic \"$ANTIBIOTIC\" --seed $SEED --force"
        run "$CMD";
      done
    done

    for SPECIES in 'Staphylococcus aureus' 'Staphylococcus epidermidis'; do
      CMD="${MAIN} --index $INDEX --species \"$SPECIES\" --antibiotic \"Gentamicin\" --seed $SEED --force"
      run "$CMD";
    done
  done
done
