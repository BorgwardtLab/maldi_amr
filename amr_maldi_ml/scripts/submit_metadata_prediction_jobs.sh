#!/usr/bin/env bash
#
# Submits metadata prediction jobs. The purpose is to test whether we
# are able to predict a metadata column such as the 'workstation', by
# only looking at spectra.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../predict_metadata.py --force "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "predict_metadata_%J.out" -R "rusage[mem=16000]"'
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
  for SPECIES in "Escherichia coli" "Klebsiella pneumoniae"; do
      CMD="${MAIN} --species \"$SPECIES\" --antibiotic \"Ceftriaxone\" --model lr --seed $SEED --column workstation" 
      run "$CMD";
  done # species

  CMD="${MAIN} --species \"Staphylococcus aureus\" --antibiotic \"Oxacillin\" --model lr --seed $SEED --column workstation"
  run "$CMD";

  # Once more with excluded values. We suspect that we are doing worse
  # in this scenario.
  for SPECIES in "Escherichia coli" "Klebsiella pneumoniae"; do
      CMD="${MAIN} --species \"$SPECIES\" --antibiotic \"Ceftriaxone\" --model lr --seed $SEED --column workstation --exclude HospitalHygiene" 
      run "$CMD";
  done # species

  CMD="${MAIN} --species \"Staphylococcus aureus\" --antibiotic \"Oxacillin\" --model lr --seed $SEED --column workstation --exclude HospitalHygiene"
  run "$CMD";

done # seed
