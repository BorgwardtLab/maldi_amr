#!/usr/bin/env bash
#
# The purpose of this script is to submit jobs to an LSF system in order
# to speed up job processing.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../curves_per_species_and_antibiotic.py --force "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -o "curve_per_species_and_antibiotic_%J.out" -R "rusage[mem=64000]"'
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
  # S. pneumoniae
  CMD="${MAIN} --antibiotic \"Penicillin\" --species \"Streptococcus pneumoniae\" --seed $SEED"
  run "$CMD";

  # H. influenzae
  for ANTIBIOTIC in 'Penicillin' 'Cefuroxime'; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Haemophilus influenzae\" --seed $SEED"
    run "$CMD";
  done

  # S. aureus jobs
  for ANTIBIOTIC in 'Clindamycin' 'Ciprofloxacin' 'Fusidic acid' 'Oxacillin' 'Penicillin'; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Staphylococcus aureus\" --seed $SEED"
    run "$CMD";
  done

  # E. coli jobs
  for ANTIBIOTIC in "Amoxicillin-Clavulanic acid" "Cefepime" "Ceftriaxone" "Ciprofloxacin" "Piperacillin-Tazobactam" "Tobramycin"; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Escherichia coli\" --seed $SEED"
    run "$CMD";
  done

  # K. pneumoniae jobs
  for ANTIBIOTIC in 'Amoxicillin-Clavulanic acid' 'Cefepime' 'Ceftriaxone' 'Ciprofloxacin' 'Meropenem' 'Piperacillin-Tazobactam' 'Tobramycin'; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Klebsiella pneumoniae\" --seed $SEED"
    run "$CMD";
  done

  # S. capitis
  for ANTIBIOTIC in 'Amoxicillin-Clavulanic acid' 'Ceftriaxone' 'Fusidic acid' 'Imipenem' 'Meropenem' 'Oxacillin' 'Piperacillin-Tazobactam' 'Tetracycline'; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Staphylococcus capitis\" --seed $SEED"
    run "$CMD";
  done

  # S. epidermidis jobs
  for ANTIBIOTIC in 'Amoxicillin-Clavulanic acid' 'Ciprofloxacin' 'Clindamycin' 'Oxacillin' 'Tetracycline'; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Staphylococcus epidermidis\" --seed $SEED"
    run "$CMD";
  done

  # M. morganii jobs
  for ANTIBIOTIC in 'Ceftriaxone' 'Ciprofloxacin' 'Imipenem'; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Morganella morganii\" --seed $SEED"
    run "$CMD";
  done

  # P. aeruginosa
  for ANTIBIOTIC in 'Ciprofloxacin' 'Meropenem' 'Piperacillin-Tazobactam' 'Tobramycin'; do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --species \"Pseudomonas aeruginosa\" --seed $SEED"
    run "$CMD";
  done

  # E. faecium
  CMD="${MAIN} --antibiotic \"Vancomycin\" --species \"Enterococcus faecium\" --seed $SEED"
  run "$CMD";
done
