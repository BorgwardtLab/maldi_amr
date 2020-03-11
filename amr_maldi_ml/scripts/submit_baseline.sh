#!/usr/bin/env bash
#
# The purpose of this script is to submit baseline computation jobs to
# an LSF system in order to speed up job processing.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../baseline.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 23:59 -R "rusage[mem=64000]"'
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
  for ANTIBIOTIC in '5-Fluorocytosine'\
      'Amikacin'\
      'Amoxicillin'\
      'Amoxicillin-Clavulanic acid'\
      'Ampicillin-Amoxicillin'\
      'Anidulafungin'\
      'Aztreonam'\
      'Caspofungin'\
      'Cefazolin'\
      'Cefepime'\
      'Cefpodoxime'\
      'Ceftazidime'\
      'Ceftriaxone'\
      'Cefuroxime'\
      'Ciprofloxacin'\
      'Clindamycin'\
      'Colistin'\
      'Daptomycin'\
      'Ertapenem'\
      'Erythromycin'\
      'Fluconazole'\
      'Fosfomycin-Trometamol'\
      'Fusidic acid'\
      'Gentamicin'\
      'Imipenem'\
      'Itraconazole'\
      'Levofloxacin'\
      'Meropenem'\
      'Micafungin'\
      'Nitrofurantoin'\
      'Norfloxacin'\
      'Oxacillin'\
      'Penicillin'\
      'Piperacillin-Tazobactam'\
      'Rifampicin'\
      'Teicoplanin'\
      'Tetracycline'\
      'Tobramycin'\
      'Trimethoprim-Sulfamethoxazole'\
      'Tigecycline'\
      'Vancomycin'\
      'Voriconazole';
  do
    CMD="${MAIN} --antibiotic \"$ANTIBIOTIC\" --seed $SEED --force"
    run "$CMD";
  done
done
