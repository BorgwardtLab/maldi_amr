#!/bin/bash
#
# Utility script for checking all `lsf.*` and `*.out` files for failure
# flags and resubmitting the jobs that failed.

RUNTIME=${RUNTIME:-23:59}
MEMORY=${MEMORY:-24000}

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB="bsub -W $RUNTIME -o \"resubmit_%J.out\" -R \"rusage[mem=$MEMORY]\""
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

for file in $(grep -L Successfully lsf.* *.out); do
  # Get the command that caused the job to fail. This is completely
  # agnostic to whatever was going on in the file.
  CMD=$(grep -A1 LSBATCH $file | tail -n 1);
  run "$CMD";
done
