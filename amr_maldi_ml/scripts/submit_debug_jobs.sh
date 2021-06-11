#!/usr/bin/env bash

for INDEX in $(seq 0 99); do
  bsub -W 00:05 -o "debug_%J.out" -R "rusage[mem=128]" groups
done
