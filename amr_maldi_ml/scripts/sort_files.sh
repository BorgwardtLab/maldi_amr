#!/usr/bin/env bash
#
# Script for sorting result files into their respective directories,
# creating them in the process if they do not exist. This is one way
# to ensure that results files are collated nicely.

if [ -z ${1+x} ]; then
  exit
fi

ROOT=$1

for MODEL in 'lightgbm' 'lr' 'rf' 'svm-linear' 'svm-rbf'; do
  mkdir -p $ROOT/$MODEL
  git mv $ROOT/*Model_$MODEL*.json $ROOT/$MODEL
done
