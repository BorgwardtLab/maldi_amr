#!/usr/bin/env bash
#
# Create visualisations of Shapley values.


for MODEL in "lr" "lightgbm" "mlp"; do 
  FILES=$(find ../../results/explained_classifiers/$MODEL/ -name "*Escherichia_coli*Ceftriaxone*.pkl" )
  poetry run python ../visualise_shapley_values.py $FILES
  FILES=$(find ../../results/explained_classifiers/$MODEL/ -name "*Klebsiella_pneumoniae*Ceftriaxone*.pkl" )
  poetry run python ../visualise_shapley_values.py $FILES
  FILES=$(find ../../results/explained_classifiers/$MODEL/ -name "*Staphylococcus_aureus*Oxacillin*.pkl" )
  poetry run python ../visualise_shapley_values.py $FILES
done
