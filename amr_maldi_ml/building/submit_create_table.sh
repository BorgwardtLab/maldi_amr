# ------------
# Table: Rejection ratio tables
# ------------
bsub -W 23:59 -o "rejection_tables_%J.out" -R "rusage[mem=16000]" "poetry run python build_finegrid_assymetric_rejection.py ../../results/calibrated_classifiers/lr"
