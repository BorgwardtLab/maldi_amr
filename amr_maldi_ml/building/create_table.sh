# ------------
# Table: Rejection ratio tables
# ------------
#python build_assymetric_rejection.py ../../results/calibrated_classifiers/lr 
#python build_assymetric_rejection.py ../../results/calibrated_classifiers/lightgbm 
#python build_assymetric_rejection.py ../../results/calibrated_classifiers/mlp 

# ------------
# Table : Create summary table for antibiotic prevalences
#         categorizes by antibiotic class
# ------------

poetry run python build_table1_categories.py --site "DRIAMS-A" \
      --save True \
      --outfile "../tables/DRIAMS-A_Table1.csv"

# ------------
# Table 1 supplement: Create full summary table for antibiotic prevalences
# ------------

poetry run python build_table1_suppl.py --site "DRIAMS-A" \
      --save True \
      --remove_empty_antibiotics True \
      --outfile "../tables/DRIAMS-A_Table1_suppl.csv"
