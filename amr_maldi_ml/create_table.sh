

# ------------
# Table 1: Create summary table for antibiotic prevalences 
#          categorizes by antibiotic class 
# ------------

python build_table1_categories.py --site "DRIAMS-A" \
      --save True \
      --outfile "plots/tables/DRIAMS-A_Table1.csv"

# ------------
# Table 1 supplement: Create full summary table for antibiotic prevalences 
# ------------

python build_table1_suppl.py --site "DRIAMS-A" \
      --save True \
      --remove_empty_antibiotics True \
      --outfile "plots/tables/DRIAMS-A_Table1_suppl.csv"


# ------------
# Table 2: Rejection ratio tables
# ------------
#python build_table2_rejection.py ../results/calibrated_classifiers \
#--outdir plots/tables/rejection 

python build_table2_assymetric_rejection.py ../results/calibrated_classifiers \
--outdir plots/tables/rejection 

# ------------
# Table 3: Replication table
# ------------
python build_table3_replication.py ../results/replication
