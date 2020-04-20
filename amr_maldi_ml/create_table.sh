

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
