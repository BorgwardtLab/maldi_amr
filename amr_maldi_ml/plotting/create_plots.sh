

# ------------
# Figure 2: Baseline barplots
# ------------
#antibiotic_list_small=("Ciprofloxacin,Amoxicillin-Clavulanic acid,Ceftriaxone,Oxacillin,Tigecycline,Colistin,Fluconazole")
#echo "$antibiotic_list_small"
#
#python plot_baseline_barplot.py --antibiotic "$antibiotic_list_small" \
#      --outfile ../plots/baseline_barplot_vmeplot/fig2_barplot_few
#
#python plot_baseline_barplot.py --antibiotic None \
#     --outfile ../plots/baseline_barplot_vmeplot/fig2_barplot_all


# ------------
# Figure 4: AUC curves per species and antibiotic
# ------------
#for model in "lr" "lightgbm" "mlp"; do
#  python plot_curves_per_species_and_antibiotic.py \
#    --species "Escherichia coli" \
#    --antibiotic \
#      "Ciprofloxacin,Ceftriaxone,Cefepime,Piperacillin-Tazobactam,Tobramycin" \
#    --model $model \
#    --outfile "../plots/curves_per_species_and_antibiotic_calibrated/Ecoli_$model"
#done
#
#for model in "lr" "lightgbm" "mlp"; do
#  python plot_curves_per_species_and_antibiotic.py \
#    --species "Staphylococcus aureus" \
#    --antibiotic "Ciprofloxacin,Fusidic acid,Oxacillin,Penicillin" \
#    --model $model \
#    --outfile "../plots/curves_per_species_and_antibiotic_calibrated/Saureus_$model"
#done
#
#for model in "lr" "lightgbm" "mlp"; do
#  python plot_curves_per_species_and_antibiotic.py \
#    --species "Klebsiella pneumoniae" \
#    --antibiotic \
#      "Ciprofloxacin,Ceftriaxone,Cefepime,Meropenem,Tobramycin" \
#    --model $model \
#    --outfile "../plots/curves_per_species_and_antibiotic_calibrated/Kpneu_$model"
#done


# ------------
# Sliding window validation
# ------------
#python plot_sliding_window_validation_major_scenarios.py --metric auroc
#python plot_sliding_window_validation_major_scenarios.py --metric auprc


# ------------
# Sensitivity-Specificity threshold
# ------------
#python plot_specificity_sensitivity_per_threshold.py ../tables/Rejection_table_assymetric_finegrid_Escherichia_coli_Ceftriaxone_calibrated_lightgbm.csv
#python plot_specificity_sensitivity_per_threshold.py ../tables/Rejection_table_assymetric_finegrid_Staphylococcus_aureus_Oxacillin_calibrated_lightgbm.csv
python plot_specificity_sensitivity_per_threshold.py ../tables/Rejection_table_assymetric_finegrid_Klebsiella_pneumoniae_Ceftriaxone_calibrated_mlp.csv 


# ------------
# Validation per species and antibiotic 
# ------------
#poetry run python plot_validation_per_species_and_antibiotic.py --species 'Escherichia coli' \
#                                                                --antibiotic Ceftriaxone \
#                                                                --outfile 'Escherichia_coli_Ceftriaxone_lightgbm' \
#                                                                -m lightgbm
#poetry run python plot_validation_per_species_and_antibiotic.py --species 'Klebsiella pneumoniae' \
#                                                                --antibiotic Ceftriaxone \
#                                                                --outfile 'Klebsiella_pneumoniae_Ceftriaxone_mlp' \
#                                                                -m mlp
#poetry run python plot_validation_per_species_and_antibiotic.py --species 'Staphylococcus aureus' \
#                                                                --antibiotic Oxacillin \
#                                                                --outfile 'Staphylococcus_aureus_Oxacillin_lightgbm' \
#                                                                -m lightgbm
