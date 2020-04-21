

# ------------
# Figure 2: barplot and vmeplot
# ------------
#antibiotic_list_small=("Ciprofloxacin,Amoxicillin-Clavulanic acid,Ceftriaxone,Oxacillin,Tigecycline,Colistin,Fluconazole")
#echo "$antibiotic_list_small"
#
#python plot_fig2_baseline_barplot.py --antibiotic "$antibiotic_list_small" \
#      --outfile plots/fig2/fig2_barplot_few
#python plot_fig2_baseline_vmeplots.py --antibiotic "$antibiotic_list_small" \
#      --outfile plots/fig2/fig2_vmeplots_few
#
#python plot_fig2_baseline_barplot.py --antibiotic None \
#     --outfile plots/fig2/fig2_barplot_all
#python plot_fig2_baseline_vmeplots.py --antibiotic None \
#     --outfile plots/fig2/fig2_vmeplots_all


# -------------
# Figure 3: Ensemble training vs. species training curves
# -------------
#python plot_fig3_ensemble_curves.py ../results/fig3_ensemble \
#        --outdir ./plots/fig3


# ------------
# Figure 4: AUC curves per species and antibiotic
# ------------
#for model in "lr" "svm-rbf" "svm-linear" "lightgbm" "rf"; do
#  python plot_fig4_curves_per_species_and_antibiotic.py \
#    --species "Escherichia coli" \
#    --antibiotic \
#      "Ciprofloxacin,Ceftriaxone,Cefepime,Piperacillin-Tazobactam,Tobramycin" \
#    --model $model \
#    --outfile "plots/fig4/fig4_Ecoli_$model"
#done
#
#for model in "lr" "svm-rbf" "svm-linear" "lightgbm" "rf"; do
#  python plot_fig4_curves_per_species_and_antibiotic.py \
#    --species "Staphylococcus aureus" \
#    --antibiotic "Ciprofloxacin,Fusidic acid,Oxacillin,Penicillin" \
#    --model $model \
#    --outfile "plots/fig4/fig4_Saureus_$model"
#done
#
#for model in "lr" "svm-rbf" "svm-linear" "lightgbm" "rf"; do
#  python plot_fig4_curves_per_species_and_antibiotic.py \
#    --species "Klebsiella pneumoniae" \
#    --antibiotic \
#      "Ciprofloxacin,Ceftriaxone,Cefepime,Amoxicillin-Clavulanic acid,Meropenem,Tobramycin,Piperacillin-Tazobactam" \
#    --model $model \
#    --outfile "plots/fig4/fig4_Kpneu_$model"
#done

# FIXME results not available yet
#for model in "lr" "svm-rbf" "svm-linear" "lightgbm" "rf"; do
#  python plot_fig4_curves_per_species_and_antibiotic.py \
#    --species "Staphylococcus capitis" \
#    --antibiotic \
#      "Oxacillin,Tetracycline,Fusidic acid,Piperacillin-Tazobactam" \
#    --model $model \
#    --outfile "plots/fig4/fig4_Scap_$model"
#done
#
#for model in "lr" "svm-rbf" "svm-linear" "lightgbm" "rf"; do
#  python plot_fig4_curves_per_species_and_antibiotic.py \
#    --species "Staphylococcus epidermidis" \
#    --antibiotic \
#      "Ciprofloxacin,Oxacillin,Amoxicillin-Clavulanic acid,Tetracyclin,Clindamycin" \
#    --model $model \
#    --outfile "plots/fig4/fig4_Sepi_$model"
#done
#
#for model in "lr" "svm-rbf" "svm-linear" "lightgbm" "rf"; do
#  python plot_fig4_curves_per_species_and_antibiotic.py \
#    --species "Morganella morganii" \
#    --antibiotic \
#      "Ciprofloxacin,Ceftriaxone,Cefepime,Amoxicillin-Clavulanic acid,Meropenem,Tobramycin,Piperacillin-Tazobactam" \
#    --model $model \
#    --outfile "plots/fig4/fig4_Mmorganii_$model"
#done
#
#for model in "lr" "svm-rbf" "svm-linear" "lightgbm" "rf"; do
#  python plot_fig4_curves_per_species_and_antibiotic.py \
#    --species "Pseudomonas aeruginosa" \
#    --antibiotic \
#      "Meropenem,Ciprofloxacin,Tobramycin,Piperacillin-Tazobactam" \
#    --model $model \
#    --outfile "plots/fig4/fig4_Paeru_$model"
#done

# ------------
# Figure 5: Validation comparison plots
# ------------

for model in "lr" "svm-rbf" "svm-linear" "lightgbm" "rf"; do
  python plot_fig5_validation.py \
    --model $model \
    --outfile "plots/fig5/fig5_$model"
done
