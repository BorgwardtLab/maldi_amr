

# ------------
# Figure 2: barplot and vmeplot
# ------------
antibiotic_list_small=("Ciprofloxacin,Amoxicillin-Clavulanic acid,Ceftriaxone,Oxacillin,Tigecycline,Colistin,Fluconazole")
echo "$antibiotic_list_small"

python plot_fig2_baseline_barplot.py --antibiotic "$antibiotic_list_small" \
      --outfile plots/fig2/fig2_barplot_few
python plot_fig2_baseline_vmeplots.py --antibiotic "$antibiotic_list_small" \
      --outfile plots/fig2/fig2_vmeplots_few

#python plot_fig2_baseline_barplot.py --antibiotic None \
#      --outfile plots/fig2/fig2_barplot_all
#python plot_fig2_baseline_vmeplots.py --antibiotic None \
#      --outfile plots/fig2/fig2_vmeplots_all

# ------------
# Figure 4: AUC curves per species and antibiotic
# ------------
#for model in "lr" "svm-rbf" "svm-linear" "lightgbm" "rf"; do
#  for species in "Escherichia coli" "Staphylococcus aureus"; do
#    python plot_fig4_curves_per_species_and_antibiotic.py --species \
#    $species --antibiotic None --model $model --outfile \
#    plots/fig4/fig4_$species_$model
#  done
#done

