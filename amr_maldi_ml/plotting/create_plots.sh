

# ------------
# Figure 2: barplot and vmeplot
# ------------
#antibiotic_list_small=("Ciprofloxacin,Amoxicillin-Clavulanic acid,Ceftriaxone,Oxacillin,Tigecycline,Colistin,Fluconazole")
antibiotic_list_small=("Ciprofloxacin,Ceftriaxone,Oxacillin,Tigecycline,Colistin,Fluconazole")
echo "$antibiotic_list_small"

python plot_baseline_barplot.py --antibiotic "$antibiotic_list_small" \
      --outfile ../plots/baseline_barplot_vmeplot/fig2_barplot_few
#python plot_baseline_vmeplots.py --antibiotic "$antibiotic_list_small" \
#      --outfile ../plots/baseline_barplot_vmeplot/fig2_vmeplots_few

python plot_baseline_barplot.py --antibiotic None \
     --outfile ../plots/baseline_barplot_vmeplot/fig2_barplot_all
#python plot_baseline_vmeplots.py --antibiotic None \
#     --outfile ../plots/baseline_barplot_vmeplot/fig2_vmeplots_all
