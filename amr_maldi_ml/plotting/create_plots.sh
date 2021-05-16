

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

      #"Ciprofloxacin,Ceftriaxone,Cefepime,Piperacillin-Tazobactam,Tobramycin" \
for model in "lr" "lightgbm" "mlp"; do
  python plot_curves_per_species_and_antibiotic.py \
    --species "Escherichia coli" \
    --antibiotic \
      "Ceftriaxone" \
    --model $model \
    --outfile "../plots/curves_per_species_and_antibiotic_calibrated/Ecoli_$model"
done

    #--antibiotic "Ciprofloxacin,Fusidic acid,Oxacillin,Penicillin" \
for model in "lr" "lightgbm" "mlp"; do
  python plot_curves_per_species_and_antibiotic.py \
    --species "Staphylococcus aureus" \
    --antibiotic "Oxacillin" \
    --model $model \
    --outfile "../plots/curves_per_species_and_antibiotic_calibrated/Saureus_$model"
done

      #"Ciprofloxacin,Ceftriaxone,Cefepime,Meropenem,Tobramycin" \
for model in "lr" "lightgbm" "mlp"; do
  python plot_curves_per_species_and_antibiotic.py \
    --species "Klebsiella pneumoniae" \
    --antibiotic \
      "Ceftriaxone" \
    --model $model \
    --outfile "../plots/curves_per_species_and_antibiotic_calibrated/Kpneu_$model"
done
