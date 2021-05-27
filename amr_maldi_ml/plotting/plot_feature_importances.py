"""Plot feature importances."""

import os
import argparse
import json
import glob

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

# Input files containing feature importances of major scenarios.
prjdir = '../../results/feature_importance_values/calibrated_classifiers'
input_files = [

    prjdir+'/lightgbm/Site_DRIAMS-A_Model_lightgbm_Species_Escherichia_coli_Antibiotic_Ceftriaxone_Seed_164-172-188-270-344-35-409-480-545-89_average.json',  
    prjdir+'/mlp/Site_DRIAMS-A_Model_mlp_Species_Klebsiella_pneumoniae_Antibiotic_Ceftriaxone_Seed_164-172-188-270-344-35-409-480-545-89_average.json',
    prjdir+'/lightgbm/Site_DRIAMS-A_Model_lightgbm_Species_Staphylococcus_aureus_Antibiotic_Oxacillin_Seed_164-172-188-270-344-35-409-480-545-89_average.json',
    				]

scenario_map = {
    'Escherichia_coli': 'E-CEF',
    'Klebsiella_pneumoniae': 'K-CEF',
    'Staphylococcus_aureus': 'S-OXA',
}

plt.close('all')
#sns.set_style("whitegrid")
fig, ax = plt.subplots(3, 1, figsize=(20,12))


# Go through eac input file and plot importances.
for i, filename in enumerate(input_files):
    print(i)
    with open(filename) as f:
        data = json.load(f)    
    
    site = data['site']
    model = data['model']
    species = data['species'].replace(' ', '_')
    antibiotic = data['antibiotic']
    mean_feature_weights = data['mean_feature_weights']

    # Plot feature weights as barplot.

    median3 = 3*np.median(np.abs(mean_feature_weights))
    if model=='lightgbm':
        colors=[sns.color_palette()[0] if w_ > median3 else sns.color_palette('pastel')[7] for w_ in mean_feature_weights]
    else:
        colors=[sns.color_palette()[1] if np.abs(w_) > median3 else sns.color_palette('pastel')[7] for w_ in mean_feature_weights]

    # FIXME for debugging
    #mean_feature_weights = mean_feature_weights[:30]

    n_feat = len(mean_feature_weights)

    sns.barplot(
        x=[j+1 for j in range(n_feat)],
        y=mean_feature_weights,
        ax=ax[i],
        palette=colors,
    )     
    ax[i].yaxis.tick_right()
    ax[i].set_xticks([])
    ax[i].set_ylabel(f'{scenario_map[species]} ({model})', fontsize=15)

# Hide x labels and tick labels for all but bottom plot.
for axis in ax:
    axis.label_outer()
plt.subplots_adjust(hspace=0)

# Save to file
outfile = f'all_scenarios.png'

plt.savefig(os.path.join('../plots/feature_importances',outfile))
