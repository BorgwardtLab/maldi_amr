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

# Go through eac input file and plot importances.
for filename in input_files:

    with open(filename) as f:
        data = json.load(f)    
    
    site = data['site']
    model = data['model']
    species = data['species'].replace(' ', '_')
    antibiotic = data['antibiotic']
    mean_feature_weights = data['mean_feature_weights']

    # Plot feature weights as barplot.
    print(species, np.mean(np.abs(mean_feature_weights)))
    print(species, np.median(np.abs(mean_feature_weights)))
    plt.close('all')
    fig, ax = plt.subplots(figsize=(20,5))

    median3 = 3*np.median(np.abs(mean_feature_weights))
    if model=='lightgbm':
        colors=['blue' if w_ > median3 else 'lightgrey' for w_ in mean_feature_weights]
    else:
        colors=['red' if np.abs(w_) > median3 else 'lightgrey' for w_ in mean_feature_weights]

    sns.barplot(
        x=[i+1 for i in range(6000)],
        y=mean_feature_weights,
        ax=ax,
        palette=colors,
    )     
    ax.set_xticks([])

    # Save to file
    outfile = f'Site_{site}_Model_{model}_Species_{species}_Antibiotic_{antibiotic}.png'
	
    plt.tight_layout()
    plt.savefig(os.path.join('../plots/feature_importances',outfile))
