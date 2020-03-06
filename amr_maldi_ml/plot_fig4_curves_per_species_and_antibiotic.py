"""
Plotting script to create

Figure 4: area-under-the-curve plots for AUROC, AUPRC and AUVME

MALDI-TOF spectra based AMR prediction at species level
"""

import os
import json
import argparse

import pandas as pd
import matplotlib as plt
import seaborn as sns


def plot_figure4(args):

    PATH_fig4 = '/links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/results/fig4_curves_per_species_and_antibiotics/'


    # create dataframe giving an overview of all files in path
    file_list = []
    for (_, _, filenames) in os.walk(PATH_fig4):
        [file_list.append(f) for f in filenames if '.json' in f]
        break

    content = pd.DataFrame(columns=['filename',
                                    'species',  
                                    'antibiotic',
                                    'site',
                                    'seed',
                                    ])

    for filename in file_list:
        with open(PATH_fig4 + filename) as f:
            data = json.load(f)

            content = content.append(
                pd.DataFrame({
                    'filename': [filename],
                    'species': [data['species']],
                    'antibiotic': [data['antibiotic']],
                    'site': [data['site']],
                    'seed': [data['seed']],
                    }),
                    ignore_index=True
                )

    # subset dataframe only relevant entries
    content = content.query('species==@args.species') 
    content = content.query('antibiotic==@args.antibiotic') 
    print(content)

    # extract y_test and y_score from json files
    y_score_total = []
    y_test_total = []

    for filename in content['filename'].values:
        with open(PATH_fig4 + filename) as f:
            data = json.load(f)
            y_test_total.extend(data['y_test'])


    # plot


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--species', 
                         type=str,
                         default='Escherichia coli')
    parser.add_argument('--antibiotic', 
                         type=str,
                         default='Ciprofloxacin')
    args = parser.parse_args()

    plot_figure4(args)
