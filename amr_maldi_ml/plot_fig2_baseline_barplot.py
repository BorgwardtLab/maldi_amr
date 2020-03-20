"""
Plotting script to create

Figure 4: area-under-the-curve plots for AUROC, AUPRC and AUVME

MALDI-TOF spectra based AMR prediction at species level
"""

import os
import json
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
#from maldi_learn.metrics import very_major_error_score, major_error_score, vme_curve, vme_auc_score
from utilities import maldi_col_map

def plot_figure2(args):

    PATH_fig2 = '/links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/results/fig2_baseline/'

    # --------------
    # create dataframe giving an overview of all files in path
    # --------------
    file_list = []
    for (_, _, filenames) in os.walk(PATH_fig2):
        [file_list.append(f) for f in filenames if '.json' in f]
        break

    content = pd.DataFrame(columns=['filename',
                                    'species',  
                                    'antibiotic',
                                    'site',
                                    'seed',
                                    ])

    for filename in file_list:
        with open(PATH_fig2 + filename) as f:
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

    # TODO give option to take antibiotic list from args.antibiotics 
    # or take everything otherwise

    # ------------
    # for each antibiotic, get avg metrics for 'all' and 'all (w/o spectra)'
    # ------------

    values = pd.DataFrame(columns=['antibiotic',
                                   'auroc_all',
                                   'auroc_all_wo_spectra',
                                  ])

    # add lines for each antibiotic
    for antibiotic in set(content['antibiotic']):
        print(antibiotic)
        content_ab = content.query('antibiotic==@antibiotic')
        assert content_ab.shape == (20,5)
        
        content_spectra = content_ab.query("species=='all'")
        content_wo_spectra = content_ab.query("species=='all (w/o spectra)'") 
        
        col_ab = maldi_col_map[antibiotic]

        # 'all': extract y_test and y_score from json files
        y_score_total = []
        y_test_total = []

        for filename in content_spectra['filename'].values:
            with open(PATH_fig2 + filename) as f:
                data = json.load(f)
                y_score_total.extend([sc[1] for sc in data['y_score']])
                y_test_total.extend(data['y_test'])

        auroc_all = round(roc_auc_score(y_test_total, y_score_total), 3)

        # 'all (w/o spectra)': extract y_test and y_score from json files
        y_score_total = []
        y_test_total = []

        for filename in content_wo_spectra['filename'].values:
            with open(PATH_fig2 + filename) as f:
                data = json.load(f)
                y_score_total.extend([sc[1] for sc in data['y_score']])
                y_test_total.extend(data['y_test'])

        auroc_all_wo_spectra = round(roc_auc_score(y_test_total, y_score_total), 3)

        values = values.append(
            pd.DataFrame({
                'antibiotic': [antibiotic],
                'auroc_all': [auroc_all],
                'auroc_all_wo_spectra': [auroc_all_wo_spectra],
                }),
                ignore_index=True
            )

    print(values)

    # -------------
    # plot barplot
    # -------------

    values = values.sort_values(by=['auroc_all'], ascending=False)
    n_ab = len(values)
    
    sns.set(style="whitegrid")
    fix, ax = plt.subplots(figsize=(22,15))

    sns.barplot(x="antibiotic", y="auroc_all", ax=ax, data=values, color=sns.color_palette()[0])
    sns.barplot(x="antibiotic", y="auroc_all_wo_spectra", ax=ax, data=values, color='firebrick')
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.ylabel('AUROC')
    plt.xlabel('')
    plt.ylim(0.5, 1.05)
    plt.xlim(0-0.5, n_ab-0.5)

    # TODO correct Cotrimoxazole spelling
    # TODO include class ratios
    # TODO include std bars
    # TODO include p-values

    plt.tight_layout()
    plt.savefig('./test.png')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--antibiotic', 
                         type=str,
                         default='Ciprofloxacin')
    args = parser.parse_args()

    plot_figure2(args)
