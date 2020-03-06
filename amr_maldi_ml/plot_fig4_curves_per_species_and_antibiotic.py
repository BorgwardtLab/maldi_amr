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
            y_score_total.extend([sc[1] for sc in data['y_score']])
            y_test_total.extend(data['y_test'])

    # ------------
    # plot
    # ------------
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 3, figsize=(30,10))
    
    # ------------
    # panel1: ROC curve
    # ------------
    fpr, tpr, thresholds = roc_curve(y_test_total, y_score_total)
    rocauc = round(roc_auc_score(y_test_total, y_score_total), 3)

    # TODO add zero to string of AUROC does not have 3 digits after comma
    lab = args.antibiotic + ' ' + str(rocauc)

    ax[0].plot(fpr, tpr, linewidth=3.0)
    ax[0].plot([0, 1], [0, 1], color='black', linestyle='--')

    # ------------
    # panel2: PRAUC-1 curve
    # ------------
    precision, recall, thresholds = precision_recall_curve(y_test_total, y_score_total)
    prauc = round(average_precision_score(y_test_total, y_score_total, average='weighted'), 3)

    lab = args.antibiotic + 'AUPRC: ' + str(prauc)

    ax[1].step(recall, precision, alpha=1.0, where='post', linewidth=3.0)

    # ------------
    # panel3: VME curve
    # ------------
    #vme, me_inv, thresholds = vme_auc_curve(list_ytrue[i], list_yprob[i])
    #me = 1-me_inv
    #vme_score = round(vme_auc_score(list_ytrue[i], list_yprob[i]),3)
    #pretty_vme = [str(pr)+'0' if len(str(pr))==3 else str(pr) for pr in [vme_score]]

    #lab = lab+'AUVME: '+pretty_vme[0]

    #ax[2].step(vme, me, color=col, label=lab, alpha=1.0, where='post', linewidth=3.0)


    # ------------
    # axes limits and labels
    # ------------
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[2].set_xlabel('Very major error')
    ax[2].set_ylabel('Major error')
    #ax[0].legend(bbox_to_anchor=(0.99, 0.01), loc='lower right', prop={'family': 'DejaVu Sans Mono', 'size': 15})
    #ax[1].legend(bbox_to_anchor=(0.01, 0.01), loc='lower left', prop={'family': 'DejaVu Sans Mono', 'size': 15})
    #ax[2].legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', prop={'family': 'DejaVu Sans Mono', 'size': 15})

    ax[0].set_xlim([-0.01,1.0])
    ax[0].set_ylim([0.0,1.01])
    ax[1].set_xlim([-0.01,1.0])
    ax[1].set_ylim([0.0,1.01])
    ax[2].set_xlim([-0.01,1.0])
    ax[2].set_ylim([0.0,1.01])

    plt.tight_layout()
    plt.savefig('./test.png')



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
