"""
Plotting script to create

Figure 4: area-under-the-curve plots for AUROC, AUPRC and AUVME

MALDI-TOF spectra based AMR prediction at species level
"""

import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from maldi_learn.metrics import very_major_error_score, major_error_score, vme_curve, vme_auc_score
from utilities import maldi_col_map

def plot_figure4(args):

    PATH_fig4 = os.path.join('../results/fig4_curves_per_species_and_antibiotics',
                             f'{args.model}/')

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
                ignore_index=True,
                )

    # subset dataframe only relevant entries
    content = content.query('species==@args.species')

    if args.antibiotic != 'None':
        antibiotic_list = args.antibiotic.split(',')
    else:
        antibiotic_list = set(content['antibiotic'])

    # ------------
    # plot
    # ------------
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))

    # add lines for each antibiotic
    for antibiotic in antibiotic_list:
        print(antibiotic)
        content_ab = content.query('antibiotic==@antibiotic')
        col_ab = maldi_col_map[antibiotic]

        # extract y_test and y_score from json files
        y_score_total = []
        y_test_total = []

        for filename in content_ab['filename'].values:
            with open(PATH_fig4 + filename) as f:
                data = json.load(f)
                y_score_total.extend([sc[1] for sc in data['y_score']])
                y_test_total.extend(data['y_test'])

        # ------------
        # panel1: ROC curve
        # ------------
        fpr, tpr, thresholds = roc_curve(y_test_total, y_score_total)
        rocauc = round(roc_auc_score(y_test_total, y_score_total), 3)

        # add zero to string of AUROC if the value does not have 3
        # digits after comma
        pretty_rocauc = [str(roc)+'0' if len(str(roc)) in [3, 4] else str(roc) for roc in [rocauc]]
        lab = '{}\t'.format(antibiotic).expandtabs()
        while len(lab) < 32:
            lab = '{}\t'.format(lab).expandtabs()
        lab = lab+'AUROC: '+pretty_rocauc[0]

        ax[0].plot(fpr, tpr, color=col_ab, label=lab, linewidth=3.0)
        ax[0].plot([0, 1], [0, 1], color='black', linestyle='--')

        # ------------
        # panel2: PRAUC curve
        # ------------
        precision, recall, thresholds = precision_recall_curve(y_test_total,
                                                               y_score_total)
        # TODO did we use weighted average in the main scripts?
        prauc = round(average_precision_score(y_test_total, y_score_total,
                                              average='weighted'), 3)

        # add zero to string of AUPRC if the value does not have 3
        # digits after comma
        pretty_prauc = [str(pr)+'0' if len(str(pr)) in [3, 4] else str(pr) for pr in [prauc]]
        lab = '{}\t'.format(antibiotic).expandtabs()
        while len(lab) < 32:
            lab = '{}\t'.format(lab).expandtabs()
        lab = lab+'AUPRC: '+pretty_prauc[0]

        ax[1].step(recall, precision, color=col_ab, label=lab,
                   alpha=1.0, where='post', linewidth=3.0)

        # ------------
        # panel3: VME curve
        # ------------
        # TODO check all functions for mistakes regarding the class
        # label change
        vme, me_inv, thresholds = vme_curve(y_test_total, y_score_total)
        me = 1-me_inv
        vme_score = round(vme_auc_score(y_test_total, y_score_total), 3)

        # add zero to string of AUVME values and align AUVME values
        pretty_vme = [str(pr)+'0' if len(str(pr)) == 3 else str(pr) for pr in [vme_score]]
        lab = '{}\t'.format(antibiotic).expandtabs()
        while len(lab) < 32:
            lab = '{}\t'.format(lab).expandtabs()
        lab = lab+'AUVME: '+pretty_vme[0]

        ax[2].step(vme, me, color=col_ab, label=lab,
                   alpha=1.0, where='post', linewidth=3.0)

    # ------------
    # axes limits and labels
    # ------------
    xy_label_fs = 20
    ax[0].set_xlabel('False Positive Rate', fontsize=xy_label_fs)
    ax[0].set_ylabel('True Positive Rate', fontsize=xy_label_fs)
    ax[1].set_xlabel('Recall', fontsize=xy_label_fs)
    ax[1].set_ylabel('Precision', fontsize=xy_label_fs)
    ax[2].set_xlabel('Very major error', fontsize=xy_label_fs)
    ax[2].set_ylabel('Major error', fontsize=xy_label_fs)
    ax[0].legend(bbox_to_anchor=(0.99, 0.01), loc='lower right',
                 prop={'family': 'DejaVu Sans Mono', 'size': 15})
    ax[1].legend(bbox_to_anchor=(0.99, 0.99), loc='upper right',
                 prop={'family': 'DejaVu Sans Mono', 'size': 15})
    ax[2].legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', 
                 prop={'family': 'DejaVu Sans Mono', 'size': 15})

    ax[0].set_xlim([-0.01, 1.0])
    ax[0].set_ylim([0.0, 1.01])
    ax[1].set_xlim([-0.01, 1.0])
    ax[1].set_ylim([0.0, 1.01])
    ax[2].set_xlim([-0.01, 1.0])
    ax[2].set_ylim([0.0, 1.01])

    plt.tight_layout()
    plt.savefig(f'./{args.outfile}.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--species',
                        type=str,
                        default='Escherichia coli')
    parser.add_argument('--antibiotic',
                        type=str,
                        default='None')
    parser.add_argument('--outfile',
                        type=str,
                        default='fig4')
    parser.add_argument('--model',
                        type=str,
                        default='lr')
    args = parser.parse_args()

    plot_figure4(args)
