"""
Plotting script to create

Figure 2 - barplots: compare prediction from spectra vs. from species
information.

MALDI-TOF spectra based AMR prediction using an ensemble of all species
"""

import os
import json
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

def plot_figure2(args):

    PATH_fig2 = '../results/fig2_baseline/'

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
                ignore_index=True,
                )

    if args.antibiotic != 'None':
        antibiotic_list = args.antibiotic.split(',')
    else:
        antibiotic_list = set(content['antibiotic'])

    # ------------
    # for each antibiotic, get avg metrics for 'all' and 'all (w/o spectra)'
    # ------------

    values = pd.DataFrame(columns=['antibiotic',
                                   'auroc_all',
                                   'auroc_all_wo_spectra',
                                   ])

    # add lines for each antibiotic
    for antibiotic in antibiotic_list:
        content_ab = content.query('antibiotic==@antibiotic')
        print(content_ab)
        assert content_ab.shape == (20, 5)

        content_spectra = content_ab.query("species=='all'")
        content_wo_spectra = content_ab.query("species=='all (w/o spectra)'")

        # 'all': extract y_test and y_score from json files
        aurocs = []
        class_ratios = []

        for filename in content_spectra['filename'].values:
            with open(PATH_fig2 + filename) as f:
                data = json.load(f)
                aurocs.append(roc_auc_score(data['y_test'],
                              [sc[1] for sc in data['y_score']]))
                assert np.all([x in [0, 1] for x in data['y_test']])
                class_ratios.append(float(sum(data['y_test']))/len(data['y_test'
                                                                        ]))
        auroc_mean_all = round(np.mean(aurocs), 3)
        auroc_std_all = round(np.std(aurocs), 3)
        class_ratio = '{:0.2f}'.format(np.mean(class_ratios))

        # 'all (w/o spectra)': extract y_test and y_score from json files
        aurocs_wo_spectra = []

        for filename in content_wo_spectra['filename'].values:
            with open(PATH_fig2 + filename) as f:
                data = json.load(f)
                aurocs_wo_spectra.append(roc_auc_score(data['y_test'],
                                         [sc[1] for sc in data['y_score']]))

        auroc_mean_all_wo_spectra = round(np.mean(aurocs_wo_spectra), 3)
        auroc_std_all_wo_spectra = round(np.std(aurocs_wo_spectra), 3)
        _, pval = ttest_ind(aurocs, aurocs_wo_spectra, equal_var=False)

        # add to values dataframe
        values = values.append(
            pd.DataFrame({
                'antibiotic': [antibiotic],
                'label': [f'{antibiotic} [{class_ratio}]'],
                'pvals': [pval],
                'auroc_all': [auroc_mean_all],
                'auroc_all_wo_spectra': [auroc_mean_all_wo_spectra],
                'auroc_std_all': [auroc_std_all],
                'auroc_std_all_wo_spectra': [auroc_std_all_wo_spectra],
                }),
            ignore_index=True,
            sort=False,
            )

    # correct Cotrimoxazole spelling
    values = values.replace({'Cotrimoxazol': 'Cotrimoxazole'})

    # -------------
    # plot barplot
    # -------------

    values = values.sort_values(by=['auroc_all'], ascending=False)
    n_ab = len(values)

    sns.set(style="whitegrid",
            font_scale=2)
    fig, ax = plt.subplots(figsize=(22, 15))

    # barplots
    sns.barplot(x="label", y="auroc_all",
                ax=ax, data=values, color=sns.color_palette()[0])
    sns.barplot(x="label", y="auroc_all_wo_spectra",
                ax=ax, data=values, color='firebrick')

    # errorbars
    ax.errorbar(list(range(0, n_ab)),
                values['auroc_all'].values,
                yerr=values['auroc_std_all'].values,
                fmt='o',
                color='black')

    # p-values
    pval_string = ['*' if pv < 0.05 else '' for pv in
                   values['pvals'].values]
    for i, yval in enumerate(values['auroc_all'].values):
        ax.annotate(
                    # '{:.1e}'.format(values['pvals'].iloc[i]),
                    pval_string[i],
                    xy=(i, yval),
                    xytext=(i-0.15,  # -0.2 for text, -0.15 for stars
                            yval+values['auroc_std_all'].iloc[i]+0.01),
                    color='black',
                    fontsize=16,
                    rotation=90)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.ylabel('AUROC')
    plt.xlabel('')
    plt.ylim(0.5, 1.06)
    plt.xlim(0-0.5, n_ab-0.5)
    plt.tight_layout()
    plt.savefig(f'./{args.outfile}.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--antibiotic',
                        type=str,
                        default='None')
    parser.add_argument('--outfile',
                        type=str,
                        default='fig2_barplot')
    args = parser.parse_args()

    plot_figure2(args)
