"""
Plotting script to create

Figure 5

MALDI-TOF spectra based AMR prediction using an ensemble of all species
"""

import os
import json
import argparse
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

def plot_figure5(args):

    # --------------
    # create dataframe giving an overview of all files in path
    # --------------
    files = os.path.join(
                '../../results/validation_per_species_and_antibiotic',
                f'{args.model}/*.json',
                         )
    file_list = glob.glob(files)

    content = pd.DataFrame(columns=[])

    for filename in file_list:
        with open(filename) as f:
            data = json.load(f)
            content = content.append(
                pd.DataFrame({
                    'filename': [filename],
                    'antibiotic': [data['antibiotic']],
                    'species': [data['species']],
                    'train_site': [data['train_site']],
                    'test_site': [data['test_site']],
                    'model': [data['model']],
                    'seed': [data['seed']],
                    }),
                ignore_index=True,
                )

    # ------------
    # for each antibiotic, get avg metrics for 'all' and 'all (w/o spectra)'
    # ------------

    values = pd.DataFrame(columns=[])

    # add lines for all train-test scenarios
    content = content.query('antibiotic==@args.antibiotic')
    content = content.query('species==@args.species')
    content = content.query("model==@args.model")
    print(content)

    for tr_site in ['DRIAMS-D','DRIAMS-C','DRIAMS-B','DRIAMS-A']:
        for te_site in ['DRIAMS-D','DRIAMS-C','DRIAMS-B','DRIAMS-A']:
            content_scenario = content.query("train_site==@tr_site")
            content_scenario = content_scenario.query("test_site==@te_site")
            if content_scenario.shape[0]>0:
                assert content_scenario.shape[0]==10

                results = []
                for filename in content_scenario['filename'].values:
                    with open(filename) as f:
                        data = json.load(f)
                        # TODO make dynamic for different metrics
                        results.append(roc_auc_score(data['y_test'],
                                      [sc[1] for sc in data['y_score']]))
                        assert np.all([x in [0, 1] for x in data['y_test']])
                result_mean_all = round(np.mean(results), 3)
                result_std_all = round(np.std(results), 3)

                # add to values dataframe
                values = values.append(
                    pd.DataFrame({
                        'antibiotic': [args.antibiotic],
                        'species': [args.species],
                        'train_site': [tr_site],
                        'test_site': [te_site],
                        'result': [result_mean_all],
                        'result_std_all': [result_std_all],
                        }),
                    ignore_index=True,
                    sort=False,
                    )
            else:
                values = values.append(
                    pd.DataFrame({
                        'antibiotic': [args.antibiotic],
                        'species': [args.species],
                        'train_site': [tr_site],
                        'test_site': [te_site],
                        'result': [np.nan],
                        'result_std_all': [np.nan],
                        }),
                    ignore_index=True,
                    sort=False,
                    )

    # -------------
    # plot barplot
    # -------------
    print(f'plotting.. {args.outfile}')
    assert len(values) == 16
    rc = {
            'legend.fontsize': 8,
            'axes.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.labelsize': 10,
          }

    sns.set(style="whitegrid",
            rc=rc,
            font_scale=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # heatmap 
    matrix = values.pivot("train_site", "test_site", "result")
    print(matrix)
    ax = sns.heatmap(matrix, 
                     annot=True, 
                     fmt=".2f",
                     linewidths=.75,
                     vmin=0.5, 
                     vmax=1,
                     cmap=sns.cubehelix_palette(8, start=.5, rot=-.75),
                     )

    # adjust y axis label position
    yticks = ax.get_yticks()
    ax.set_yticklabels(ax.get_xticklabels())
    ax.set_yticks([i-0.3 for i in yticks])

    ax.set_ylabel('training')    
    ax.set_xlabel('testing')
    plt.tight_layout()
    plt.savefig(f'../plots/validation_per_species_and_antibiotic/{args.outfile}.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--species',
                        type=str,
                        default='None')
    parser.add_argument('--antibiotic',
                        type=str,
                        default='None')
    parser.add_argument('--metric',
                        type=str,
                        default='auroc')
    parser.add_argument('-m', '--model',
                        type=str,
                        default='lr')
    parser.add_argument('--outfile',
                        type=str,
                        default='validation_per_species_and_antibiotic')
    args = parser.parse_args()

    plot_figure5(args)
