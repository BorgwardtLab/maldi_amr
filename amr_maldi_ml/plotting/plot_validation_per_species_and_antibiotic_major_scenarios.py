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

from utils import scenario_map

scenarios = [
    ('Escherichia coli', 'Ceftriaxone', 'lightgbm'),
    ('Klebsiella pneumoniae', 'Ceftriaxone', 'mlp'),
    ('Staphylococcus aureus', 'Oxacillin', 'lightgbm'),
]

map_models = {
    'lightgbm': 'LightGBM',
    'mlp': 'MLP',
}

def plot_figure5(args):

    # --------------
    # create dataframe giving an overview of all files in path
    # --------------
    file_list = []
    for scenario in scenarios:
        searchstr = os.path.join(
                    '../../results/validation_per_species_and_antibiotic',
                    f'{scenario[2]}/*{scenario[1]}*.json')

        files = glob.glob(searchstr)
        file_list.extend(files)

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
    pd.options.display.max_rows = 999

    # ------------
    # for each antibiotic, get avg metrics for 'all' and 'all (w/o spectra)'
    # ------------

    values = [pd.DataFrame(columns=[])]*len(scenarios)

    for i, scenario in enumerate(scenarios):

        # add lines for all train-test scenarios
        content_ = content.copy()
        content_ = content_.query('species==@scenario[0]')
        content_ = content_.query('antibiotic==@scenario[1]')
        content_ = content_.query('model==@scenario[2]')

        for tr_site in ['DRIAMS-D','DRIAMS-C','DRIAMS-B','DRIAMS-A']:
            for te_site in ['DRIAMS-D','DRIAMS-C','DRIAMS-B','DRIAMS-A']:
                content_scenario = content_.query("train_site==@tr_site")
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
                    values[i] = values[i].append(
                        pd.DataFrame({
                            'antibiotic': [scenario[1]],
                            'species': [scenario[0]],
                            'train_site': [tr_site],
                            'test_site': [te_site],
                            'result': [result_mean_all],
                            'result_std_all': [result_std_all],
                            }),
                        ignore_index=True,
                        sort=False,
                        )
                else:
                    values[i] = values[i].append(
                        pd.DataFrame({
                            'antibiotic': [scenario[1]],
                            'species': [scenario[0]],
                            'train_site': [tr_site],
                            'test_site': [te_site],
                            'result': [np.nan],
                            'result_std_all': [np.nan],
                            }),
                        ignore_index=True,
                        sort=False,
                        )

    # -------------
    # plot heatplot
    # -------------
    print(f'plotting.. {args.outfile}')
    #assert len(values) == 16
    rc = {
            'legend.fontsize': 8,
            'axes.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.labelsize': 10,
          }

    plt.close('all')
    sns.set(style="whitegrid",
            rc=rc,
            font_scale=1.1
    )
   
    fig, ax = plt.subplots(1, 3, figsize=(20,8))

    for i in range(len(ax)):
        # heatmap 
        matrix = values[i].pivot("train_site", "test_site", "result")
        print(i)
        print(matrix)
        sns.heatmap(matrix, 
                    annot=True, 
                    fmt=".2f",
                    linewidths=.75,
                    vmin=0.5, 
                    vmax=1,
                    ax=ax[i],
                    cmap=sns.cubehelix_palette(8, start=.5, rot=-.75),
                    cbar=True if i==len(ax)-1 else False,
                    cbar_kws={'label': 'AUROC'}
        )

        # adjust y axis label position
        yticks = ax[i].get_yticks()
        ax[i].set_yticks([i-0.3 for i in yticks])
        ax[i].set_yticklabels(ax[i].get_xticklabels())

        ax[i].set_title(f'{scenario_map[scenarios[i][0].replace(" ", "_")]} ({map_models[scenarios[i][2]]})')
        ax[i].set_ylabel('training')    
        ax[i].set_xlabel('testing')

    for ax_ in ax:
        ax_.label_outer()
    # adjust one yticks explicitly, as they are not covered
    # by the command above
    ax[2].set_yticks([])
    ax[2].set_ylabel('')
    plt.subplots_adjust(wspace=0.01)
    plt.savefig(f'../plots/validation_per_species_and_antibiotic/{args.outfile}_combined.png')
    plt.savefig(f'../plots/validation_per_species_and_antibiotic/{args.outfile}_combined.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric',
                        type=str,
                        default='auroc')
    parser.add_argument('--outfile',
                        type=str,
                        default='validation_per_species_and_antibiotic')
    args = parser.parse_args()

    plot_figure5(args)
