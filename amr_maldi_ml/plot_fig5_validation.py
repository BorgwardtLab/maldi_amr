"""
Plotting script to create

Figure 5

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

def plot_figure5(args):

    PATH_fig5 = '../results/fig5_validation/'

    # --------------
    # create dataframe giving an overview of all files in path
    # --------------
    file_list = []
    for (_, _, filenames) in os.walk(PATH_fig5):
        [file_list.append(f) for f in filenames if '.json' in f]
        break

    content = pd.DataFrame(columns=[])

    for filename in file_list:
        with open(PATH_fig5 + filename) as f:
            data = json.load(f)
            content = content.append(
                pd.DataFrame({
                    'filename': [filename],
                    'antibiotic': [data['antibiotic']],
                    'train_site': [data['train_site']],
                    'test_site': [data['test_site']],
                    'model': [data['model']],
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

    values = pd.DataFrame(columns=[])
    # append empty entry
    for antibiotic in antibiotic_list:
        values = values.append(
                pd.DataFrame({
                    'antibiotic': [antibiotic],
                    'train_test': ['EMPTY'],
                    'auroc': [0.00],
                    'auroc_std_all': [0.00],
                    }),
                ignore_index=True,
                sort=False,
                )

    # add lines for each antibiotic
    for antibiotic in antibiotic_list:
        content_ab = content.query('antibiotic==@antibiotic')
        content_model = content_ab.query("model==@args.model")

        for tr_site in ['DRIAMS-C','DRIAMS-B','DRIAMS-A']:
            for te_site in ['DRIAMS-C','DRIAMS-B','DRIAMS-A']:
                aurocs = []
                class_ratios = []
                
                content_tr = content_model.query("train_site==@tr_site")
                content_scenario = content_tr.query("test_site==@te_site")
                assert content_scenario.shape[0]==10

                for filename in content_scenario['filename'].values:
                    with open(PATH_fig5 + filename) as f:
                        data = json.load(f)
                        aurocs.append(roc_auc_score(data['y_test'],
                                      [sc[1] for sc in data['y_score']]))
                        assert np.all([x in [0, 1] for x in data['y_test']])
                        class_ratios.append(float(sum(data['y_test']))/len(data['y_test'
                                                                                ]))
                auroc_mean_all = round(np.mean(aurocs), 3)
                auroc_std_all = round(np.std(aurocs), 3)
                class_ratio = '{:0.2f}'.format(np.mean(class_ratios))

                # add to values dataframe
                values = values.append(
                    pd.DataFrame({
                        'antibiotic': [antibiotic],
                        'train_test': [tr_site+'_'+te_site],
                        'auroc': [auroc_mean_all],
                        'auroc_std_all': [auroc_std_all],
                        }),
                    ignore_index=True,
                    sort=False,
                    )

    # correct Cotrimoxazole spelling
    values = values.replace({'Cotrimoxazol': 'Cotrimoxazole'})
    values = values.sort_values(by=['antibiotic'])
    print(values['train_test'].unique())
    print(values.loc[values['antibiotic']=='Ciprofloxacin'])

    # -------------
    # plot barplot
    # -------------
    print('plotting..')
    n_ab = len(antibiotic_list)
    rc = {'legend.fontsize': 10}

    sns.set(style="whitegrid",
            rc=rc,
            font_scale=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # barplots BACK
    sns.barplot(x="antibiotic", y="auroc", 
                hue="train_test", 
                hue_order=['DRIAMS-A_DRIAMS-A',
                           'DRIAMS-B_DRIAMS-B',
                           'DRIAMS-C_DRIAMS-C',
                           ],
                data=values,  
                ax=ax,
                )

    # barplots FRONT
    sns.barplot(x="antibiotic", y="auroc", 
                hue="train_test", 
                hue_order=[
                           'EMPTY',
                           'DRIAMS-A_DRIAMS-B',
                           'DRIAMS-A_DRIAMS-C',
                           ],
                data=values,  
                ax=ax, hatch='//'
                )

    # adjust legend
    legend = ax.get_legend()
    legend.legendHandles = [legend.legendHandles[i] for i in [0,1,2,4,5]]
    ax.legend(handles=legend.legendHandles,
              labels=['train: A - test: A',
                      'train: B - test: B',
                      'train: C - test: C',
                      'train: A - test: B',
                      'train: A - test: C',
                      ],
               loc='center left', 
               bbox_to_anchor=(1.03,0.5))

    sns.despine(left=True)
    plt.xticks(rotation=90)
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
                        default='fig5_plot')
    parser.add_argument('--model',
                        type=str,
                        default='lr')
    args = parser.parse_args()

    plot_figure5(args)
