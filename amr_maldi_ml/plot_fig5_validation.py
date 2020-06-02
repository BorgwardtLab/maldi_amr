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

    PATH_fig5 = os.path.join('../results/fig5_validation/',
                             f'{args.model}')

    # --------------
    # create dataframe giving an overview of all files in path
    # --------------
    file_list = []
    for (_, _, filenames) in os.walk(PATH_fig5):
        [file_list.append(f) for f in filenames if '.json' in f]
        break

    content = pd.DataFrame(columns=[])

    for filename in file_list:
        with open(os.path.join(PATH_fig5, filename)) as f:
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
                    'result': [0.00],
                    'result_std_all': [0.00],
                    }),
                ignore_index=True,
                sort=False,
                )

    # add lines for each antibiotic
    for antibiotic in antibiotic_list:
        content_ab = content.query('antibiotic==@antibiotic')
        content_model = content_ab.query("model==@args.model")

        for tr_site in ['DRIAMS-D','DRIAMS-C','DRIAMS-B','DRIAMS-A']:
            for te_site in ['DRIAMS-D','DRIAMS-C','DRIAMS-B','DRIAMS-A']:
                results = []
                class_ratios = []
                
                content_tr = content_model.query("train_site==@tr_site")
                content_scenario = content_tr.query("test_site==@te_site")
                print(tr_site, te_site)
                print(content_scenario)
                assert content_scenario.shape[0]==10

                for filename in content_scenario['filename'].values:
                    with open(os.path.join(PATH_fig5, filename)) as f:
                        data = json.load(f)
                        #results.append(roc_auc_score(data['y_test'],
                        #              [sc[1] for sc in data['y_score']]))
                        results.append(data[f'{args.metric}'])
                        assert np.all([x in [0, 1] for x in data['y_test']])
                        class_ratios.append(float(sum(data['y_test']))/len(data['y_test'
                                                                                ]))
                result_mean_all = round(np.mean(results), 3)
                result_std_all = round(np.std(results), 3)
                class_ratio = '{:0.2f}'.format(np.mean(class_ratios))

                # add to values dataframe
                values = values.append(
                    pd.DataFrame({
                        'antibiotic': [antibiotic],
                        'train_test': [tr_site+'_'+te_site],
                        'result': [result_mean_all],
                        'result_std_all': [result_std_all],
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
    print(f'plotting.. {args.outfile}')
    n_ab = len(antibiotic_list)
    rc = {
            'legend.fontsize': 8,
            'axes.labelsize': 8,
            'xtick.labelsize': 8,
          }

    sns.set(style="whitegrid",
            rc=rc,
            font_scale=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # barplots BACK
    sns.barplot(x="antibiotic", y=f"result", 
                hue="train_test", 
                hue_order=['DRIAMS-A_DRIAMS-A',
                           'DRIAMS-B_DRIAMS-B',
                           'DRIAMS-C_DRIAMS-C',
                           'DRIAMS-D_DRIAMS-D',
                           ],
                data=values,  
                ax=ax,
                )

    # barplots FRONT
    sns.barplot(x="antibiotic", y=f"result", 
                hue="train_test", 
                hue_order=[
                           'EMPTY',
                           'DRIAMS-A_DRIAMS-B',
                           'DRIAMS-A_DRIAMS-C',
                           'DRIAMS-A_DRIAMS-D',
                           ],
                data=values,  
                ax=ax, hatch='//'
                )

    # adjust legend
    legend = ax.get_legend()
    legend.legendHandles = [legend.legendHandles[i] for i in [0,1,2,3,5,6,7]]
    ax.legend(handles=legend.legendHandles,
              labels=['train: A - test: A',
                      'train: B - test: B',
                      'train: C - test: C',
                      'train: D - test: D',
                      'train: A - test: B',
                      'train: A - test: C',
                      'train: A - test: D',
                      ],
               loc='center left', 
               #fontsize='small',
               bbox_to_anchor=(1.03,0.5),
               )

    ylabel_map = {
            'auroc': 'AUROC',
            'auprc': 'AUPRC',
            'accuracy': 'accuracy',
            }

    sns.despine(left=True)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel_map[args.metric])
    plt.xlabel('')
    if args.metric=='auroc':
        plt.ylim(0.5, 1.03)
    else:
        plt.ylim(0.0, 1.06)
    plt.xlim(0-0.5, n_ab-0.5)
    plt.tight_layout()
    plt.savefig(f'./{args.outfile}.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--antibiotic',
                        type=str,
                        default='None')
    parser.add_argument('--metric',
                        type=str,
                        default='auroc')
    parser.add_argument('--model',
                        type=str,
                        default='lr')
    parser.add_argument('--outfile',
                        type=str,
                        default='fig5_plot')
    args = parser.parse_args()

    plot_figure5(args)
