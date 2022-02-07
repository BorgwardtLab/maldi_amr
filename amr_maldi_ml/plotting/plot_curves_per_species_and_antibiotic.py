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

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from maldi_learn.metrics import specificity_sensitivity_curve
from utils import maldi_col_map_seaborn
from warnings import simplefilter


def interpolate_at(df, x, ascending=True):
    """Interpolate a data frame at certain positions.

    This is an auxiliary function for interpolating an indexed data
    frame at a certain position or at certain positions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame; must have index that is compatible with `x`.

    x : scalar or iterable
        Index value(s) to interpolate the data frame at. Must be
        compatible with the data type of the index.

    ascending : bool
        Determines sorting order of index column. Sorting is
        always performed to ensure proper interpolation.

    Returns
    -------
    pandas.DataFrame
        Data frame evaluated at the specified index positions.
    """
    # Check whether object support iteration. If yes, we can build
    # a sequence index; if not, we have to convert the object into
    # something iterable.
    try:
        _ = (a for a in x)
        new_index = pd.Index(x)
    except TypeError:
        new_index = pd.Index([x])

    # Ensures that the data frame is sorted correctly based on its
    # index. We use `mergesort` in order to ensure stability. This
    # set of options will be reused later on.
    sort_options = {
        'ascending': ascending,
        'kind': 'mergesort',
    }
    df = df.sort_index(**sort_options)

    # TODO: have to decide whether to keep first index reaching the
    # desired level or last. The last has the advantage that it's a
    # more 'pessimistic' estimate since it will correspond to lower
    # thresholds (in case the index is sorted in descending order).
    df = df[~df.index.duplicated(keep='last')]

    # Include the new index, sort again and then finally interpolate the
    # values.
    df = df.reindex(df.index.append(new_index).unique())
    df = df.sort_index(**sort_options)
    df = df.interpolate()

    return df.loc[new_index]


def plot_figure4(args):
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    PATH_fig4 = os.path.join('../../results/calibrated_classifiers',
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
                    'test_calibrated_auprc': [data['test_calibrated_auprc']],
                    'test_calibrated_auroc': [data['test_calibrated_auroc']],
                    }),
                ignore_index=True,
                )

    # subset dataframe only relevant entries
    content = content.query('species==@args.species')

    if args.antibiotic != 'None':
        antibiotic_list = args.antibiotic.split(',')
    else:
        antibiotic_list = set(content['antibiotic'])

    print(content)
    # ------------
    # plot
    # ------------
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(2, 1, figsize=(10, 20))

    # add lines for each antibiotic
    for antibiotic in antibiotic_list:
        content_ab = content.query('antibiotic==@antibiotic')
        col_ab = maldi_col_map_seaborn[antibiotic]

        # extract y_test and y_score from json files
        y_score_total = []
        y_test_total = []

        for filename in content_ab['filename'].values:
            with open(PATH_fig4 + filename) as f:
                data = json.load(f)
                y_score_total.extend([sc[1] for sc in data['y_score']])
                y_test_total.extend(data['y_test'])

        class_ratio = round(
            float(sum(y_test_total)) / len(y_test_total) * 100,
            1,
                            )

        # ------------
        # panel1: ROC curve
        # ------------
        fpr, tpr, thresholds = roc_curve(y_test_total, y_score_total)
        #rocauc = round(roc_auc_score(y_test_total, y_score_total), 2)
        rocauc = np.mean(content_ab['test_calibrated_auroc'].values)
        rocauc = round(rocauc, 2)

        # add zero to string of AUROC if the value does not have 3
        # digits after comma
        pretty_rocauc = [str(roc)+'0' if len(str(roc)) in [3] else str(roc) for roc in [rocauc]]
        lab = '{}\t'.format(antibiotic.lower()).expandtabs()
        while len(lab) < 32:
            lab = '{}\t'.format(lab).expandtabs()
        lab = lab+'AUROC: '+pretty_rocauc[0]

        ax[0].plot(fpr, tpr, color=col_ab, label=lab, linewidth=3.0)
        ax[0].plot([0, 1], [0, 1], color='black', linestyle='--')

        if args.export:
            df = pd.DataFrame.from_dict(
                    {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'threshold': thresholds.tolist(),
                    }
            )

            df = df.set_index('threshold')
            df = interpolate_at(
                df,
                np.linspace(thresholds[0], thresholds[-1], 100, endpoint=True)
            )

            df = df.to_csv(
                f'./{args.outfile}_{antibiotic.lower()}_roc.csv',
                index=False
            )

        # ------------
        # panel2: PRAUC curve
        # ------------
        precision, recall, thresholds = precision_recall_curve(y_test_total,
                                                               y_score_total)
        #prauc = round(average_precision_score(y_test_total, y_score_total,
        #                                      average='weighted'), 2)
        prauc = np.mean(content_ab['test_calibrated_auprc'].values)
        prauc = round(prauc, 2)

        # add zero to string of AUPRC if the value does not have 3
        # digits after comma
        pretty_prauc = [str(pr)+'0' if len(str(pr)) in [3] else str(pr) for pr in [prauc]]
        lab = '{}\t'.format(antibiotic.lower()).expandtabs()
        while len(lab) < 32:
            lab = '{}\t'.format(lab).expandtabs()
        #lab = lab+'AUPRC: '+pretty_prauc[0]
        lab = f'AUPRC: {pretty_prauc[0]} ({class_ratio}%)' 

        ax[1].step(recall, precision, color=col_ab, label=lab,
                   alpha=1.0, where='post', linewidth=3.0)

        if args.export:
            # Add 'fake' threshold to account for length differences.
            thresholds = thresholds.tolist() + [thresholds[-1] + 1e-4]

            df = pd.DataFrame.from_dict(
                    {
                        'precision': precision.tolist(),
                        'recall': recall.tolist(),
                        'threshold': thresholds,
                    }
            )

            df = df.set_index('threshold')
            df = interpolate_at(
                df,
                np.linspace(thresholds[0], thresholds[-1], 100, endpoint=True)
            )

            df = df.to_csv(
                f'./{args.outfile}_{antibiotic.lower()}_pr.csv',
                index=False
            )

    # ------------
    # axes limits and labels
    # ------------
    xy_label_fs = 20
    ax[0].set_xlabel('False Positive Rate (1 - specificity)', fontsize=xy_label_fs)
    ax[0].set_ylabel('True Positive Rate (sensitivity)', fontsize=xy_label_fs)
    ax[1].set_xlabel('Recall', fontsize=xy_label_fs)
    ax[1].set_ylabel('Precision', fontsize=xy_label_fs)
    ax[0].legend(bbox_to_anchor=(0.99, 0.01), loc='lower right',
                 prop={'family': 'DejaVu Sans Mono', 'size': 15})
    ax[1].legend(bbox_to_anchor=(0.99, 0.99), loc='upper right',
                 prop={'family': 'DejaVu Sans Mono', 'size': 15})

    ax[0].set_xlim([-0.01, 1.0])
    ax[0].set_ylim([0.0, 1.01])
    ax[1].set_xlim([-0.01, 1.0])
    ax[1].set_ylim([0.0, 1.01])

    plt.tight_layout()
    plt.savefig(f'./{args.outfile}.png')
    plt.savefig(f'./{args.outfile}.pdf')

    if args.export:
        df_summary = content.groupby(
                ['antibiotic', 'site']).agg({
                    'test_calibrated_auprc': ['mean', 'std'],
                    'test_calibrated_auroc': ['mean', 'std']
                }
        )

        df_summary.to_csv(f'./{args.outfile}_summary.csv', index=False)


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
    parser.add_argument(
        '-e', '--export',
        action='store_true',
        help='If set, export data in CSV format.'
    )

    args = parser.parse_args()

    plot_figure4(args)
