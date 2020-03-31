"""
Plotting script to create

Figure 2 - VME curve plots: very-major-error vs. major-erro curves

MALDI-TOF spectra based AMR prediction using an ensemble of all species
"""

import os
import json
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from maldi_learn.metrics import vme_curve
from utilities import maldi_col_map


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

    # -------------
    # plot vme plots
    # -------------

    sns.set(style="whitegrid",
            font_scale=4)
    fig, ax = plt.subplots(figsize=(22, 22))

    # add lines for each antibiotic
    for antibiotic in antibiotic_list:
        print(antibiotic)
        content_ab = content.query('antibiotic==@antibiotic')
        print(content_ab)
        assert content_ab.shape == (20, 5)

        content_spectra = content_ab.query("species=='all'")

        col_ab = maldi_col_map[antibiotic]

        # extract y_test and y_score from json files
        y_score_total = []
        y_test_total = []

        for filename in content_spectra['filename'].values:
            with open(PATH_fig2 + filename) as f:
                data = json.load(f)
                y_score_total.extend([sc[1] for sc in data['y_score']])
                y_test_total.extend(data['y_test'])

        # plot VME curve
        vme, me_inv, thresholds = vme_curve(y_test_total, y_score_total)
        me = 1-me_inv
        ax.step(vme, me, color=col_ab,
                label=antibiotic,
                alpha=1.0,
                where='post',
                linewidth=3.0)

    if args.antibiotic == 'None':
        plt.legend(loc='upper right', fontsize='x-small')
    else:
        plt.legend(loc='upper right', fontsize='medium')
    plt.ylabel('very major error')
    plt.xlabel('major error')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(f'./{args.outfile}.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--antibiotic',
                        type=str,
                        default=None)
    parser.add_argument('--outfile',
                        type=str,
                        default='fig2_vmeplot')
    args = parser.parse_args()

    plot_figure2(args)
