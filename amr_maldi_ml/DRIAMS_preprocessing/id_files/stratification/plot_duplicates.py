"""Plot all spectra with same stratification id from folders."""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_duplicates(args):
    filenames = args.INPUT

    sns.set_context("paper", font_scale=2)
    fig, ax = plt.subplots(figsize=(100,18))

    for i, f in enumerate(filenames):
        df = pd.read_table(f, sep=' ')
        ax.plot('bin_index',
                'binned_intensity',
                data=df,
                label=str(i),
               )
    
    ax.set_xlabel('bin number')
    ax.set_ylabel('intensity')

    split_id = f.split('/')[-2]
    if args.output:
        filename = args.output+'.png'
    else:
        filename = f'./plots/{split_id}' + \
                   '.png'

    plt.tight_layout()
    plt.savefig(filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT',
                        nargs='+',
                        required=True,
                        )
    parser.add_argument('--output', '-o',
                        default=None,
                        type=str)
    args = parser.parse_args()

    plot_duplicates(args)
