""" Script to plot sensitivity-specificity pairs of rejection threshold. """

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str)
    args = parser.parse_args()

    # read in threshold table
    df = pd.read_csv(args.INPUT)

    # plot sensitivity vs. specificity
    plt.figure(figsize=(10,10))
    sns.set(style="whitegrid")

    plt.scatter(x=df['specificity'], 
                y=df['sensitivity'], 
                s=4, 
                cmap='tab10',
                c=df['percentage rejected samples'],
                vmin=0,
                vmax=100,
                )
    plt.xlabel('specificity (percentage of susceptible samples correctly identified)')
    plt.ylabel('sensitivity (percentage of resistant samples correctly identified)')
    plt.xlim((0.0,1.0))
    plt.ylim((0.0,1.0))

    plt.colorbar()

    filename = os.path.basename(args.INPUT).replace('.csv', '')
    plt.savefig(f'../plots/sensitivity_vs_specificity/{filename}.png')

    for thresh_ in np.linspace(0,90,10):

        thresh_ = round(thresh_)
        df_ = df.loc[df['percentage rejected samples'] >= thresh_]
        print()
        print(thresh_)
        df_ = df_.loc[df_['percentage rejected samples'] < thresh_+10]

        # plot sensitivity vs. specificity
        plt.figure(figsize=(10,10))
        sns.set(style="whitegrid")

        plt.scatter(x=df_['specificity'], 
                    y=df_['sensitivity'], 
                    s=4, 
                    cmap='tab10',
                    c=df_['percentage rejected samples'],
                    vmin=0,
                    vmax=100,
                    )
        plt.xlabel('specificity (percentage of susceptible samples correctly identified)')
        plt.ylabel('sensitivity (percentage of resistant samples correctly identified)')
        plt.xlim((0.0,1.0))
        plt.ylim((0.0,1.0))
        plt.title(f'{thresh_}')

        plt.colorbar()

        filename = os.path.basename(args.INPUT).replace('.csv', '')
        plt.savefig(f'../plots/sensitivity_vs_specificity/{filename}_{thresh_}.png')
