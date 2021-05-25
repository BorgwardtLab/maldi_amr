""" Script to plot sensitivity-specificity pairs of rejection threshold. """

import os
import argparse

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
                )
    plt.xlabel('specificity (percentage of susceptible samples correctly identified)')
    plt.ylabel('sensitivity (percentage of resistant samples correctly identified)')

    plt.colorbar()

    filename = os.path.basename(args.INPUT).replace('.csv', '')
    plt.savefig(f'../plots/sensitivity_vs_specificity/{filename}.png')
