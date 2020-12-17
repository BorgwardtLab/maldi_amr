"""Plot sliding window validation experiment data."""

import argparse
import json

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, nargs='+', help='Input file(s)')

    args = parser.parse_args()

    rows = []

    for filename in args.INPUT:
        with open(filename) as f:
            data = json.load(f)
            rows.append(data)

    df = pd.DataFrame(rows)

    for column in df.columns:
        if '_to' in column or '_from' in column:
            df[column] = pd.to_datetime(df[column])

    df = df.sort_values('train_to')
    df['auroc'] *= 100

    print(
        df.groupby(['train_to', 'species', 'antibiotic']).agg(
            {
                'auroc': [np.mean, np.std]
            }
        )
    )

    df['scenario'] = df['species'] + ' (' + df['antibiotic'] + ')'

    g = sns.lineplot(
        x='train_to',
        y='auroc',
        data=df,
        hue='scenario'
    )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
