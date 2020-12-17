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
        if 'date' in column:
            df[column] = pd.to_datetime(df[column])

    for group, df_grouped in df.groupby(['species', 'model', 'antibiotic']):
        df_ = df_grouped.groupby('train_to').agg(
            {
                'auroc': [np.mean, np.std]
            }
        )

        df_.columns = df_.columns.to_flat_index()

        df_.rename(
            columns={
                column: '_'.join(column) for column in df_.columns
            },
            inplace=True
        )

        df_.reset_index(inplace=True)

        sns.lineplot(x='train_to', y='auroc_mean', data=df_)

    plt.show()
