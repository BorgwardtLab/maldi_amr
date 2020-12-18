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
    parser.add_argument(
        '-m', '--metric',
        type=str,
        default='auroc',
        help='Metric to use for plotting'
    )

    parser.add_argument(
        '-d', '--date-column',
        type=str,
        default='train_to',
        help='Date column to use for visualisation'
    )

    args = parser.parse_args()

    # Create new data frame by concatenating all the JSON files there
    # are. This assumes that the fields are the same in all files. It
    # will inevitably *fail* if this is not the case.

    rows = []

    for filename in args.INPUT:
        with open(filename) as f:
            data = json.load(f)
            rows.append(data)

    df = pd.DataFrame(rows)

    # Convert all columns to dates. This simplifies the `seaborn`
    # handling below as it will recognise the special structure of
    # the data frame.

    for column in df.columns:
        if column.endswith('_to') or column.endswith('_from'):
            df[column] = pd.to_datetime(df[column])

    # Create new column that describes the whole scenario.
    df['scenario'] = df['species'] + ' (' + df['antibiotic'] + ')'

    df = df.sort_values(['scenario', args.date_column])
    df[args.metric] *= 100

    # Some debug output, just so all values can be seen in all their
    # glory.
    print(
        df.groupby([args.date_column, 'species', 'antibiotic']).agg(
            {
                args.metric: [np.mean, np.std]
            }
        )
    )

    g = sns.lineplot(
        x=args.date_column,
        y=args.metric,
        data=df,
        hue='scenario'
    )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
