#!/usr/bin/env python3

import argparse
import glob
import json
import os

import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('DIRECTORY')

    parser.add_argument(
        '-m', '--metric',
        type=str,
        help='Performance metric to monitor',
        default='auroc'
    )

    parser.add_argument(
        '-t', '--threshold',
        type=float,
        help='Threshold for the given performance metric',
        default='0.90',
    )

    args = parser.parse_args()

    filenames = sorted(glob.glob(os.path.join(args.DIRECTORY, '*.json')))

    # Will contain the rows for a nice data frame that assesses the
    # dependency between samples and performance metrics.
    rows = []

    for filename in tqdm(filenames, desc='File'):
        with open(filename) as f:
            data = json.load(f)

            rows.append({
                'n_samples': data['n_samples'],
                'accuracy': data['accuracy'],
                'auprc': data['auprc'],
                'auroc': data['auroc'],
                'species': data['species'],
                'antibiotic': data['antibiotic']
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(['n_samples', 'species', 'antibiotic'])

    # Keep only the *first* row, i.e. the row with the smallest number
    # of samples.
    df = df[df[args.metric] >= args.threshold].groupby(
        ['species', 'antibiotic']
    ).first()

    print(f'Metric: {args.metric}')
    print(f'Threshold: {args.threshold}')

    n_samples_mean = df['n_samples'].mean()
    print(f'Require on average {n_samples_mean:.0f} samples')
