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
            })

    df = pd.DataFrame(rows)
    print(df.to_csv(index=False))
