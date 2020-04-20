#!/usr/bin/env python
#
# Collection script for all results. Will create a table based on the
# species and the antibiotic and summarise the performance measures.

import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str)

    args = parser.parse_args()

    metrics = ['auroc', 'auprc', 'accuracy']

    rows = []
    filenames = args.INPUT

    # Check if we are getting a directory here, in which case we have to
    # create the list of filenames manually.
    if len(filenames) == 1:
        if os.path.isdir(filenames[0]):
            filenames = glob.glob(os.path.join(filenames[0], '*.json'))
            filenames = sorted(filenames)

    for filename in tqdm(filenames, desc='Loading'):
        with open(filename) as f:

            # Ensures that we can parse normal JSON files
            pos = 0

            for line in f:

                # We found *probably* the beginning of the JSON file, so
                # we can start the parse process from here, having to do
                # a reset.
                if line.startswith('{'):
                    f.seek(pos)
                    break
                else:
                    pos += len(line)

            # Check whether file is empty for some reason. If so, we
            # skip it.
            line = f.readline()
            if line == '':
                continue

            # Not empty, so we need to reset the file pointer
            else:
                f.seek(pos)

            data_raw = json.load(f)

        # Create one row in the table containing the relevant
        # information for now.
        row = {
            'species': data_raw.get('species', 'all'),
            'antibiotic': data_raw['antibiotic'],
            # If no model was found, default to `lr`. This makes us
            # compatible with older files.
            'model': data_raw.get('model', 'lr'),
        }

        # Check whether information about the train and test site is
        # available. If so, we can automatically stratify accordingly.
        if 'train_site' in data_raw and 'test_site' in data_raw:
            row.update({
                'train_site': data_raw['train_site'],
                'test_site': data_raw['test_site']
            })

        # CHeck which metrics are *actually* available in the data. This
        # accounts for experiments with specific train/test valiues, for
        # example.
        metrics_ = list(itertools.chain.from_iterable(
                [[key for key in data_raw if metric in key] for metric
                in metrics]
        ))

        metrics = sorted(metrics_)

        for metric in metrics:
            row[metric] = data_raw[metric] * 100.0

        rows.append(row)

    pd.options.display.max_rows = 999
    pd.options.display.float_format = '{:,.2f}'.format

    df = pd.DataFrame(rows)

    group_columns = ['species', 'antibiotic', 'model']
    if 'train_site' in df.columns and 'test_site' in df.columns:
        group_columns += ['train_site', 'test_site']

    df = df.groupby(group_columns).agg(
        {
            metric: [np.mean, np.std] for metric in metrics
        }
    )

    print(df)
