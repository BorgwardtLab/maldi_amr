#!/usr/bin/env python
#
# Collection script for all results. Will create a table based on the
# species and the antibiotic and summarise the performance measures.

import argparse
import json

import numpy as np
import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str)

    args = parser.parse_args()

    metrics = ['auroc', 'auprc', 'accuracy']

    rows = []

    for filename in tqdm(args.INPUT, desc='Loading'):
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
            'species': data_raw['species'],
            'antibiotic': data_raw['antibiotic'],
        }

        for metric in metrics:
            row[metric] = data_raw[metric] * 100.0

        rows.append(row)

    pd.options.display.max_rows = 999
    pd.options.display.float_format = '{:,.2f}'.format

    df = pd.DataFrame(rows)
    df = df.groupby(['species', 'antibiotic']).agg(
        {
            metric: [np.mean, np.std] for metric in metrics
        }
    )

    print(df)
