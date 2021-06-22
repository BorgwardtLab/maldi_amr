#!/usr/bin/env python
#
# Collect results of curves_per_species_and_antibiotics results
# to compare results of different classifiers in one table.

import itertools
import json
import os
import glob

import numpy as np
import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':

    metrics = [
        'auroc',
        'auprc',
        'accuracy',
    ]

    rows = []
    prjdir = '../../results/calibrated_classifiers'

    filenames = []
    for model in ['lr', 'lightgbm', 'mlp']:
        filenames += glob.glob(prjdir+f'/{model}/*_Antibiotic_Ceftriaxone*')
        filenames += glob.glob(prjdir+f'/{model}/*_Antibiotic_Oxacillin*')


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
            'model': data_raw['model'],
        }

        # Check which metrics are *actually* available in the data. This
        # accounts for experiments with specific train/test values, for
        # example.
        metrics_ = list(itertools.chain.from_iterable(
                [[key for key in data_raw if metric in key] for metric
                 in metrics]
        ))

        # Remove all 'train' metrics.
        metrics_ = [
            metric for metric in metrics_ if 'train' not in metric
        ]
        metrics_ = [
            metric for metric in metrics_ if 'calibrate' not in metric
        ]

        metrics = sorted(metrics_)

        for metric in metrics:
            # We collate here for simplicity reasons...
            if type(data_raw[metric]) is list:
                data_raw[metric] = np.mean(data_raw[metric])

            row[metric] = data_raw[metric]

        rows.append(row)

    pd.options.display.max_rows = 999
    pd.options.display.float_format = '{:,.2f}'.format

    df = pd.DataFrame(rows)

    group_columns = ['species', 'antibiotic', 'model']

    aggregation_fns = [np.mean, np.std]

    # Create a data frame that contains metrics over all the different
    # seeds. Each species--antibiotic combination is represented here.
    df = df.groupby(group_columns).agg(
            {
                metric: aggregation_fns for metric in metrics
            }
        )

    df = df.round(2)
    
    # combine mean and std in one column to fit the layout
    df['accuracy'] = df[('test_accuracy', 'mean')].astype('str') + \
                     ' ± ' + \
                     df[('test_accuracy', 'std')].astype('str')
    df['AUROC'] = df[('test_auroc', 'mean')].astype('str') + \
                  ' ± ' + \
                  df[('test_auroc', 'std')].astype('str')
    df['AUPRC'] = df[('test_auprc', 'mean')].astype('str') + \
                  ' ± ' + \
                  df[('test_auprc', 'std')].astype('str')

    # delete columns not meant to show up in the final table
    df.drop(columns=['test_auroc'], inplace=True)
    df.drop(columns=['test_auprc'], inplace=True)
    df.drop(columns=['test_accuracy'], inplace=True)

    df.to_csv('../tables/curves_per_species_table.csv', sep=',')
    print(df)
