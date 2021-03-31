#!/usr/bin/env python
#
# Collection script for all results. Will create a table based on the
# species and the antibiotic and summarise the performance measures.

import argparse
import itertools
import json
import os
import tabulate

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score


def get_files(directory):
    """Walk a directory structure and return JSON filenames.

    This function is a helper function for finding JSON filenames in
    a recursive fashion, i.e. by fully walking a directory and *all*
    its subdirectories.

    Parameters
    ----------
    directory : str
        Root folder for the directory enumeration

    Returns
    -------
    List of JSON files (ready for reading; the whole path of each file
    will be returned).
    """
    result = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.json':
                result.append(os.path.join(root, filename))

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str)

    parser.add_argument(
        '-a', '--aggregate',
        type=str,
        default=['mean', 'std'],
        nargs='+',
        help='Provide aggregation functions for each metric. By default, '
             'mean and standard deviation will be calculated.'
    )

    parser.add_argument(
        '-i', '--ignore',
        type=str,
        help='If set, ignores files that contain the specified string.'
    )

    parser.add_argument(
        '-m', '--metrics',
        type=str,
        default='auroc,auprc,accuracy',
        help='String to define metrics that will be displayed, separated by '
             'comma.'
    )

    args = parser.parse_args()

    metrics = args.metrics.split(',')

    metric_fn = {
        'auroc': accuracy_score,
        'auprc': average_precision_score,
        'accuracy': accuracy_score,
    }

    rows = []
    filenames = args.INPUT

    # Check if we are getting a directory here, in which case we have to
    # create the list of filenames manually.
    if len(filenames) == 1:
        if os.path.isdir(filenames[0]):
            filenames = get_files(filenames[0])
            filenames = sorted(filenames)

    if args.ignore is not None:
        filenames = [fn for fn in filenames if args.ignore not in fn]

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

        # Ditto for train years.
        if 'train_years' in data_raw and 'test_years' in data_raw:
            row.update({
                'train_years': ' '.join(data_raw['train_years']),
                'test_years': ' '.join(data_raw['test_years'])
            })

        # Ditto for workstations.
        if 'meta_test_workstation' in data_raw and 'y_score' in data_raw:
            row.update({
                'meta_test_workstation': data_raw['meta_test_workstation'],
                'y_test': data_raw['y_test'],
                'y_pred': data_raw['y_pred'],
                'y_score': [pair[1] for pair in data_raw['y_score']],
            })

        rows.append(row)

    pd.options.display.max_rows = 999
    pd.options.display.float_format = '{:,.2f}'.format

    df = pd.DataFrame(rows)

    group_columns = ['species', 'antibiotic', 'model']
    if 'train_site' in df.columns and 'test_site' in df.columns:
        group_columns += ['train_site', 'test_site']
    if 'train_years' in df.columns and 'test_years' in df.columns:
        group_columns += ['train_years', 'test_years']

    # Create a data frame that contains metrics over all the different
    # seeds. Each species--antibiotic combination is represented here.
    df = df.groupby(group_columns).agg({'sum'})
    df.columns = df.columns.get_level_values(0)

    # Aggregate dataframe which appends the lists of y scores and labels
    # to have one large list for metric calculation.
    workstations = np.unique(
             df['meta_test_workstation'].agg({'sum'}).values[0]
                             ).tolist()

    # Calculate metrics per workstation
    n = len(df)
    rows = []

    for row_index in range(n):
        row = {}
        for workstation in workstations:
            # Skip if workstation not present in current scenario.
            if workstation not in df['meta_test_workstation'].iloc[row_index]:
                row[f'{workstation}_{metric}'] = 'NA'
                continue 

            mask = np.array(df['meta_test_workstation'].iloc[row_index]) == workstation
            m = len(mask)
            y_test_ws = [df['y_test'].iloc[row_index][i] for i in range(m) if mask[i]] 
            y_score_ws = [df['y_score'].iloc[row_index][i] for i in range(m) if mask[i]] 
            y_pred_ws = [df['y_pred'].iloc[row_index][i] for i in range(m) if mask[i]] 
            
            for metric in metrics:
                # Due to low sample size, often only one class will be present.
                if len(np.unique(y_test_ws)) == 1:
                    row[f'{workstation}_{metric}'] = np.nan
                else: 
                    row[f'{workstation}_{metric}'] = metric_fn[metric](y_test_ws, y_pred_ws)
                

        rows.append(row)

    df_ = pd.DataFrame(rows)
    df_.index = df.index

    # append with rows
    for col in df_.columns:
        df[col] = df_[col]
    df = df.drop(columns=['y_test', 'y_pred', 'y_score', 'meta_test_workstation'])

    print(df)
