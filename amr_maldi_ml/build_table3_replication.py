#!/usr/bin/env python
#
# Collection script for all results. Will create a table based on the
# species and the antibiotic and summarise the performance measures.

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from tqdm import tqdm


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
        '-i', '--ignore',
        type=str,
        help='If set, ignores files that contain the specified string.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='plots/tables/Table3_replication.csv',
        help='If set, output table to filename.'
    )

    args = parser.parse_args()
    
    metrics = [
            'test_source_auroc', 
            'test_target_auroc',
            'test_source_auprc',
            'test_source_auprc',
               ]

    folds = ['0', '1', '2']

    models = ['lr', 'lightgbm']

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

        for fold in folds:
            for metric in metrics:
                row[metric] = data_raw[fold][metric] * 100.0
            rows.append(row.copy())

    pd.options.display.max_rows = 999
    pd.options.display.float_format = '{:,.2f}'.format

    df = pd.DataFrame(rows)
    
    # subset dataframe by define models.
    df = df.loc[df['model'].isin(models)]

    # lowercase antibiotic names
    df['antibiotic'] = df['antibiotic'].str.lower()

    group_columns = ['species', 'antibiotic', 'model']
    if 'train_site' in df.columns and 'test_site' in df.columns:
        group_columns += ['train_site', 'test_site']

    # Create a data frame that contains metrics over all the different
    # seeds. Each species--antibiotic combination is represented here.
    df = df.groupby(group_columns).agg(
            {
                metric: [np.mean, np.std] for metric in metrics
            }
        )

    # TODO Order columns that metrics which should be compared are next 
    # to each other

    # round to two digits
    df = df.round(2)

    df.to_csv(args.output)
    print(df)
