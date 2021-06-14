#!/usr/bin/env python
#
# Collect results of union_of_sites_validation experiments
# and compare results of different unions.

import itertools
import argparse
import json
import os
import glob

import numpy as np
import pandas as pd

from tqdm import tqdm


def join_list(data):
    """Join items of a list. Passes through strings."""
    if type(data) == str:
        return data
    else:
        return ' '.join(data)

def process_file(filename):
    """Process single file and return its rows."""
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
            'train_site': join_list(data_raw['train_site']),
            # FIXME Check first if all test sites are the same
            #'test_site': join_list(data_raw['test_site'])
        })

    # Ditto for train years.
    if 'train_years' in data_raw and 'test_years' in data_raw:
        row.update({
            'train_years': ' '.join(data_raw['train_years']),
            'test_years': ' '.join(data_raw['test_years'])
        })

    global metrics

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

    # No metrics founds; check whether we have folds to traverse and
    # collect the data.
    if len(metrics_) == 0:
        folds = [key for key in data_raw if key.isnumeric()]

        # FIXME: this could be extracted automatically, but it is
        # easier for now. Will patch this later.
        metrics_ = [
            # FIXME: disabled for the time being 'test_source_accuracy',
            'test_source_auprc',
            'test_source_auroc',
            # FIXME: disabled for the time being 'test_target_accuracy',
            'test_target_auprc',
            'test_target_auroc'
        ]

        for fold in folds:
            for metric in metrics_:
                name = fold.replace('test_', '')
                name = name.replace('source', 'src')

                row.update({
                    'fold': name,
                    metric: data_raw[fold][metric]
                })

        metrics = metrics_
    else:
        metrics = sorted(metrics_)

        for metric in metrics:
            # We collate here for simplicity reasons...
            if type(data_raw[metric]) is list:
                data_raw[metric] = np.mean(data_raw[metric])

            row[metric] = data_raw[metric] * 100.0

    return row

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--target_site',
        type=str,
        default='DRIAMS-B',
    )
    parser.add_argument(
        '-m', '--metrics',
        type=str,
        default='auroc,auprc',
    )
    args = parser.parse_args()

    metrics = args.metrics.split(',')

    prjdir = '../../results/union_of_sites_validation'

    keep_scenarios = {
        'DRIAMS-B': ['DRIAMS-A', 'DRIAMS-B', 'DRIAMS-A DRIAMS-B', 'DRIAMS-A DRIAMS-C DRIAMS-D', 'DRIAMS-A DRIAMS-B DRIAMS-C DRIAMS-D'],
        'DRIAMS-C': ['DRIAMS-A', 'DRIAMS-C', 'DRIAMS-A DRIAMS-C', 'DRIAMS-A DRIAMS-B DRIAMS-D', 'DRIAMS-A DRIAMS-C DRIAMS-B DRIAMS-D'],
    }

    filenames = []
    filenames.append(glob.glob(prjdir+f'/lightgbm/*Test_site_{args.target_site}*_Species_Escherichia_coli_Antibiotic_Ceftriaxone*'))
    filenames.append(glob.glob(prjdir+f'/mlp/*Test_site_{args.target_site}*_Species_Klebsiella_pneumoniae_Antibiotic_Ceftriaxone*'))
    filenames.append(glob.glob(prjdir+f'/lightgbm/*Test_site_{args.target_site}*_Species_Staphylococcus_aureus_Antibiotic_Oxacillin*'))

    for flist in filenames:
        rows = []
        #for filename in tqdm(flist, desc='Loading'):
        for filename in flist:
            rows.append(process_file(filename))

        pd.options.display.max_rows = 999
        pd.options.display.float_format = '{:,.2f}'.format

        df = pd.DataFrame(rows)

        group_columns = [
                    'species', 'antibiotic', 'model',
                    'train_site',
                    #'test_site',
        ]
        if df['train_years'].isna().any() or df['test_years'].isna().any():
            pass
        else:
            group_columns += ['train_years', 'test_years']

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
        if 'accuracy' in metrics:
            df['accuracy'] = df[('accuracy', 'mean')].astype('str') + \
                             ' ± ' + \
                             df[('accuracy', 'std')].astype('str')
            df.drop(columns=['accuracy'], inplace=True)

        if 'auroc' in metrics:
            df['AUROC'] = df[('auroc', 'mean')].astype('str') + \
                          ' ± ' + \
                          df[('auroc', 'std')].astype('str')
            df.drop(columns=['auroc'], inplace=True)

        if 'auprc' in metrics:
            df['AUPRC'] = df[('auprc', 'mean')].astype('str') + \
                          ' ± ' + \
                          df[('auprc', 'std')].astype('str')
            df.drop(columns=['auprc'], inplace=True)

        print(df)
        keep_indices = [idx for idx in df.index if idx[3] in keep_scenarios[args.target_site]] 
        df = df.loc[keep_indices]
        print('\n\n')
        print(df.transpose())
