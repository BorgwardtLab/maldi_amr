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
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_temporal_validation(df):

    scenarios = [
            ('Staphylococcus aureus', 'oxacillin'),
            ('Escherichia coli', 'ceftriaxone'),
            ('Klebsiella pneumoniae', 'ceftriaxone'),
            ]

    metric = 'auprc'

    models = ['lr', 'lightgbm']

    single_years = ['2015', '2016', '2017']
    cumulative_years = ['2015 2016 2017', '2016 2017', '2017'] 

    model_map = {
            'lr': sns.color_palette()[0],
            'lightgbm': sns.color_palette()[1],
                 }

    metric_map = {
            'auprc': 'AUPRC',
            'auroc': 'AUROC',
            }
    species_map = {
            'Staphylococcus aureus': 'S. aureus',
            'Escherichia coli': 'E. coli',
            'Klebsiella pneumoniae': 'K. pneumoniae',
                    }
    # plot
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for i, (species, antibiotic) in enumerate(scenarios):
        print((species, antibiotic))
        for model in models:

            df_ = df.loc[(
                f'{species}', f'{antibiotic}', model, slice(None),'2018'
                          ), 
                         (
                f'{metric}', 'mean'
                          ),
                         ].reset_index()
            print(df_)


            # plot single years curve
            df__ = df_.loc[
                    df_['train_years'].isin(single_years)
                           ]
            
            y_vals = []
            for year in single_years:
                y_vals.append(df__.loc[
                    df__['train_years']==year
                                       ][(f'{metric}','mean')].values[0])
            ax[i].plot(
                y_vals, 
                color=model_map[model],
                linestyle='--',
                marker='v',
                label=f'{model} (single years)'
                           )
            
            # plot cumulative years curve
            df__ = df_.loc[
                    df_['train_years'].isin(cumulative_years)
                           ]

            y_vals = []
            for year in cumulative_years:
                y_vals.append(df__.loc[
                    df__['train_years']==year
                                       ][(f'{metric}','mean')].values[0])
            ax[i].plot(
                y_vals, 
                color=model_map[model],
                marker='o',
                label=f'{model} (cumulative years)',
                           )

            ax[i].set_title(f'{species_map[species]} ({antibiotic})')
            ax[i].set_ylim((0.10,0.80))
            ax[i].set_ylabel(metric_map[metric])
            if i==0:
                ax[i].legend(loc='lower right')

            ax[i].set_xticklabels(['', 'test_year+3', '', 'test_year+2', '', 'test_year+1'])

    plt.tight_layout()
    plt.savefig(f'plots/temporal_validation/temporal_validation.png')



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
    
    metrics = [ 'auroc', 'auprc', 'accuracy']


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

        # Ditto for train years
        elif 'train_years' in data_raw and 'test_years' in data_raw:
            row.update({
                'train_years': ' '.join(data_raw['train_years']),
                'test_years': ' '.join(data_raw['test_years']),
            })

        for metric in metrics:
            row[metric] = data_raw[metric]
        
        rows.append(row)

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
    if 'train_years' in df.columns and 'test_years' in df.columns:
        group_columns += ['train_years', 'test_years']

    # Create a data frame that contains metrics over all the different
    # seeds. Each species--antibiotic combination is represented here.
    df = df.groupby(group_columns).agg(
            {
                metric: [np.mean, np.std] for metric in metrics
            }
        )

    print(df)

    # Plot results for temporal validation
    plot_temporal_validation(df)
