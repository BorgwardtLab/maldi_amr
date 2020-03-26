#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd

from amr_maldi_ml.utilities import ab_name_map


def clean_data(filename, outfile):
    df = pd.read_csv(filename, low_memory=False, encoding='utf8')
    print(f'\nInput file: {filename}')
    print(f'ID file starting shape: {df.shape}')

    # delete artifact column
    #print(df['Cefepim.1'].isna().sum())
    #print(df['Ceftazidim.1'].isna().sum())
    df = df.drop(columns=['Cefepim.1', 'Ceftazidim.1'])

    # Select columns for deletion. We want to remove columns that could
    # potentially leak patient information.
    columns_to_delete = [
        'strain',
        'TAGESNUMMER',
        'Value',
        'A',
        'Score1',
        'Organism(second best match)',
        'Score2',
        'SPEZIES_MALDI',
        'GENUS',
        'SPEZIES_MLAB',
        'MATERIAL',
        'AUFTRAGSNUMMER',
        'STATION',
        'PATIENTENNUMMER',
        'GEBURTSDATUM',
        'GESCHLECHT',
        'EINGANGSDATUM',
        'LOKALISATION'
    ]
    print(f'Remove columns: {columns_to_delete}')

    df = df.drop(columns=columns_to_delete)     # remove obsolete columns
    df = df.dropna(subset=['code'])             # remove missing codes
    df = df.drop_duplicates()                   # drop full duplicates
    df = df.dropna(subset=['Organism(best match)'])
    print(f'ID file shape after basic clean-up: {df.shape}')

    duplicate_codes = df[df.duplicated('code')]['code'].values
    # remove entries with duplicated ids
    df = df.drop_duplicates(subset=['code'], keep=False) 
    print(f'Number of non-unique codes: {len(duplicate_codes)}')
    print(f'ID file final shape: {df.shape}')

    df = df.rename(columns={
        'KEIM': 'laboratory_species',
        'Organism(best match)': 'species',
    })

    # assert no duplicates in code
    assert len(df['code'].unique()) == df.shape[0], 'codes not unique.'

    # rename columns to standard antibiotic names
    print('Columns not covered by antibiotic name maps:\n{}'.format([n for n in df.columns if n not in ab_name_map.keys()]))
    df = df.rename(columns=ab_name_map)

    # remove Dummy antibiotic
    if 'Dummy' in list(df.columns):
        df = df.drop(columns='Dummy')

    df.to_csv(outfile, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument('OUTPUT', type=str, help='Output file')

    args = parser.parse_args()

    clean_data(args.INPUT, args.OUTPUT)
