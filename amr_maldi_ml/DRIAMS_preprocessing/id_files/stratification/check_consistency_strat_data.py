#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd

from amr_maldi_ml.utilities import ab_name_map


def check_consistency(fileA, fileB):
    df_A = pd.read_csv(fileA, low_memory=False, encoding='utf8')
    df_B = pd.read_csv(fileB, low_memory=False, encoding='utf8')
    print(f'\nCompare: {fileA} {fileB}')
    print(f'File shapes: {df_A.shape} {df_B.shape}')

    df_A = df_A.drop(columns=['Cefepim.1', 'Ceftazidim.1'])
    df_B = df_B.drop(columns=['Cefepim.1', 'Ceftazidim.1'])

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

    for df in [df_A, df_B]:
        df.drop(columns=columns_to_delete, inplace=True)     # remove obsolete columns
        df.dropna(subset=['code'], inplace=True)             # remove missing codes
        df.drop_duplicates(inplace=True)                     # drop full duplicates
        df.dropna(subset=['Organism(best match)'], inplace=True)
    print(f'Prev id shape after basic clean-up: {df_A.shape}')
    print(f'Strat id shape after basic clean-up: {df_B.shape}')
    total = df_B.shape[0]

    df_B.dropna(subset=['PATIENTENNUMMER_id'], inplace=True)             # remove missing codes
    print(f'Stat id shape after dropna patient number: {df_B.shape}')
    df_B.dropna(subset=['FALLNUMMER_id'], inplace=True)             # remove missing codes
    print(f'Stat id shape after dropna case number: {df_B.shape}')
    df_B.dropna(subset=['AUFTRAGSNUMMER_id'], inplace=True)             # remove missing codes
    print(f'Stat id shape after dropna task number: {df_B.shape}')
    after = df_B.shape[0]
    
    print(f'Dropped percentage: {1 - (after / float(total))}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prev', 
                        type=str, 
                        required=True,
                        help='Previous DRIAMS-A id file')
    parser.add_argument('--strat', 
                        type=str, 
                        required=True,
                        help='Stratification DRIAMS-A id file')

    args = parser.parse_args()
    check_consistency(args.prev, args.strat)
