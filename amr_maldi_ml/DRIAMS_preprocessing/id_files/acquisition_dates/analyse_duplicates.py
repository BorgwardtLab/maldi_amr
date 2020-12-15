#!/usr/bin/env python3

import os
import dotenv
import argparse

import numpy as np
import pandas as pd

from maldi_learn.driams import load_driams_dataset

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

#strat_ids = ['PATIENTENNUMMER_id', 'FALLNUMMER_id', 'AUFTRAGSNUMMER_id']
col_strat_id = 'FALLNUMMER_id'
col_strat_count = col_strat_id + '_n_duplicates'

def find_duplicate_spectra(args):
    # extract unique stratification ids
    filename = './match_strat_to_code.csv'
    df = pd.read_csv(filename, low_memory='False')
    df_duplicates = df.loc[
                    df[col_strat_count] > 1
                           ]
    duplicated_strat_ids = df_duplicates[
                                    col_strat_id
                                         ].unique()

    # create dict to match strat_id with Bruker codes
    duplicate_dict = {}
    for strat_id in duplicated_strat_ids:
        df_ = df.loc[df[col_strat_id] == strat_id] 
        duplicate_dict[strat_id] = list(df_['code'].values)

    # load DRIAMS
    driams_dataset = load_driams_dataset(
                DRIAMS_ROOT,
                'DRIAMS-A',
                '*',
                args.species,
                args.antibiotic,
                spectra_type='binned_6000',
                #nrows=1000, #FIXME
            )
    codes_in_X = driams_dataset.y.code.values

    # for one strat_id, load MALDI-TOF spectra
    for strat_id in duplicated_strat_ids:
        spectra = []
        for code in duplicate_dict[strat_id]:
            if sum(codes_in_X == code) != 0:
                idx = np.where(codes_in_X == code)[0][0]
                spectra.append(driams_dataset.X[idx].intensities)
        spectra = np.array(spectra)
        print(f'{strat_id} {spectra.shape}. max n of spectra {len(duplicate_dict[strat_id])}')
             
    # copy files into ne folder structure

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--species',
                        default='*',
                        type=str)
    parser.add_argument('--antibiotic',
                        default='Ceftriaxone',
                        type=str)

    args = parser.parse_args()
    find_duplicate_spectra(args)
