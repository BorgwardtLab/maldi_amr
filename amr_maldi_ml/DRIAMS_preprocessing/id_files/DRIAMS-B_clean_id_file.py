import argparse

import numpy as np
import pandas as pd

from amr_maldi_ml.utilities import ab_name_map


def clean_data(filename, outfile):

    df = pd.read_csv(filename, sep=';', low_memory=False)
    print(df.shape)
    print(df.columns)

    list_antibiotics = list(df['Antibiotic'].unique())

    # remove useless columns
    df = df.drop(columns=['Unnamed: 0', 'No'])
    # delete rows with Bruker code not 36
    df = df.loc[df['Bruker'].str.len() == 36]
    # delete rows with no clear species identification
    df = df.loc[df['Organism_best_match'] != 'not reliable identification']

    # make combined ID, with Bruker code, Score1, Score2
    df['combined_code'] = df[['Bruker', 'Score1', 'Score2']].apply(
                                                    lambda x: ''.join(x),
                                                    axis=1)

    # remove lines that do not have unique combined_code and antibiotic
    print(df.shape)
    df = df.drop_duplicates(subset=['combined_code', 'Antibiotic'],
                            keep=False)

    # intialize dataframe
    list_id_columns = ['species', 'code', 'combined_code'] + list_antibiotics
    id_clean_indices = list(df['combined_code'].unique())
    id_clean = pd.DataFrame(columns=list_id_columns, index=id_clean_indices)
    id_clean.set_index('combined_code')

    # fill up dataframe
    for index, row in df.iterrows():
        idx = row['combined_code']
        id_clean.at[idx, 'species'] = row['Organism_best_match']
        id_clean.at[idx, 'code'] = row['Bruker']
        assert id_clean.isna().at[idx, row['Antibiotic']] == True, f'Antibiotic entry already exists. {row}'
        id_clean.at[idx, row['Antibiotic']] = row['Resultat']

    id_clean = id_clean.drop(columns=[np.nan])

    # replace column names with converted names
    print('Columns not covered by antibiotic name maps:\n{}'.format(
        [n for n in id_clean.columns if n not in ab_name_map.keys()]))
    id_clean = id_clean.rename(columns=ab_name_map)
    id_clean = id_clean.replace({'RES': 'R', 'INT': 'I',
                                 'negativ': '0', 'positiv': '1'})
    # write to file
    id_clean.to_csv(outfile, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument('OUTPUT', type=str, help='Output file')

    args = parser.parse_args()

    clean_data(args.INPUT, args.OUTPUT)
