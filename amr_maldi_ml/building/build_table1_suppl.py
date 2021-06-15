"""
Collect summary results and print Table 1.
"""

import os
import json
import argparse

import pandas as pd


def create_table(args):
    
    PATH_TABLE = '../../results/DRIAMS_summary'
    
    # get list of all .json files in directory
    file_list = []
    for (_, _, filenames) in os.walk(PATH_TABLE):
        [file_list.append(f) for f in filenames if '.json' in f]
        break

    table = pd.DataFrame(columns=['antibiotic',
                                  'site',
                                  'number of samples',
                                  'percentage positive',
                                  'most frequent species',
                                  ])    

    # read data and append to dataframe
    for filename in file_list:
        with open(os.path.join(PATH_TABLE, filename)) as f:
            data = json.load(f)

            # construct string of common species
            common_species_tuples = zip(data['most frequent species'], data['most frequent species counts'])
            common_species_list = [f'{tup[0]} ({tup[1]})' for tup in common_species_tuples]
            common_species_string = ', '.join(common_species_list)

            table = table.append(
                pd.DataFrame({
                    'antibiotic' : [data['antibiotic']],
                    'site': [data['site']],
                    'number of samples': [data['number spectra with AMR profile']],
                    'percentage positive': [round(data['positive class ratio'], 3)],
                    'most frequent species': [common_species_string],
                }),
                ignore_index=True
                )    
    table = table.sort_values(by=['number of samples'], ascending=False)

    # lowercase antibiotic names
    table['antibiotic'] = table['antibiotic'].str.lower()
    
    # subset to site 
    table = table.loc[table['site'] == args.site]    
    table = table.drop(columns=['site'])
    
    if args.remove_empty_antibiotics==True:
        table = table.loc[table['number of samples'] != 0]

    if args.save == True:
        table.to_csv(f'{args.outfile}', index=False)
    
    print(table)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--site',
                        type=str,
                        default='DRIAMS-A')
    parser.add_argument('--save',
                        type=bool,
                        default=False)
    parser.add_argument('--remove_empty_antibiotics',
                        type=bool,
                        default=False)
    parser.add_argument('--outfile',
                        type=str,
                        default='table1_suppl.csv')
    args = parser.parse_args()

    create_table(args)
