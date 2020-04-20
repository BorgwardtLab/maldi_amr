"""
Collect summary results and print Table 1.
"""

import os
import json
import argparse

import pandas as pd
from utilities import ab_cat_map


def create_table(args):
    
    PATH_TABLE = '../results/DRIAMS_summary'
    
    # get list of all .json files in directory
    file_list = []
    for (_, _, filenames) in os.walk(PATH_TABLE):
        [file_list.append(f) for f in filenames if '.json' in f]
        break

    table = pd.DataFrame(columns=['antibiotic',
                                  'site',
                                  'number of samples',
                                  'percentage positive',
                                  ])    

    # read data and append to dataframe
    for filename in file_list:
        with open(os.path.join(PATH_TABLE, filename)) as f:
            data = json.load(f)

            table = table.append(
                pd.DataFrame({
                    'antibiotic' : [data['antibiotic']],
                    'site': [data['site']],
                    'number of samples': [data['number spectra with AMR profile']],
                    'percentage positive': [round(data['positive class ratio'], 3)],
                }),
                ignore_index=True
                )    
    
    # subset to site 
    table = table.loc[table['site']==args.site]    
    
    # remove empty antibiotics
    table = table.loc[table['number of samples']!=0] 

    
    # create new table with categorized antibiotics 
    table_categorized = pd.DataFrame(columns=['antimicrobial class',
                                              'antibiotic',
                                              'site',
                                              'number of samples',
                                              'percentage positive',
                                              ])    

    for index, row in table.iterrows():
        table_categorized = table_categorized.append(
            pd.DataFrame({
                'antimicrobial class' : [ab_cat_map[row['antibiotic']]],
                'antibiotic' : [row['antibiotic']],
                'site': [row['site']],
                'number of samples': [row['number of samples']],
                'percentage positive': [row['percentage positive']],
            }),
            ignore_index=True
            ) 

    # create column with positive numbers
    table_categorized['number positive samples'] = table_categorized['number of samples'] * table_categorized['percentage positive']
    
    # add 'total' row 
    table_categorized = table_categorized.append(
        pd.DataFrame({
            'antimicrobial class' : ['total'],
            'antibiotic' : [''],
            'site': [args.site],
            'number of samples': [table_categorized['number of samples'].sum()],
            'number positive samples': [table_categorized['number positive samples'].sum()],
        }),
        ignore_index=True
        ) 

    # group rows by antimicrobial class
    table_categorized = table_categorized.groupby('antimicrobial class').agg({'antibiotic': lambda x: ', '.join(x), 
                                                                   'number of samples': 'sum',
                                                                   'number positive samples': 'sum'})
    # recreate column with positive class percentage
    table_categorized['percentage positive'] = table_categorized['number positive samples'] / table_categorized['number of samples']    
    table_categorized = table_categorized.drop(columns=['number positive samples'])
    table_categorized['percentage positive'] = table_categorized['percentage positive'].round(3)

    # sort rows by sample size, 'total' row last
    table_categorized = table_categorized.sort_values(by=['number of samples'],
                                                      ascending=False) 
    # TODO put 'total' row last
    
    # sort columns in order in which they appear in the paper
    table_categorized = table_categorized[['number of samples',
                                           'percentage positive',
                                           'antibiotic']]

    # rename column headers to the final header naming
    table_categorized = table_categorized.rename(columns={'antibiotic': 'antimicrobials', 
                                                          'number of samples': 'number of samples in class',
                                                          'percentage positive': 'avg. positive sample ratio'})


    if args.save == True:
        table_categorized.to_csv(f'{args.outfile}', index=True)
    
    print(table_categorized)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--site',
                        type=str,
                        default='DRIAMS-A')
    parser.add_argument('--save',
                        type=bool,
                        default=False)
    parser.add_argument('--outfile',
                        type=str,
                        default='table1.csv')
    args = parser.parse_args()
    
    create_table(args)
