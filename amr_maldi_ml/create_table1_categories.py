"""
Collect summary results and print Table 1.
"""

import os
import json

import pandas as pd
from utilities import ab_cat_map


def create_table(site, save=False, remove_empty_antibiotics=True):
    
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
    table = table.loc[table['site']==site]    
    
    # remove empty antibiotics
    table = table.loc[table['number of samples']!=0] 

    
    # create new table with categorized antibiotics 
    table_categorized = pd.DataFrame(columns=['category',
                                              'antibiotic',
                                              'site',
                                              'number of samples',
                                              'percentage positive',
                                              ])    

    for index, row in table.iterrows():
        table_categorized = table_categorized.append(
            pd.DataFrame({
                'category' : [ab_cat_map[row['antibiotic']]],
                'antibiotic' : [row['antibiotic']],
                'site': [row['site']],
                'number of samples': [row['number of samples']],
                'percentage positive': [row['percentage positive']],
            }),
            ignore_index=True
            ) 

    # create column with positive numbers
    table_categorized['number positive samples'] = table_categorized['number of samples'] * table_categorized['percentage positive']
    
    # group rows by category
    table_categorized = table_categorized.groupby('category').agg({'antibiotic': lambda x: ', '.join(x), 
                                                                   'number of samples': 'sum',
                                                                   'number positive samples': 'sum'})
    # create column with positive class percentage
    table_categorized['percentage positive'] = table_categorized['number positive samples'] / table_categorized['number of samples']    
    table_categorized = table_categorized.drop(columns=['number positive samples'])
    table_categorized['percentage positive'] = table_categorized['percentage positive'].round(3)

    # rename column headers to the final header naming
    table_categorized = table_categorized.rename(columns={'antibiotic': 'antimicrobials', 
                                                          'category': 'antimicrobial class',
                                                          'number of samples': 'number of samples in class',
                                                          'percentage positive': 'avg. positive sample ratio'})



    if save == True:
        table_categorized.to_csv(os.path.join(PATH_TABLE,f'{site}_Table1.csv'), index=True)
    
    print(table_categorized)


if __name__=='__main__':
    
    site = 'DRIAMS-A'
    create_table(site, save=True)
