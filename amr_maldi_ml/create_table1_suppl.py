"""
Collect summary results and print Table 1.
"""

import os
import json

import pandas as pd


def create_table(site, mode='print'):
    
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
                                  'species',
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
                    'species': [data['species']],
                }),
                ignore_index=True
                )    
    table = table.sort_values(by=['number of samples'], ascending=False)
    print('table', table)
    
    # subset to site 
    table = table.loc[table['site']==site]    


    if mode=='save':
        table.to_csv(os.path.join(PATH_TABLE,f'{site}.csv'), index=False)
    else:
        print(table)
    pass

if __name__=='__main__':
    
    site = 'DRIAMS-A'
    create_table(site, mode='save')
