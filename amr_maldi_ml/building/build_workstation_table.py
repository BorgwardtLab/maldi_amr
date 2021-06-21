import os
import json
import argparse
import dotenv

import pandas as pd
import numpy as np
from maldi_learn.driams import load_driams_dataset
from maldi_learn.filters import DRIAMSBooleanExpressionFilter

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

site = 'DRIAMS-A'

def create_table(args):
    
    major_scenarios = [
        ('Escherichia coli', 'Ceftriaxone', 'lightgbm'),
        ('Klebsiella pneumoniae', 'Ceftriaxone', 'mlp'),
        ('Staphylococcus aureus', 'Oxacillin', 'lightgbm'),
    ]
   
    id_suffix = 'strat' if site == 'DRIAMS-A' else 'clean'

    # get list of all .json files in directory
    rows = []

    for scenario in major_scenarios:
        driams_dataset = load_driams_dataset(
            DRIAMS_ROOT,
            site,
            '*',
            species=scenario[0],
            antibiotics=scenario[1],
            handle_missing_resistance_measurements='remove_if_all_missing',
            spectra_type='binned_6000',
            id_suffix=id_suffix,
        )
        
        ws_ = driams_dataset.y['workstation'].values
        row = {
            'species': scenario[0],  
            'antibiotics': scenario[1],  
            'Total': len(ws_),
        }
        
        for ws in np.unique(ws_):
            ws_count = len([j for j in ws_ if j==ws])
            row[ws] = ws_count
        rows.append(row)

    df = pd.DataFrame.from_records(rows)
    df = df.fillna(0)
    df['Stool'] = df['Stool'].astype(int)

    
    for ws in ['HospitalHygiene', 'Blood', 'DeepTissue', 'Genital',
       'Respiratory', 'Stool', 'Urine', 'Varia', 'Total']:
    
        # calculate percentage column
        df[ws+'_perc'] = df[ws]/df['Total'] 
        df[ws+'_perc'] = df[ws+'_perc'].round(4) *100
        df[ws+'_perc'] = df[ws+'_perc'].round(1)

        # construct string to be printed in table
        df[ws.lower()] = df[ws].astype(str) + ' (' + df[ws+'_perc'].astype(str) + ')'

        # clean-up table
        df = df.drop(columns=[ws, ws+'_perc']) 


    if args.save:
        df.to_csv(f'{args.outfile}', index=False)
    
    print(df)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save',
                        action='store_true')
    parser.add_argument('--outfile',
                        type=str,
                        default='../tables/table_workstations.csv')
    args = parser.parse_args()

    create_table(args)
