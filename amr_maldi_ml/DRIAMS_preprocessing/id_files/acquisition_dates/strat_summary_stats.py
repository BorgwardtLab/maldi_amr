#!/usr/bin/env python3

import os
import argparse

import numpy as np
import pandas as pd


strat_ids = ['PATIENTENNUMMER_id', 'FALLNUMMER_id', 'AUFTRAGSNUMMER_id']

strat_files = [
    '2018_strat_clean.csv',
    '2017_strat_clean.csv',
    '2016_strat_clean.csv',
    '2015_strat_clean.csv',
]
    

def check_consistency(args):
    for strat_file in strat_files:
        filename = os.path.join(args.path, strat_file)
        df = pd.read_csv(filename, low_memory=False, encoding='utf8')

        print(f'\n\n\nCheck: {filename}')
        print(f'Antibiotic {args.antibiotic}')
        print(df.head())
        print(f'File shape: {df.shape}')

        if args.antibiotic:
            df.dropna(subset=[args.antibiotic], inplace=True)          
            print(f'File shape after antibiotic subsetting: {df.shape}')
        
        for strat_id in strat_ids:
            # remove all samples with missing stratification id
            df.dropna(subset=[strat_id], inplace=True)          
            # change dtype of stratification id columns
            df = df.astype({strat_id: int}).astype({strat_id: str})
            # append species to the stratification id
            df[strat_id] = df[strat_id] + '_' + df['species']  
        print(f'File shape after strat-id clean-up: {df.shape}')
        
        # summary statistics per stratification id
        n = df.shape[0]

        for strat_id in strat_ids:
            print(f'\n{strat_id}') 
            n_unique = len(df[strat_id].unique())
            print(f'In {n} samples, {n_unique} {strat_id}s are unique.\n') 
            val_counts = df[strat_id].value_counts()
            n_ids_without_duplicates = sum(val_counts == 1)
            duplicate_counts = val_counts.value_counts().sort_index().rename(f'# {strat_id}')
            duplicate_counts = duplicate_counts.reset_index().rename(columns={'index': '# duplicates'})
            print(duplicate_counts)
            print('\nRead table as follows:')
            print(f'In {n_unique} {strat_id}s, {n_ids_without_duplicates} have only one entry in the dataset.') 
            print(f'The maximal number of duplicates is {val_counts.max()}.')


def write_checkfile(args):
    print('\nwriting checkfile..')
    df = pd.DataFrame({})

    for strat_file in strat_files:
        # recreate the dataframe
        filename = os.path.join(args.path, strat_file)
        df_year = pd.read_csv(filename, low_memory=False, encoding='utf8')

        if args.antibiotic:
            df_year.dropna(subset=[args.antibiotic], inplace=True)      
    
        for strat_id in strat_ids:
            df_year.dropna(subset=[strat_id], inplace=True)          
            df_year = df_year.astype({strat_id: int}).astype({strat_id: str})
            df_year[strat_id] = df_year[strat_id] + '_' + strat_file[:4] + '_' + df_year['species'] 
        df = df.append(df_year)

    # remove all antibiotics columns
    columns_to_delete = [
    'Penicillin_without_meningitis',
    'Cefuroxime.1',
    'Bacitracin',
    'Isoniazid_.4mg-l',
    'Ticarcillin-Clavulan acid',
    'Penicillin',
    'Ceftriaxone',
    'Vancomycin',
    'Piperacillin-Tazobactam',
    'Ciprofloxacin',
    'Cefepime',
    'Cotrimoxazole',
    'Meropenem',
    'Moxifloxacin',
    'Amoxicillin-Clavulanic acid',
    'Colistin',
    'Tobramycin',
    'Ceftazidime',
    'Ceftolozane-Tazobactam',
    'Ceftazidime-Avibactam',
    'Ceftobiprole',
    'Quinolones',
    'Tigecycline',
    'Levofloxacin',
    'Fosfomycin',
    'Amikacin',
    'Imipenem',
    'Minocycline',
    'Gentamicin',
    'Ceftarolin',
    'Ampicillin-Sulbactam',
    'Gentamicin_high_level',
    'Aztreonam',
    'Clindamycin',
    'Amoxicillin',
    'Metronidazole',
    'Daptomycin',
    'Ampicillin-Amoxicillin',
    'Caspofungin',
    'Voriconazole',
    'Posaconazole',
    'Amphotericin B',
    'Itraconazole',
    'Fluconazole',
    'Erythromycin',
    'Doxycycline',
    'Isavuconazole',
    'Anidulafungin',
    '5-Fluorocytosine',
    'Micafungin',
    'Tetracycline',
    'Azithromycin',
    'Ertapenem',
    'Fosfomycin-Trometamol',
    'Norfloxacin',
    'Cefpodoxime',
    'Nitrofurantoin',
    'Aminoglycosides',
    'Chloramphenicol',
    'Rifampicin_1mg-l',
    'Rifampicin',
    'Linezolid',
    'Amoxicillin-Clavulanic acid_uncomplicated_HWI',
    'Strepomycin_high_level',
    'Teicoplanin',
    'Cefuroxime',
    'Penicillin_with_endokarditis',
    'Penicillin_without_endokarditis',
    'Meropenem_with_meningitis',
    'Meropenem_without_meningitis',
    'Cefazolin',
    'Oxacillin',
    'Fusidic acid',
    'Streptomycin',
    'Isoniazid_.1mg-l',
    'Pyrazinamide',
    'Ethambutol_5mg-l',
    'Cefixime',
    'Mupirocin',
    'Vancomycin_GRD',
    'Teicoplanin_GRD',
    'Cefoxitin_screen',
    'Penicillin_with_meningitis',
    'Clarithromycin',
    'Penicillin_with_other_infections',
    'Penicillin_with_pneumonia',
    'Meropenem_with_pneumonia',
]
    df = df.drop(columns=columns_to_delete) 
    
    for strat_id in strat_ids:
        # get strat_id counts
        val_counts = df[strat_id].value_counts().rename(strat_id+'_n_duplicates')
        val_counts = val_counts.reset_index().rename(columns={'index': strat_id})
        print('dataframes')
        print(val_counts)
        df = df.merge(val_counts, on=strat_id, how='left')

    print(df.head())
    df.to_csv(os.path.join(args.path, 'match_strat_to_code.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', 
                        type=str, 
                        required=True,
                        help='Input id filepath')
    parser.add_argument('--antibiotic',
                        type=str,
                        default=None,
                        help='If present, only samples with RSI label for this antibiotic will be considered')
    parser.add_argument('--write_checkfile',
                        action='store_true',
                        help='Set if output should be written')

    args = parser.parse_args()
    #check_consistency(args)

    if args.write_checkfile: 
        write_checkfile(args)
