#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd


def clean_data(filename, outfile):
    df = pd.read_csv(filename, low_memory=False, encoding='utf8')
    print(f'\nInput file: {filename}')
    print(f'ID file starting shape: {df.shape}')

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
    print(f'ID file shape after basic clean-up: {df.shape}')

    duplicate_codes = df[df.duplicated('code')]['code'].values
    df = df.drop_duplicates(subset=['code'], keep=False) # remove entries with duplicated ids    
    print(f'Number of non-unique codes: {len(duplicate_codes)}')
    print(f'ID file final shape: {df.shape}')

    df = df.rename(columns={
        'KEIM': 'laboratory_species',
        'Organism(best match)': 'species',
    })

    
    ab_name_map = {
        'Amoxicillin...Clavulansaeure.bei.unkompliziertem.HWI': 'Amoxicillin-Clavulans.unkompl.HWI',
        'Amoxicillin-Clavulansaeure.unkompl.HWI':
        'Amoxicillin-Clavulansaeure_uncomplicated_HWI',
        'Ampicillin...Amoxicillin': 'Ampicillin-Amoxicillin',
        'Amphotericin.B': 'Amphotericin-B',
        'Piperacillin...Tazobactam': 'Piperacillin-Tazobactam',
        'Amoxicillin...Clavulansaeure': 'Amoxicillin-Clavulanic acid',
        'Fusidinsaeure': 'Fusidic acid',
        # TODO Ceftazidim vs Ceftazidim.1
        'Ceftazidim.1': 'Ceftazidime',
        'Ceftazidim.Avibactam': 'Ceftazidime-Avibactam',
        'X5.Fluorocytosin': '5-Fluorocytosin',
        'Fosfomycin.Trometamol': 'Fosfomycin-Trometamol',
        'Ceftolozan...Tazobactam': 'Ceftolozane-Tazobactam',
        'Cefepim': 'Cefepime',
        'Posaconazol': 'Posaconazole',
        'Tigecyclin': 'Tigecycline',
        'Cefpodoxim': 'Cefpodoxime',
        'Ceftobiprol': 'Ceftobiprole',
        'Fluconazol': 'Fluconazole',
        'Cefuroxim': 'Cefuroxime',
        'Tetracyclin': 'Tetracycline',
        'Ticarcillin...Clavulansaeure': 'Ticarcillin-Clavulanic acid',
        'Ceftriaxon': 'Ceftriaxone',
        'Itraconazol': 'Itraconazole',
        'Cotrimoxazol': 'Trimethoprim-Sulfamethoxazole',
        'Minocyclin': 'Minocycline',
        'Voriconazol': 'Voriconazole',
        'Metronidazol': 'Metronidazole',
        'Aminoglykoside': 'Aminoglycosides',
        'Chinolone': 'Quinolones',
        'Doxycyclin': 'Doxycycline',
        'Cefixim': 'Cefixime',
        'Meropenem.bei.Meningitis': 'Meropenem_with_meningitis',
        'Meropenem.bei.Pneumonie': 'Meropenem_with_pneumonia',
        'Meropenem.ohne.Meningitis': 'Meropenem_without_meningitis',
        'Isoniazid.0.1.mg.l': 'Isoniazid_.1mg-l',
        'Isoniazid.0.4.mg.l': 'Isoniazid_.4mg-l',
        'Ethambutol.5.0.mg.l': 'Ethambutol_5mg-l',
        'Pyrazinamid.100.0.mg.l': 'Pyrazinamid_100mg-l',
        'Streptomycin.1.0.mg.l': 'Streptomycin_1mg-l',
        'Rifampicin.1.0.mg.l': 'Rifampicin_1mg-l',
        'Gentamicin.High.level': 'Gentamicin_high_level',
        'Penicillin.bei.Endokarditis': 'Penicillin_with_endokarditis',
        'Penicillin.bei.Meningitis': 'Penicillin_with_meningitis', 
        'Penicillin.bei.Pneumonie': 'Penicillin_with_pneumonia', 
        'Penicillin.bei.anderen.Infekten': 'Penicillin_other_infections_present', 
        'Penicillin.ohne.Endokarditis': 'Penicillin_without_endokarditis',
        # TODO
        'Vancomycin.GRD': 'Vancomycin.GRD',
        'Cefepim.1': 'Cefepim.1',
        'Cefoxitin.Screen': 'Cefoxitin.Screen',
    }


    # assert no duplicates in code
    assert len(df['code'].unique()) == df.shape[0], 'codes not unique.'

    # rename columns to standard antibiotic names
    df = df.rename(columns=ab_name_map)

    # remove Dummy antibiotic
    if 'Dummy' in list(df.columns): df = df.drop(columns='Dummy')

    df.to_csv(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument('OUTPUT', type=str, help='Output file')

    args = parser.parse_args()

    clean_data(args.INPUT, args.OUTPUT)
