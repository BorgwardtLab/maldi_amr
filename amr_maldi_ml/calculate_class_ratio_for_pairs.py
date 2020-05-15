"""
"""

import glob
import os

import pandas as pd
from dotenv import load_dotenv
from maldi_learn.driams import load_driams_dataset
from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')
explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

list_sites = ['DRIAMS-A', 'DRIAMS-B', 'DRIAMS-C', 'DRIAMS-D']

list_pairs = [
    ('Escherichia coli', 'Ciprofloxacin'),
    ('Escherichia coli', 'Ceftriaxone'),
    ('Escherichia coli', 'Cefepime'),
    ('Escherichia coli', 'Piperacillin-Tazobactam'),
    ('Escherichia coli', 'Tobramycin'),

    ('Staphylococcus aureus', 'Oxacillin'),
    ('Staphylococcus aureus', 'Penicillin'),
    ('Staphylococcus aureus', 'Ciprofloxacin'),
    ('Staphylococcus aureus', 'Fusidic acid'),

    ('Klebsiella pneumoniae', 'Ciprofloxacin'),
    ('Klebsiella pneumoniae', 'Ceftriaxone'),
    ('Klebsiella pneumoniae', 'Cefepime'),
    ('Klebsiella pneumoniae', 'Amoxicillin-Clavulanic acid'),
    ('Klebsiella pneumoniae', 'Meropenem'),
    ('Klebsiella pneumoniae', 'Tobramycin'),
    ('Klebsiella pneumoniae', 'Piperacillin-Tazobactam'),
              ]


def determine_class_ratio(pair, site):
    year = explorer.available_years(site)

    driams = load_driams_dataset(
    DRIAMS_ROOT,
    site,
    year,
    pair[0],
    pair[1],
    encoder=DRIAMSLabelEncoder(),
    handle_missing_resistance_measurements='remove_if_any_missing',
    spectra_type='binned_6000',
    )

    assert len(driams.y[pair[1]].unique()) == 2, \
        f'More than two labels present for antibiotic {pair[1]}.'

    return round(driams.y[pair[1]
                          ].value_counts(normalize=True).loc[1.0], 
                 3)



if __name__=='__main__':
    
    for pair in list_pairs:
        pos_class_ratio = determine_class_ratio(pair, 'DRIAMS-A')
        print(f'positive class ratio for {pair[1]} in {pair[0]}: {pos_class_ratio}')
