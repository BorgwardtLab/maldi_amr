"""Extract all codes for given species and write to txt file."""

import argparse
import dotenv
import json
import os

import numpy as np

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These parameters should remain fixed for this particular
# experiment. We always train on the same data set, using
# *all* available years.
site = 'DRIAMS-A'
years = ['2015', '2016', '2017', '2018']

def extract_codes_per_species(args):

    driams_dataset = load_driams_dataset(
            DRIAMS_ROOT,
            site,
            years,
            species=args.species,
            antibiotics='Ciprofloxacin', 
            encoder=DRIAMSLabelEncoder(),
            handle_missing_resistance_measurements='keep',
            spectra_type='binned_6000',
    )
    
    codes = driams_dataset.y['code']
    print(driams_dataset.y.head())
    filename = f'codes_per_species_' + \
               f'{args.species.replace(" ","_")}' + \
                '.txt'
    codes.to_csv(filename, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s', '--species',
        type=str,
        help='Species for which to extract the codes.',
    )

    parser.add_argument(
        '-o', '--output',
        default='./',
        type=str,
        help='Output path for storing the results.'
    )

    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='If set, overwrites all files. Else, skips existing files.'
    )

    args = parser.parse_args()
    extract_codes_per_species(args)
