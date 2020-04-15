#!/usr/bin/env python3
#
# Performs an analysis of prevalence values for certain
# species--antibiotic combinations, aggregated by site.

import dotenv
import os
import json

import numpy as np
import pandas as pd

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder
from maldi_learn.driams import load_driams_dataset

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')
outdir = '../results/DRIAMS_summary'

antibiotics = [
    'Amikacin',
    'Amoxicillin-Clavulanic acid',
    'Cefepime',
    'Ceftriaxone',
    'Ciprofloxacin',
    'Clindamycin',
    'Gentamicin',
    'Imipenem',
    'Piperacillin-Tazobactam',
]

# Could also be configured automatically, as we have the option to query
# available sites. However, the interest of simplicity, let us configure
# everything manually.
sites = [
    'DRIAMS-A',
    'DRIAMS-B',
    'DRIAMS-C',
    'DRIAMS-D',
]

if __name__ == '__main__':

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

    for site in sites:
        metadata_fingerprints = explorer.metadata_fingerprints(site)
        available_years = explorer.available_years(site)
        # extract available antibiotics
        available_antibiotics_per_year = explorer.available_antibiotics(site)
        available_antibiotics = set(available_antibiotics_per_year[available_years[0]])
        for year in available_years[1:]:
            available_antibiotics = available_antibiotics.intersection(set(available_antibiotics_per_year[year]))
        available_antibiotics = list(available_antibiotics)

        # load driams data
        driams_dataset = load_driams_dataset(
                DRIAMS_ROOT,
                site,
                available_years,
                '*',
                antibiotics=available_antibiotics,
                encoder=DRIAMSLabelEncoder(),
                handle_missing_resistance_measurements='remove_if_all_missing',
                nrows=2000,  # FIXME: remove after debugging
        )

        for antibiotic in available_antibiotics:  
            print(site, antibiotic)
            y = driams_dataset.y[antibiotic].dropna().values
            
            # Calculate summary characteristics
            if len(y) != 0:
                pos_class_ratio = float(sum(y)/len(y))
                num_samples = len(y)
            else:
                pos_class_ratio = 0
                num_samples = 0

            # Create dictionary to be saved in json output file
            d_summary = {
                         'antibiotic': antibiotic,
                         'site': site,
                         'year': available_years,
                         'positive class ratio': pos_class_ratio,
                         'number spectra with AMR profile': num_samples,
                         }

            print(d_summary)
            outfile = os.path.join(outdir, f'{site}_{antibiotic}.json')
           
            # save to file 
            with open(outfile, 'w') as f:
                json.dump(d_summary, f, indent=4)
