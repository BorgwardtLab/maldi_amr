#!/usr/bin/env python3
#
# Performs an analysis of prevalence values for certain
# species--antibiotic combinations, aggregated by site.

import dotenv
import os

import pandas as pd

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder
from maldi_learn.driams import load_driams_dataset

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

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

        driams_dataset = load_driams_dataset(
                DRIAMS_ROOT,
                site,
                available_years,
                antibiotics=antibiotics,
                encoder=DRIAMSLabelEncoder(),
                nrows=2000,  # FIXME: remove after debugging
        )

        print(driams_dataset.y)
