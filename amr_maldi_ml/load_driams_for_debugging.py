import argparse
import dotenv
import json
import logging
import pathlib
import os

import numpy as np

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import stratify_by_species_and_label

from models import run_experiment

from utilities import generate_output_filename
from utilities import load_stratify_split_data

from sklearn.model_selection import train_test_split

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These parameters should remain fixed for this particular
# experiment. We always train on the same data set, using
# *all* available years.
site = 'DRIAMS-A'
years = ['2015', '2016', '2017', '2018']

driams_dataset = load_driams_dataset(
            DRIAMS_ROOT,
            site,
            years,
            species='*',
            antibiotics='Amoxicillin-Clavulanic acid',  # Only a single one for this run
            encoder=DRIAMSLabelEncoder(),
            handle_missing_resistance_measurements='keep',
            spectra_type='binned_6000',
            on_error='warn'
    )
print(driams_dataset.y.shape)


X_train, y_train, X_test, y_test, meta_train, meta_test = load_stratify_split_data(
    DRIAMS_ROOT,
    site,
    years,
    '*',
    antibiotics='Amoxicillin-Clavulanic acid',  # Only a single one for this run
    42,
)
print(X_train.shape)
print(X_test.shape)
