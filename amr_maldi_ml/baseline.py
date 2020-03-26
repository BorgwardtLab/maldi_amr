"""Measure baseline performance.

The purpose of this script is to measure the baseline performance, i.e.
the performance that we obtain by looking at:

    1. *All* species and their respective spectra; accumulating their
       classification performance

    2. *Only* the species information without considering any spectra
"""

import argparse
import dotenv
import joblib
import logging
import os
import json
import pathlib
import warnings

import numpy as np

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import stratify_by_species_and_label

from models import run_experiment

from utilities import generate_output_filename

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These parameters should remain fixed for this particular
# experiment. We always train on the same data set, using
# *all* available years.
site = 'DRIAMS-A'
years = ['2015', '2016', '2017', '2018']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-a', '--antibiotic',
        type=str,
        help='Antibiotic for which to run the experiment',
        required=True,
    )

    parser.add_argument(
        '-m', '--model',
        default='lr',
        help='Selects model to use for subsequent training'
    )

    parser.add_argument(
        '-S', '--seed',
        type=int,
        help='Random seed to use for the experiment',
        required=True
    )

    name = 'fig2_baseline'

    parser.add_argument(
        '-o', '--output',
        default=pathlib.Path(__file__).resolve().parent.parent / 'results'
                                                               / name,
        type=str,
        help='Output path for storing the results.'
    )

    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='If set, overwrites all files. Else, skips existing files.'
    )

    args = parser.parse_args()

    # Create the output directory for storing all results of the
    # individual combinations.
    os.makedirs(args.output, exist_ok=True)

    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    metadata_fingerprints = explorer.metadata_fingerprints(site)

    logging.info(f'Site: {site}')
    logging.info(f'Years: {years}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Antibiotic: {args.antibiotic}')

    driams_dataset = load_driams_dataset(
        DRIAMS_ROOT,
        site,
        years,
        '*',
        antibiotics=args.antibiotic,
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
    )

    logging.info(f'Loaded data set for {args.antibiotic}')

    # Having loaded the data set, we have to generate two different
    # feature vectors:
    #
    #   1. The 'regular' feature vector as returned by our data set
    #      loader. This necessitates no additional transformation.
    #
    #   2. The feature vector that we obtain by throwing away *all*
    #      information about the spectra, leaving us only with sets
    #      of one-hot-encoded species information.
    #
    # The purpose of the second experiment is to assess to what extent
    # microbial resistance can be be predicted based on information
    # about the species.

    ohe = OneHotEncoder(sparse=False)
    X_species = ohe.fit_transform(
        driams_dataset.y['species'].values.reshape(-1, 1)
    )

    logging.info('Created species-only feature vector')

    # Create feature matrix from the binned spectra. We only need to
    # consider the second column of each spectrum for this.
    X_spectra = np.asarray(
                    [spectrum.intensities for spectrum in driams_dataset.X]
                )

    train_index, test_index = stratify_by_species_and_label(
        driams_dataset.y,
        antibiotic=args.antibiotic,
        random_state=args.seed,
    )

    # Labels are shared for both of these experiments, so they only
    # need to be created once.
    y = driams_dataset.to_numpy(args.antibiotic)

    for X, t in zip([X_species, X_spectra], ['no_spectra', '']):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # Prepare the output dictionary containing all information to
        # reproduce the experiment.
        output = {
            'site': site,
            'seed': args.seed,
            'model': args.model,
            'antibiotic': args.antibiotic,
            'years': years,
        }

        # Add fingerprint information about the metadata files to make sure
        # that the experiment is reproducible.
        output['metadata_versions'] = metadata_fingerprints

        output_filename = generate_output_filename(
            args.output,
            output,
            suffix=t
        )

        # Add this information after generating a file name because
        # I want it to be kept out of there. This is slightly hacky
        # but only required for this one experiment.
        output['species'] = 'all' if not t else 'all (w/o spectra)'

        # Only write if we either are running in `force` mode, or the
        # file does not yet exist.
        if not os.path.exists(output_filename) or args.force:

            n_folds = 5

            results = run_experiment(
                X_train, y_train,
                X_test, y_test,
                args.model,
                n_folds,
                random_state=args.seed,  # use seed whenever possible
                verbose=True             # want info about best model etc.
            )

            output.update(results)

            logging.info(f'Saving {os.path.basename(output_filename)}')

            with open(output_filename, 'w') as f:
                json.dump(output, f, indent=4)
        else:
            logging.warning(
                f'Skipping {output_filename} because it already exists.'
            )
