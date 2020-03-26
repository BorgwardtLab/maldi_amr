"""Measure validation performance.

The purpose of this script is to measure the validation performance of
all classifier, i.e. the performance that we obtain by training on one
site and testing on another.
"""

import argparse
import dotenv
import json
import logging
import os
import pathlib

import numpy as np

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import stratify_by_species_and_label

from models import run_experiment

from utilities import generate_output_filename

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

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

    name = 'fig5_validation'

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

    parser.add_argument(
        '--train-site',
        type=str,
        help='Site to use for training (e.g. DRIAMS-A)',
        required=True
    )

    parser.add_argument(
        '--test-site',
        type=str,
        help='Site to use for testing (e.g. DRIAMS-B)',
        required=True
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

    train_site = args.train_site
    test_site = args.test_site

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    metadata_fingerprints_train = explorer.metadata_fingerprints(train_site)
    metadata_fingerprints_test = explorer.metadata_fingerprints(test_site)

    train_years = explorer.available_years(train_site)
    test_years = explorer.available_years(test_site)

    logging.info(f'Train site: {train_site}')
    logging.info(f'Train years: {train_years}')
    logging.info(f'Test site: {test_site}')
    logging.info(f'Test years: {test_years}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Model: {args.model}')
    logging.info(f'Antibiotic: {args.antibiotic}')

    driams_dataset_train = load_driams_dataset(
        DRIAMS_ROOT,
        train_site,
        train_years,
        '*',
        antibiotics=args.antibiotic,
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
    )

    logging.info('Loaded training site data')

    driams_dataset_test = load_driams_dataset(
        DRIAMS_ROOT,
        test_site,
        test_years,
        '*',
        antibiotics=args.antibiotic,
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
    )

    logging.info('Loaded test site data')

    train_index, _ = stratify_by_species_and_label(
        driams_dataset_train.y,
        antibiotic=args.antibiotic,
        random_state=args.seed,
    )

    X_train = np.asarray(
        [spectrum.intensities for spectrum in driams_dataset_train.X]
    )
    y_train = driams_dataset_train.to_numpy(args.antibiotic)

    _, test_index = stratify_by_species_and_label(
        driams_dataset_test.y,
        antibiotic=args.antibiotic,
        random_state=args.seed,
    )

    X_test = np.asarray(
        [spectrum.intensities for spectrum in driams_dataset_test.X]
    )
    y_test = driams_dataset_test.to_numpy(args.antibiotic)

    # Subset *twice* because `X_train` initially refers to the *whole*
    # training data set, but we only get a smaller subset based on the
    # split we defined earlier.
    X_train, y_train = X_train[train_index], y_train[train_index]
    X_test, y_test = X_test[test_index], y_test[test_index]

    # Prepare the output dictionary containing all information to
    # reproduce the experiment.
    output = {
        'train_site': train_site,
        'train_years': train_years,
        'test_site': test_site,
        'test_years': test_years,
        'seed': args.seed,
        'model': args.model,
        'antibiotic': args.antibiotic,
    }

    # Add fingerprint information about the metadata files to make sure
    # that the experiment is reproducible.
    output['metadata_versions_train'] = metadata_fingerprints_train
    output['metadata_versions_test'] = metadata_fingerprints_test

    output_filename = generate_output_filename(
        args.output,
        output,
    )

    output['species'] = 'all'

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
