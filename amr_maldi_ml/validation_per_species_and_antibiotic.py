"""Measure validation performance.

The purpose of this script is to measure the validation performance of
all classifier, i.e. the performance that we obtain by training on one
site and testing on another. In contrast to the other scenario, we use
a *limited* data set (stratified by species and antibiotic) here.
"""

import argparse
import dotenv
import json
import logging
import os
import pathlib

import numpy as np

from maldi_learn.driams import DRIAMSDatasetExplorer

from maldi_learn.filters import DRIAMSBooleanExpressionFilter

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import case_based_stratification
from maldi_learn.utilities import stratify_by_species_and_label

from models import run_experiment

from utilities import generate_output_filename

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')


def _load_data(
    site,
    years,
    species,
    antibiotic,
    seed,
):
    """Load data set and return it in partitioned form."""
    extra_filters = []
    if site == 'DRIAMS-A':
        extra_filters.append(
            DRIAMSBooleanExpressionFilter('workstation != HospitalHygiene')
        )

    id_suffix = 'clean'
    strat_fn = stratify_by_species_and_label

    if site == 'DRIAMS-A':
        id_suffix = 'strat'
        strat_fn = case_based_stratification

    driams_dataset = load_driams_dataset(
        DRIAMS_ROOT,
        site,
        '*',
        species=species,
        antibiotics=antibiotic,
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
        on_error='warn',
        id_suffix=id_suffix,
        extra_filters=extra_filters,
    )

    logging.info(f'Loaded data set for {species} and {antibiotic}')

    X = np.asarray([spectrum.intensities for spectrum in driams_dataset.X])

    logging.info('Finished vectorisation')

    # Stratified train--test split
    train_index, test_index = strat_fn(
        driams_dataset.y,
        antibiotic=antibiotic,
        random_state=seed,
    )

    logging.info('Finished stratification')

    # Use the column containing antibiotic information as the primary
    # label for the experiment. All other columns will be considered
    # metadata. The remainder of the script decides whether they are
    # being used or not.
    y = driams_dataset.to_numpy(antibiotic)
    meta = driams_dataset.y.drop(columns=antibiotic)

    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    meta_train, meta_test = meta.iloc[train_index], meta.iloc[test_index]

    return X_train, y_train, X_test, y_test, meta_train, meta_test


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

    parser.add_argument(
        '-s', '--species',
        type=str,
        help='Species for which to run the experiment',
        required=True,
    )

    name = 'validation_per_species_and_antibiotic'

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
    logging.info(f'Species: {args.species}')

    X_train, y_train, _, _, meta_train, _ = _load_data(
        args.train_site,
        train_years,
        args.species,
        args.antibiotic,
        args.seed,
    )

    logging.info('Loaded training site data')

    _, _, X_test, y_test, _, meta_test =  _load_data(
        args.test_site,
        test_years,
        args.species,
        args.antibiotic,
        args.seed,
    )

    logging.info('Loaded test site data')

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
        'species': args.species,
    }

    # Add fingerprint information about the metadata files to make sure
    # that the experiment is reproducible.
    output['metadata_versions_train'] = metadata_fingerprints_train
    output['metadata_versions_test'] = metadata_fingerprints_test

    output_filename = generate_output_filename(
        args.output,
        output,
    )

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
            verbose=True,            # want info about best model etc.
            meta_train=meta_train,
            meta_test=meta_test,
        )

        output.update(results)

        logging.info(f'Saving {os.path.basename(output_filename)}')

        with open(output_filename, 'w') as f:
            json.dump(output, f, indent=4)
    else:
        logging.warning(
            f'Skipping {output_filename} because it already exists.'
        )
