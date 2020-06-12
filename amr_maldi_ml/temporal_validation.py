"""Measure temporal validation performance.

The purpose of this script is to measure the validation performance of
all classifier, i.e. the performance that we obtain by training on the
same site but testing for different years.
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

from sklearn.utils import shuffle

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

    parser.add_argument(
        '-s', '--species',
        type=str,
        help='Species for which to run the experiment',
        required=True,
    )

    name = 'fig5_temporal_validation'

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
        '--train-years',
        type=str,
        nargs='+',
        help='Years to use for training on a site',
        required=True,
    )

    parser.add_argument(
        '--test-years',
        type=str,
        nargs='+',
        help='Years to use for testing on a site',
        required=True,
    )

    parser.add_argument(
        '--site',
        default='DRIAMS-A',
        help='Site to use for the temporal validation scenario',
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
    metadata_fingerprints = explorer.metadata_fingerprints(args.site)

    train_years = args.train_years
    test_years = args.test_years

    logging.info(f'Site: {args.site}')
    logging.info(f'Train years: {train_years}')
    logging.info(f'Test years: {test_years}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Model: {args.model}')
    logging.info(f'Antibiotic: {args.antibiotic}')

    driams_dataset_train = load_driams_dataset(
        DRIAMS_ROOT,
        args.site,
        train_years,
        args.species,
        antibiotics=args.antibiotic,
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
    )

    logging.info('Loaded training data')

    driams_dataset_test = load_driams_dataset(
        DRIAMS_ROOT,
        args.site,
        test_years,
        args.species,
        antibiotics=args.antibiotic,
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
    )

    logging.info('Loaded test data')

    X_train = np.asarray(
        [spectrum.intensities for spectrum in driams_dataset_train.X]
    )
    y_train = driams_dataset_train.to_numpy(args.antibiotic)

    X_test = np.asarray(
        [spectrum.intensities for spectrum in driams_dataset_test.X]
    )
    y_test = driams_dataset_test.to_numpy(args.antibiotic)

    # The shuffling is technically only required for the `train` data
    # set because we expect this to contain multiple years. We make a
    # better effort, however, in order to be prepared for anything.
    X_train, y_train = shuffle(X_train, y_train, random_state=args.seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=args.seed)

    # Prepare the output dictionary containing all information to
    # reproduce the experiment.
    output = {
        'site': args.site,
        'train_years': train_years,
        'test_years': test_years,
        'species': args.species,
        'seed': args.seed,
        'model': args.model,
        'antibiotic': args.antibiotic,
    }

    # Add fingerprint information about the metadata files to make sure
    # that the experiment is reproducible.
    output['metadata_versions'] = metadata_fingerprints

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
