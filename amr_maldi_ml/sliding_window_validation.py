"""Measure temporal validation performance using sliding windows."""

import argparse
import dateparser
import dotenv
import json
import logging
import os
import pathlib

import numpy as np
import pandas as pd

from maldi_learn.driams import DRIAMSDatasetExplorer

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import stratify_by_species_and_label

from models import run_experiment

from sklearn.utils import resample
from sklearn.utils import shuffle

from utilities import generate_output_filename

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')


class KeepAllBeforeFilter:
    def __init__(self, date):
        self.date = dateparser.parse(date)

    def __call__(self, row):
        row_date = dateparser.parse(row['acquisition_date'])
        return row_date <= self.date


class KeepRangeFilter:
    def __init__(self, date_from, date_to):
        self.date_from = dateparser.parse(date_from)
        self.date_to = dateparser.parse(date_to)

    def __call__(self, row):
        row_date = dateparser.parse(row['acquisition_date'])
        return self.date_from <= row_date <= self.date_to


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

    name = 'sliding_window_validation'

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

    logging.info(f'Site: {args.site}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Model: {args.model}')
    logging.info(f'Antibiotic: {args.antibiotic}')

    driams_dataset = load_driams_dataset(
        DRIAMS_ROOT,
        args.site,
        '*',
        species=args.species,
        antibiotics=args.antibiotic,
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
        id_suffix='strat_acqu',
        nrows=1000,
    )

    logging.info(f'Loaded full data set')

    # Contains all spectra of the data set. Need to get this into an
    # `np.array` in order to slice it later on.
    X = np.asarray(
        [spectrum.intensities for spectrum in driams_dataset.X]
    )

    # TODO: make configurable
    test_from = '2018-04-30' 

    date_filter = KeepAllBeforeFilter(date=test_from)
    mask = driams_dataset.y.apply(date_filter, axis=1)
    X_train = X[mask]
    y_train = driams_dataset.y[mask]

    X_test = X[~mask]
    y_test = np.asarray(
        driams_dataset.y[~mask][args.antibiotic].values,
        dtype=int
    )

    date_range = pd.date_range(
        start='2015-11-01',
        end='2018-04-30',
        freq='M',
    )

    df = pd.DataFrame(index=date_range)
    for i, window in enumerate(df.rolling(6)):
        # Ignore first periods that do not contain sufficient data.
        if i <= 4:
            continue

        date_from = window.index[0].strftime('%Y-%m-%d')
        date_to = window.index[-1].strftime('%Y-%m-%d')

        date_filter = KeepRangeFilter(date_from, date_to)
        mask = y_train.apply(date_filter, axis=1)

        # Local data sets to use for training at this range.
        X_train_ = X_train[mask]
        y_train_ = np.asarray(
            y_train[mask][args.antibiotic].values,
            dtype=int
        )

        # Prepare the output dictionary containing all information to
        # reproduce the experiment.
        output = {
            'site': args.site,
            'years': '*',
            'species': args.species,
            'seed': args.seed,
            'model': args.model,
            'antibiotic': args.antibiotic,
            'train_from': date_from,
            'train_to': date_to,
            'test_from': test_from,
        }

        # Add fingerprint information about the metadata files to make sure
        # that the experiment is reproducible.
        output['metadata_versions'] = metadata_fingerprints

        suffix = f'{date_from}_{date_to}'

        output_filename = generate_output_filename(
            args.output,
            output,
            suffix=suffix
        )

        # Only write if we either are running in `force` mode, or the
        # file does not yet exist.
        if not os.path.exists(output_filename) or args.force:

            n_folds = 5

            results = run_experiment(
                # Use local variants of the train data set because they
                # have been subset to a range of months.
                X_train_, y_train_,
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
