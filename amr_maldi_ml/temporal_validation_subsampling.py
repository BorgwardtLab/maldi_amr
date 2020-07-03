"""Measure temporal validation performance.

The purpose of this script is to measure the validation performance of
all classifier, i.e. the performance that we obtain by training on the
same site but testing for different years.

In contrast to the simple scenario, this script uses subsampling for a
more realistic view on sample occurrence.
"""

import argparse
import dotenv
import json
import logging
import os
import pathlib

import numpy as np

from imblearn.over_sampling import RandomOverSampler

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import stratify_by_species_and_label

from models import run_experiment

from sklearn.utils import resample
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

    name = 'fig5_temporal_validation_subsampling'

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
        '-r', '--resample',
        action='store_true',
        help='If set, resamples the data set to the size of the one used '
             'for training.'
    )

    parser.add_argument(
        '--train-years',
        type=str,
        nargs='+',
        help='Years to use for training on a site',
        required=True,
    )

    parser.add_argument(
        '--test-year',
        type=str,
        help='Year to use for testing on a site',
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
    test_year = args.test_year

    logging.info(f'Site: {args.site}')
    logging.info(f'Train years: {train_years}')
    logging.info(f'Test year: {test_year}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Model: {args.model}')
    logging.info(f'Antibiotic: {args.antibiotic}')

    # Load the test data set first because it is easier to handle for
    # us; we will only ever use the `test` portion of this year since
    # it is possible that the *same* year occurs also for training.
    driams_dataset_test = load_driams_dataset(
        DRIAMS_ROOT,
        args.site,
        test_year,
        args.species,
        antibiotics=args.antibiotic,
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
    )

    logging.info(f'Loaded test data')

    # Let's never look at the training data; we need the index, though,
    # in order to remove duplicate samples later on.
    train_index, test_index = stratify_by_species_and_label(
        driams_dataset_test.y,
        antibiotic=args.antibiotic,
        random_state=args.seed,
    )

    X_test = np.asarray(
        [spectrum.intensities for spectrum in driams_dataset_test.X]
    )
    y_test = driams_dataset_test.to_numpy(args.antibiotic)

    # Again, subset the data correctly so that we are never using any
    # test data.
    X_test = X_test[test_index]
    y_test = y_test[test_index]

    # Determine the class ration on the test data set
    class_ratios = np.bincount(y_test) / len(y_test)
    logging.info(f'Desired minority class ratio: {class_ratios[1]:.2f}')

    # Check that we are in the right scenario; class 1 must be the
    # minority class.
    assert class_ratios[0] > class_ratios[1]

    X_train = []
    y_train = []

    for train_year in train_years:
        driams_dataset = load_driams_dataset(
            DRIAMS_ROOT,
            args.site,
            train_year,
            args.species,
            antibiotics=args.antibiotic,
            encoder=DRIAMSLabelEncoder(),
            handle_missing_resistance_measurements='remove_if_all_missing',
            spectra_type='binned_6000',
        )

        logging.info(f'Loaded training data for {train_year}')

        X = np.asarray(
            [spectrum.intensities for spectrum in driams_dataset.X]
        )
        y = driams_dataset.to_numpy(args.antibiotic)

        # Remove the samples that we already used for testing; this
        # ensures that there is no leakage.
        #
        # This procedure is not required for any other year because
        # each year is treated individually.
        if train_year == test_year:
            X = X[train_index]
            y = y[train_index]

        # Again, check the correctness of this scenario; class 1 needs
        # to be the minority class.
        n0, n1 = np.bincount(y)
        assert n0 > n1

        desired_ratio = class_ratios[1] / class_ratios[0]

        # Need random oversampling because the actual ratio is smaller
        # than the desired ratio.
        if n1 / n0 < desired_ratio:
            ros = RandomOverSampler(
                sampling_strategy=desired_ratio,
                random_state=args.seed,
            )

            X, y = ros.fit_resample(X, y)

        # Just pick the desired number of points at random and subset
        # `X` and `y` accordingly.
        else:
            # Be generous with the number of points that we have to
            # resample. This makes a difference of a single sample,
            # so we should be good.
            n_points_to_sample = int(desired_ratio * n0 + 0.5)

            # The resampling only pertains to the positive, i.e. the
            # minority, class.
            indices = np.nonzero(y == 1)[0]

            indices = resample(
                indices,
                n_samples=n_points_to_sample,
                replace=False,
                random_state=args.seed
            )

            # Add the remaining indices as well (of the majority class);
            # we do not need to perform any resampling here.
            indices = np.concatenate(
                    (indices, np.nonzero(y == 0)[0])
            )

            X, y = X[indices], y[indices]

        class_ratio = np.bincount(y)[1] / len(y)

        logging.info(f'Achieved minority class ratio of {class_ratio:.2f} '
                     f'for {train_year}')

        X_train.extend(X)
        y_train.extend(y)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if args.resample:
        n_samples = len(train_index)

        logging.info(
            f'Resampling `X_train` and `y_train` to {n_samples}'
        )

        X_train, y_train = resample(
            X_train, y_train,
            n_samples=n_samples,
            replace=False,
            stratify=y_train,
            random_state=args.seed,
        )

    class_ratio = np.bincount(y_train)[1] / len(y_train)
    logging.info(f'Achieved overall minority class ratio of {class_ratio:.2f}')

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
        'test_years': [test_year],        # required for compatibility
        'species': args.species,
        'seed': args.seed,
        'model': args.model,
        'antibiotic': args.antibiotic,
        'resample': args.resample,        # to be able to know what is going on
        'n_samples_train': len(X_train),  # potentially useful
        'n_samples_test': len(X_test),    # ditto
    }

    # Add fingerprint information about the metadata files to make sure
    # that the experiment is reproducible.
    output['metadata_versions'] = metadata_fingerprints

    suffix = None if not args.resample else 'resample'

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
