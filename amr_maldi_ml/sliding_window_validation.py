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

from maldi_learn.filters import DRIAMSBooleanExpressionFilter

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import load_driams_dataset

from models import calculate_metrics
from models import get_feature_weights
from models import run_experiment

from utilities import generate_output_filename

from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

from imblearn.over_sampling import RandomOverSampler

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
        '-c', '--cumulative',
        action='store_true',
        help='If set, uses cumulative window'
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
        '-l', '--log-codes',
        action='store_true',
        help='If set, log spectra codes in output file.'
    )

    parser.add_argument(
        '--site',
        default='DRIAMS-A',
        help='Site to use for the temporal validation scenario',
    )

    parser.add_argument(
        '--duration',
        default=5,
        type=int,
        help='Duration in months of the sliding window',
    )

    parser.add_argument(
        '--oversampling',
        action='store_true',
        help='If set, training data will be oversampled to match test data.',
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
    logging.info(f'Duration: {args.duration}')
    logging.info(f'Antibiotic: {args.antibiotic}')

    extra_filters = []
    if args.site == 'DRIAMS-A':
        extra_filters.append(
            DRIAMSBooleanExpressionFilter('workstation != HospitalHygiene')
        )

    id_suffix = 'strat' if args.site == 'DRIAMS-A' else 'clean'

    driams_dataset = load_driams_dataset(
        DRIAMS_ROOT,
        args.site,
        '*',
        species=args.species,
        antibiotics=args.antibiotic,
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
        extra_filters=extra_filters,
        id_suffix=id_suffix,
    )

    logging.info('Loaded full data set')

    # Contains all spectra of the data set. Need to get this into an
    # `np.array` in order to slice it later on.
    X = np.asarray(
        [spectrum.intensities for spectrum in driams_dataset.X]
    )

    # TODO: make configurable
    train_from = '2015-11-01'
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

    if args.log_codes:
        y_codes = np.asarray(
            driams_dataset.y[~mask]['code'].values,
            dtype=str
        )

    date_range = pd.date_range(
        start=train_from,
        end=test_from,
        freq='M',
    )

    # Set last date for training. If mode is not cumulative, this
    # variable will be modified.
    date_to = test_from

    # Create data frame that only contains the date ranges, making it
    # super easy to collect them subsequently.
    df = pd.DataFrame(index=date_range)

    for i, window in enumerate(df.rolling(args.duration + 1)):
        # Ignore first periods that do not contain sufficient data. This
        # is not an off-by-one error because `i` is an index.
        if i < args.duration:
            continue

        # Always has to be adjusted
        date_from = window.index[0].strftime('%Y-%m-%d')

        # Only adjust for proper sliding window operation, but *not* for
        # cumulative operation.
        if not args.cumulative:
            date_to = window.index[-1].strftime('%Y-%m-%d')

        date_filter = KeepRangeFilter(date_from, date_to)
        mask = y_train.apply(date_filter, axis=1)

        # Local data sets to use for training at this range.
        X_train_ = X_train[mask]
        y_train_ = np.asarray(
            y_train[mask][args.antibiotic].values,
            dtype=int
        )

        # If args.oversampling is set, oversample X_train_ and y_train_
        # to match X_test and y_test in terms of class ratio.
        if args.oversampling:
            # determine desired class ratio
            class_ratios = np.bincount(y_test) / len(y_test)
            assert class_ratios[0] > class_ratios[1], \
                        'Class 1 must be the minority class.'
            
            # This is not a bug; the desired ratio required by RandomOverSampler
            # is _not_ the positive class ratio in the dataset.
            desired_ratio = class_ratios[1] / class_ratios[0]
            
            n0, n1 = np.bincount(y_train_)
            # Need random oversampling because the actual ratio is smaller
            # than the desired ratio.
            if n1 / n0 < desired_ratio:
                try:
                    ros = RandomOverSampler(
                            sampling_strategy=desired_ratio,
                            random_state=args.seed,
                    )

                    X_train_, y_train_ = ros.fit_resample(X_train_, y_train_)
                except:
                    logging.info(f'RandomOverSampler threw an error:'
                                  ' y_train_ {n1/n0} y_test {desired_ratio}')
            # Just pick the desired number of points at random and subset
            # `X_train_` and `y_train_` accordingly.
            else:
                # Be generous with the number of points that we have to
                # resample. This makes a difference of a single sample,
                # so we should be good.
                n_points_to_sample = int(desired_ratio * n0 + 0.5)

                # The resampling only pertains to the positive, i.e. the
                # minority, class.
                indices = np.nonzero(y_train_ == 1)[0]

                indices = resample(
                    indices,
                    n_samples=n_points_to_sample,
                    replace=False,
                    random_state=args.seed
                )

                # Add the remaining indices as well (of the majority class);
                # we do not need to perform any resampling here.
                indices = np.concatenate(
                        (indices, np.nonzero(y_train_ == 0)[0])
                )

                X_train_, y_train_ = X_train_[indices], y_train_[indices]

        class_ratio = np.bincount(y_train_)[1] / len(y_train_)

        logging.info(f'Achieved minority class ratio of {class_ratio:.2f}.')

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
            'oversampling': 'True' if args.oversampling else 'False',
        }

        # Add fingerprint information about the metadata files to make sure
        # that the experiment is reproducible.
        output['metadata_versions'] = metadata_fingerprints

        # add time interval to output_filename
        delta = dateparser.parse(f'{date_to}') \
            - dateparser.parse(f'{date_from}')

        suffix = f'TimeDelta_{delta.days}_{date_from}_{date_to}'

        if args.cumulative:
            suffix += '_cumulative'

        if args.log_codes:
            suffix += '_codes'
            output['y_codes'] = y_codes.tolist()

        output_filename = generate_output_filename(
            args.output,
            output,
            suffix=suffix
        )

        # Only write if we either are running in `force` mode, or the
        # file does not yet exist.
        if not os.path.exists(output_filename) or args.force:

            n_folds = 5

            results, clf = run_experiment(
                # Use local variants of the train data set because they
                # have been subset to a range of months.
                X_train_, y_train_,
                X_test, y_test,
                args.model,
                n_folds,
                random_state=args.seed,      # use seed whenever possible
                verbose=True,                # want info about best model etc.
                return_best_estimator=True,  # want best estimator
            )

            feature_weights = get_feature_weights(clf, args.model)

            # Calibrate the classifier as well so that we get the best
            # scores for the prediction; they should already reflect a
            # probability ideally.
            cccv = CalibratedClassifierCV(
                clf,
                cv=5,              # This is the default anyway
                ensemble=False,    # We want a single classifier
                method='sigmoid',
            )

            cccv.fit(X_train_, y_train_)
            y_pred_calibrated = cccv.predict(X_test).tolist()
            y_score_calibrated = cccv.predict_proba(X_test)

            test_metrics_calibrated = calculate_metrics(
                y_test,
                y_pred_calibrated,
                y_score_calibrated,
                prefix='test_calibrated'
            )

            output.update(results)
            output.update(test_metrics_calibrated)

            output.update({
                'y_pred_calibrated': y_pred_calibrated,
                'y_score_calibrated': y_score_calibrated.tolist(),
                'feature_weights': feature_weights,
            })

            # include sample size to output
            output['train_sample_size'] = len(y_train_)

            logging.info(f'Saving {os.path.basename(output_filename)}')

            with open(output_filename, 'w') as f:
                json.dump(output, f, indent=4)
        else:
            logging.warning(
                f'Skipping {output_filename} because it already exists.'
            )
