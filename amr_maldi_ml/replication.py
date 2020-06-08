"""Measure replication performance.

The purpose of this script is to measure the replication performance of
a classifier, i.e. the performance that we obtain by training on one
site and testing on another, with the *same* samples being measured in
both sites. All the differences in performance are thus due to some of
the differences in hospital procedure.
"""

import argparse
import collections
import dotenv
import joblib
import json
import logging
import os
import pathlib
import warnings

import numpy as np

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from models import calculate_metrics
from models import get_pipeline_and_parameters

from utilities import generate_output_filename

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')


def run_experiment(
    X_source, y_source,
    X_target, y_target,
    model,
    n_folds,
    random_state=None,
    verbose=False,
):
    """Run experiment for given train--test split.

    We need to provide our own code here because for the replication
    scenario, we are interested in the performance evaluated using a
    10-fold cross-validation procedure.

    Parameters
    ----------
    X_source : array-like
        Training data coming from the 'source' site, such as DRIAMS-E.

    y_source : list
        Labels for the training data

    X_target : array_like
        Test data coming from the 'target' site, such as DRIAMS-F.

    y_target : list
        Labels for the test data

    model : str
        Specifies a model whose pipeline will be queried and set up by
        this function. Must be a valid model according to the function
        `get_pipeline_and_parameters()`.

    n_folds : int
        Number of folds for internal cross-validation

    random_state : int, `RandomState` instance, or None
        If set, propagates random state to a model. This is *not*
        required or used for all models.

    Returns
    -------
    A dictionary containing measurement descriptions and their
    corresponding values for each fold.
    """
    pipeline, param_grid = get_pipeline_and_parameters(
        model,
        random_state
    )

    cv = StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=random_state
    )

    # Prepare the results dictionary for this experiment. Depending on
    # the input parameters of this function, additional information is
    # added.
    #
    # The primary index of this `dict` will be the fold index.
    results = collections.defaultdict(dict)

    for i, (train_index, test_index) in enumerate(cv.split(
        X_source, y_source)
    ):
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=n_folds,
            scoring='roc_auc',
            n_jobs=-1,
        )

        X_train = X_source[train_index]
        y_train = y_source[train_index]

        logging.info(f'Starting grid search for {len(X_train)} samples')

        # Ignore these warnings only for the grid search process. The
        # reason is that some of the jobs will inevitably *fail* to
        # converge because of bad `C` values. We are not interested in
        # them anyway.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            with joblib.parallel_backend('threading', -1):
                grid_search.fit(X_train, y_train)

        # Calculate metrics for the training data fully in-line because we
        # only ever want to save the results.
        train_metrics = calculate_metrics(
            y_train,
            grid_search.predict(X_train),
            grid_search.predict_proba(X_train),
            prefix='train'
        )

        # Ensures that we are performing the same predictions at all
        # times, since the two sites are supposed to measure the same
        # samples.
        #
        # If this is not the case, we log it.
        if not np.equal(y_source[test_index], y_target[test_index]).all():
            logging.warning(
                f'Found label discrepancy:\n'
                f'  - source: {y_source[test_index]}\n'
                f'  - target: {y_target[test_index]}'
            )

        # Use the labels of the source site for the training. This is
        # consistent with how the cross-validation is structured.
        y_test = y_source[test_index]

        y_pred_source = grid_search.predict(X_source[test_index])
        y_score_source = grid_search.predict_proba(X_source[test_index])

        test_metrics_source = calculate_metrics(
            y_test,
            y_pred_source,
            y_score_source,
            prefix='test_source',
        )

        y_pred_target = grid_search.predict(X_target[test_index])
        y_score_target = grid_search.predict_proba(X_target[test_index])

        test_metrics_target = calculate_metrics(
            y_test,
            y_pred_target,
            y_score_target,
            prefix='test_target',
        )

        results[i].update(train_metrics)
        results[i].update(test_metrics_source)
        results[i].update(test_metrics_target)

    return results


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

    name = 'replication'

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
        help='Site to use for training (e.g. DRIAMS-E)',
        default='DRIAMS-E',
    )

    parser.add_argument(
        '--test-site',
        type=str,
        help='Site to use for testing (e.g. DRIAMS-F)',
        default='DRIAMS-F',
    )

    args = parser.parse_args()

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

    # If this is not the same, the replication itself does not make any
    # sense.
    assert train_years == test_years

    logging.info(f'Train site: {train_site}')
    logging.info(f'Train years: {train_years}')
    logging.info(f'Test site: {test_site}')
    logging.info(f'Test years: {test_years}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Model: {args.model}')
    logging.info(f'Antibiotic: {args.antibiotic}')
    logging.info(f'Species: {args.species}')

    driams_dataset_train = load_driams_dataset(
        DRIAMS_ROOT,
        train_site,
        train_years,
        species=args.species,
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
        species=args.species,
        antibiotics=args.antibiotic,
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
    )

    logging.info('Loaded test site data')

    X_train = np.asarray(
        [spectrum.intensities for spectrum in driams_dataset_train.X]
    )
    y_train = driams_dataset_train.to_numpy(args.antibiotic)

    X_test = np.asarray(
        [spectrum.intensities for spectrum in driams_dataset_test.X]
    )
    y_test = driams_dataset_test.to_numpy(args.antibiotic)

    # Ensures that the samples are loaded in the proper order. Else, our
    # stratification will *not* work.

    id_train = driams_dataset_train.y.id.values
    id_test = driams_dataset_test.y.id.values

    assert np.equal(id_train, id_test).all()

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
