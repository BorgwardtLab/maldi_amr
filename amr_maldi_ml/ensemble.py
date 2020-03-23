"""Measure ensemble training performance.

The purpose of this script is to measure ensemble training performance,
i.e. the performance that we obtain by *training* on all species, but
evaluating only on a single one (which gives us more data, but also a
higher variance to contend with). This is contrasted by the usual
training scenario, where we train on specific species and evaluate on it
as well.

This script reproduces Figure 3 in the main paper.
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
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These parameters should remain fixed for this particular
# experiment. We always train on the same data set, using
# *all* available years.
site = 'DRIAMS-A'
years = ['2015', '2016', '2017', '2018']


def get_min_samples(n_folds, y_train):
    """Calculate number of minimum samples for CV-based scenarios.

    This helper function checks the prevalence of the positive class and
    uses it to calculate the number of minimum samples required in
    a given cross validation scenario.

    Parameters
    ----------
    n_folds : int
        Number of folds to be used

    y_train : array-like
        Labels to be used for training

    Returns
    -------
    Minimum number of samples required.
    """
    # Ensures that we *always* base the minimum number of samples on the
    # minority class. This should, in most cases, be the positive class,
    # i.e. `1`, but there are some combinations for which that might not
    # be the case.
    minority_class = np.argmin(np.bincount(y_train))
    prevalence = (y_train == minority_class).sum() / len(y_train)
    min_samples = int(np.ceil(n_folds / prevalence))

    return min_samples


def subset_indices(y, indices, species):
    """Calculate subset of species-relevant indices and labels.

    The purpose of this function is to return a set of labels for use
    in a downstream classification task, based on a prior *selection*
    of indices.

    Parameters
    ----------
    y : `pandas.DataFrame`
        Data frame containing labels and species information. Must have
        a 'species' column to be valid.

    indices : array-like, int
        Indices (integer-based) referring to valid positions within the
        label vector `y`.

    species : str
        Identifier of a species whose subset of labels should be
        calculated.

    Returns
    -------
    Tuple of indices (array of length n, where n is the number of
    samples) and subset of data frame. The indices correspond to the
    subset of species-specific indices. The subset can be used for
    accessing vectors of spectra.
    """
    y = y.iloc[indices].copy()
    y.index = indices
    y = y.query('species == @args.species')

    indices = y.index.values
    return indices, y


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
        '-s', '--species',
        type=str,
        help='Species for which to run the experiment',
        required=True
    )

    parser.add_argument(
        '-S', '--seed',
        type=int,
        help='Random seed to use for the experiment',
        required=True
    )

    parser.add_argument(
        '--n-scenarios', '-n',
        type=int,
        default=20,
        help='Number of scenarios (subsampling draws) to run',
    )

    parser.add_argument(
        '--index', '-i',
        type=int,
        help='Index of scenario (subsampling draw) to run',
        required=True,
    )

    name = 'fig3_ensemble'

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

    if not (0 <= args.index < args.n_scenarios):
        raise RuntimeError(f'Scenario index {args.index} is out of range')

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
    logging.info(f'Model: {args.model}')
    logging.info(f'Antibiotic: {args.antibiotic}')
    logging.info(f'Species: {args.species}')

    driams_dataset = load_driams_dataset(
        DRIAMS_ROOT,
        site,
        years,
        '*',   # This is correct; we initially want *all* species
        antibiotics=args.antibiotic,
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
    )

    logging.info(f'Loaded data set')

    # Create feature matrix from the binned spectra. We only need to
    # consider the second column of each spectrum for this.
    X = np.asarray([spectrum.intensities for spectrum in driams_dataset.X])

    # Do the split only once in order to ensure that we are
    # predicting on the same samples subsequently.
    train_index, test_index, train_stratify, _ = stratify_by_species_and_label(
        driams_dataset.y,
        antibiotic=args.antibiotic,
        return_stratification=True,
        random_state=args.seed,
    )

    # Required only once because we have to use the *same* indices in
    # the downstream classification task.
    test_index_, y_test = subset_indices(
        driams_dataset.y,
        test_index,
        args.species
    )

    # Actual label vector (`y_test` is a data frame prior to the call of
    # this function)
    y_test = driams_dataset.to_numpy(args.antibiotic, y=y_test)

    for ensemble in [True, False]:

        # Not in ensemble mode; select all samples with the proper
        # species that we can train on.
        if not ensemble:
            train_index_, y_train = subset_indices(
                driams_dataset.y,
                train_index,
                args.species
            )

            # Sets all indices that are *valid* to be used in the
            # subsequent subset of the indices.
            mask = np.isin(
                    train_index,
                    train_index_,
                    assume_unique=True)

            train_stratify_ = train_stratify[mask]

        # Ensemble mode: just use the regular train index. The reason
        # for choosing a different variable name is because I do not
        # want to overwrite anything here.
        else:
            train_index_ = train_index
            y_train = driams_dataset.y.iloc[train_index]
            train_stratify_ = train_stratify

        # Main split required for the data set. Will be subsampled later
        # on (this pertains to both `y_train` and `X_train`, but not to
        # `X_test` and `y_test`).

        y_train = driams_dataset.to_numpy(args.antibiotic, y=y_train)

        X_train = X[train_index_]
        X_test = X[test_index_]

        # Subsequently, we will add more and more information to this
        # dictionary.
        output = {
            'site': site,
            'model': args.model,
            'species': args.species,
            'seed': args.seed,
            'antibiotic': args.antibiotic,
            'years': years,
            'n_samples': [],
        }

        n_folds = 5
        n_scenarios = args.n_scenarios

        min_samples = get_min_samples(n_folds, y_train)
        if min_samples >= len(y_train):
            logging.warning(f'Require {min_samples} samples, but only '
                            f'{len(y_train)} are present. Decreasing '
                            f'`n_folds` to 3.')

            # TODO: could make this adjustment for a *range* of possible
            # values and see which one works best.
            n_folds = 3
            min_samples = get_min_samples(n_folds, y_train)

        max_samples = len(train_index_)
        sample_sizes = np.linspace(
                           min_samples,
                           max_samples,
                           n_scenarios,
                           dtype=int)

        n_samples = sample_sizes[args.index]

        indices = resample(
                    np.arange(max_samples),
                    n_samples=n_samples,
                    stratify=y_train,
                    random_state=args.seed,
                    replace=False
                )

        # Store number of samples; the type cast to `int` ensures
        # that the value can be serialised.
        output['n_samples'] = int(n_samples)

        # Add fingerprint information about the metadata files to make sure
        # that the experiment is reproducible.
        output['metadata_versions'] = metadata_fingerprints

        # Prepare suffix for output
        suffix = 'ensemble' if ensemble else 'single'
        suffix += f'_index_{args.index:02}'

        output_filename = generate_output_filename(
            args.output,
            output,
            suffix=suffix
        )

        # Only write if we either are running in `force` mode, or the
        # file does not yet exist.
        if not os.path.exists(output_filename) or args.force:
            logging.info(f'Saving {os.path.basename(output_filename)}')

            results = run_experiment(
                X_train=X_train[indices],
                y_train=y_train[indices],
                X_test=X_test,
                y_test=y_test,
                n_folds=n_folds,
                model=args.model,
            )

            output.update(results)

            with open(output_filename, 'w') as f:
                json.dump(output, f, indent=4)
        else:
            logging.warning(
                f'Skipping {output_filename} because it already exists.'
            )
