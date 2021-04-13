"""Calculate per-workstation performance curves with resampling."""

import argparse
import dotenv
import json
import logging
import pathlib
import os

import numpy as np

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.filters import DRIAMSBooleanExpressionFilter

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import case_based_stratification

from models import run_experiment

from utilities import generate_output_filename

from sklearn.utils import resample


dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These parameters should remain fixed for this particular
# experiment. We always train on the same data set, using
# *all* available years.
site = 'DRIAMS-A'
years = ['2015', '2016', '2017', '2018']


def _load_dataset(
    root,
    site,
    years,
    species,
    antibiotic,
    filter_expression,
    seed,
    n_samples=0,
):
    """Load data set with a specific filter expression."""
    # Encode type of spectra to use for the remainder of this experiment;
    # this ensures that there are no magic strings flying around in the
    # code.
    spectra_type = 'binned_6000'

    extra_filters = [
        DRIAMSBooleanExpressionFilter(filter_expression)
    ]

    data = load_driams_dataset(
        root,
        site,
        years,
        species=species,
        antibiotics=antibiotic,  # Only a single one for this run
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        id_suffix='strat',
        on_error='warn',
        spectra_type=spectra_type,
        extra_filters=extra_filters,
    )

    if n_samples != 0:

        # Should never be violated; we only want *fewer* samples in the
        # exclusion scenario.
        assert n_samples < len(data.X)

        data.X, data.y = resample(
            data.X, data.y,
            n_samples=n_samples,
            replace=False,
            random_state=seed
        )

    return data


def _run_experiment(
    driams_dataset,
    fingerprints,
    species,
    antibiotic,
    seed,
    output_path,
    force,
    model,
    suffix,
    n_jobs=-1
):
    """Run a single experiment for a given species--antibiotic combination."""
    # Create feature matrix from the binned spectra. We only need to
    # consider the second column of each spectrum for this.
    X = np.asarray([spectrum.intensities for spectrum in driams_dataset.X])

    logging.info('Finished vectorisation')

    # Stratified train--test split
    train_index, test_index = case_based_stratification(
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

    # Prepare the output dictionary containing all information to
    # reproduce the experiment.
    output = {
        'site': site,
        'seed': seed,
        'model': model,
        'antibiotic': antibiotic,
        'species': species,
        'years': years,
        'test_size_obtained': len(y_test) / (len(y_train) + len(y_test)),
        'prevalence_train': (np.bincount(y_train) / len(y_train)).tolist(),
        'prevalence_test': (np.bincount(y_test) / len(y_test)).tolist(),
    }

    output_filename = generate_output_filename(
        output_path,
        output,
        suffix=suffix,
    )

    # Add fingerprint information about the metadata files to make sure
    # that the experiment is reproducible.
    output['metadata_versions'] = fingerprints

    # Only write if we either are running in `force` mode, or the
    # file does not yet exist.
    if not os.path.exists(output_filename) or force:

        n_folds = 5

        results = run_experiment(
            X_train, y_train,
            X_test, y_test,
            model,
            n_folds,
            verbose=True,
            random_state=seed,
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


if __name__ == '__main__':

    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-a', '--antibiotic',
        default='Ceftriaxone',
        type=str,
        help='Antibiotic for which to run the experiment'
    )

    parser.add_argument(
        '-s', '--species',
        default='Escherichia coli',
        type=str,
        help='Species for which to run the experiment'
    )

    parser.add_argument(
        '-S', '--seed',
        type=int,
        default=123,
        help='Random seed to use for the experiment'
    )

    parser.add_argument(
        '-m', '--model',
        default='lr',
        help='Selects model to use for subsequent training'
    )

    parser.add_argument(
        '-w', '--workstation',
        required=True,
        type=str,
        help='Sets name of workstation on which to perform the analysis'
    )

    parser.add_argument(
        '-W', '--exclude-workstation',
        required=True,
        type=str,
        help='Sets name of workstation that is to be excluded from the '
             'analysis. Resampling happens on the workstation that was '
             'specified using `--workstation`.'
    )

    name = 'resample_per_workstation'

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

    logging.info(f'Site: {site}')
    logging.info(f'Species: {args.species}')
    logging.info(f'Antibiotic: {args.antibiotic}')
    logging.info(f'Years: {years}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Workstation: {args.workstation}')

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    metadata_fingerprints = explorer.metadata_fingerprints(
        site,
        id_suffix='strat'
    )

    # How many jobs to use to run this experiment. Should be made
    # configurable ideally.
    n_jobs = 24

    only_workstation_data = _load_dataset(
        explorer.root,
        site,
        years,
        args.species,
        args.antibiotic,
        f'workstation == {args.workstation}',
        args.seed
    )

    n_samples = len(only_workstation_data.X)

    logging.info(f'Per-workstation data set contains {n_samples} samples')

    _run_experiment(
        only_workstation_data,
        metadata_fingerprints,
        args.species,
        args.antibiotic,
        args.seed,
        args.output,
        args.force,
        args.model,
        f'only_{args.workstation}',
        n_jobs,
    )

    logging.info(
        f'Excluding workstation {args.exclude_workstation} data for '
        f'resampled training (n = {n_samples}).'
    )

    no_workstation_data = _load_dataset(
        explorer.root,
        site,
        years,
        args.species,
        args.antibiotic,
        f'workstation != {args.exclude_workstation}',
        args.seed,
        n_samples=n_samples,
    )

    _run_experiment(
        no_workstation_data,
        metadata_fingerprints,
        args.species,
        args.antibiotic,
        args.seed,
        args.output,
        args.force,
        args.model,
        f'no_{args.exclude_workstation}_resample_{args.workstation}',
        n_jobs,
    )
