"""Domain adaptation for shallow (i.e. non-deep) models."""

import argparse
import dotenv
import json
import logging
import os
import pathlib

import numpy as np

from libtlda.iw import ImportanceWeightedClassifier

from sklearn.preprocessing import StandardScaler

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.filters import DRIAMSBooleanExpressionFilter

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import case_based_stratification
from maldi_learn.utilities import stratify_by_species_and_label

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


def _run_experiment(X, y, Z, z_true, random_state=None):
    """Run domain adaptation experiment."""
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Z = scaler.fit_transform(Z)

    clf = ImportanceWeightedClassifier(
        loss_function='logistic',
        l2_regularization=0.01,
        weight_estimator='lr',  # Use LR to discriminate the domains
    )

    clf.fit(X, y, Z)
    z_pred = clf.predict(Z)

    print(clf.iw)
    print(max(clf.iw))

    print(sum(z_true))
    print(sum(z_pred))

    # TODO: need to return `dict` with additional information about
    # experiment, such as performance etc.


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-a', '--antibiotic',
        default='Ceftriaxone',
        type=str,
        help='Antibiotic for which to run the experiment',
    )

    parser.add_argument(
        '-S', '--seed',
        type=int,
        help='Random seed to use for the experiment',
        required=True
    )

    parser.add_argument(
        '-s', '--species',
        default='Escherichia coli',
        type=str,
        help='Species for which to run the experiment',
    )

    name = 'domain_adaptation_shallow'

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
        '--source-site',
        type=str,
        help='Site to use as source (e.g. DRIAMS-A)',
        required=True
    )

    parser.add_argument(
        '--target-site',
        type=str,
        help='Site to use as target (e.g. DRIAMS-B)',
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

    source_site = args.source_site
    target_site = args.target_site

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    metadata_fingerprints_source = explorer.metadata_fingerprints(source_site)
    metadata_fingerprints_target = explorer.metadata_fingerprints(target_site)

    source_years = explorer.available_years(source_site)
    target_years = explorer.available_years(target_site)

    logging.info(f'Source site: {source_site}')
    logging.info(f'Source years: {source_years}')
    logging.info(f'Target site: {target_site}')
    logging.info(f'Target years: {target_years}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Antibiotic: {args.antibiotic}')
    logging.info(f'Species: {args.species}')

    # TODO: we are losing some samples here because we ignore the
    # `X_test` split; on the other hand, this is compatible  with
    # our validation scenario(s).
    X, y, *_ = _load_data(
        args.source_site,
        source_years,
        args.species,
        args.antibiotic,
        args.seed,
    )

    logging.info('Loaded source site data')

    # TODO: need test split here as well
    Z, z_true, *_ = _load_data(
        args.target_site,
        target_years,
        args.species,
        args.antibiotic,
        args.seed,
    )

    logging.info('Loaded target site data')

    # Prepare the output dictionary containing all information to
    # reproduce the experiment.
    output = {
        'train_site': source_site,
        'source_years': source_years,
        'test_site': target_site,
        'target_years': target_years,
        'seed': args.seed,
        'antibiotic': args.antibiotic,
        'species': args.species,
        'model': 'ImportanceWeightedClassifier',  # TODO: Make adjustable
    }

    # Add fingerprint information about the metadata files to make sure
    # that the experiment is reproducible.
    output['metadata_versions_source'] = metadata_fingerprints_source
    output['metadata_versions_target'] = metadata_fingerprints_target

    output_filename = generate_output_filename(
        args.output,
        output,
    )

    # Only write if we either are running in `force` mode, or the
    # file does not yet exist.
    if not os.path.exists(output_filename) or args.force:

        results = _run_experiment(
            X, y, Z, z_true,
            random_state=args.seed,  # use seed whenever possible
        )

        output.update(results)

        logging.info(f'Saving {os.path.basename(output_filename)}')

        with open(output_filename, 'w') as f:
            json.dump(output, f, indent=4)
    else:
        logging.warning(
            f'Skipping {output_filename} because it already exists.'
        )
