"""Calculate performance curves for species--antibiotic combinations."""

import argparse
import dotenv
import json
import logging
import pathlib
import os

import numpy as np
import pandas as pd

from maldi_learn.driams import DRIAMSDatasetExplorer

from sklearn.model_selection import train_test_split

from models import run_experiment

from utilities import generate_output_filename
from utilities import load_stratify_split_data


dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

def _subsample(y, proportion, seed):
    n = len(y)
    indices = range(n)
    sample_size = int(n*proportion)

    if sample_size == n:
        return indices, sample_size

    if sample_size == 0:
        return np.array([]), sample_size

    if sample_size == n-1:
        sample_size -= 1
    elif sample_size == 1:
        sample_size += 1
        
    indices_sampled, _ = train_test_split(indices, 
                             train_size=sample_size,
                             stratify=y,
                             random_state=seed,
                             ) 
    return indices_sampled, sample_size
    

def _run_experiment(
    root,
    fingerprints,
    train_site,
    test_site,
    species,
    antibiotic,
    seed,
    output_path,
    force,
    model,
    train_proportions,
    filter_expression,
    n_jobs=-1
):
    """Run a single experiment for a given species--antibiotic combination."""
    X_train, y_train, meta_train, sample_sizes = [], [], [], []

    assert len(train_site) == len(train_proportions), \
        'Number of train sites and train proportions not equal.'

    for i, site in enumerate(train_site):
        X, y, *_, meta, _ = load_stratify_split_data(
            root,
            site,
            '*',
            species,
            antibiotic,
            seed
        )
        
        idx, sample_size = _subsample(y, train_proportions[i], args.seed)
        
        sample_sizes.append(sample_size)
        X_train.append(X[idx, :])
        y_train.append(y[idx])
        meta_train.append(meta.iloc[idx])

    X_train, y_train = map(
        np.concatenate, [X_train, y_train]
    )

    meta_train = pd.concat(meta_train)

    X_test, y_test, meta_test = [], [], []

    for site in test_site:
        _, _, X, y, _, meta = load_stratify_split_data(
            root,
            site,
            '*',
            species,
            antibiotic,
            seed
        )

        X_test.append(X)
        y_test.append(y)
        meta_test.append(meta)

    X_test, y_test = map(
        np.concatenate, [X_test, y_test]
    )

    meta_test = pd.concat(meta_test)

    # Prepare the output dictionary containing all information to
    # reproduce the experiment.
    output = {
        'train_site': train_site,
        'test_site': test_site,
        'seed': seed,
        'model': model,
        'antibiotic': antibiotic,
        'species': species,
        'years': '*',
        'test_size_obtained': len(y_test) / (len(y_train) + len(y_test)),
        'prevalence_train': (np.bincount(y_train) / len(y_train)).tolist(),
        'prevalence_test': (np.bincount(y_test) / len(y_test)).tolist(),
        'sample_sizes_per_train_site': sample_sizes,
        'train_proportions': train_proportions,
    }

    suffix = None

    # Generate new suffix based on filter expression. This ensures that
    # different filters result in different filenames.
    if filter_expression:
        # This can surely be solved in a more elegant fashion, but what
        # the heck?
        suffix = filter_expression.replace(' ', '_')
        suffix = suffix.replace('==', '')
        suffix = suffix.replace('!=', 'no')
        suffix = suffix.replace('__', '_')

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
        '-F', '--filter-expression',
        type=str,
        help='Optional filter expression to use for reducing the input '
             'data set. Can be `workstation == Blood`, for instance, to '
             'keep only certain samples.'
    )

    parser.add_argument(
        '--train-site',
        nargs='+',
        default=['DRIAMS-A'],
        type=str,
        help='Train site(s)'
    )

    parser.add_argument(
        '--train-proportions',
        nargs='+',
        type=float,
        help='Train site proportions'
    )

    parser.add_argument(
        '--test-site',
        nargs='+',
        default=['DRIAMS-A'],
        type=str,
        help='Test site(s)'
    )

    name = 'curves_per_species_and_antibiotics_case_based_stratification'

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

    logging.info(f'Train site(s): {args.train_site}')
    logging.info(f'Test site(s): {args.test_site}')
    logging.info(f'Species: {args.species}')
    logging.info(f'Antibiotic: {args.antibiotic}')
    logging.info(f'Seed: {args.seed}')

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    metadata_fingerprints = {
        site: explorer.metadata_fingerprints(
                site,
                id_suffix='strat' if site == 'DRIAMS-A' else 'clean')
        for site in args.train_site + args.test_site
    }

    # How many jobs to use to run this experiment. Should be made
    # configurable ideally.
    n_jobs = 24


    if args.train_proportions:
        train_proportions = args.train_proportions
    else:
        train_proportions = [1.0]*len(args.train_site)

    _run_experiment(
        explorer.root,
        metadata_fingerprints,
        args.train_site,
        args.test_site,
        args.species,
        args.antibiotic,
        args.seed,
        args.output,
        args.force,
        args.model,
        train_proportions,
        args.filter_expression,
        n_jobs
    )
