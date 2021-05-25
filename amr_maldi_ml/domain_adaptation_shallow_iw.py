"""Domain adaptation for shallow (i.e. non-deep) models."""

import argparse
import dotenv
import json
import logging
import os
import pathlib

import numpy as np

from importance_weighting import run_iw_experiment

from sklearn.preprocessing import StandardScaler

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.filters import DRIAMSBooleanExpressionFilter

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import case_based_stratification
from maldi_learn.utilities import stratify_by_species_and_label

from utilities import generate_output_filename
from utilities import load_stratify_split_data

from models import calculate_metrics

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')



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
        '-m', '--model',
        default='lr',
        help='Selects model to use for subsequent training'
    )

    parser.add_argument(
        '-s', '--species',
        default='Escherichia coli',
        type=str,
        help='Species for which to run the experiment',
    )

    name = 'domain_adaptation_shallow_iw'

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
    X_train, y_train, X_test, y_test, *_ = load_stratify_split_data(
        args.source_site,
        source_years,
        args.species,
        args.antibiotic,
        args.seed,
    )

    logging.info('Loaded source site data')

    # We don't need `z_train`, i.e. the labels on the training domain,
    # because we never look at them anyway.
    Z_train, z_train, Z_test, z_test, *_ = load_stratify_split_data(
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
        'model': args.model, 
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

        n_folds = 5
        
        results = run_iw_experiment(
                X_train, y_train,
                X_test, y_test,
                Z_train, z_train,
                Z_test, z_test,
                model=args.model,
                n_folds=n_folds,
                random_state=args.seed,
        )

        output.update(results)

        logging.info(f'Saving {os.path.basename(output_filename)}')

        with open(output_filename, 'w') as f:
            json.dump(output, f, indent=4)
    else:
        logging.warning(
            f'Skipping {output_filename} because it already exists.'
        )
