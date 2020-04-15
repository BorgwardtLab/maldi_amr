"""Calculate feature importance values from specific scenarios."""

import argparse
import dotenv
import json
import logging
import pathlib
import os

import numpy as np

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import stratify_by_species_and_label

from models import load_pipeline

from utilities import generate_output_filename

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These parameters should remain fixed for this particular
# experiment. We always train on the same data set, using
# *all* available years.
site = 'DRIAMS-A'
years = ['2015', '2016', '2017', '2018']


if __name__ == '__main__':

    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'INPUT',
        type=str,
        help='Input file',
    )

    name = 'feature_importance_values'

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

    pipeline, data = load_pipeline(args.INPUT)

    antibiotic = data['antibiotic']
    site = data['site']
    years = data['years']
    seed = data['seed']
    species = data['species']
    best_params = data['best_params']
    model = data['model']

    logging.info(f'Site: {site}')
    logging.info(f'Years: {years}')
    logging.info(f'Seed: {seed}')

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    metadata_fingerprints = explorer.metadata_fingerprints(site)

    driams_dataset = load_driams_dataset(
            DRIAMS_ROOT,
            site,
            years,
            species=species,
            antibiotics=antibiotic,  # Only a single one for this run
            encoder=DRIAMSLabelEncoder(),
            handle_missing_resistance_measurements='remove_if_all_missing',
            spectra_type='binned_6000',
    )

    logging.info(f'Loaded data set for {species} and {antibiotic}')

    # Create feature matrix from the binned spectra. We only need to
    # consider the second column of each spectrum for this.
    X = np.asarray([spectrum.intensities for spectrum in driams_dataset.X])

    logging.info('Finished vectorisation')

    # Stratified train--test split
    train_index, test_index = stratify_by_species_and_label(
        driams_dataset.y,
        antibiotic=antibiotic,
        random_state=seed,
    )

    logging.info('Finished stratification')

    # Create labels
    y = driams_dataset.to_numpy(antibiotic)

    X_train, y_train = X[train_index], y[train_index]

    pipeline.fit(X_train, y_train)
    clf = pipeline[model]

    output = {
        'site': site,
        'years': years,
        'seed': seed,
        'antibiotic': antibiotic,
        'species': species,
        'model': model,
        'best_params': best_params,
        'metadata_versions': metadata_fingerprints,
        'feature_importance': clf.coef_.tolist(),
    }

    output_filename = generate_output_filename(
        args.output,
        output
    )

    if not os.path.exists(output_filename) or args.force:
        logging.info(f'Saving {os.path.basename(output_filename)}')

        with open(output_filename, 'w') as f:
            json.dump(output, f, indent=4)
    else:
        logging.warning(
            f'Skipping {output_filename} because it already exists.'
        )
