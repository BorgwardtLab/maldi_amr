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

from models import get_pipeline_and_parameters

from utilities import generate_output_filename

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These parameters should remain fixed for this particular
# experiment. We always train on the same data set, using
# *all* available years.
site = 'DRIAMS-A'
years = ['2015', '2016', '2017', '2018']


def _run_experiment(
    root,
    fingerprints,
    species,
    antibiotic,
    seed,
    output_path,
    force,
    model,
    n_jobs=-1
):
    """Run a single experiment for a given species--antibiotic combination."""
    driams_dataset = load_driams_dataset(
            root,
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
    X_test, y_test = X[test_index], y[test_index]

    # Prepare the output dictionary containing all information to
    # reproduce the experiment.
    output = {
        'site': site,
        'seed': seed,
        'model': model,
        'antibiotic': antibiotic,
        'species': species,
        'years': years,
    }

    output_filename = generate_output_filename(
        output_path,
        output
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

    # Load configuration and output (which we will subsequently ignore,
    # though, because we do not need it).
    with open(args.INPUT) as f:
        data = json.load(f)

    # Create the output directory for storing all results of the
    # individual combinations.
    os.makedirs(args.output, exist_ok=True)

    antibiotic = data['antibiotic']
    site = data['site']
    years = data['years']
    seed = data['seed']
    species = data['species']
    best_params = data['best_params']

    # This accommodates older versions of the code that do not save
    # their models to the file.
    model = data.get('model', 'lr')

    logging.info(f'Site: {site}')
    logging.info(f'Years: {years}')
    logging.info(f'Seed: {seed}')

    pipeline, _ = get_pipeline_and_parameters(model, random_state=seed) 
    pipeline.set_params(**best_params)

    # Scaling needs some manual adjustments because we just store
    # whether a scaler was used or not.
    if pipeline['scaler'] == 'StandardScaler':
        pipeline.set_params(scaler=StandardScaler())

    raise 'heck'

    # Create input grid for the subsequent experiments. Not all
    # combinations are useful, hence we specify them in a list.
    # This will only be used if '-A' or '--all' was specified.
    input_grid = ParameterGrid([
        {
            'species': ['Escherichia coli'],
            'antibiotic': ['Ciprofloxacin',
                           'Amoxicillin-Clavulanic acid',
                           'Ceftriaxone',
                           'Tobramycin',
                           'Piperacillin-Tazobactam',
                           'Cefepime'],
            'seed': seeds,
        },
        {
            'species': ['Klebsiella pneumoniae'],
            'antibiotic': ['Ciprofloxacin',
                           'Amoxicillin-Clavulanic acid',
                           'Ceftriaxone',
                           'Tobramycin',
                           'Piperacillin-Tazobactam',
                           'Meropenem',
                           'Cefepime'],
            'seed': seeds,
        },
        {
            'species': ['Staphylococcus aureus'],
            'antibiotic': ['Ciprofloxacin',
                           'Penicillin',
                           'Oxacillin',
                           'Fusidic acid'],
            'seed': seeds,
        }
    ])

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    metadata_fingerprints = explorer.metadata_fingerprints(site)

    # How many jobs to use to run this experiment. Should be made
    # configurable ideally.
    n_jobs = 24

    # Run all combinations and ignore everything else. Other arguments
    # may be supplied, but we will not use them.
    if args.all:

        logging.info('Running all experiments for pre-defined grid.')
        logging.info('Ignoring *all* other parameters.')

        for combination in input_grid:
            species = combination['species']
            antibiotic = combination['antibiotic']
            seed = combination['seed']

            _run_experiment(
                explorer.root,
                metadata_fingerprints,
                species,
                antibiotic,
                seed,
                args.output,
                args.force,
                args.model,
                n_jobs
            )
    # Run a specific experiment: species, antibiotic, and seed have to
    # be specified.
    else:
        assert args.species is not None
        assert args.antibiotic is not None
        assert args.seed is not None

        _run_experiment(
            explorer.root,
            metadata_fingerprints,
            args.species,
            args.antibiotic,
            args.seed,
            args.output,
            args.force,
            args.model,
            n_jobs
        )
