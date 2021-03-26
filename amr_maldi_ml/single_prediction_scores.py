"""Calculate prediction score for single DRIAMS entries from specific scenarios."""

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

def single_prediction_scores(args):
    # Create the output directory for storing all results of the
    # individual combinations.
    os.makedirs(args.output, exist_ok=True)

    # Check if INPUT is a list of files or a single file
    files = list(args.INPUT)
    print(f'\nLength INPUT is {len(files)}')
    print(f'\nFile to read model from {files}')
    print(f'i\nID to classify {args.input}')

    # Create empty lists for average feature importances
    all_antibiotics = []
    all_sites = []
    all_years = []
    all_seeds = []
    all_species = []
    all_models = [] 
    all_scores = []
    all_predictions = []
    all_metadata_versions = []

    for f in files:
        pipeline, data = load_pipeline(f)

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
        logging.info(f'Model: {model}')

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

        # include prediction of input sample
        if args.input in driams_dataset.y['code'].values:
            logging.info('Target code found in loaded dataset')
            df = driams_dataset.y.reset_index()
            target_index = df[df['code'] == args.input].index.values
            assert df.shape[0] == X.shape[0]
            X_target = X[target_index]
        else:
            logging.info('Target code NOT found in loaded dataset. Abort.')
            break 

        # Append to lists for average feature importance output
        all_antibiotics.append(antibiotic)
        all_sites.append(site)
        if years not in all_years:
            all_years.append(years)
        all_seeds.append(seed)
        all_species.append(species)
        all_models.append(model) 
        all_scores.append(pipeline.predict_proba(X_target)[0,1])
        all_predictions.append(pipeline.predict(X_target))
        all_metadata_versions.append(metadata_fingerprints)

        output = {
            'site': site,
            'years': years,
            'seed': seed,
            'antibiotic': antibiotic,
            'species': species,
            'model': model,
            'best_params': best_params,
            'metadata_versions': metadata_fingerprints,
            'scores_positive': pipeline.predict_proba(X_target)[0,1].tolist(),
            'predictions': pipeline.predict(X_target).tolist(),
        }

        output_filename = generate_output_filename(
            args.output,
            output,
            suffix=f'{args.input}',
        )

        if not os.path.exists(output_filename) or args.force:
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
        nargs='+',
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input ids to classify',
        required=True,
    )

    name = 'single_prediction_scores/data_collection'

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

    single_prediction_scores(args)

