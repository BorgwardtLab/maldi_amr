"""Calculate mean spectra of a given scenario for both classes."""

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

def feature_importances(args):
    # Create the output directory for storing all results of the
    # individual combinations.
    os.makedirs(args.output, exist_ok=True)

    # Check if INPUT is a list of files or a single file
    print(list(args.INPUT))
    files = list(args.INPUT)

    # Create empty lists for average feature importances
    all_antibiotics =[]
    all_sites = []
    all_years = []
    all_seeds = []
    all_species = []
    all_models = [] 
    all_feature_importances = []
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

        # Append to lists for average feature importance output
        all_antibiotics.append(antibiotic)
        all_sites.append(site)
        if years not in all_years:
            all_years.append(years)
        all_seeds.append(seed)
        all_species.append(species)
        all_models.append(model) 
        all_feature_importances.append(clf.coef_.tolist())
        all_metadata_versions.append(metadata_fingerprints)
        print(f'\nLength feat importances {len(clf.coef_.tolist())}')

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

    if len(files) > 1:
        mean_feature_importances = np.mean(
                     np.array(all_feature_importances),
                     axis=0).tolist()
        std_feature_importances = np.std(
                     np.array(all_feature_importances),
                     axis=0).tolist()
        #metadata_fingerprints = list(set(all_metadata_versions))
        sites = list(set(all_sites))
        antibiotics = list(set(all_antibiotics))
        species = list(set(all_species))
        models = list(set(all_models))

        print(f'\nLength mean feat importances {len(mean_feature_importances)}')

        # Stop if files from more than one antibiotics-species-model scenario 
        # were given as input
        if any([len(l) > 1 for l in [all_years,
                                     sites,
                                     antibiotics,
                                     species,
                                     models]]):
            print('Cannot include more than one scenario in average \
                   feature importance vectors.')
            return

        output = {
            'site': sites[0],
            'years': all_years[0],
            'seed': all_seeds,
            'antibiotic': antibiotics[0],
            'species': species[0],
            'model': models[0],
            #'best_params': best_params,
            #'metadata_versions': metadata_fingerprints,
            'mean_feature_importance': mean_feature_importances,
            'std_feature_importance': std_feature_importances,
        }
        for k in output.keys():
            if k != 'feature_importance':
                print(type(output[k]), output[k])

        output_print = output.copy()
        output_print['seed'] = '-'.join([str(seed) for seed in all_seeds])     
 
        output_filename = generate_output_filename(
            args.output,
            output_print,
            suffix='average',
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

    feature_importances(args)

