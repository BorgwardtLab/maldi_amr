"""Calculate feature importance values from specific scenarios."""

import argparse
import dotenv
import json
import logging
import pathlib
import os

import numpy as np

from maldi_learn.driams import load_driams_dataset

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
    print('\nFiles used to calculate average:')
    [print(file) for file in list(args.INPUT)]
    files = list(args.INPUT)

    # Create empty lists for average feature importances
    antibiotics =[]
    sites = []
    seeds = []
    species = []
    models = [] 
    feature_weights = []

    for file in files:
        with open(file) as f:
            data = json.load(f)

        # Append to lists for average feature importance output
        antibiotics.append(data['antibiotic'])
        sites.append(data['site'])
        seeds.append(data['seed'])
        species.append(data['species'])
        models.append(data['model']) 
        feature_weights.append(data['feature_weights'])

    # Stop if files from more than one antibiotics-species-model scenario 
    # were given as input
    if any([len(set(l)) > 1 for l in [sites,
                                 antibiotics,
                                 species,
                                 models]]):
        print('Cannot include more than one scenario in average \
               feature importance vectors.')
        return

    # Condense experiments to mean or single value.
    site = list(set(sites))[0]
    antibiotic = list(set(antibiotics))[0]
    species = list(set(species))[0]
    model = list(set(models))[0]
    mean_feature_weights = np.mean(
                 np.array(feature_weights),
                 axis=0).tolist()
    std_feature_weights = np.std(
                 np.array(feature_weights),
                 axis=0).tolist()

    output = {
        'site': site,
        'seed': '-'.join([str(seed) for seed in seeds]),
        'antibiotic': antibiotic,
        'species': species,
        'model': model,
        'mean_feature_weights': mean_feature_weights,
        'std_feature_weights': std_feature_weights,
    }

    output_filename = generate_output_filename(
        args.output,
        output,
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

