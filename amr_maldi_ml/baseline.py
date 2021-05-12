"""Measure baseline performance.

The purpose of this script is to measure the baseline performance, i.e.
the performance that we obtain by looking at:

    1. *All* species and their respective spectra; accumulating their
       classification performance

    2. *Only* the species information without considering any spectra
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

from models import run_experiment

from utilities import generate_output_filename
from utilities import load_stratify_split_data

from sklearn.preprocessing import OneHotEncoder

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These parameters should remain fixed for this particular
# experiment. We always train on the same data set, using
# *all* available years.
site = 'DRIAMS-A'
years = ['2015', '2016', '2017', '2018']

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
        '-S', '--seed',
        type=int,
        help='Random seed to use for the experiment',
        required=True
    )

    name = 'baseline_case_based_stratification'

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

    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    metadata_fingerprints = explorer.metadata_fingerprints(
        site,
        id_suffix='strat'
    )

    logging.info(f'Site: {site}')
    logging.info(f'Years: {years}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Antibiotic: {args.antibiotic}')

    X_train, y_train, X_test, y_test, meta_train, meta_test = load_stratify_split_data(
        DRIAMS_ROOT,
        site,
        years,
        'Escherichia coli',
        args.antibiotic,
        args.seed
    )

    logging.info(f'Loaded data set for {args.antibiotic}')
    print(meta_train.head())

    # Having loaded the data set, we have to generate two different
    # feature vectors:
    #
    #   1. The 'regular' feature vector as returned by our data set
    #      loader. This necessitates no additional transformation.
    #
    #   2. The feature vector that we obtain by throwing away *all*
    #      information about the spectra, leaving us only with sets
    #      of one-hot-encoded species information.
    #
    # The purpose of the second experiment is to assess to what extent
    # microbial resistance can be be predicted based on information
    # about the species.

    ohe = OneHotEncoder(sparse=False)
    species_vector = np.r_[meta_train['species'].values, meta_test['species'].values]
    
    ohe.fit(species_vector.reshape(-1,1))
    X_species_train = ohe.transform(
        meta_train['species'].values.reshape(-1, 1)
    )
    X_species_test = ohe.transform(
        meta_test['species'].values.reshape(-1, 1)
    )

    logging.info('Created species-only feature vectors')

    for [X_train, X_test], t in zip(
                [[X_species_train, X_species_test], [X_train, X_test]], 
                ['no_spectra', '']
                    ):
        # Prepare the output dictionary containing all information to
        # reproduce the experiment.
        output = {
            'site': site,
            'seed': args.seed,
            'model': args.model,
            'antibiotic': args.antibiotic,
            'years': years,
        }

        # Add fingerprint information about the metadata files to make sure
        # that the experiment is reproducible.
        output['metadata_versions'] = metadata_fingerprints

        output_filename = generate_output_filename(
            args.output,
            output,
            suffix=t
        )

        # Add this information after generating a file name because
        # I want it to be kept out of there. This is slightly hacky
        # but only required for this one experiment.
        output['species'] = 'all' if not t else 'all (w/o spectra)'

        # Only write if we either are running in `force` mode, or the
        # file does not yet exist.
        if not os.path.exists(output_filename) or args.force:

            n_folds = 5

            results = run_experiment(
                X_train, y_train,
                X_test, y_test,
                args.model,
                n_folds,
                random_state=args.seed,  # use seed whenever possible
                verbose=True             # want info about best model etc.
            )

            output.update(results)

            logging.info(f'Saving {os.path.basename(output_filename)}')

            with open(output_filename, 'w') as f:
                json.dump(output, f, indent=4)
        else:
            logging.warning(
                f'Skipping {output_filename} because it already exists.'
            )
