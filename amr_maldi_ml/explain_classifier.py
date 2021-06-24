"""Explain classifier based on Shapley values."""

import argparse
import dotenv
import json
import logging
import pathlib
import os
import shap

from maldi_learn.driams import DRIAMSDatasetExplorer

from models import load_pipeline

from utilities import generate_output_filename
from utilities import load_stratify_split_data


dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')


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

    name = 'explained_classifiers'

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

    X_train, y_train, X_test, y_test, *_ = load_stratify_split_data(
        DRIAMS_ROOT,
        site,
        years,
        species,
        antibiotic,
        seed
    )

    pipeline.fit(X_train, y_train)

    # We have to transform the data set, i.e. use all but the last step
    # of the pipeline. This ensures that Shapley values can *always* be
    # calculated (regardless of pipeline length).
    if len(pipeline) > 1:
        X_train = pipeline[:-1].transform(X_train)
        X_test = pipeline[:-1].transform(X_test)

    if model == 'mlp':

        X_train = shap.kmeans(X_train, 25)

        explainer = shap.KernelExplainer(
            pipeline[-1].predict_proba,
            X_train
        )

        shapley_values = explainer.shap_values(X_test)
    else:
        explainer = shap.Explainer(
            pipeline[-1],
            X_train
        )

        shapley_values = explainer(X_test)

    output = {
        'site': site,
        'years': years,
        'seed': seed,
        'antibiotic': antibiotic,
        'species': species,
        'model': model,
        'best_params': best_params,
        'metadata_versions': metadata_fingerprints,
        'shapley_values': shapley_values,
    }

    output_filename = generate_output_filename(
        args.output,
        output,
        extension='.pkl',
    )

    if not os.path.exists(output_filename) or args.force:
        logging.info(f'Saving {os.path.basename(output_filename)}')

        import pickle
        with open(output_filename, 'wb') as f:
            pickle.dump(output, f)
    else:
        logging.warning(
            f'Skipping {output_filename} because it already exists.'
        )
