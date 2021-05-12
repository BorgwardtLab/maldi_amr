"""Calibrate classifier for specific scenarios."""

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

from models import calculate_metrics
from models import load_pipeline

from utilities import generate_output_filename
from utilities import load_stratify_split_data

from sklearn.calibration import CalibratedClassifierCV

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

    name = 'calibrated_classifiers'

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

    train_metrics = calculate_metrics(
        y_train,
        pipeline.predict(X_train),
        pipeline.predict_proba(X_train),
        prefix='train',
    )

    # Store information *prior* to calibration. We require the actual
    # labels, as well as the scores, to calculate a calibration curve
    # later on (from the output of this experiment). As a convenience
    # to users, we also store the actual predictions.
    y_test = y_test.tolist()
    y_pred = pipeline.predict(X_test).tolist()
    y_score = pipeline.predict_proba(X_test)

    test_metrics = calculate_metrics(
        y_test,
        y_pred,
        y_score,
        prefix='test',
    )

    cccv = CalibratedClassifierCV(
        pipeline,
        cv=5,              # This is the default anyway
        method='sigmoid' 
    )

    cccv.fit(X_train, y_train)
    y_pred_calibrated = cccv.predict(X_test).tolist()
    y_score_calibrated = cccv.predict_proba(X_test)

    test_metrics_calibrated = calculate_metrics(
        y_test,
        y_pred_calibrated,
        y_score_calibrated,
        prefix='test_calibrated'
    )

    output = {
        'site': site,
        'years': years,
        'seed': seed,
        'antibiotic': antibiotic,
        'species': species,
        'model': model,
        'best_params': best_params,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_score': y_score.tolist(),
        'y_pred_calibrated': y_pred_calibrated,
        'y_score_calibrated': y_score_calibrated.tolist(),
        'metadata_versions': metadata_fingerprints,
    }

    output.update(train_metrics)
    output.update(test_metrics)
    output.update(test_metrics_calibrated)

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
