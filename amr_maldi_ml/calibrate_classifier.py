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

from maldi_learn.utilities import stratify_by_species_and_label

from models import calculate_metrics
from models import load_pipeline

from utilities import generate_output_filename

from sklearn.calibration import CalibratedClassifierCV

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
    X_test, y_test = X[test_index], y[test_index]

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
        cv='prefit',
        method='isotonic'  # We assume that sufficient data are available
    )

    # This looks wrong but it is correct: the calibration happens on the
    # *test* data (and the calibrated values are also reported on it).
    cccv.fit(X_test, y_test)
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
