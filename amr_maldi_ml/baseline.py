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

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import stratify_by_species_and_label

from utilities import generate_output_filename

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

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
        '-S', '--seed',
        type=int,
        help='Random seed to use for the experiment',
        required=True
    )

    parser.add_argument(
        '-o', '--output',
        default=pathlib.Path(__file__).resolve().parent.parent / 'results',
        type=str,
        help='Output path for storing the results.'
    )

    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='If set, overwrites all files. Else, skips existing files.'
    )

    args = parser.parse_args()

    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    metadata_fingerprints = explorer.metadata_fingerprints(site)

    logging.info(f'Site: {site}')
    logging.info(f'Years: {years}')
    logging.info(f'Seed: {args.seed}')
    logging.info(f'Antibiotic: {args.antibiotic}')

    driams_dataset = load_driams_dataset(
        DRIAMS_ROOT,
        site,
        years,
        '*',
        antibiotics=args.antibiotic,
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
    )

    logging.info(f'Loaded data set for {args.antibiotic}')

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
    X_species = ohe.fit_transform(
        driams_dataset.y['species'].values.reshape(-1, 1)
    )

    logging.info('Created species-only feature vector')

    # Create feature matrix from the binned spectra. We only need to
    # consider the second column of each spectrum for this.
    X_spectra = np.asarray(
                    [spectrum.intensities for spectrum in driams_dataset.X]
                )

    train_index, test_index = stratify_by_species_and_label(
        driams_dataset.y,
        antibiotic=args.antibiotic,
        random_state=args.seed,
    )

    # Labels are shared for both of these experiments, so they only
    # need to be created once.
    y = driams_dataset.to_numpy(args.antibiotic, dtype=float)

    for X, t in zip([X_species, X_spectra], ['no_spectra', '']):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        n_folds = 5

        # Fit the classifier and start calculating some summary metrics.
        # All of this is wrapped in cross-validation based on the grid
        # defined above.
        lr = LogisticRegression(solver='saga', max_iter=500)

        pipeline = Pipeline(
            steps=[
                ('scaler', None),
                ('lr', lr),
            ]
        )

        param_grid = [
            {
                'scaler': ['passthrough', StandardScaler()],
                'lr__C': 10.0 ** np.arange(-3, 4),  # 10^{-3}..10^{3}
                'lr__penalty': ['l1', 'l2'],
            },
            {
                'scaler': ['passthrough', StandardScaler()],
                'lr__penalty': ['none'],
            }
        ]

        grid_search = GridSearchCV(
                        pipeline,
                        param_grid=param_grid,
                        cv=n_folds,
                        scoring='roc_auc',
                        n_jobs=-1,
        )

        logging.info('Starting grid search')

        # Ignore these warnings only for the grid search process. The
        # reason is that some of the jobs will inevitably *fail* to
        # converge because of bad `C` values. We are not interested in
        # them anyway.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            with joblib.parallel_backend('threading', -1):
                grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)
        y_score = grid_search.predict_proba(X_test)

        accuracy = accuracy_score(y_pred, y_test)

        auprc = average_precision_score(
                    y_test, y_score[:, 1],
                    average='weighted'
        )

        auroc = roc_auc_score(y_test, y_score[:, 1], average='weighted')

        # Replace information about the standard scaler prior to writing out
        # the `best_params_` grid. The reason for this is that we cannot and
        # probably do not want to serialise the scaler class. We only need
        # to know *if* a scaler has been employed.
        if 'scaler' in grid_search.best_params_:
            scaler = grid_search.best_params_['scaler']
            if scaler != 'passthrough':
                grid_search.best_params_['scaler'] = type(scaler).__name__

        # Prepare the output dictionary containing all information to
        # reproduce the experiment.

        output = {
            'site': site,
            'seed': args.seed,
            'antibiotic': args.antibiotic,
            'best_params': grid_search.best_params_,
            'years': years,
            'y_score': y_score.tolist(),
            'y_pred': y_pred.tolist(),
            'y_test': y_test.tolist(),
            'accuracy': accuracy,
            'auprc': auprc,
            'auroc': auroc,
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
            logging.info(f'Saving {os.path.basename(output_filename)}')

            with open(output_filename, 'w') as f:
                json.dump(output, f, indent=4)
        else:
            logging.warning(
                f'Skipping {output_filename} because it already exists.'
            )
