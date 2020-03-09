"""Calculate performance curves for species--antitbiotic combinations."""

import argparse
import dotenv
import joblib
import json
import logging
import pathlib
import os
import warnings

import numpy as np

from utilities import generate_output_filename

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import stratify_by_species_and_label

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
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
    species,
    antibiotic,
    seed,
    output_path,
    force,
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

    logging.info(len(train_index))
    logging.info(len(test_index))

    logging.info('Finished stratification')

    # Create labels
    y = driams_dataset.to_numpy(antibiotic)

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
                    n_jobs=n_jobs,
    )

    # Ignore these warnings only for the grid search process. The
    # reason is that some of the jobs will inevitably *fail* to
    # converge because of bad `C` values. We are not interested in
    # them anyway.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        with joblib.parallel_backend('threading', n_jobs):
            grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    y_score = grid_search.predict_proba(X_test)

    accuracy = accuracy_score(y_pred, y_test)

    auprc = average_precision_score(
                y_test, y_score[:, 1],
                average='weighted'
    )

    auroc = roc_auc_score(y_test, y_score[:, 1], average='weighted')

    # Prepare the output dictionary containing all information to
    # reproduce the experiment.

    output = {
        'site': site,
        'seed': seed,
        'antibiotic': antibiotic,
        'species': species,
        'best_params': grid_search.best_params_,
        'years': years,
        'y_score': y_score.tolist(),
        'y_pred': y_pred.tolist(),
        'y_test': y_test.tolist(),
        'accuracy': accuracy,
        'auprc': auprc,
        'auroc': auroc,
    }

    output_filename = generate_output_filename(
        output_path,
        output
    )

    # Only write if we either are running in `force` mode, or the
    # file does not yet exist.
    if not os.path.exists(output_filename) or force:
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
        '-A', '--all',
        action='store_true',
        help='Calculate curves for all pre-defined combinations'
    )

    parser.add_argument(
        '-a', '--antibiotic',
        type=str,
        help='Antibiotic for which to run the experiment'
    )

    parser.add_argument(
        '-s', '--species',
        type=str,
        help='Species for which to run the experiment'
    )

    parser.add_argument(
        '-S', '--seed',
        type=int,
        help='Random seed to use for the experiment'
    )

    name = 'fig4_curves_per_species_and_antibiotics'

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

    # Only required in case the pre-defined grid is being used.
    seeds = [344, 172, 188, 270, 35, 164, 545, 480, 89, 409]

    # Create the output directory for storing all results of the
    # individual combinations.
    os.makedirs(args.output, exist_ok=True)

    logging.info(f'Site: {site}')
    logging.info(f'Years: {years}')

    if args.seed:
        logging.info(f'Seed: {args.seed}')
    else:
        logging.info(f'Seeds: {seeds}')

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
                species,
                antibiotic,
                seed,
                args.output,
                args.force,
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
            args.species,
            args.antibiotic,
            args.seed,
            args.output,
            args.force,
            n_jobs
        )
