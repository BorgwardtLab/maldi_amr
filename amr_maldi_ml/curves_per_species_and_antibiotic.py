"""Calculate performance curves for species--antitbiotic combinations."""

import argparse
import dotenv
import json
import pathlib
import os
import warnings

import numpy as np

from utilities import generate_output_filename

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from maldi_learn.vectorization import BinningVectorizer
from maldi_learn.utilities import stratify_by_species_and_label

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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

    site = 'DRIAMS-A'
    years = ['2015', '2017']
    _seeds = [123, 321]

    # Create the output directory for storing all results of the
    # individual combinations.
    os.makedirs(args.output, exist_ok=True)

    # Create input grid for the subsequent experiments. Not all
    # combinations are useful, hence we specify them in a list.
    input_grid = ParameterGrid([
        {
            'species': ['Escherichia coli'],
            'antibiotic': ['Ciprofloxacin'],
            'seed': _seeds,
        },
        {
            'species': ['Staphylococcus aureus'],
            'antibiotic': ['Ciprofloxacin',
                           'Ceftriaxone',
                           'Amoxicillin-Clavulanic acid'],
            'seed': _seeds,
        }
    ])

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

    for combination in input_grid:
        driams_dataset = load_driams_dataset(
                explorer.root,
                site,
                years,
                combination['species'],
                combination['antibiotic'],
                encoder=DRIAMSLabelEncoder(),
                handle_missing_resistance_measurements='remove_if_all_missing',
                nrows=400,
        )

        # Bin spectra
        bv = BinningVectorizer(100, min_bin=2000, max_bin=20000)
        X = bv.fit_transform(driams_dataset.X)

        # Stratified train--test split
        train_index, test_index = stratify_by_species_and_label(
            driams_dataset.y,
            antibiotic=combination['antibiotic'],
            random_state=combination['seed'],
        )

        # Create labels
        y = driams_dataset.to_numpy(combination['antibiotic'])

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        param_grid = [
            {
                'C': 10.0 ** np.arange(-3, 4),  # 10^{-3}..10^{3}
                'penalty': ['l1', 'l2'],
            },
            {
                'penalty': ['none'],
            }
        ]

        n_folds = 5

        # Fit the classifier and start calculating some summary metrics.
        # All of this is wrapped in cross-validation based on the grid
        # defined above.
        lr = LogisticRegression(solver='saga', max_iter=500)
        grid_search = GridSearchCV(
                        lr,
                        param_grid=param_grid,
                        cv=n_folds,
                        scoring='roc_auc'
        )

        # Ignore these warnings only for the grid search process. The
        # reason is that some of the jobs will inevitably *fail* to
        # converge because of bad `C` values. We are not interested in
        # them anyway.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

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
            'seed': combination['seed'],
            'antibiotic': combination['antibiotic'],
            'species': combination['species'],
            'years': years,
            'y_score': y_score.tolist(),
            'y_pred': y_pred.tolist(),
            'y_test': y_test.tolist(),
            'accuracy': accuracy,
            'aupprc': auprc,
            'auroc': auroc,
        }

        output_filename = generate_output_filename(
            args.output,
            output
        )

        # Only write if we either are running in `force` mode, or the
        # file does not yet exist.
        if not os.path.exists(output_filename) or args.force:
            with open(output_filename, 'w') as f:
                json.dump(output, f)
