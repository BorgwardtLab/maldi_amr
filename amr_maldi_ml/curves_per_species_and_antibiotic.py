"""Calculate performance curves for species--antitbiotic combinations."""

import argparse
import dotenv
import json
import os

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from maldi_learn.vectorization import BinningVectorizer
from maldi_learn.utilities import stratify_by_species_and_label

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-A', '--all',
        action='store_true',
        help='If specified, use *all* available antibiotics and species.'
    )

    args = parser.parse_args()
    
    site = 'DRIAMS-A'
    years = ['2015', '2017']
    _seeds = [123, 321]

    #  create param grid for experiments
    grid = ParameterGrid([
        {'species': ['Escherichia coli'],
        'antibiotics': ['Ciprofloxacin'],
        'seed': _seeds,
        },
        {
        'species': ['Staphylococcus aureus'],
        'antibiotics': ['Ciprofloxacin', 
                        'Ceftriaxone', 
                        'Amoxicillin-Clavulanic acid'],
        'seed': _seeds,
        }
    ])

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

    # Set of default parameters; should be made adjustable for running
    # the comparison at larger scales.

    for combination in grid:
        print(combination)

        driams_dataset = load_driams_dataset(
                    explorer.root,
                    site,
                    years,
                    combination['species'],
                    combination['antibiotics'],
                    encoder=DRIAMSLabelEncoder(),
                    handle_missing_resistance_measurements='remove_if_all_missing',
                    nrows=2000,
        )

        # Bin spectra
        bv = BinningVectorizer(100, min_bin=2000, max_bin=20000)
        X = bv.fit_transform(driams_dataset.X)

        # Stratified train--test split
        train_index, test_index = stratify_by_species_and_label(
            driams_dataset.y,
            antibiotic=combination['antibiotics'],  # TODO: support more than one antibiotic
            random_state=combination['seed'],
        )

        # Create labels
        y = driams_dataset.to_numpy(combination['antibiotics'])

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # Fit the classifier and start calculating some summary metrics
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_score = lr.predict_proba(X_test)

        accuracy = accuracy_score(y_pred, y_test)
        auprc = average_precision_score(y_test, y_score[:, 1], average='weighted')
        auroc = roc_auc_score(y_test, y_score[:, 1], average='weighted')

        # Prepare the output dictionary containing all information to
        # reproduce the experiment.

        output = {
            'site': site,
            'antibiotics': combination['antibiotics'],
            'species': combination['species'],
            'years': years,
            'y_score': y_score.tolist(),
            'y_pred': y_pred.tolist(),
            'y_test': y_test.tolist(),
            'accuracy': accuracy,
            'aupprc': auprc,
            'auroc': auroc,
        }

        print(json.dumps(
            output,
            indent=4
        ))

    # TODO: generate filename for input arguments
