"""Calculate performance curves for species--antitbiotic combinations."""

import argparse
import dotenv
import os

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from maldi_learn.vectorization import BinningVectorizer
from maldi_learn.utilities import stratify_by_species_and_label

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

    # Set of default parameters; should be made adjustable for running
    # the comparison at larger scales.
    site = 'DRIAMS-A'
    years = ['2015', '2017']
    species = 'Staphylococcus aureus'
    antibiotics = ['Ciprofloxacin', 'Penicillin.ohne.Meningitis']

    driams_dataset = load_driams_dataset(
                explorer.root,
                site,
                years,
                species,
                antibiotics,
                encoder=DRIAMSLabelEncoder(),
                handle_missing_resistance_measurements='remove_if_all_missing',
    )

    # Bin spectra
    bv = BinningVectorizer(100, min_bin=2000, max_bin=20000)
    X = bv.fit_transform(driams_dataset.X)

    # Stratified train--test split
    index_train, index_test = stratify_by_species_and_label(
        driams_dataset.y,
        antibiotic='Ciprofloxacin'  # TODO: support more than one antibiotic
    )

    # Create labels
    y = driams_dataset.to_numpy('Ciprofloxacin')

    # Fit the classifier
    lr = LogisticRegression()
    lr.fit(X[index_train], y[index_train])
    y_pred = lr.predict(X[index_test])

    print(accuracy_score(y_pred, y[index_test]))
