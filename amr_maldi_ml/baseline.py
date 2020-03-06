"""Measure baseline performance.

The purpose of this script is to measure the baseline performance, i.e.
the performance that we obtain by looking at:

    1. *All* species and their respective spectra; accumulating their
       classification performance

    2. *Only* the species information without considering any spectra
"""

import dotenv
import joblib
import logging
import os
import json
import pathlib
import warnings

import numpy as np

from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

#from maldi_learn.utilities import stratify_by_species_and_label
from maldi_learn.vectorization import BinningVectorizer

from utilities import generate_output_filename

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder


dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These parameters should remain fixed for this particular
# experiment. We always train on the same data set, using
# *all* available years.
site = 'DRIAMS-A'
years = ['2015']                 # TODO: more years: '2016', '2017', '2018'
antibiotics = ['Ciprofloxacin']  # TODO: more antibiotics

# TODO: make configurable
seed = 42
output_path = pathlib.Path(__file__).resolve().parent.parent / 'results'
force = False


def stratify_by_species_and_label(
    y,
    antibiotic,
    test_size=0.2,
    random_state=123
):
    """Stratification by species and antibiotic label.

    This function performs a stratified train--test split, taking into
    account species *and* label information.

    Parameters
    ----------
    y : pandas.DataFrame
        Label data frame containing information about the species, the
        antibiotics, and other (optional) information, which is ignored
        by this function.

    antibiotic : str
        Specifies the antibiotic for the stratification. This must be
        a valid column in `y`.

    test_size: float
        Specifies the size of the test data set returned by the split.
        This function cannot guarantee that a specific test size will
        lead to a valid split. In this case, it will fail.

    random_state:
        Specifies the random state to use for the split.

    Returns
    -------
    Tuple of train and test indices.
    """
    n_samples = y.shape[0]

    from sklearn.preprocessing import LabelEncoder 

    # Encode species information to simplify the stratification. Every
    # combination of antibiotic and species will be encoded as an
    # individual class.
    le = LabelEncoder()

    species_transform = le.fit_transform(y.species)
    labels = y[antibiotic].values

    # Creates the *combined* label required for the stratification.
    stratify = np.vstack((species_transform, labels)).T
    stratify = stratify.astype('int')

    _, indices, counts = np.unique(
        stratify,
        axis=0,
        return_index=True,
        return_counts=True
    )

    # Get indices of all elements that appear an insufficient number of
    # times to be used in the stratification.
    invalid_indices = indices[counts < 2]

    # Replace all of them by a 'fake' class whose numbers are guaranteed
    # *not* to occur in the data set (because labels are encoded from 0,
    # and the binary label is either 0 or 1).
    stratify[invalid_indices, :] = [-1, -1]

    from sklearn.model_selection import train_test_split

    train_index, test_index = train_test_split(
        range(n_samples),
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    train_index = np.asarray(train_index)
    train_index = train_index[
                    np.isin(train_index,
                            invalid_indices,
                            assume_unique=True,
                            invert=True)
                ]

    test_index = np.asarray(test_index)
    test_index = test_index[
                    np.isin(test_index,
                            invalid_indices,
                            assume_unique=True,
                            invert=True)
               ]

    return train_index, test_index


if __name__ == '__main__':

    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    driams_dataset = load_driams_dataset(
        DRIAMS_ROOT,
        site,
        years,
        '*',
        antibiotics=antibiotics,
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
    )

    logging.info('Loaded data set')

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

    bv = BinningVectorizer(
            100,
            min_bin=2000,
            max_bin=20000,
            n_jobs=-1,
        )

    # TODO: should load pre-processed data here
    X_spectra = bv.fit_transform(driams_dataset.X)

    for antibiotic in antibiotics:
        logging.info(f'Performing experiment for {antibiotic}')

        train_index, test_index = stratify_by_species_and_label(
            driams_dataset.y,
            antibiotic=antibiotic,
            random_state=seed,
        )

        # Labels are shared for both of these experiments, so they only
        # need to be created once.
        y = driams_dataset.to_numpy(antibiotic)

        for X, t in zip([X_species, X_spectra], ['no_spectra', '']):
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
                            scoring='roc_auc',
                            n_jobs=-1,
            )

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

            # Prepare the output dictionary containing all information to
            # reproduce the experiment.

            output = {
                'site': site,
                'seed': seed,
                'antibiotic': antibiotic,
                'best_params': grid_search.best_params_,
                'years': years,
                'y_score': y_score.tolist(),
                'y_pred': y_pred.tolist(),
                'y_test': y_test.tolist(),
                'accuracy': accuracy,
                'auprc': auprc,
                'auroc': auroc,
            }

            print(t)
            print(output_path)
            output_filename = generate_output_filename(
                output_path,
                output,
                suffix=t
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
