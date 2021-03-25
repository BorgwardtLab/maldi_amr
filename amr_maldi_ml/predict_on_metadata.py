"""Perform predictions based on metadata instead of spectra."""

import argparse
import logging

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

from maldi_learn.driams import DRIAMSLabelEncoder


def make_dataframes(filename, args):
    """Create data frames from file, based on mode."""
    df = pd.read_csv(
        args.FILE,
        sep=';',
        na_values='-',
        keep_default_na=True,
        # Required to ensure that different data types can be inferred
        # for the columns.
        low_memory=False
    )

    for col in ['Score1', 'Score2']:
        df[col] = df[col].apply(lambda x: np.isscalar(x))

    # Get only the species we are interested in.
    df = df.query('`Organism.best.match.` == @args.species')

    # Retrieve all metadata columns. Since the file format differs
    # between sites, we cannot rely on the ordering. However, each
    # site has the same 21 columns, and the last 6 columns store a
    # bunch of other data.
    n_columns = len(df.columns)
    metadata_columns = df.columns[np.r_[:21, n_columns-6:n_columns]]

    # Split the data set into two parts: one containing metadata
    # columns, the other one only containing resistance info. We
    # do a quick sanity check here.
    df_metadata = df[metadata_columns]
    df_resistance = df.drop(columns=metadata_columns)

    # Get all potential resistance values, but ignore all NaNs during
    # the set construction.
    resistance_values = set(df_resistance.values.ravel())
    resistance_values = set(filter(lambda x: x == x , resistance_values))

    assert 'I' in resistance_values
    assert 'R' in resistance_values

    encoder = DRIAMSLabelEncoder()

    # Only use a specific antibiotic for the prediction task. This could
    # also be achieved in the query above.
    df_resistance = encoder.fit_transform(df_resistance)
    df_resistance = df_resistance[args.antibiotic]
    
    # Remove all values of the prediction data frame that contain NaNs
    # because we cannot assign them a proper label.
    df_metadata = df_metadata[~df_resistance.isna()]
    df_resistance = df_resistance[~df_resistance.isna()]

    # The labels should be integers such that categorical predictions
    # work as expected. We lose the ability to represent NaNs, but we
    # can live with this because we removed them anyway.
    y = df_resistance.values.astype(int)

    # TODO: we remove all NaN values from the metadata because it is
    # unclear how to convert them.
    #
    # Need to drop NaN before converting to `numpy` data frame. We could
    # probably be smarter here.
    df_metadata = df_metadata.dropna(axis='columns')

    #df_metadata = df_metadata.drop(
    #    columns=[
    #        c for c in df_metadata.columns if c.endswith('_id')
    #    ]
    #)

    if args.mode == 'numerical':
        logging.info('Only including numerical columns:')
        logging.info(
            f'{df_metadata.select_dtypes(include=[np.number]).columns}'
        )

        X = df_metadata.select_dtypes(include=[np.number])
    elif args.mode == 'categorical':
        logging.info('Only including categorical columns:')
        logging.info(
            f'{df_metadata.select_dtypes(include=[np.object]).columns}'
        )

        X = df_metadata.select_dtypes(include=[np.object])
        X = pd.get_dummies(X).to_numpy()
    elif args.mode == 'all':
        logging.info('Including all column types:')
        logging.info(f'{df_metadata.columns}')

        X = pd.get_dummies(df_metadata).to_numpy()

    return X, y


if __name__ == '__main__':
    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('FILE')
    parser.add_argument(
        '-a', '--antibiotic',
        type=str,
        default='Ceftriaxon'
    )
    parser.add_argument(
        '-s', '--species',
        type=str,
        default='Escherichia coli'
    )
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['numerical', 'categorical', 'all'],
        help='Choice of variable pre-selection',
        default='numerical'
    )

    args = parser.parse_args()

    X, y = make_dataframes(args.FILE, args)

    clf = LogisticRegressionCV(
        cv=5,
        scoring='average_precision',
        class_weight='balanced'
    )
    clf.fit(X, y)

    y_pred = clf.predict(X)
    y_score = clf.predict_proba(X)

    print(f'Accuracy: {accuracy_score(y, y_pred):.2f}')
    print(f'AUPRC: {average_precision_score(y, y_score[:, 1]):.2f}')
    print(classification_report(y, y_pred, zero_division=0))

    result = permutation_importance(
        clf,
        X, y,
        n_repeats=5,
        random_state=42,
        scoring='average_precision'
    )
    print(result.importances_mean)
