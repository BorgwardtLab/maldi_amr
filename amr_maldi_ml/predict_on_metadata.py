"""Perform predictions based on metadata instead of spectra."""

import argparse

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

from maldi_learn.driams import DRIAMSLabelEncoder


if __name__ == '__main__':
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

    args = parser.parse_args()

    df = pd.read_csv(
        args.FILE,
        sep=';',
        na_values='-',
        keep_default_na=True,
        low_memory=False
    )
    df = df.query('`Organism.best.match.` == @args.species')

    metadata_columns = df.columns[np.r_[:21, 102:108]]
    df_metadata = df[metadata_columns]
    df_resistance = df.drop(columns=metadata_columns)

    encoder = DRIAMSLabelEncoder()

    # Only use a specific antibiotic for the prediction task.
    df_resistance = encoder.fit_transform(df_resistance)
    df_resistance = df_resistance[args.antibiotic]

    # Remove all values of the prediction data frame that contain NaNs.
    df_metadata = df_metadata[~df_resistance.isna()]
    df_resistance = df_resistance[~df_resistance.isna()]

    # Need to drop NaN before converting to `numpy` data frame. We could
    # probably be smarter here.
    df_metadata = df_metadata.dropna(axis='columns')

    X = pd.get_dummies(df_metadata).to_numpy()
    y = df_resistance.values.astype(int)

    clf = LogisticRegressionCV(cv=5)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    y_score = clf.predict_proba(X)

    print(f'Accuracy: {accuracy_score(y, y_pred):.2f}')
    print(f'AUPRC: {average_precision_score(y, y_score[:, 1]):.2f}')
    print(classification_report(y, y_pred, zero_division=0))
