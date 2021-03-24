"""Perform predictions based on metadata instead of spectra."""

import argparse

import numpy as np
import pandas as pd

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

    df = pd.read_csv(args.FILE, sep=';')
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
