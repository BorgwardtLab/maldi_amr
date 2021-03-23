"""Perform predictions based on metadata instead of spectra."""

import argparse

import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE')

    args = parser.parse_args()

    df = pd.read_csv(args.FILE, sep=';')

    metadata_columns = df.columns[np.r_[:21, 102:108]]
    df_metadata = df[metadata_columns]
    df_resistance = df.drop(columns=metadata_columns)
