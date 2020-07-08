#!/usr/bin/env python3
#
# Calculate mean over different spectra. This can be used to quickly
# 'bin' multiple spectra.

import argparse
import json
import os

from maldi_learn.driams import load_spectrum
from maldi_learn.vectorization import BinningVectorizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'INPUT',
        type=str,
        nargs='+',
        help='Input file(s)'
    )

    parser.add_argument(
        '-b', '--bins',
        type=int,
        default=6000,
        help='Number of bins'
    )

    args = parser.parse_args()

    X = [
        load_spectrum(f) for f in args.INPUT
    ]

    bv = BinningVectorizer(
        args.bins,
        min_bin=2000,
        max_bin=20000,
        n_jobs=-1  # Use all available cores to perform the processing
    )

    X = bv.fit_transform(X)

    # Prepares output dictionary. Filenames are the key, and binned
    # spectra are the values.
    output = {}

    for spectrum, filename in zip(X, args.INPUT):
        name = os.path.basename(filename)
        name = os.path.splitext(name)[0]

        output[name] = spectrum[:, 1].tolist()

    print(json.dumps(output, indent=4))
