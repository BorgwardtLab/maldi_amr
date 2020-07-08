#!/usr/bin/env python3
#
# Converts a JSON file to an Excel file. This script assumes that each
# key in the file corresponds to a single column.

import argparse
import json

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('FILE', type=str, help='Input file')
    parser.add_argument('OUTPUT', type=str, help='Output file')
    parser.add_argument(
        '-k', '--key',
        type=str,
        help='Key to use for restricting the data frame. If not set, uses all '
             'available keys.'
    )

    args = parser.parse_args()

    with open(args.FILE) as f:
        data = json.load(f)

    if args.key is not None:
        data = data[args.key]

    df = pd.DataFrame.from_dict(data)
    df.to_excel(args.OUTPUT, index=False)
