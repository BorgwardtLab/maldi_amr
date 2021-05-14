import argparse
import json

import numpy as np
import pandas as pd


def json_to_csv(args):
    for filename in args.INPUT:
        with open(filename) as f:
            data = json.load(f)

        df = pd.DataFrame({k: data[k] for k in ['mean_feature_weights', 'std_feature_weights']})

        rank = [sorted(
           -np.abs(df['mean_feature_weights'])
                       ).index(x) for x in -np.abs(df['mean_feature_weights'])
                ]
        df['rank'] = [r+1 for r in rank]

        df['bin_start'] = np.linspace(2000, 19997, 6000).astype(int).tolist()
        df['bin_end'] = np.linspace(2003, 20000, 6000).astype(int).tolist()

        outname = filename.replace('.json', '.csv')
        df.to_csv(outname, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'INPUT',
        type=str,
        help='Input file',
        nargs='+',
    )
    
    args = parser.parse_args()
    
    json_to_csv(args)
