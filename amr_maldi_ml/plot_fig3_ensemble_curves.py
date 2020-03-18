"""Plot curves for ensemble experiment.

The purpose of this script is to plot the curves of the ensemble and
mixture experiment. The script is sufficiently smart to collect data
automatically and create the plots according to availability.
"""


import argparse
import glob
import json
import os

from tqdm import tqdm


# Global metadata information; this will be updated by the script to
# ensure that we are working with data files from the *same* sources
# in order to create curves.
metadata_versions = {}


def _add_or_compare(metadata):
    if not metadata_versions:
        metadata_versions.update(metadata)
    else:
        for key, value in metadata.items():
            assert metadata_versions[key] == value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input directory')

    args = parser.parse_args()

    # Stores data rows corresponding to individual scenarios. Each
    # scenario involves the same antibiotic (used as the key here).
    scenarios = {}

    for filename in tqdm(sorted(glob.glob(os.path.join(args.INPUT,
                                          '*.json'))), desc='File'):

        with open(filename) as f:
            data = json.load(f)

        antibiotic = data['antibiotic']

        _add_or_compare(data['metadata_versions'])
