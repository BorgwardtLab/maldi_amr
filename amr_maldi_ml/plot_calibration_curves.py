"""Plot calibration curves.

The purpose of this script is to plot the calibration curves of
different classifiers to show to what extent calibration can be
used to improve classification results. The script is smart and
will collect data automatically so that plots are created based
on their availability.
"""


import argparse
import collections
import glob
import json
import os

from sklearn.calibration import calibration_curve

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


# Global metadata information; this will be updated by the script to
# ensure that we are working with data files from the *same* sources
# in order to create curves.
metadata_versions = {}


def _add_or_compare(metadata):
    if not metadata_versions:
        metadata_versions.update(metadata)
    else:
        # Smarter way to compare the entries of the dictionaries with
        # each other.
        assert metadata_versions == metadata


def plot_curves(df, output, metric='auroc'):
    """Plot curves that are contained in a data frame.

    This function is performing the main work for a single data frame.
    It will automatically collect the relevant curves and prepare a plot
    according to the data.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing individual measurements for each species,
        site, type, and seed. The following columns need to exist:

            - 'site'
            - 'species'
            - 'type'
            - 'n_samples'
            - 'seed'

        All other columns are optional; they are taken to correspond to
        performance metrics such as AUROC.

    metric : str, optional
        Specifies a column in the data frame that is used as
        a performance metric.

    Returns
    -------
    Nothing. As a side-effect of calling this function, plots will be
    generated.
    """
    # Will store the individual curves for subsequent plotting. Every
    # curve is indexed by a tuple of its species, an antibiotic, plus
    # a classifier.
    curves = {}

    for (antibiotic, species, model), df_ in df.groupby(['antibiotic',
                                                         'species',
                                                         'model']):

        y_test = np.vstack(df_['y_test']).ravel()
        y_score = np.vstack(df_['y_score'])

        minority_class = np.argmin(np.bincount(y_test))

        curve = calibration_curve(y_test, y_score[:, minority_class])
        curves[(antibiotic, species, model)] = curve

    sns.set(style='whitegrid')

    fig, ax = plt.subplots()
    fig.suptitle(df.antibiotic.unique()[0])

    palette = sns.color_palette()

    supported_species = [
        'Escherichia coli',
        'Klebsiella pneumoniae',
        'Staphylococcus aureus',
    ]

    species_to_colour = {
        species: palette[i] for i, species in enumerate(supported_species)
    }

    for (antibiotic, species, model), curve in curves.items():

        colour = species_to_colour[species]

        ax.plot(
            curve[0],
            curve[1],
            c=colour,
            label=species + ' ' + model,
        )

    ax.set_ylabel(str(metric).upper())
    ax.set_xlabel('Number of samples')
    ax.legend(loc='lower right')

    #plt.savefig(os.path.join(outdir, f'fig3_{df.antibiotic.unique()[0]}.png'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input directory')
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output directory'
    )

    args = parser.parse_args()

    # Stores data rows corresponding to individual scenarios. Each
    # scenario involves the same antibiotic (used as the key here).
    scenarios = collections.defaultdict(list)

    # Keys to skip when creating a single row in the data dictionary
    # above. This ensures that we only add single pieces of data and
    # have an easier time turning every scenario into a data frame.
    skip_keys = ['years', 'metadata_versions']

    files_to_load = []
    for root, dirs, files in os.walk(args.INPUT):
        files_to_load.extend([
            os.path.join(root, fn) for fn in files if
            os.path.splitext(fn)[1] == '.json'
        ])

    for filename in tqdm(files_to_load, desc='File'):

        with open(filename) as f:
            data = json.load(f)

        antibiotic = data['antibiotic']
        species = data['species']

        _add_or_compare(data['metadata_versions'])

        row = {
            key: data[key] for key in data.keys() if key not in skip_keys
        }

        basename = os.path.basename(filename)

        scenarios[(antibiotic, species)].append(row)

    for antibiotic in sorted(scenarios.keys()):

        rows = scenarios[antibiotic]
        df = pd.DataFrame.from_records(rows)

        plot_curves(df, args.output)
