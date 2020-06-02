"""Plot curves for ensemble experiment.

The purpose of this script is to plot the curves of the ensemble and
mixture experiment. The script is sufficiently smart to collect data
automatically and create the plots according to availability.
"""


import argparse
import collections
import glob
import json
import os

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


def plot_curves(df, outdir, metric='auroc'):
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
    # curve is indexed by a tuple containing its species and its type
    # to indicate whether we are plotting an ensemble or not.
    curves = {}

    for (species, type_), df_ in df.groupby(['species', 'type']):
        curve = df_.groupby(['n_samples']).agg({
            metric: [np.mean, np.std, 'count']
        })

        # This will just reduce the curves that we can draw around the
        # mean, but it will at least make it possible to always render
        # a curve in the end.
        curve = curve.fillna(0)

        curves[(species, type_)] = curve

    sns.set(style='whitegrid')

    fig, ax = plt.subplots(figsize=(9,6))
    fig.suptitle(df.antibiotic.unique()[0].lower())

    palette = sns.color_palette()

    supported_species = [
        'Escherichia coli',
        'Klebsiella pneumoniae',
        'Staphylococcus aureus',
    ]

    species_to_colour = {
        species: palette[i] for i, species in enumerate(supported_species)
    }

    for (species, type_), curve in curves.items():

        # Ignore all species that are not in the list of support species
        # above (the reason for being excluded might be, for example, an
        # insufficient number of samples).
        if species not in supported_species:
            continue

        colour = species_to_colour[species]

        x = curve.index
        mean = upper = curve[metric]['mean']

        upper = curve[metric]['mean'] + curve[metric]['std']
        lower = curve[metric]['mean'] - curve[metric]['std']

        linestyle = 'solid' if type_ == 'ensemble' else 'dashdot'

        mean_n_samples = np.mean(curve[metric]['count'])

        ax.plot(
            x,
            mean,
            c=colour,
            label=species + f' ({type_})',
            linestyle=linestyle
        )

        ax.fill_between(
            x,
            lower,
            upper,
            facecolor=colour,
            alpha=0.25,
            linestyle=linestyle,
        )

    ax.set_ylabel(str(metric).upper())
    ax.set_xlabel('Number of samples')
    ax.legend(loc='lower right')

    plt.savefig(os.path.join(outdir, f'fig3_{df.antibiotic.unique()[0]}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input directory')
    parser.add_argument('--outdir', type=str, 
                        default='.', help='Output directory')

    args = parser.parse_args()

    # Stores data rows corresponding to individual scenarios. Each
    # scenario involves the same antibiotic (used as the key here).
    scenarios = collections.defaultdict(list)

    # Keys to skip when creating a single row in the data dictionary
    # above. This ensures that we only add single pieces of data and
    # have an easier time turning every scenario into a data frame.
    skip_keys = ['years', 'metadata_versions']

    for filename in tqdm(sorted(glob.glob(os.path.join(args.INPUT,
                                          '*.json'))), desc='File'):

        with open(filename) as f:
            data = json.load(f)

        antibiotic = data['antibiotic']

        _add_or_compare(data['metadata_versions'])

        row = {
            key: data[key] for key in data.keys() if key not in skip_keys
        }

        basename = os.path.basename(filename)
        row['type'] = 'ensemble' if 'ensemble' in basename else 'single'

        scenarios[antibiotic].append(row)

    for antibiotic in sorted(scenarios.keys()):

        rows = scenarios[antibiotic]
        df = pd.DataFrame.from_records(rows)

        plot_curves(df, args.outdir)
