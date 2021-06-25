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

from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from utils import scenario_map

# Global metadata information; this will be updated by the script to
# ensure that we are working with data files from the *same* sources
# in order to create curves.
metadata_versions = {}

map_models = {
    'lightgbm': 'LightGBM',
    'mlp': 'MLP',
}

def _add_or_compare(metadata):
    if not metadata_versions:
        metadata_versions.update(metadata)
    else:
        # Smarter way to compare the entries of the dictionaries with
        # each other.
        assert metadata_versions == metadata


def interpolate_at(df, x):
    """Interpolate a data frame at certain positions.

    This is an auxiliary function for interpolating an indexed data
    frame at a certain position or at certain positions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame; must have index that is compatible with `x`.

    x : scalar or iterable
        Index value(s) to interpolate the data frame at. Must be
        compatible with the data type of the index.

    Returns
    -------
    Data frame evaluated at the specified index positions.
    """
    # Check whether object support iteration. If yes, we can build
    # a sequence index; if not, we have to convert the object into
    # something iterable.
    try:
        _ = (a for a in x)
        new_index = pd.Index(x)
    except TypeError:
        new_index = pd.Index([x])

    # Ensures that the data frame is sorted correctly based on its
    # index. We use `mergesort` in order to ensure stability. This
    # set of options will be reused later on.
    sort_options = {
        'ascending': False,
        'kind': 'mergesort',
    }
    df = df.sort_index(**sort_options)

    # TODO: have to decide whether to keep first index reaching the
    # desired level or last. The last has the advantage that it's a
    # more 'pessimistic' estimate since it will correspond to lower
    # thresholds.
    df = df[~df.index.duplicated(keep='last')]

    # Include the new index, sort again and then finally interpolate the
    # values.
    df = df.reindex(df.index.append(new_index).unique())
    df = df.sort_index(**sort_options)
    df = df.interpolate()

    return df.loc[new_index]


def plot_curves(df, outdir):
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

    Returns
    -------
    Nothing. As a side-effect of calling this function, plots will be
    generated.
    """

    metrics = ['auroc', 'auprc']

    # Will store the individual curves for subsequent plotting. Every
    # curve is indexed by a tuple containing its species and its type
    # to indicate whether we are plotting an ensemble or not.
    curves = {}

    for (
     species, type_, model, ab,
         ), df_ in df.groupby([
            'species', 'type', 'model', 'antibiotic'
                                ]):

        # Find first set of `n_samples` values that we will use to
        # quantise the remainder of the curves.
        stop = (np.argmin(np.diff(df_['n_samples']) > 0))
        values = df_['n_samples'].values[:stop + 1]

        # Will store the updated curves. This does not work inline, at
        # least not trivially, because of the `groupby` operation.
        all_curves = []

        for seed, df_curve in df_.groupby('seed'):

            # Get number columns and use `n_samples` to make the
            # interpolation work later on.
            number_columns = df_curve.select_dtypes(np.number).columns
            numbers = df_curve[number_columns].set_index('n_samples')
            numbers = interpolate_at(numbers, values)
            numbers = numbers.fillna(method='ffill')

            # Need to ensure that we can actually assign all columns
            # properly.
            numbers = numbers.reset_index()
            numbers = numbers.rename(columns={'index': 'n_samples'})

            # Set proper index again so that we can assign the values
            # below.
            numbers = numbers.set_index(df_curve.index)

            df_curve[number_columns] = numbers[number_columns]
            all_curves.append(df_curve)

        all_curves = pd.concat(all_curves)

        for metric in metrics:
            curve = all_curves.groupby(['n_samples']).agg({
                metric: [np.mean, np.std, 'count']
            })

            # This will just reduce the curves that we can draw around the
            # mean, but it will at least make it possible to always render
            # a curve in the end.
            curve = curve.fillna(0)

            curves[(species, type_, model, ab, metric)] = curve

    sns.set(style='whitegrid')

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    palette = sns.color_palette()

    supported_species = [
        'Escherichia coli',
        'Klebsiella pneumoniae',
        'Staphylococcus aureus',
    ]

    species_to_colour = {
        species: palette[i] for i, species in enumerate(supported_species)
    }

    for i, metric in enumerate(metrics):
        for (species, type_, model, ab, metric_), curve in curves.items():

            # Ignore all species that are not in the list of support species
            # above (the reason for being excluded might be, for example, an
            # insufficient number of samples).
            if species not in supported_species:
                continue
            
            # Skip curve if wrong metric.
            if metric_ != metric:
                continue

            colour = species_to_colour[species]

            x = curve.index
            mean = curve[metric]['mean']

            upper = curve[metric]['mean'] + curve[metric]['std']
            lower = curve[metric]['mean'] - curve[metric]['std']

            linestyle = 'solid' if type_ == 'ensemble' else 'dashdot'

            ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            ax[i].plot(
                x,
                mean,
                c=colour,
                label=scenario_map[species.replace(' ','_')] + f' ({map_models[model]}) ({type_})',
                linestyle=linestyle
            )

            ax[i].fill_between(
                x,
                lower,
                upper,
                facecolor=colour,
                alpha=0.25,
                linestyle=linestyle,
            )

        ax[i].set_ylabel(str(metric).upper())
        ax[i].set_xlabel('Number of samples')
        ax[i].set_xlim(-1, 20001)
        if metric == 'auprc':
            ax[i].legend(loc='upper right')
        else:
            legend = ax[i].legend()
            legend.remove()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'combined.png'))


if __name__ == '__main__':
    # Stores data rows corresponding to individual scenarios. 
    rows = []

    # Keys to skip when creating a single row in the data dictionary
    # above. This ensures that we only add single pieces of data and
    # have an easier time turning every scenario into a data frame.
    skip_keys = ['years', 'metadata_versions']

    scenarios = [
        ('Escherichia coli', 'Ceftriaxone', 'lightgbm'),
        ('Klebsiella pneumoniae', 'Ceftriaxone', 'mlp'),
        ('Staphylococcus aureus', 'Oxacillin', 'lightgbm'),
    ]

    input_dir = '../../results/ensemble'

    input_files = glob.glob(
        os.path.join(input_dir, '**/*.json'), recursive=True
    )

    for filename in tqdm(sorted(input_files), desc='File'):

        with open(filename) as f:
            data = json.load(f)

        # Ignore input files that are not part of the major scenarios.
        if (data['species'], data['antibiotic'], data['model']) not in \
                scenarios:
            continue

        antibiotic = data['antibiotic']

        #_add_or_compare(data['metadata_versions'])

        row = {
            key: data[key] for key in data.keys() if key not in skip_keys
        }

        basename = os.path.basename(filename)
        row['type'] = 'ensemble' if 'ensemble' in basename else 'single'

        rows.append(row)

    df = pd.DataFrame.from_records(rows)
    plot_curves(df, '../plots/ensemble')
