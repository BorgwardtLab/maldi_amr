"""Visualisation of Shapley values."""

import argparse
import os
import pickle
import shap

import numpy as np

import matplotlib.pyplot as plt


def pool(shap_values):
    """Pool `shap.Explanation` objects."""
    values = np.vstack([v.values for v in shap_values])
    base_values = np.hstack([v.base_values for v in shap_values])
    data = np.vstack([v.data for v in shap_values])

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=data
    )


def make_plots(
    shap_values,
    prefix='',
    out_dir=None,
):
    """Create all possible plots.

    Parameters
    ----------
    shap_values : `shap.Explanation` objectplot
        Shapley values to visualise. The explanation object will be
        visualised automatically by downstream methods. 

    prefix : str, optional
        If set, adds a prefix to all filenames. Typically, this prefix
        can come from the model/run that was used to create the Shapley
        values. Will be ignored if not set.

    out_dir : str or `None`
        Output directory for creating visualisations. If set to `None`,
        will default to temporary directory.
    """
    if out_dir is None:
        out_dir = '/tmp'

    filename_prefix = os.path.join(
        out_dir, prefix + '_shapley'
    )

    shap.summary_plot(
        shap_values,
        feature_names=[f'{int(n)}-{int(n)+3}kDa' for n in np.linspace(2000,19997,6000)],
        color_bar_label='feature value (intensity)',
        show=False,
    )
    plt.tight_layout()
    plt.savefig(filename_prefix, dpi=300)
    plt.cla()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o', '--outdir',
        type=str,
        default='../plots/shapley',
        help='Output directory where plots will be stored.',
    )
    parser.add_argument(
        'FILE',
        nargs='+',
        type=str,
        help='Input file(s). Must contain Shapley values.',
    )

    args = parser.parse_args()

    all_shap_values = []

    for filename in args.FILE:
        print(f'Reading {filename}...')
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            shap_values = data['shapley_values']

        all_shap_values.append(shap_values)

    # TODO: could remove 'Seed' from filename and store one file per
    # scenario (with a nicer name, but this will do as well for now)
    prefix = os.path.basename(args.FILE[0])
    prefix = os.path.splitext(prefix)[0]

    make_plots(
        pool(all_shap_values),
        prefix=prefix,
        out_dir=args.outdir,
    )
