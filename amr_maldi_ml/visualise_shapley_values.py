"""Visualisation of Shapley values."""

import argparse
import os
import pickle
import shap

import matplotlib.pyplot as plt


def make_plots(
    shap_values,
    prefix='',
    out_dir=None,
):
    """Create all possible plots.

    Parameters
    ----------
    shap_values : `shap.Explanation` object
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

    for plot in ['bar', 'dot']:
        shap.summary_plot(
            shap_values,
            plot_type=plot,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(filename_prefix + f'_{plot}.png', dpi=300)
        plt.cla()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'FILE',
        type=str,
        help='Input file`. Must contain Shapley values.',
    )

    args = parser.parse_args()

    with open(args.FILE, 'rb') as f:
        data = pickle.load(f)
        shap_values = data['shapley_values']

    prefix = os.path.basename(args.FILE)
    prefix = os.path.splitext(prefix)[0]

    make_plots(
        shap_values,
        prefix=prefix,
    )
