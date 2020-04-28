"""Plot calibration curves.

The purpose of this script is to plot the calibration curves of
different classifiers to show to what extent calibration can be
used to improve classification results. The script is smart and
will collect data automatically so that plots are created based
on their availability.
"""


import argparse
import collections
import json
import os

from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from maldi_learn.metrics import specificity_score
from maldi_learn.metrics import sensitivity_score
from utilities import _encode

# Global metadata information; this will be updated by the script to
# ensure that we are working with data files from the *same* sources
# in order to create curves.
metadata_versions = {}

# To improve the captions of the curves. The advantage of this
# dictionary is that the script automatically raises an error
# upon encountering an unknown model.
model_to_name = {
    'lr': 'Logistic regression',
    'rf': 'Random forest',
    'svm-rbf': 'SVM (RBF kernel)',
    'svm-linear': 'SVM (linear kernel)',
    'lightgbm': 'LightGBM'
}


def _add_or_compare(metadata):
    if not metadata_versions:
        metadata_versions.update(metadata)
    else:
        # Smarter way to compare the entries of the dictionaries with
        # each other.
        assert metadata_versions == metadata


def make_calibration_curve(y_true, y_score):
    """Create calibration curve from scores.

    Parameters
    ----------
    y_true : `numpy.array` or `list`
        True labels

    y_score : `numpy.array` or `list`
        Classifier prediction scores

    Returns
    -------
    Tuple of (x, y) values, corresponding to the *predicted*
    probabilities and the *true* probabilities, respectively,
    of the calibration curve.
    """
    minority_class = np.argmin(np.bincount(y_true))

    prob_true, prob_pred = calibration_curve(y_true,
                                             y_score[:, minority_class],
                                             n_bins=5)

    # Note the change in inputs here; it is customary to show the
    # predicted probabilities on the x axis.
    return prob_pred, prob_true


def plot_calibration_curves(df, outdir):
    """Plot calibration curves based on data frame.

    This function is performing the main work for a single data frame.
    It will automatically collect the relevant curves and prepare a plot
    according to the data.

    Parameters
    ----------
    df : `pandas.DataFrame`
        TBD

    Returns
    -------
    Nothing. As a side-effect of calling this function, plots will be
    generated.
    """
    # Will store the individual curves for subsequent plotting. Every
    # curve is indexed by a classifier and by its type (if a model is
    # calibrated or not).
    curves = {}

    # The way the data are handed over to this function, there is only
    # a single model.
    model = df.model.unique()[0]

    for (species, antibiotic), df_ in df.groupby(['species', 'antibiotic']):
        y_test = np.vstack(df_['y_test']).ravel()
        y_score = np.vstack(df_['y_score'])
        y_score_calibrated = np.vstack(df_['y_score_calibrated'])

        curves[(species, antibiotic, 'raw')] = make_calibration_curve(
            y_test, y_score
        )

        curves[(species, antibiotic, 'calibrated')] = make_calibration_curve(
            y_test, y_score_calibrated
        )

    sns.set(style='whitegrid')

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    fig.suptitle(f'{model_to_name[model]}')

    palette = sns.color_palette()

    supported_species = [
        'Escherichia coli',
        'Klebsiella pneumoniae',
        'Staphylococcus aureus',
        'Staphylococcus epidermidis'
    ]

    species_to_colour = {
        species: palette[i] for i, species in enumerate(supported_species)
    }

    ax.plot([0, 1], [0, 1], 'k', label='Perfectly calibrated')

    for (species, antibiotic, curve_type), curve in curves.items():

        colour = species_to_colour[species]

        ax.plot(
            curve[0],
            curve[1],
            c=colour,
            linestyle='dotted' if curve_type == 'raw' else 'solid',
            label=f'{species} ({antibiotic})',
        )

    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('True probability')
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    filename = f'Calibration_curve_'     \
               f'{_encode(species)}_'    \
               f'{_encode(antibiotic)}_' \
               f'{model}.png'

    plt.savefig(
        os.path.join(outdir, filename),
        bbox_inches='tight'
    )


def make_rejection_curve(y_true, y_score, metric):
    """Calculate rejection curve based on scores and evaluation metric.

    This function simulates a rejection scenario based on prediction
    scores for a given classifier. To this end, potential probability
    thresholds are generated and the classifier is simulated with the
    corresponding rejection rate.

    Parameters
    ----------
    y_true : `numpy.array` or `list`
        True labels

    y_score : `numpy.array` or `list`
        Classifier prediction scores

    metric : str
        Metric to use for evaluating the classifier in a specific
        rejection scenario.

    Returns
    -------
    Tuple of (x, y) values, corresponding to the simulated scenario.
    """
    thresholds = np.linspace(0.5, 1.0, 20)
    y_score_max = np.amax(y_score, axis=1)
    n_samples = len(y_score_max)

    # TODO: is this required?
    minority_class = np.argmin(np.bincount(y_true))

    # Will be filled with the `x` and `y` values of the current curve
    x = []
    y = []
    ratio_keep = []

    for threshold in thresholds:

        # Get the indices that we want to *keep*, i.e. those test
        # samples whose maximum probability exceeds the threshold
        indices = y_score_max > threshold

        # Subset the predictions and the labels according to these
        # indices and calculate the desired metric.
        #
        # TODO: make metric configurable
        y_true_ = y_true[indices]
        y_pred_proba_ = y_score[indices][:, minority_class]

        # Predict the positive class if the prediction threshold is
        # larger than the one we use for this iteration.
        y_pred = np.zeros_like(y_pred_proba_)
        y_pred[y_pred_proba_ > threshold] = 1.0

        # Ensures that we are working with the proper scenario here;
        # we need two different classes to perform the calculation.
        if len(set(y_true_)) != 2:
            break

        # Calculate threshold for which a certain percentage of samples
        # is rejected
        ratio_keep.append(sum(indices)/float(n_samples))

        average_precision = average_precision_score(y_true_, y_pred_proba_)
        accuracy = accuracy_score(y_true_, y_pred)
        roc_auc = roc_auc_score(y_true_, y_pred_proba_)
        specificity = specificity_score(y_true_, y_pred)
        sensitivity = sensitivity_score(y_true_, y_pred)

        x.append(threshold)

        if metric == 'auroc':
            y.append(roc_auc)
        elif metric == 'auprc':
            y.append(average_precision)
        elif metric == 'accuracy':
            y.append(accuracy)
        elif metric == 'specificity':
            y.append(specificity)
        elif metric == 'sensitivity':
            y.append(sensitivity)
    
    return x, y, ratio_keep


def plot_rejection_curves(df, metric, outdir):
    """Plot rejection curves based on data frame.

    This function simulates different rejection scenarios based on the
    classifier scores. It will calculate rejection curves and plot the
    different scenarios in the data frame.

    Parameters
    ----------
    df : `pandas.DataFrame`
        TBD

    metric : str
        Metric for assessing the rejection curves. Can be either one of

            - accuracy
            - auprc (average precision score)
            - auroc (area under the ROC curve)
            - specificity (also called recall)
            - sensitivity

    outdir : str
        Output directory; this is where the plots will be stored.

    Returns
    -------
    Nothing. As a side-effect of calling this function, plots will be
    generated.
    """
    # Will store the individual curves for subsequent plotting. Every
    # curve is indexed by a classifier and by its type (if a model is
    # calibrated or not).
    curves = {}

    # The way the data are handed over to this function, there is only
    # a single model.
    model = df.model.unique()[0]

    for (species, antibiotic), df_ in df.groupby(['species', 'antibiotic']):

        y_test = np.vstack(df_['y_test']).ravel()
        y_score = np.vstack(df_['y_score'])
        y_score_calibrated = np.vstack(df_['y_score_calibrated'])

        curves[(species, antibiotic, 'raw')] = make_rejection_curve(
            y_test, y_score, metric
        )

        curves[(species, antibiotic, 'calibrated')] = make_rejection_curve(
            y_test, y_score_calibrated, metric
        )

    sns.set(style='whitegrid')

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    fig.suptitle(f'{model_to_name[model]}')

    palette = sns.color_palette()

    supported_species = [
        'Escherichia coli',
        'Klebsiella pneumoniae',
        'Staphylococcus aureus',
        'Staphylococcus epidermidis'
    ]

    species_to_colour = {
        species: palette[i] for i, species in enumerate(supported_species)
    }

    kr_to_fmt = {
        0.95: 'vr',
        0.9: 'or',
        0.75: '^r',
    }

    for (species, antibiotic, curve_type), curve in curves.items():

        colour = species_to_colour[species]

        ax.plot(
            curve[0],
            curve[1],
            c=colour,
            linestyle='dotted' if curve_type == 'raw' else 'solid',
            label=f'{species} ({antibiotic})',
        )
        
        for kr in [0.95, 0.9, 0.75]:
            if sum([val<kr for val in curve[2]])==0:
                continue
            kr_index = next(idx for idx, val in enumerate(curve[2]) 
                                if val<kr)
            ax.plot(curve[0][kr_index], curve[1][kr_index],
                    kr_to_fmt[kr])

        
    metric_to_label = {
        'accuracy': 'accuracy',
        'auprc': 'AUPRC',
        'auroc': 'AUROC',
        'specificity': 'specificity',
        'sensitivity': 'sensitivity',
    }

    ax.set_xlabel('Threshold')
    ax.set_ylabel(metric_to_label[metric])
    if metric == 'accuracy':
        ax.set_ylim((0.75, 1))
    elif metric =='specificity':
        ax.set_ylim((0.9, 1))
    elif metric == 'sensitivity':
        ax.set_ylim((0.0, 1))
    else:
        ax.set_ylim((0.3, 1))
    ax.set_xlim((0.5, 1))
    ax.legend(loc='lower left')

    filename = f'Rejection_curve_'       \
               f'{_encode(species)}_'    \
               f'{_encode(antibiotic)}_' \
               f'{metric}_'              \
               f'{model}.png'

    plt.savefig(
        os.path.join(outdir, filename),
        bbox_inches='tight'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input directory')
    parser.add_argument(
        '--outdir',
        type=str,
        default='.',
        help='Output directory'
    )
    parser.add_argument(
        '-m', '--metric',
        type=str,
        default='auroc'
    )

    args = parser.parse_args()

    # Stores data rows corresponding to individual scenarios. Each
    # scenario involves the same model (plus multiple antibiotics,
    # or species, combinations).
    scenarios = collections.defaultdict(list)

    # Keys to skip when creating a single row in the data dictionary
    # above. This ensures that we only add single pieces of data and
    # have an easier time turning every scenario into a data frame.
    skip_keys = ['years', 'metadata_versions']

    # Contains the combinations of species--antibiotic that we want to
    # plot in the end. Anything else is ignored.
    selected_combinations = [
        ('Escherichia coli', 'Cefepime'),
        ('Klebsiella pneumoniae', 'Ceftriaxone'),
        ('Staphylococcus aureus', 'Oxacillin')
    ]

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
        model = data['model']

        if (species, antibiotic) not in selected_combinations:
            continue

        _add_or_compare(data['metadata_versions'])

        row = {
            key: data[key] for key in data.keys() if key not in skip_keys
        }

        basename = os.path.basename(filename)

        scenarios[model].append(row)

    for model in sorted(scenarios.keys()):

        rows = scenarios[model]
        df = pd.DataFrame.from_records(rows)

        plot_calibration_curves(df, args.outdir)
        plot_rejection_curves(df, args.metric, args.outdir)
