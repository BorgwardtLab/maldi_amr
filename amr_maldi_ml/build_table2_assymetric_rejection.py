"""Build rejection tables with metric information.

This script will create tables listing the confusion table metrics,
sensitivity and specificity at different rejection thresholds. Also
included is the percentage of samples which are rejected in each
threshold.
A seperate table will be created for each model and species-antibiotic 
scenario.
"""


import argparse
import collections
import json
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

from tqdm import tqdm

from maldi_learn.metrics import specificity_score
from maldi_learn.metrics import sensitivity_score
from utilities import _encode

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


def calc_metrics_for_rejection_threshold(y_true, y_score,
        threshold_lower, threshold_upper):
    """Calculate metrics for a certain rejection threshold.

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

    threshold_lower: `int`
        Rejection threshold w.r.t. to positive class. 
        All samples with a predicted positive class
        probability above this threshold and below threshold_upper
        will be discarded.

    threshold_upper: `int`
        Rejection threshold w.r.t. to positive class. 
        All samples with a predicted positive class
        probability below this threshold and above threshold_lower
        will be discarded.

    Returns
    -------
    Pandas dataframe with all metrics, rejection threshold and rejection
    percentage as columns.
    """
    # Determine maximum class probability to apply threshold cut-off
    # on both classes. 
    n_samples = len(y_score)
    minority_class = np.argmin(np.bincount(y_true))

    # Get the indices that we want to *keep*, i.e. those test
    # samples whose maximum probability exceeds the threshold
    indices_upper = y_score[:, minority_class] > threshold_upper
    indices_lower = y_score[:, minority_class] < threshold_lower
    indices = indices_upper + indices_lower

    # Subset the predictions and the labels according to these
    # indices and calculate the desired metrics.
    y_true_ = y_true[indices]
    y_pred_proba_ = y_score[indices][:, minority_class]

    # Predict the positive class if the prediction threshold is
    # larger than the one we use for this iteration.
    y_pred = np.zeros_like(y_pred_proba_)
    y_pred[y_pred_proba_ > threshold_upper] = 1.0

    # Ensures that we are working with the proper scenario here;
    # we need two different classes to perform the calculation.
    if len(set(y_true_)) != 2:
        return None
            
    # Calculate percentage of rejected samples 
    n_rejected = len(y_true) - len(y_true_) 
    percentage_rejected = round(n_rejected / n_samples, 5)


    # Calculate confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_true_, 
                                      y_pred,
                                      labels=[0,1]
                                      ).ravel() 

    # Calculate metrics
    average_precision = average_precision_score(y_true_, y_pred_proba_)
    accuracy = accuracy_score(y_true_, y_pred)
    roc_auc = roc_auc_score(y_true_, y_pred_proba_)
    specificity = round(specificity_score(y_true_, y_pred), 5)
    sensitivity = round(sensitivity_score(y_true_, y_pred), 5)

    # Convert into pd.DataFrame format for appending the overall
    # dataframe 
    row = pd.DataFrame({
            'threshold_lower': [threshold_lower],
            'threshold_upper': [threshold_upper],
            'percentage rejected samples': [percentage_rejected],
            'specificity': [specificity],
            'sensitivity': [sensitivity],
            'TP': [TP],
            'FP': [FP],
            'TN': [TN],
            'FN': [FN],
        })

    return row


def build_rejection_table(df, outdir, curve_type='calibrated'):
    """Contruct table with different rejection threshold in each row.

    This function simulates different rejection scenarios based on the
    classifier scores. Each line represents the metrics and percentage
    of rejected samples for a different rejection ratio.

    Parameters
    ----------
    df : `pandas.DataFrame`
        TBD

    outdir : str
        Output directory; this is where the plots will be stored.
    
    curve_type : str
        Type of classifier, either `calibrated` or `raw`.

    Returns
    -------
    Nothing. As a side-effect of calling this function, tables will be
    generated.
    """

    thresholds_lower = np.linspace(0.0, 0.5, 21)
    thresholds_upper = np.linspace(0.5, 1.0, 21)

    # The way the data are handed over to this function, there is only
    # a single model.
    model = df.model.unique()[0]

    for (species, antibiotic), df_ in df.groupby(['species', 'antibiotic']):
        table_df = pd.DataFrame()

        y_test = np.vstack(df_['y_test']).ravel()
        y_score = np.vstack(df_['y_score'])
        y_score_calibrated = np.vstack(df_['y_score_calibrated'])

        for threshold_upper in thresholds_upper:
            for threshold_lower in thresholds_lower:

                if curve_type=='calibrated':
                    row = calc_metrics_for_rejection_threshold(
                        y_test, 
                        y_score_calibrated, 
                        threshold_lower,
                        threshold_upper,
                    )

                else:
                    row = calc_metrics_for_rejection_threshold(
                        y_test, 
                        y_score, 
                        threshold_lower,
                        threshold_upper,
                    )
                
                if row is not None:
                    table_df = table_df.append(row)
    
        print(table_df)
        filename = (
                f'Rejection_table_' +
                f'assymetric_' +
                f'{species}_{antibiotic}_' + 
                f'{curve_type}_{model}.csv'
        )
        table_df.to_csv(
            os.path.join(outdir, filename),
            index=False
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

        build_rejection_table(df, args.outdir)
