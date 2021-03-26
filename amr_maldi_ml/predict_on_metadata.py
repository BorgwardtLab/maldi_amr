"""Perform predictions based on metadata instead of spectra."""

import argparse
import logging
import uniplot
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from maldi_learn.driams import DRIAMSLabelEncoder


def make_dataframes(files, args):
    """Create data frames from file(s), based on mode."""
    df = []

    for filename in files:
        df_ = pd.read_csv(
            filename,
            sep=';',
            na_values='-',
            keep_default_na=True,
            # Required to ensure that different data types can be inferred
            # for the columns.
            low_memory=False
        )

        df.append(df_)

    df = pd.concat(df)

    # Ensures that these columns are always numerical. This will fill
    # them up with NaNs.
    for col in ['Score1', 'Score2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Get only the species we are interested in.
    df = df.query('`Organism.best.match.` == @args.species')

    # Retrieve all metadata columns. Since the file format differs
    # between sites, we cannot rely on the ordering. However, each
    # site has the same 21 columns, and the last 6 columns store a
    # bunch of other data.
    n_columns = len(df.columns)
    metadata_columns = df.columns[np.r_[:21, n_columns-6:n_columns]]

    # Remove columns that are (almost) unique for each sample, thus
    # making prediction trivial.
    columns_to_remove = [
        #'code',
        #'strain',
        #'Value',
        #'A',
        #'acquisition_date',
        #'Organism.best.match.',
        #'Organism.second.best.match.',
        #'GENUS',
        #'KEIM',
        #'acquisition_time',
        #'EINGANGSDATUM',
        #'SPEZIES_MALDI',
        #'SPEZIES_MLAB',
    ]

    for col in columns_to_remove:
        metadata_columns = metadata_columns.drop(col)

    logging.info(f'All metadata columns: {metadata_columns}')

    # Split the data set into two parts: one containing metadata
    # columns, the other one only containing resistance info. We
    # do a quick sanity check here.
    df_metadata = df[metadata_columns]
    df_resistance = df.drop(columns=metadata_columns)

    # Get all potential resistance values, but ignore all NaNs during
    # the set construction.
    resistance_values = set(df_resistance.values.ravel())
    resistance_values = set(filter(lambda x: x == x , resistance_values))

    assert 'I' in resistance_values
    assert 'R' in resistance_values

    encoder = DRIAMSLabelEncoder()

    # Only use a specific antibiotic for the prediction task. This could
    # also be achieved in the query above.
    df_resistance = encoder.fit_transform(df_resistance)
    df_resistance = df_resistance[args.antibiotic]
    
    # Remove all values of the prediction data frame that contain NaNs
    # because we cannot assign them a proper label.
    df_metadata = df_metadata[~df_resistance.isna()]
    df_resistance = df_resistance[~df_resistance.isna()]

    # The labels should be integers such that categorical predictions
    # work as expected. We lose the ability to represent NaNs, but we
    # can live with this because we removed them anyway.
    y = df_resistance.values.astype(int)

    # TODO: we remove all NaN values from the metadata because it is
    # unclear how to convert them.
    #
    # Need to drop NaN before converting to `numpy` data frame. We could
    # probably be smarter here.
    df_metadata = df_metadata.dropna(axis='columns')

    if args.drop_id:
        df_metadata = df_metadata.drop(
            columns=[
                c for c in df_metadata.columns if c.endswith('_id')
            ]
        )

        logging.info('Dropped ID columns from metadata')
    
    # Select columns/variables to use for the prediction task. This way,
    # we store also their names.
    columns = []

    if args.mode == 'numerical':
        columns = df_metadata.select_dtypes(include=[np.number]).columns
        logging.info(f'Only including numerical columns: {columns}')
    elif args.mode == 'categorical':
        columns = df_metadata.select_dtypes(include=[np.object]).columns
        logging.info(f'Only including categorical columns: {columns}')
    elif args.mode == 'all':
        columns = df_metadata.columns
        logging.info('Including all column types: {columns}')

    if args.single:
        X = []
        for col in columns:
            if df_metadata[col].dtype == np.object:
                X.append(
                    (col, pd.get_dummies(df_metadata[col]).to_numpy())
                )
            else:
                X.append(
                    (col, df_metadata[col].to_numpy().reshape(-1, 1))
                )
    else:
        X = pd.get_dummies(df_metadata[columns]).to_numpy()

    return X, y


def train_and_predict(X, y, name=None):
    """Train and predict on a feature matrix."""
    clf = LogisticRegressionCV(
        cv=5,
        scoring='average_precision',
        class_weight='balanced',
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        clf.fit(X, y)

    logging.info(f'Using feature matrix of shape {X.shape}...')

    y_pred = clf.predict(X)
    y_score = clf.predict_proba(X)

    if name is not None:
        logging.info(f'--- {name} ---')

    logging.info(f'  Accuracy: {accuracy_score(y, y_pred):.2f}')
    logging.info(f'  Precision: {precision_score(y, y_pred):.2f}')
    logging.info(f'  Recall: {recall_score(y, y_pred):.2f}')
    logging.info(f'  AUPRC: {average_precision_score(y, y_score[:, 1]):.2f}')
    logging.info(f'  AUROC: {roc_auc_score(y, y_score[:, 1]):.2f}')

    precision, recall, thres = precision_recall_curve(y, y_score[:, 1])
    uniplot.plot(
        precision,
        recall,
        lines=True,
        y_min=-0.05, y_max=1.05,
        x_min=-0.05, x_max=1.05,
        title='Precision--Recall'
    )

    uniplot.histogram(
        np.max(y_score, axis=1),
        bins=10,
        # The histogram function has some issues with plotting
        # everything properly under certain circumstances.
        x_min=0.49,
        x_max=0.99,
        title='Prediction Probabilities'
    )


if __name__ == '__main__':
    # Basic log configuration to ensure that we see where the process
    # spends most of its time.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'FILE',
        nargs='+',
        help='Input file(s)'
    )

    parser.add_argument(
        '-a', '--antibiotic',
        type=str,
        default='Ceftriaxon'

    )
    parser.add_argument(
        '-s', '--species',
        type=str,
        default='Escherichia coli'
    )

    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['numerical', 'categorical', 'all'],
        help='Choice of variable pre-selection',
        default='numerical'
    )

    parser.add_argument(
        '-d', '--drop-id',
        action='store_true',
        help='If set, drops ID columns'
    )

    parser.add_argument(
        '--single',
        action='store_true',
        help='If set, analyses each single variable individually'
    )

    args = parser.parse_args()

    X, y = make_dataframes(args.FILE, args)

    class_prevalence = np.min(np.bincount(y) / len(y))
    logging.info(f'Class prevalence: {class_prevalence:.2f}')

    if type(X) is list:
        for name, X_ in X:
            train_and_predict(X_, y, name=name)
    else:
        train_and_predict(X, y)
