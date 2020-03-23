"""Describes models for subsequent training."""

import logging
import joblib
import warnings

import numpy as np

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_pipeline_and_parameters(model):
    """Return pipeline and parameters for a given model.

    This function creates a full training pipeline for a given model
    name, making it possible to harmonise training across different
    scripts and scenarios. Moreover, it returns a parameter grid for
    a subsequent grid search.

    Parameters
    ----------
    model : str
        Describes model whose pipeline should be returned. Current
        supported values are:

            - 'lr' for logistic regression
            - 'rbf-svm' for a support vector machine with an RBF kernel

    Returns
    -------
    Tuple of `sklearn.pipeline.Pipeline` and `dict`, representing the
    pipeline for training a model and its parameters, respectively.
    """
    if model == 'lr':
        lr = LogisticRegression(solver='saga', max_iter=500)

        pipeline = Pipeline(
            steps=[
                ('scaler', None),
                ('lr', lr),
            ]
        )

        param_grid = [
            {
                'scaler': ['passthrough', StandardScaler()],
                'lr__C': 10.0 ** np.arange(-3, 4),  # 10^{-3}..10^{3}
                'lr__penalty': ['l1', 'l2'],
            },
            {
                'scaler': ['passthrough', StandardScaler()],
                'lr__penalty': ['none'],
            }
        ]

        return pipeline, param_grid


def run_experiment(X_train, y_train, X_test, y_test, model, n_folds):
    """Run experiment for given train--test split.

    This is the main function for training and testing a classifier. The
    function is oblivious to the context of the training. It performs an
    in-depth grid search and evaluates the results.

    Parameters
    ----------
    X_train : array-like
        Training data

    y_train : list
        Labels for the training data

    X_test : array_like
        Test data

    y_test : list
        Labels for the rest data

    model : str
        Specifies a model whose pipeline will be queried and set up by
        this function. Must be a valid model according to the function
        `get_pipeline_and_parameters()`.

    n_folds : int
        Number of folds for internal cross-validation

    Returns
    -------
    A dictionary containing measurement descriptions and their
    corresponding values.
    """
    pipeline, param_grid = get_pipeline_and_parameters(model)

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=n_folds,
        scoring='roc_auc',
        n_jobs=-1,
    )

    logging.info(f'Starting grid search for {len(X_train)} samples')

    # Ignore these warnings only for the grid search process. The
    # reason is that some of the jobs will inevitably *fail* to
    # converge because of bad `C` values. We are not interested in
    # them anyway.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        with joblib.parallel_backend('threading', -1):
            grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    y_score = grid_search.predict_proba(X_test)

    accuracy = accuracy_score(y_pred, y_test)

    # Automatically choose the proper evaluation method for measuring
    # requiring the selection of a minority class.
    minority_class = np.argmin(np.bincount(y_test))

    auprc = average_precision_score(y_test, y_score[:, minority_class])
    auroc = roc_auc_score(y_test, y_score[:, minority_class])

    # Replace information about the standard scaler prior to writing out
    # the `best_params_` grid. The reason for this is that we cannot and
    # probably do not want to serialise the scaler class. We only need
    # to know *if* a scaler has been employed.
    if 'scaler' in grid_search.best_params_:
        scaler = grid_search.best_params_['scaler']
        if scaler != 'passthrough':
            grid_search.best_params_['scaler'] = type(scaler).__name__

    # Prepare the results dictionary for this experiment.
    results = {
        'accuracy': accuracy,
        'auprc': auprc,
        'auroc': auroc,
    }

    return results
