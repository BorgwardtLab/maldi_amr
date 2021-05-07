"""Describes models for subsequent training."""

import logging
import joblib
import json
import warnings

import numpy as np

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models import get_pipeline_and_parameters
from models import load_pipeline
from models import calculate_metrics


def run_experiment(
    X_train, y_train,
    X_test, y_test,
    Z_train, z_train,
    Z_test, z_test,
    model,
    n_folds,
    random_state=None,
    verbose=False,
    scoring='roc_auc',
    class_weight='balanced',
    meta_train=None,
    meta_test=None,
):
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

    random_state : int, `RandomState` instance, or None
        If set, propagates random state to a model. This is *not*
        required or used for all models.

    scoring : str, optional
        Specifies scoring function to use. Should be a string that can
        be understood by `GridSearchCV`.

    verbose : bool, optional
        If set, will add verbose information about the trained model in
        the form of adding the best parameters as well as information
        about predicted labels and scores.

    meta_train : `pd.DataFrame`, optional
        If set, will add additional information about training samples
        based on metadata. Only applies if `verbose = True`.

    meta_test : `pd.DataFrame`, optional
        If set, will add additional information about test samples based
        on metadata. Only applies if `verbose = True`.

    Returns
    -------
    A dictionary containing measurement descriptions and their
    corresponding values.
    """
    pipeline, param_grid = get_pipeline_and_parameters(
        model,
        random_state,
        class_weight=class_weight,
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=n_folds,
        scoring=scoring,
        n_jobs=-1,
    )

    # Construct domain classification dataset 
    n, dim_X = X_train.shape 
    m, dim_Z = Z_train.shape
    assert dim_X == dim_Z, 'Dimensionalities do not match between X and Z.'

    XZ_train = np.concatenate((X_train, Z_train), axis=0)
    classes = np.r_[(np.zeros(n), np.ones(m))]
    
    logging.info(f'Starting grid search for importance classifiers on {len(XZ_train)} training samples')

    # Ignore these warnings only for the grid search process. The
    # reason is that some of the jobs will inevitably *fail* to
    # converge because of bad `C` values. We are not interested in
    # them anyway.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        with joblib.parallel_backend('threading', -1):
            grid_search.fit(XZ_train, classes)

    # Calculate importance weighting
    X_importance_weights = grid_search.predict_proba(X_train)
    print(X_importance_weights.min())
    print(X_importance_weights.max())

    # Ignore these warnings only for the grid search process. The
    # reason is that some of the jobs will inevitably *fail* to
    # converge because of bad `C` values. We are not interested in
    # them anyway.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        with joblib.parallel_backend('threading', -1):
            grid_search.fit(X_train, y_train, 
                            sample_weight=X_importance_weights,
                            )

    # Calculate metrics for the training data fully in-line because we
    # only ever want to save the results.
    train_metrics = calculate_metrics(
        y_train,
        grid_search.predict(X_train),
        grid_search.predict_proba(X_train),
        prefix='train'
    )

    y_pred = grid_search.predict(X_test)
    y_score = grid_search.predict_proba(X_test)

    test_metrics = calculate_metrics(y_test, y_pred, y_score)

    # Replace information about the standard scaler prior to writing out
    # the `best_params_` grid. The reason for this is that we cannot and
    # probably do not want to serialise the scaler class. We only need
    # to know *if* a scaler has been employed.
    if 'scaler' in grid_search.best_params_:
        scaler = grid_search.best_params_['scaler']
        if scaler != 'passthrough':
            grid_search.best_params_['scaler'] = type(scaler).__name__

    # Prepare the results dictionary for this experiment. Depending on
    # the input parameters of this function, additional information is
    # added.
    results = {
        'scoring': scoring,
    }

    # Add additional information about split. This is relevant for
    # scenarios with a case-based stratification.
    results.update({
        'test_size_obtained': len(y_test) / (len(y_train) + len(y_test)),
        'prevalence_train': (np.bincount(y_train) / len(y_train)).tolist(),
        'prevalence_test': (np.bincount(y_test) / len(y_test)).tolist(),
    })

    if verbose:
        results.update({
            'best_params': grid_search.best_params_,
            'y_score': y_score.tolist(),
            'y_pred': y_pred.tolist(),
            'y_test': y_test.tolist(),
            'importance_weights': X_importance_weights.tolist(),
        })

        # Only include certain columns of the meta data. This is
        # future-proof in case we need more columns or more data
        # later on.
        meta_columns = ['workstation']

        for col in meta_columns:
            if meta_train is not None and col in meta_train:
                results.update({
                    f'meta_train_{col}': meta_train[col].values.tolist(),
                })
            if meta_test is not None and col in meta_test:
                results.update({
                    f'meta_test_{col}': meta_test[col].values.tolist()
                })

    # Add information that *always* needs to be available.
    results.update(train_metrics)
    results.update(test_metrics)

    return results
