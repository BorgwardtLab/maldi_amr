"""Describes models for subsequent training."""

import logging
import joblib
import json
import warnings

import numpy as np

from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_pipeline_and_parameters(model, random_state):
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
            - 'svm-rbf' for a support vector machine with an RBF kernel
            - 'svm-linear' for a support vector machine with a linear kernel
            - 'rf' for a random forest
            - 'lightgbm' for a LightGBM model

    random_state : int, `RandomState` instance, or None
        If set, propagates random state to a model. This is *not*
        required or used for all models.

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

    elif model == 'svm-rbf':
        svm = SVC(
            kernel='rbf',
            max_iter=500,
            probability=True,
            class_weight='balanced'
        )

        pipeline = Pipeline(
            steps=[
                ('scaler', None),
                ('svm', svm),
            ]
        )

        param_grid = {
            'scaler': ['passthrough', StandardScaler()],
            'svm__C': 10.0 ** np.arange(-3, 4),  # 10^{-3}..10^{3}
            'svm__gamma': ['scale', 'auto'],
        }

        return pipeline, param_grid

    elif model == 'svm-linear':
        svm = SVC(
            kernel='linear',
            max_iter=500,
            probability=True,
            class_weight='balanced'
        )

        pipeline = Pipeline(
            steps=[
                ('scaler', None),
                ('svm', svm),
            ]
        )

        param_grid = {
            'scaler': ['passthrough', StandardScaler()],
            'svm__C': 10.0 ** np.arange(-3, 4),  # 10^{-3}..10^{3}
        }

        return pipeline, param_grid

    elif model == 'rf':

        # Make sure that we set a random state here; else, the results
        # are not reproducible.
        if random_state is None:
            warnings.warn(
                '`random_state` is not set for random '
                'forest classifier.'
            )

        rf = RandomForestClassifier(
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
        )

        pipeline = Pipeline(
            steps=[
                ('rf', rf),
            ]
        )

        param_grid = [
            {
                'rf__criterion': ['gini', 'entropy'],
                'rf__bootstrap': [False],
                'rf__n_estimators': [25, 50, 100, 200],
                'rf__max_features': ['auto', 'sqrt', 'log2']
            },
            {
                'rf__criterion': ['gini', 'entropy'],
                'rf__bootstrap': [True],
                'rf__oob_score': [True, False],
                'rf__n_estimators': [25, 50, 100, 200],
                'rf__max_features': ['auto', 'sqrt', 'log2']
            },
        ]

        return pipeline, param_grid

    elif model == 'lightgbm':

        # Make sure that we set a random state here; else, the results
        # are not reproducible.
        if random_state is None:
            warnings.warn(
                '`random_state` is not set for LightGBM classifier.'
            )

        lightgbm = LGBMClassifier(
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
        )

        pipeline = Pipeline(
            steps=[
                ('lightgbm', lightgbm),
            ]
        )

        param_grid = {
            'lightgbm__boosting_type': ['gbdt', 'dart', 'goss', 'rf'],
            'lightgbm__n_estimators': [25, 50, 100, 200],
            'lightgbm__learning_rate': 10.0 ** np.arange(-3, 4),
        }

        return pipeline, param_grid

    # If we reached this point, we should signal that we are not aware
    # of the currently-selected model.
    raise RuntimeError(
        f'No pipeline or configuration for "{model}" available.'
    )


def load_pipeline(filename):
    """Load ready-to-use pipeline from configuration file.

    This utility function is capable of taking a configuration file in
    JSON format and using it to obtain a ready-to-use pipeline. Notice
    that *no* cross-validation will be used.

    Parameters
    ----------
    filename : str
        Input filename

    Returns
    -------
    Tuple of pipeline and configuration file in JSON format. The latter
    is useful to extract additional parameters, depending on which kind
    of experiment is to be run. The configuration file is guaranteed to
    contain a `model` parameter. If none is set, defaults to `lr`.
    """
    with open(filename) as f:
        data = json.load(f)

    seed = data['seed']
    best_params = data['best_params']

    # This accommodates older versions of the code that do not save
    # their models to the file.
    model = data.get('model', 'lr')

    # Ensures that the key exists afterwards, regardless of whether we
    # are loading an old file or not.
    data['model'] = model

    pipeline, _ = get_pipeline_and_parameters(model, random_state=seed)
    pipeline.set_params(**best_params)

    if 'scaler' in pipeline.named_steps:
        # Scaling needs some manual adjustments because we just store
        # whether a scaler was used or not.
        if pipeline['scaler'] == 'StandardScaler':
            pipeline.set_params(scaler=StandardScaler())

    return pipeline, data


def calculate_metrics(y_true, y_pred, y_score, prefix=None):
    """Calculate performance metrics from scores and predictions.

    This is a convenience function for computing performance metrics
    from scores and predictions of the data. Currently the following
    metrics will be calculated:

    - accuracy
    - AUROC
    - AUPRC

    Parameters
    ----------
    y_true : `numpy.array`
        True labels

    y_pred : `numpy.array`
        Predicted labels

    y_score : `numpy.array`
        Prediction scores

    prefix : str, optional
        If set, adds a prefix to the calculated metrics of the form
        'prefix_NAME', where 'NAME' is the name of the metric.

    Returns
    -------
    Dictionary whose keys are the names of the calculated metrics, with
    an optional prefix, and whose values are the respective performance
    measures.
    """
    accuracy = accuracy_score(y_pred, y_true)

    # Automatically choose the proper evaluation method for measures
    # requiring the selection of a minority class.
    minority_class = np.argmin(np.bincount(y_true))

    auprc = average_precision_score(y_true, y_score[:, minority_class])
    auroc = roc_auc_score(y_true, y_score[:, minority_class])

    kv = [
        ('accuracy', accuracy),
        ('auprc', auprc),
        ('auroc', auroc)
    ]

    if prefix is not None:
        kv = [(f'{prefix}_{key}', value) for key, value in kv]

    return dict(kv)


def run_experiment(
    X_train, y_train,
    X_test, y_test,
    model,
    n_folds,
    random_state=None,
    verbose=False,
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

    verbose : bool, optional
        If set, will add verbose information about the trained model in
        the form of adding the best parameters as well as information
        about predict labels and scores.

    Returns
    -------
    A dictionary containing measurement descriptions and their
    corresponding values.
    """
    pipeline, param_grid = get_pipeline_and_parameters(
        model,
        random_state
    )

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
    results = {}

    if verbose:
        results.update({
            'best_params': grid_search.best_params_,
            'y_score': y_score.tolist(),
            'y_pred': y_pred.tolist(),
            'y_test': y_test.tolist(),
        })

    # Add information that *always* needs to be available.
    results.update(train_metrics)
    results.update(test_metrics)

    return results
