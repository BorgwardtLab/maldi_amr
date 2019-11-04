# -*- coding: utf-8 -*-
# @Author: weisc
# @Date:   2019-08-07 10:21:32
# @Last Modified by:   weisc
# @Last Modified time: 2019-09-18 17:27:51



import numpy as np
import six

from collections 				import Counter
from itertools 					import compress
from sklearn.model_selection 	import train_test_split, GridSearchCV
from sklearn.feature_selection 	import SelectKBest
from sklearn.linear_model 		import LogisticRegression
from sklearn.metrics 			import roc_auc_score, confusion_matrix, f1_score, classification_report, make_scorer, recall_score
from sklearn.svm 				import SVC
from sklearn.ensemble 			import RandomForestClassifier
from sklearn.preprocessing 		import label_binarize, normalize, LabelEncoder, StandardScaler
from scipy.spatial.distance 	import cdist
from sklearn.pipeline 			import make_pipeline

from MaldiAI_pub1.utils 		import calc_pos_class_ratio, calc_sample_size


class Experiment():

	def __init__(self, phenotype):
		self.phenotype = phenotype

	def LogisticRegression(self, X_train, y_train, n_folds=5, scaling=False):
		self.clf_type = 'LogisticRegression'
		param_grid_lr = {
		'logisticregression__C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
			}

		if len(np.unique(y_train)) < 2:
			print('All samples have the same class label.')
			return
		elif len(np.unique(y_train)) == 2:
			# binary classification
			scoring = {
					'ROC-AUC': 'roc_auc',
					'PR-AUC' : 'average_precision'
				}
		elif len(np.unique(y_train)) > 2:
			# multiclass classification
			f1_scorer = make_scorer(f1_score, average='weighted')
			recall_scorer = make_scorer(recall_score, average='weighted')
			scoring = {
				'f1': f1_scorer,
				'recall' : recall_scorer
			}


		# set solver
		if len(np.unique(y_train)) > 2:
			print('Multiclass classification: solver lbfgs')
			solver='lbfgs'
		elif len(np.unique(y_train)) == 2:
			print('Binary classification: solver liblinear')
			solver='liblinear'


		if scaling:
			lr = make_pipeline(
				StandardScaler(),
				LogisticRegression(random_state=123, solver=solver))
		else:
			lr = make_pipeline(LogisticRegression(random_state=123, solver=solver))


		grid_search = GridSearchCV(estimator = lr, param_grid = param_grid_lr, scoring = scoring, cv = n_folds, refit = '{}'.format(scoring.keys()[0]), 
			n_jobs = n_folds).fit(X_train, y_train)
		self.grid_search = grid_search
		self.stds = grid_search.cv_results_['std_test_{}'.format(scoring.keys()[0])][grid_search.best_index_]
		self.estimator = grid_search.best_estimator_
		self.best_score = grid_search.best_score_



	def RandomForest(self, X_train, y_train, n_folds=5, scaling=False):
		self.clf_type = 'RandomForest'
		param_grid_rf = {
		'randomforestclassifier__min_samples_split': [2],
		'randomforestclassifier__max_depth': [2, 5, 10, 15],
		'randomforestclassifier__max_features': [2, 3, 5, 10, 20],
		'randomforestclassifier__min_samples_leaf': [1],
		'randomforestclassifier__n_estimators': [10, 20, 50, 100, 200]
			}
		if np.shape(X_train)[1] < 20:
			param_grid_rf = {
			'randomforestclassifier__min_samples_split': [2],
			'randomforestclassifier__max_depth': [2, 5, 10, 15],
			'randomforestclassifier__max_features': [2, 3, 5, 10],
			'randomforestclassifier__min_samples_leaf': [1],
			'randomforestclassifier__n_estimators': [10, 20, 50, 100, 200]
				}
		if np.shape(X_train)[1] < 10:
			param_grid_rf = {
			'randomforestclassifier__min_samples_split': [2],
			'randomforestclassifier__max_depth': [2, 5, 10, 15],
			'randomforestclassifier__max_features': [1],
			'randomforestclassifier__min_samples_leaf': [1],
			'randomforestclassifier__n_estimators': [10, 20, 50, 100, 200]
				}	


		if len(np.unique(y_train)) == 2:
			# binary classification
			scoring = {
					'ROC-AUC': 'roc_auc',
					'PR-AUC' : 'average_precision'
				}
		elif len(np.unique(y_train)) > 2:
			# multiclass classification
			f1_scorer = make_scorer(f1_score, average='weighted')
			recall_scorer = make_scorer(recall_score, average='weighted')
			scoring = {
				'f1': f1_scorer,
				'recall' : recall_scorer
			}


		if scaling:
			rf = make_pipeline(
				StandardScaler(),
				RandomForestClassifier(random_state=123))
		else:
			rf = make_pipeline(RandomForestClassifier(random_state=123))


		grid_search = GridSearchCV(estimator = rf, param_grid = param_grid_rf, scoring = scoring, cv = n_folds, refit = '{}'.format(scoring.keys()[0]), 
			n_jobs = n_folds).fit(X_train, y_train)
		self.grid_search = grid_search
		self.stds = grid_search.cv_results_['std_test_{}'.format(scoring.keys()[0])][grid_search.best_index_]
		self.estimator = grid_search.best_estimator_
		self.best_score = grid_search.best_score_	



	def LinearRegression(self, X_train, y_train, n_folds=5, scaling=False):
		self.clf_type = 'LinearRegression'
		param_grid_rf = {
		'randomforestclassifier__min_samples_split': [2],
		'randomforestclassifier__max_depth': [2, 5, 10, 15],
		'randomforestclassifier__max_features': [2, 3, 5, 10, 20],
		'randomforestclassifier__min_samples_leaf': [1],
		'randomforestclassifier__n_estimators': [10, 20, 50, 100, 200]
			}
		if np.shape(X_train)[1] < 2:
			param_grid_rf = {
			'randomforestclassifier__min_samples_split': [2],
			'randomforestclassifier__max_depth': [2, 5, 10, 15],
			'randomforestclassifier__max_features': [1],
			'randomforestclassifier__min_samples_leaf': [1],
			'randomforestclassifier__n_estimators': [10, 20, 50, 100, 200]
				}	


		if len(np.unique(y_train)) == 2:
			# binary classification
			scoring = {
					'ROC-AUC': 'roc_auc',
					'PR-AUC' : 'average_precision'
				}
		elif len(np.unique(y_train)) > 2:
			# multiclass classification
			f1_scorer = make_scorer(f1_score, average='weighted')
			recall_scorer = make_scorer(recall_score, average='weighted')
			scoring = {
				'f1': f1_scorer,
				'recall' : recall_scorer
			}


		if scaling:
			rf = make_pipeline(
				StandardScaler(),
				RandomForestClassifier(random_state=123))
		else:
			rf = make_pipeline(RandomForestClassifier(random_state=123))


		grid_search = GridSearchCV(estimator = rf, param_grid = param_grid_rf, scoring = scoring, cv = n_folds, refit = '{}'.format(scoring.keys()[0]), 
			n_jobs = n_folds).fit(X_train, y_train)
		self.grid_search = grid_search
		self.stds = grid_search.cv_results_['std_test_{}'.format(scoring.keys()[0])][grid_search.best_index_]
		self.estimator = grid_search.best_estimator_
		self.best_score = grid_search.best_score_	


	def LogisticRegression_weighsamples(self, X_train, y_train, n_folds=5, scaling=False, sample_weights=None):
		print('got sample_weights {}'.format(np.shape(sample_weights)))
		

		self.clf_type = 'LogisticRegression'
		param_grid_lr = {
		'logisticregression__C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
			}

		if len(np.unique(y_train)) < 2:
			print('All samples have the same class label.')
			return
		elif len(np.unique(y_train)) == 2:
			# binary classification
			scoring = {
					'ROC-AUC': 'roc_auc',
					'PR-AUC' : 'average_precision'
				}
		elif len(np.unique(y_train)) > 2:
			# multiclass classification
			f1_scorer = make_scorer(f1_score, average='weighted')
			recall_scorer = make_scorer(recall_score, average='weighted')
			scoring = {
				'f1': f1_scorer,
				'recall' : recall_scorer
			}


		# set solver
		if len(np.unique(y_train)) > 2:
			print('Multiclass classification: solver lbfgs')
			solver='lbfgs'
		elif len(np.unique(y_train)) == 2:
			print('Binary classification: solver liblinear')
			solver='liblinear'


		if scaling:
			lr = make_pipeline(
				StandardScaler(),
				LogisticRegression(random_state=123, solver=solver))
		else:
			lr = make_pipeline(LogisticRegression(random_state=123, solver=solver))


		grid_search = GridSearchCV(estimator = lr, param_grid = param_grid_lr, scoring = scoring, cv = n_folds, refit = '{}'.format(scoring.keys()[0]), 
			n_jobs = n_folds, fit_params={'logisticregression__sample_weight': sample_weights}).fit(X_train, y_train)
		self.grid_search = grid_search
		self.stds = grid_search.cv_results_['std_test_{}'.format(scoring.keys()[0])][grid_search.best_index_]
		self.estimator = grid_search.best_estimator_
		self.best_score = grid_search.best_score_



	def RandomForest_weighsamples(self, X_train, y_train, n_folds=5, scaling=False, sample_weights=None):
		self.clf_type = 'RandomForest'
		param_grid_rf = {
		'randomforestclassifier__min_samples_split': [2],
		'randomforestclassifier__max_depth': [2, 5, 10, 15],
		'randomforestclassifier__max_features': [2, 3, 5, 10, 20],
		'randomforestclassifier__min_samples_leaf': [1],
		'randomforestclassifier__n_estimators': [10, 20, 50, 100, 200]
			}
		if np.shape(X_train)[1] < 20:
			param_grid_rf = {
			'randomforestclassifier__min_samples_split': [2],
			'randomforestclassifier__max_depth': [2, 5, 10, 15],
			'randomforestclassifier__max_features': [2, 3, 5, 10],
			'randomforestclassifier__min_samples_leaf': [1],
			'randomforestclassifier__n_estimators': [10, 20, 50, 100, 200]
				}
		if np.shape(X_train)[1] < 10:
			param_grid_rf = {
			'randomforestclassifier__min_samples_split': [2],
			'randomforestclassifier__max_depth': [2, 5, 10, 15],
			'randomforestclassifier__max_features': [1],
			'randomforestclassifier__min_samples_leaf': [1],
			'randomforestclassifier__n_estimators': [10, 20, 50, 100, 200]
				}	


		if len(np.unique(y_train)) == 2:
			# binary classification
			scoring = {
					'ROC-AUC': 'roc_auc',
					'PR-AUC' : 'average_precision'
				}
		elif len(np.unique(y_train)) > 2:
			# multiclass classification
			f1_scorer = make_scorer(f1_score, average='weighted')
			recall_scorer = make_scorer(recall_score, average='weighted')
			scoring = {
				'f1': f1_scorer,
				'recall' : recall_scorer
			}


		if scaling:
			rf = make_pipeline(
				StandardScaler(),
				RandomForestClassifier(random_state=123))
		else:
			rf = make_pipeline(RandomForestClassifier(random_state=123))


		grid_search = GridSearchCV(estimator = rf, param_grid = param_grid_rf, scoring = scoring, cv = n_folds, refit = '{}'.format(scoring.keys()[0]), 
			n_jobs = n_folds, fit_params={'randomforestclassifier__sample_weight': sample_weights}).fit(X_train, y_train)
		self.grid_search = grid_search
		self.stds = grid_search.cv_results_['std_test_{}'.format(scoring.keys()[0])][grid_search.best_index_]
		self.estimator = grid_search.best_estimator_
		self.best_score = grid_search.best_score_	
