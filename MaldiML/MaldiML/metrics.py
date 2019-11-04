# -*- coding: utf-8 -*-
# @Author: weisc
# @Date:   2019-09-18 17:25:56
# @Last Modified by:   weisc
# @Last Modified time: 2019-10-16 13:56:06


import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
from collections import Counter

from MaldiAI_pub1.utils    import calc_pos_class_ratio, calc_sample_size, vme_auc_curve, vme_auc_score

class Metrics():

    def __init__(self):
        self.TP = []


    def binary_classification(self, clf, X_test, y_test):
        assert np.all(np.isfinite(y_test)), 'Non-finite values given in y_test!'
        self.pos_class_ratio    = calc_pos_class_ratio(y_test)
        self.sample_size        = len(y_test)

        self.rocauc             = round(roc_auc_score(y_test, clf.estimator.predict_proba(X_test)[:, 1]), 3)
        #prec, rec, thresholds   = precision_recall_curve(y_test, clf.estimator.predict_proba(X_test)[:, 1])
        self.prauc              = average_precision_score(y_test, clf.estimator.predict_proba(X_test)[:, 1], average='weighted')
        self.prauc0             = average_precision_score(1-y_test, 1-clf.estimator.predict_proba(X_test)[:, 1], average='weighted')
        self.vmeauc             = vme_auc_score(y_test, clf.estimator.predict_proba(X_test)[:, 1])
        self.TN, self.FP, self.FN, self.TP = confusion_matrix(y_test, clf.estimator.predict(X_test), labels=[0,1]).ravel()
        self.recall             = round(self.TP / float(self.TP+self.FN), 3)
        self.sensitivity        = self.recall      
        self.precision          = round(self.TP / float(self.TP+self.FP), 3)
        self.accuracy           = round((self.TP+self.TN) / float(self.TN+self.FP+self.FN+self.TP), 3)
        self.f1                 = round(f1_score(y_test, clf.estimator.predict(X_test), average='binary'), 3)
        self.specificity        = round(self.TN / float(self.FP+self.TN), 3)
        self.very_major_error   = round(self.FP / float(self.FP+self.TN), 3)
        self.major_error        = round(self.FN / float(self.FN+self.TP), 3)


    def binary_classification_per_species(self, clf, X_test, y_test, species_test, min_num=20):
        species_counts = np.unique(species_test, return_counts=True)
        species_keep = []
        counts_keep = []

        for i, species in enumerate(species_counts[0]):
            if species_counts[1][i] >= min_num:
                species_keep.append(species)
                counts_keep.append(species_counts[1][i])

        dd_metrics = {}
        for species in species_keep:
            X_test_species = X_test[species_test==species,:]
            y_test_species = y_test[species_test==species]
            if len(np.unique(y_test_species)) < 2:
                continue
            if Counter(y_test_species)[0] < float(min_num)/2 or Counter(y_test_species)[1] < float(min_num)/2:
                continue
            dd_metrics[species] = Metrics()
            dd_metrics[species].binary_classification(clf, X_test_species, y_test_species)
        self.metrics_dict = dd_metrics

