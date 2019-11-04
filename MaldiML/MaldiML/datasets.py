# -*- coding: utf-8 -*-
# @Author: weisc
# @Date:   2019-08-07 10:21:41
# @Last Modified by:   weisc
# @Last Modified time: 2019-09-18 17:28:39



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


class Dataset():

	def __init__(self, X, y, list_tasks, sample_ids, sample_genera, sample_species):
		# check input
		if len(np.shape(y)) > 1:
			assert len(list_tasks) == y.shape[1]
		else: 
			assert len(list_tasks) == 1
		assert np.shape(sample_ids) == np.shape(sample_genera) == np.shape(sample_species)
		assert isinstance(list_tasks, list)
		assert isinstance(X, np.ndarray)
		assert isinstance(y, np.ndarray)

		if not isinstance(sample_ids, np.ndarray):
			sample_ids = np.array(sample_ids)
			assert sample_ids.ndim == 1
		if not isinstance(sample_genera, np.ndarray):
			sample_genera = np.array(sample_genera)
			assert sample_genera.ndim == 1
		if not isinstance(sample_species, np.ndarray):
			sample_species = np.array(sample_species)
			assert sample_species.ndim == 1


		# start building Dataset
		self.X = X
		self.y = y
		self.tasks = list_tasks
		self.sample_ids = sample_ids
		self.sample_genera = sample_genera
		self.sample_species = sample_species
		self.pos_class_ratio = []
		self.sample_size = []

		
		if len(np.shape(self.y)) == 1:
			self.pos_class_ratio.append(calc_pos_class_ratio(self.y))
			self.sample_size.append(calc_sample_size(self.y))
		else:
			for i in range(len(self.tasks)):
				self.pos_class_ratio.append(calc_pos_class_ratio(self.y[:,i]))
				self.sample_size.append(calc_sample_size(self.y[:,i]))

		# check output
		assert len(self.pos_class_ratio) == len(self.sample_size) == len(self.tasks)
		assert np.shape(self.sample_ids) == np.shape(self.sample_genera) == np.shape(self.sample_species)
		assert np.shape(self.X)[0] == np.shape(self.y)[0] == np.shape(self.sample_species)[0]


	def to_singletask(self,task):
		Singletask_Dataset = Dataset(self.X, self.y[:,self.tasks.index(task)], [task], self.sample_ids, self.sample_genera, self.sample_species)
		Singletask_Dataset.sample_size = Singletask_Dataset.sample_size[0]
		Singletask_Dataset.pos_class_ratio = Singletask_Dataset.pos_class_ratio[0]
		Singletask_Dataset.check_is_singletask()
		Singletask_Dataset.check()
		return Singletask_Dataset


	def to_multitask(self,task):
		idx = []
		for k, key in enumerate(task):
			idx.append(self.tasks.index(key))
		Multitask_Dataset = Dataset(self.X, self.y[:,idx], task, self.sample_ids, self.sample_genera, self.sample_species)
		Multitask_Dataset.check()
		return Multitask_Dataset


	def remove_nans(self):
		# check input
		if len(self.y.shape) > 1:
			if self.y.shape[1] > 1:
				print('remove_nans only possible on singletask datasets')
				return

		idx = np.where(~np.isnan(self.y))
		self.X = self.X[idx]
		self.y = self.y[idx]
		self.sample_ids = self.sample_ids[idx[0]]
		self.sample_genera = self.sample_genera[idx[0]]
		self.sample_species = self.sample_species[idx[0]]
		self.pos_class_ratio = []
		self.sample_size = []

		if len(np.shape(self.y)) == 1:
			self.pos_class_ratio.append(calc_pos_class_ratio(self.y))
			self.sample_size.append(calc_sample_size(self.y))
		else:
			for i in range(len(self.tasks)):
				self.pos_class_ratio.append(calc_pos_class_ratio(self.y[:,i]))
				self.sample_size.append(calc_sample_size(self.y[:,i]))

		self.sample_size = self.sample_size[0]
		self.pos_class_ratio = self.pos_class_ratio[0]

		# check output
		assert np.shape(self.sample_ids) == np.shape(self.sample_genera) == np.shape(self.sample_species)
		assert np.array_equal(self.y, self.y.astype(bool))


	def remove_nans_for_MTL(self, type='keep_if_one_non-nan_exists'):
		# count how many nans per line in y matrix
		y_nan = np.isnan(self.y)
		y_nansum = np.sum(y_nan,axis=1)

		if type == 'keep_if_one_non-nan_exists':
			idx = y_nansum != np.shape(self.y)[1]
		elif type == 'keep_if_none_are_nan':
			idx = y_nansum == 0

		self.X = self.X[idx,:]
		self.y = self.y[idx,:]
		self.sample_ids = self.sample_ids[idx]
		self.sample_genera = self.sample_genera[idx]
		self.sample_species = self.sample_species[idx]
		self.pos_class_ratio = []
		self.sample_size = []
		# check output
		assert np.shape(self.sample_ids) == np.shape(self.sample_genera) == np.shape(self.sample_species)


	def remove_species(self, species_to_remove_is=None, species_to_remove_startswith=None):
		if species_to_remove_is is not None:
			print('Remove species \"{}\" from Dataset'.format(species_to_remove_is))
			# remove_idx = self.sample_species == species_to_remove_is
			remove_idx = np.isin(self.sample_species, species_to_remove_is)
		elif species_to_remove_startswith is not None:
			print('Remove species that start with \"{}\" from Dataset'.format(species_to_remove_startswith))
			remove_idx = np.array([spec.startswith(species_to_remove_startswith) for spec in self.sample_species])
		else:
			print('No species given for removal!')
			return

		if sum(remove_idx)==len(self.y):
			# all samples are removed
			self.X = self.X[~remove_idx,:]
			self.y = self.y[~remove_idx]
			self.sample_ids = self.sample_ids[~remove_idx]
			self.sample_genera = self.sample_genera[~remove_idx]
			self.sample_species = self.sample_species[~remove_idx]
			
			self.pos_class_ratio = []
			self.sample_size = []
			return

		self.X = self.X[~remove_idx,:]
		self.y = self.y[~remove_idx]
		self.sample_ids = self.sample_ids[~remove_idx]
		self.sample_genera = self.sample_genera[~remove_idx]
		self.sample_species = self.sample_species[~remove_idx]
		
		self.pos_class_ratio = []
		self.sample_size = []

		if len(np.shape(self.y)) == 1:
			num1 = sum(self.y == 1)
			num0 = sum(self.y == 0)
			self.pos_class_ratio.append(num1 / float(num1 + num0))
			self.sample_size.append(num1 + num0)
		else:
			for i in range(len(self.tasks)):
				y_i = self.y[:,i]
				num1 = sum(y_i == 1)
				num0 = sum(y_i == 0)
				if num1 == 0:
					self.pos_class_ratio.append(int(0))
					self.sample_size.append(num1 + num0)
				else:
					self.pos_class_ratio.append(num1 / float(num1 + num0))
					self.sample_size.append(num1 + num0)
		self.check()			
		return


	def reduce_to_species(self, species_to_reduce_to=None, species_to_reduce_to_startswith=None):
		if species_to_reduce_to is not None:
			remove_idx = ~np.isin(self.sample_species, species_to_reduce_to)
		# elif species_to_reduce_to_startswith is not None:
		# 	remove_idx = np.array([~spec.startswith(species_to_reduce_to_startswith) for spec in self.sample_species])
		else:
			print('No species given!')

		self.X = self.X[~remove_idx,:]
		self.y = self.y[~remove_idx]
		self.sample_ids = self.sample_ids[~remove_idx]
		self.sample_genera = self.sample_genera[~remove_idx]
		self.sample_species = self.sample_species[~remove_idx]

		self.pos_class_ratio = []
		self.sample_size = []
		if len(np.shape(self.y)) == 1:
			self.pos_class_ratio.append(calc_pos_class_ratio(self.y))
			self.sample_size.append(calc_sample_size(self.y))
		else:
			for i in range(len(self.tasks)):
				self.pos_class_ratio.append(calc_pos_class_ratio(self.y[:,i]))
				self.sample_size.append(calc_sample_size(self.y[:,i]))
		
		# check output
		assert len(self.pos_class_ratio) == len(self.sample_size) == len(self.tasks)
		return


	def remove_species_less_than_min_num(self, min_num):
		remove_idx = []
		dict_counts = Counter(self.sample_species)
		remove_keys = [key for key,val in dict_counts.items() if val < min_num]
		[remove_idx.append(True) if s in remove_keys else remove_idx.append(False) for s in self.sample_species]
		remove_idx = np.array(remove_idx)

		self.X = self.X[~remove_idx,:]
		self.y = self.y[~remove_idx]
		self.sample_ids = self.sample_ids[~remove_idx]
		self.sample_genera = self.sample_genera[~remove_idx]
		self.sample_species = self.sample_species[~remove_idx]
		self.pos_class_ratio = []
		self.sample_size = []

		if len(np.shape(self.y)) == 1:
			num1 = sum(self.y == 1)
			num0 = sum(self.y == 0)
			self.pos_class_ratio.append(num1 / float(num1 + num0))
			self.sample_size.append(num1 + num0)
		else:
			for i in range(len(self.tasks)):
				y_i = self.y[:,i]
				num1 = sum(y_i == 1)
				num0 = sum(y_i == 0)
				if num1 == 0:
					self.pos_class_ratio.append(int(0))
					self.sample_size.append(num1 + num0)
				else:
					self.pos_class_ratio.append(num1 / float(num1 + num0))
					self.sample_size.append(num1 + num0)
		return


	def remove_not_reliable_identified(self):
		self.remove_species(species_to_remove_startswith='not reliable identification')
		return	


	def remove_species_not_stratifiable(self):
		self.check_is_singletask()
		self.check_contains_nans()
		self.check()

		list_all_species = list(np.unique(self.sample_species))

		for spec in list_all_species:

			spec_classes = [self.y[i] for i in range(len(self.y)) if self.sample_species[i]==spec]
			spec_classes_counts = np.unique(spec_classes, return_counts=True)
			# print spec, tuple(spec_classes_counts[1]), tuple(spec_classes_counts[1])==(1,1), tuple(spec_classes_counts[1])==(1,)

			if 1 in spec_classes_counts[1]:
				if tuple(spec_classes_counts[1])==(1,1) or tuple(spec_classes_counts[1])==(1,):
					remove_idx = np.array([s==spec for s in self.sample_species])
				else:
					class_remove = int(spec_classes_counts[0][np.where(spec_classes_counts[1]==1)])
					remove_idx1 = [s==spec for s in self.sample_species]
					remove_idx2 = [c==class_remove for c in list(self.y.flatten())]
					remove_idx = np.array([True if remove_idx1[j]==True and remove_idx2[j]==True else False for j in range(len(self.sample_species))])
					
				

				self.X = self.X[~remove_idx,:]
				self.y = self.y[~remove_idx]
				self.sample_ids = self.sample_ids[~remove_idx]
				self.sample_genera = self.sample_genera[~remove_idx]
				self.sample_species = self.sample_species[~remove_idx]
				
				self.pos_class_ratio = []
				self.sample_size = []

				# if there are no samples left, set pos_class_ratio and sample_size to 0
				if len(self.y)==0:
					print('Dataset empty!')
					if len(np.shape(self.y)) == 1:
						self.pos_class_ratio.append(0)
						self.sample_size.append(0)
					else:
						for i in range(len(self.tasks)):
							self.pos_class_ratio.append(num1 / float(num1 + num0))
							self.sample_size.append(0)
					return

				# if there are samples left, calc pos_class_ratio and sample_size 
				if len(np.shape(self.y)) == 1:
					num1 = sum(self.y == 1)
					num0 = sum(self.y == 0)
					self.pos_class_ratio.append(num1 / float(num1 + num0))
					self.sample_size.append(num1 + num0)
				else:
					for i in range(len(self.tasks)):
						y_i = self.y[:,i]
						num1 = sum(y_i == 1)
						num0 = sum(y_i == 0)
						if num1 == 0:
							self.pos_class_ratio.append(int(0))
							self.sample_size.append(num1 + num0)
						else:
							self.pos_class_ratio.append(num1 / float(num1 + num0))
							self.sample_size.append(num1 + num0)

		return	


	def train_test_split(self, test_size=0.2, random_state=123, stratify=None):
		# stratify default is using self.y 
		if stratify is None:
			stratify=self.y

		if (isinstance(stratify,six.string_types) and stratify == 'random') or stratify.ndim == 2:
			print('Random train_test_split')
			
			n_sample_range = range(self.X.shape[0])
			_, _, _, _, index_train, index_test = train_test_split(self.X, self.y, n_sample_range, test_size=test_size, random_state=random_state)

			# random singletask split
			if self.y.ndim == 1:
				Train_Dataset = Dataset(self.X[index_train,:], self.y[index_train], self.tasks, self.sample_ids[index_train], self.sample_genera[index_train], self.sample_species[index_train])
				Test_Dataset = Dataset(self.X[index_test,:], self.y[index_test], self.tasks, self.sample_ids[index_test], self.sample_genera[index_test], self.sample_species[index_test])

			# multitask split - always random	
			if self.y.shape[1] > 1:
				Train_Dataset = Dataset(self.X[index_train,:], self.y[index_train,:], self.tasks, self.sample_ids[index_train], self.sample_genera[index_train], self.sample_species[index_train])
				Test_Dataset = Dataset(self.X[index_test,:], self.y[index_test,:], self.tasks, self.sample_ids[index_test], self.sample_genera[index_test], self.sample_species[index_test])
			return Train_Dataset, Test_Dataset


		# singletask stratified split		
		elif stratify.ndim == 1:
			# print '\nsingletask stratified split'
			_, _, _, _, index_train, index_test = train_test_split(self.X, self.y, range(len(self.y)), test_size=test_size, stratify=stratify, random_state=random_state)
			Train_Dataset = Dataset(self.X[index_train,:], self.y[index_train], self.tasks, self.sample_ids[index_train], self.sample_genera[index_train], self.sample_species[index_train])
			Test_Dataset = Dataset(self.X[index_test,:], self.y[index_test], self.tasks, self.sample_ids[index_test], self.sample_genera[index_test], self.sample_species[index_test])
			return Train_Dataset, Test_Dataset
	

	def train_test_split_strat_by_species_and_class(self, test_size=0.2, random_state=123):
		self.check_is_singletask()
		self.check()

		# construct class and species vector for stratification
		le = LabelEncoder()
		species_num = le.fit_transform(self.sample_species).reshape((len(self.y),1))
		class_num = self.y.reshape((len(self.y),1))
		strat = np.c_[class_num, species_num]

		_, _, _, _, index_train, index_test = train_test_split(self.X, self.y, range(len(self.y)), test_size=test_size, stratify=strat, random_state=random_state)
		Train_Dataset = Dataset(self.X[index_train,:], self.y[index_train], self.tasks, self.sample_ids[index_train], self.sample_genera[index_train], self.sample_species[index_train])
		Test_Dataset = Dataset(self.X[index_test,:], self.y[index_test], self.tasks, self.sample_ids[index_test], self.sample_genera[index_test], self.sample_species[index_test])
		return Train_Dataset, Test_Dataset


	def check_is_singletask(self):
		assert self.y.ndim == 1, 'Dataset is NOT in singletask form.'


	def check_contains_nans(self):
		self.check_is_singletask()
		assert ~np.isnan(self.y).any(), 'Dataset contains Nans in y vector.'


	def copy(self):
		Copy_Dataset = Dataset(self.X, self.y, self.tasks, self.sample_ids, self.sample_genera, self.sample_species)
		return Copy_Dataset


	def add_X_of_other_Dataset(self, dataset):
		
		if len(dataset.sample_species) == len(self.sample_species):
			if np.all(dataset.sample_species == self.sample_species) \
			and np.all(dataset.sample_ids == self.sample_ids) \
			and np.all(dataset.tasks == self.tasks) \
			and np.all(np.nan_to_num(dataset.y) == np.nan_to_num(self.y)):
				print('adding X of same length dataset')
				new_X = np.c_[self.X, dataset.X]
				assert np.shape(new_X)[0] == np.shape(self.X)[0] == np.shape(dataset.X)[0]
				Append_Dataset = Dataset(new_X, self.y, self.tasks, self.sample_ids, self.sample_genera, self.sample_species)
				return Append_Dataset


		print('adding X of different length dataset')
		intersection_id = set(dataset.sample_ids).intersection(self.sample_ids)
		new_X = []
		new_y = []
		new_tasks = []
		new_sample_ids = []
		new_sample_genera = []
		new_sample_species = []

		for i, sid in enumerate(self.sample_ids):
			if sid in intersection_id:

				idx = np.where(dataset.sample_ids == sid)[0][0]

				assert np.all(np.nan_to_num(self.y[i,:]) == np.nan_to_num(dataset.y[idx,:]))
				assert self.sample_ids[i] == dataset.sample_ids[idx]
				assert self.sample_genera[i] == dataset.sample_genera[idx]
				assert self.sample_species[i] == dataset.sample_species[idx]

				# use row-bind since X[i,:] is 1-dimensional
				new_X.append(np.r_[self.X[i,:],dataset.X[idx,:]])
				new_y.append(self.y[i,:])
				new_sample_ids.append(self.sample_ids[i])
				new_sample_genera.append(self.sample_genera[i])
				new_sample_species.append(self.sample_species[i])
				

		new_tasks = self.tasks
		assert np.all(self.tasks == dataset.tasks)

		Append_Dataset = Dataset(np.array(new_X), np.array(new_y), new_tasks, new_sample_ids, new_sample_genera, new_sample_species)
		return Append_Dataset
	

	def count_empty_AMR(self):
		max_num = len(self.tasks)
		sample_reached_max_num = [np.sum(np.isnan(self.y[i,:])) == max_num for i in range(self.y.shape[0])]
		return np.sum(sample_reached_max_num)


	def reduce_to_nonempty_AMR(self):
		self.check()
		max_num = len(self.tasks)
		sample_reached_max_num = [np.sum(np.isnan(self.y[i,:])) == max_num for i in range(self.y.shape[0])]
		remove_idx = np.array(sample_reached_max_num)

		self.X = self.X[~remove_idx,:]
		self.y = self.y[~remove_idx]
		self.sample_ids = self.sample_ids[~remove_idx]
		self.sample_genera = self.sample_genera[~remove_idx]
		self.sample_species = self.sample_species[~remove_idx]

		self.pos_class_ratio = []
		self.sample_size = []
		if len(np.shape(self.y)) == 1:
			self.pos_class_ratio.append(calc_pos_class_ratio(self.y))
			self.sample_size.append(calc_sample_size(self.y))
		else:
			for i in range(len(self.tasks)):
				self.pos_class_ratio.append(calc_pos_class_ratio(self.y[:,i]))
				self.sample_size.append(calc_sample_size(self.y[:,i]))
		
		# check output
		self.check()


	def check(self):
		if len(np.shape(self.y)) > 1:
			assert len(self.tasks) == self.y.shape[1]
		else: 
			assert len(self.tasks) == 1
		assert np.shape(self.sample_ids) == np.shape(self.sample_genera) == np.shape(self.sample_species)
		assert isinstance(self.tasks, list)
		assert isinstance(self.X, np.ndarray)
		assert isinstance(self.y, np.ndarray)

		assert isinstance(self.sample_ids, np.ndarray)
		assert self.sample_ids.ndim == 1
		assert isinstance(self.sample_genera, np.ndarray)
		assert self.sample_genera.ndim == 1
		assert isinstance(self.sample_species, np.ndarray)
		assert self.sample_species.ndim == 1
