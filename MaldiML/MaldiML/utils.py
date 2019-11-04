# -*- coding: utf-8 -*-
# @Author: weisc
# @Date:   2019-09-17 13:02:55
# @Last Modified by:   weisc
# @Last Modified time: 2019-11-04 11:34:57
## -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------
# utils file for main_ms.py
#
# Feb 2018 C. Weis
#-----------------------------------------------------------------------------
import numpy as np
import scipy as sp
import cPickle as pickle
import os, xlrd, matplotlib, csv, re, six, operator
from string import ascii_letters
from collections import Counter, namedtuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from matplotlib.ticker import NullFormatter
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE, MDS, SpectralEmbedding, Isomap

from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from scipy.stats import chisquare, ttest_ind
from sklearn.metrics.ranking import _binary_clf_curve

from itertools import compress
from collections import defaultdict as ddict
matplotlib.pyplot.switch_backend('agg')



def calc_pos_class_ratio(y):
    num1 = sum(y == 1)
    num0 = sum(y == 0)
    if num1 == 0:
        pos_class_ratio = int(0)
    elif num1 == 0 and num0 == 0:
        pos_class_ratio = int(0)
    else:
        pos_class_ratio = num1 / float(num1 + num0)
    return pos_class_ratio


def calc_sample_size(y):
    num1 = sum(y == 1)
    num0 = sum(y == 0)
    sample_size = num1 + num0
    return sample_size

#-----------------------------------------------------------------------------
# Maldi Spectra Read-in functions
#-----------------------------------------------------------------------------


def read_single_spectra(spectra_dir, filename):
    mz = []
    intensity = []
    file = open(os.path.join(spectra_dir, filename), 'r')
    for line in file:
        if line.startswith('#') or line.startswith('COM') or line.startswith('"'):
            pass
        else:
            splitline = line.strip('\n').split(' ')
            if len(splitline)>1:
                try:
                    intensity.append(float(splitline[1]))
                    mz.append(float(splitline[0]))
                except Exception as e:
                    print('Failed to append mx and intensity: ' + str(e))    

    if len(mz) != len(intensity):
        mz = []
        intensity = []
    # assert len(mz) == len(intensity)
    return mz, intensity


def divide_by_fileids(all_fileid_i, train_fileids, test_fileids, val_fileids, all_bin_intesities_i, all_resistancies_i, i):
    train_idx = np.in1d(all_fileid_i,train_fileids)
    test_idx = np.in1d(all_fileid_i,test_fileids)
    val_idx = np.in1d(all_fileid_i,val_fileids)
    X_train = all_bin_intesities_i[train_idx,:]
    X_test = all_bin_intesities_i[test_idx,:]
    X_val = all_bin_intesities_i[val_idx,:]
    y_train = all_resistancies_i[train_idx,i]
    y_test = all_resistancies_i[test_idx,i]
    y_val = all_resistancies_i[val_idx,i]
    return [X_train, X_test, X_val, y_train, y_test, y_val]


def bin_separation(mz, intensity, bin_size):
    # 2000 mz to 20000 mz
    binsteps = np.arange(2000,20000+int(bin_size),int(bin_size))
    histogram_intensities = np.zeros(len(binsteps))
    #np.histogram(mz, bins=binsteps)

    for i,bin in enumerate(binsteps):
        idx = np.where((mz > binsteps[i]) & (mz < binsteps[i]+int(bin_size)))
        intensity = np.transpose(intensity)
        histogram_intensities[i] = sum(intensity[idx[0]])
    return histogram_intensities


def bin_separation_new(mz, intensity, bin_size):
    # 2000 mz to 20000 mz
    binsteps = np.arange(2000,20000+int(bin_size),int(bin_size))
    index = np.digitize(np.array(mz),binsteps)
    histogram_intensities = [sum(list(compress(intensity, index == i))) for i in range(1,len(binsteps)+1)]
    return np.asarray(histogram_intensities)


def read_maldi(spectra_dir, resist_dir, bin_size):
    list_spectra = [g for g in os.listdir(spectra_dir) if not g.startswith('.')]
    all_bin_intesities = []
    all_resistancies = []
    all_fileid = []

    # go through all files in spectra directory
    for i,filename in enumerate(list_spectra):
        [mz, intensity] = read_single_spectra(spectra_dir, filename)

        # separate spectra into bins
        if len(mz) > 0:
            fileid = filename.split('-')[0]
            all_fileid.append(fileid)

            bin_intensities = bin_separation(mz, intensity, bin_size)
            # add single spectra to full dataset
            if all_bin_intesities == []:
                all_bin_intesities = bin_intensities[np.newaxis,:]
            else:
                all_bin_intesities = np.r_[all_bin_intesities, bin_intensities[np.newaxis,:]]

            
            # read in resistancy file
            [resistancy, names_reagents] = read_resist(resist_dir,fileid)

            if all_resistancies == []:
                all_resistancies = resistancy[np.newaxis,:]
            else:
                all_resistancies = np.r_[all_resistancies, resistancy[np.newaxis,:]]
    all_fileid = np.asarray(all_fileid)
    return [all_bin_intesities, all_resistancies, all_fileid, names_reagents]


def read_and_bin_preprocessed_spectra(spectra_dir, bin_size, id_lower=1, id_upper=10, verbose=False):
    list_spectra = [g for g in os.listdir(spectra_dir) if not g.startswith('.')]
    d_spectra = {}


    # go through all files in spectra directory
    for i,filename in enumerate(list_spectra):
        swapid = re.split('/',filename)[-1].replace('_spectraPreprocessed.txt', '')        

        # include for Aarau
        # swapid = '_'.join([swapid, re.split('/',filename)[-1].split('_')[1]])
        
        if verbose: print('current sample ID: {}'.format(swapid))

        # check constraints for spectra id length
        if len(swapid)<id_lower or len(swapid)>id_upper:
            if verbose: print('SamplesID length out of bounds {} and {}. continue.'.format(id_lower, id_upper))
            continue

        # read in data
        [mz, intensity] = read_single_spectra(spectra_dir, filename)
        if mz == [] or intensity == []:
            if verbose: print('Empty spectra file. continue.')
            continue

        # separate spectra into bins
        if len(mz) > 0:
            bin_intensities = bin_separation(mz, intensity, bin_size)

            # add single spectra to full dataset
            # if all_bin_intesities == []:
            #     all_bin_intesities = bin_intensities[np.newaxis,:]
            # else:
            #     all_bin_intesities = np.r_[all_bin_intesities, bin_intensities[np.newaxis,:]]

            d_spectra[swapid]=np.array(bin_intensities)
        else:
            if verbose: print('len(mz)!>0')
# 
    # all_bin_intesities = np.array(all_bin_intesities)
    # return [all_bin_intesities, all_fileid]
    return d_spectra


# def save_binned_data(name, bin_size, all_bin_intesities, all_species = None):
#     # 
#     # [all_bin_intesities, all_species] = utils.read_maldi_RKI(spectra_dir, bin_size)
#     # utils.save_binned_data('RKI_', bin_size, all_bin_intesities, all_species)
#     #

#     if all_species == None:
#         print("all_species empty")
#         all_species = np.repeat(0, all_bin_intesities.shape[0])
#     if type(all_bin_intesities) is not np.ndarray:
#         all_bin_intesities = np.asarray(all_bin_intesities)
#     if type(all_species) is not np.ndarray:
#         all_species = np.asarray(all_species)
#     assert type(all_bin_intesities) is np.ndarray
#     assert type(all_species) is np.ndarray
#     assert all_species.shape[0] == all_bin_intesities.shape[0]
#     print(all_species[0:3])
#     dataname = '/links/groups/borgwardt/Projects/maldi_tof_diagnostics/binned_spectra/' + name + str(bin_size) + '.txt'
#     phenoname = '/links/groups/borgwardt/Projects/maldi_tof_diagnostics/binned_spectra/' + name + str(bin_size) + '_pheno.txt'
#     np.savetxt(dataname, all_bin_intesities, delimiter=',') 
#     f = open(phenoname, 'w')
#     for item in all_species:
#         f.write("%s\n" % item)
    
#     #np.savetxt(phenoname, all_species, delimiter=',') 
#     return 


# def load_binned_data(name, bin_size):
#     data = {}
#     dataname = '/links/groups/borgwardt/Projects/maldi_tof_diagnostics/binned_spectra/' + name + str(bin_size) + '.txt'
#     phenoname = '/links/groups/borgwardt/Projects/maldi_tof_diagnostics/binned_spectra/' + name + str(bin_size) + '_pheno.txt'
#     all_bin_intesities = np.loadtxt(dataname, delimiter=',') 
#     with open(phenoname) as f:
#         for i, line in enumerate(f):
#             ph = line.strip('\n')
#             assert type(ph) is str
#             data[ph] = all_bin_intesities[i,:]
#     assert type(data) is dict
#     return data


def check_data(spectra, pheno, logging, min_num_classes=2, min_num_samples=30, min_num_samples_per_class=4, minority_class1=True):
    if len(np.unique(pheno)) < min_num_classes:
        logging.info('Only samples of one class for this chemical!')
        check = False
    elif spectra.shape[0] < min_num_samples:
        logging.info('Less than 30 samples left after removing NaNs')
        check = False    
    elif Counter(pheno)[0] < min_num_samples_per_class or Counter(pheno)[1] < min_num_samples_per_class:
        logging.info('Too few samples of one class provided.')
        check = False
    else:
        check=True


    # flip class labels if necessary
    if minority_class1==True:
        logging.info('Is label flipping necessary?')
        logging.info(Counter(pheno))
        if sum(pheno) > sum(1-pheno):
            pheno = 1-pheno
            logging.info('Flip!')
            logging.info(Counter(pheno))
    return check, spectra, pheno


def spectra_scale(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


#-----------------------------------------------------------------------------
# Load phenotype functions
#-----------------------------------------------------------------------------


def idres_metadata(csvname):
    #csvname = '/links/groups/borgwardt/Data/ms_diagnostics/201711-12_merge_IDRES.csv'
    SampleMetadata = namedtuple('SampleMetadata', ['tagesnr', 'genus', 'species', 'resist'], verbose=False)
    meta_dict = {}

    with open(csvname,'r') as f:
        ff = csv.reader(f)
        for j, row in enumerate(ff):
            if j==0:
                meta_dict['AB_names'] = list(row)[21:]
            else:
                tagesnr = list(row)[1]
                genus = list(row)[9]
                species = list(row)[4]
                resist = list(row)[21:]
                sample = SampleMetadata(tagesnr, genus, species, resist)
                meta_dict[tagesnr] = sample
    return meta_dict


def idres_metadata_BrID(csvname, verbose=False):
    SampleMetadata = namedtuple('SampleMetadata', ['BrukerID', 'list_AB', 'genus', 'species', 'resist'], verbose=False)
    meta_dict = {}
    multiplied_ID = set()

    n_pop = 0

    with open(csvname,'r') as f:
        ff = csv.reader(f, dialect=csv.excel)
        for j, row in enumerate(ff):
            if j==0:
                header = list(row)
                idx_Bruker = header.index('code')
                idx_species = header.index('Organism(best match)')
                idx_genus = header.index('GENUS')
                ab_names = list(row)[21:]
                if verbose: print('Names of antibiotics {}: '.format(ab_names))
            else:
                brukerID = list(row)[idx_Bruker]
                # if key is duplicated, remove
                if brukerID in meta_dict.keys() or brukerID in multiplied_ID:
                    multiplied_ID.add(brukerID)
                    if brukerID in meta_dict.keys(): 
                        meta_dict.pop(brukerID, None)
                        n_pop += 1
                    continue

                genus = list(row)[idx_genus]
                species = list(row)[idx_species]
                resist = list(row)[21:]
                sample = SampleMetadata(brukerID, ab_names, genus, species, resist)
                meta_dict[brukerID] = sample
    if verbose: print('{} spectra removed due to duplicated naming: {}'.format(n_pop, multiplied_ID))
    return meta_dict



def idres_metadata_validation(csvname):
    SampleMetadata = namedtuple('SampleMetadata', ['BrukerID', 'list_AB', 'genus', 'species', 'resist'], verbose=False)
    meta_dict = {}

    with open(csvname,'r') as f:
        ff = csv.reader(f , dialect=csv.excel)
        for j, row in enumerate(ff):
            if j==0:
                header = list(row)
                idx_Bruker = header.index('sampleID')
                idx_species = header.index('species')
                idx_genus = header.index('genus')
                
                idx_antibiotics = range(len(header))
                idx_antibiotics.remove(idx_Bruker)
                idx_antibiotics.remove(idx_species)
                idx_antibiotics.remove(idx_genus)

                ab_names = [header[i] for i in idx_antibiotics]
                # meta_dict['AB_names'] = ab_names
            else:
                # brukerID = list(row)[idx_Bruker].replace('-','') # necessary for Aarau
                brukerID = list(row)[idx_Bruker]
                genus = list(row)[idx_genus]
                species = list(row)[idx_species]
                resist = [list(row)[i] for i in idx_antibiotics]
                sample = SampleMetadata(brukerID, ab_names, genus, species, resist)
                meta_dict[brukerID] = sample
    return meta_dict

def idres_metadata_Aarau(csvname):
    SampleMetadata = namedtuple('SampleMetadata', ['BrukerID', 'genus', 'species', 'resist'], verbose=False)
    meta_dict = {}

    with open(csvname,'r') as f:
        ff = csv.reader(f , dialect=csv.excel)
        for j, row in enumerate(ff):
            if j==0:
                header = list(row)
                idx_Bruker = header.index('sampleID')
                idx_species = header.index('species')
                idx_genus = header.index('genus')
                
                idx_antibiotics = range(len(header))
                idx_antibiotics.remove(idx_Bruker)
                idx_antibiotics.remove(idx_species)
                idx_antibiotics.remove(idx_genus)

                meta_dict['AB_names'] = [header[i] for i in idx_antibiotics]
            else:
                brukerID = list(row)[idx_Bruker]
                genus = list(row)[idx_genus]
                species = list(row)[idx_species]
                resist = [list(row)[i] for i in idx_antibiotics]
                sample = SampleMetadata(brukerID, genus, species, resist)
                meta_dict[brukerID] = sample
    return meta_dict


def read_resist(resist_dir,fileid):
    path = os.path.join(resist_dir, '50_Klebsiella_biochem_ABres.xlsx')
    wb = xlrd.open_workbook(path, 'r')
    sheet = wb.sheet_by_index(0)
    #file = open(os.path.join(resist_dir, '50_Klebsiella_biochem_ABres.txt'), 'r')
    #lines = file.readlines()
    resistance = []
    curr_row = 0

    names_unicode = sheet.row_values(0, start_colx=2, end_colx=None)
    names = [s.encode('utf-8') for s in names_unicode]
    names.insert(0, sheet.cell(0, 0).value.encode('utf-8'))

    num_rows = sheet.nrows - 1
    while curr_row < num_rows:
        # first go through all ids and find the matching lineid
        curr_row += 1
        cell = sheet.cell(curr_row, 1)
        lineid = int(cell.value)
        if lineid == int(fileid):
            #include species first
            resistance.append(str(sheet.cell(rowx=curr_row,colx=0).value))
            for j in range(2,sheet.ncols):
                resistance.append(str(sheet.cell(rowx=curr_row,colx=j).value))

    resistance = np.array(resistance)
    # encode chemical and antibiotic resistance
    resistance[resistance == 'n'] =     '0'
    resistance[resistance == '(n)'] =     '0'
    resistance[resistance == 'p'] =     '1'
    resistance[resistance == 'R'] =     '0'
    resistance[resistance == 'S'] =     '1'
    resistance[resistance == 'I'] =     '0'#wrong   #TODO clean up
    resistance[resistance == 'I*'] =     '0'#wrong
    resistance[resistance == '-'] =     'NaN'#wrong
    resistance[resistance == ''] =         'NaN'#wrong
    resistance[resistance == ' '] =     'NaN'#wrong
    resistance[resistance == 'U'] =     'NaN'#wrong
    resistance[resistance == 'pneumoniae'] =      '1'#multiclass
    resistance[resistance == 'variicola'] =     '2'#multiclass
    resistance[resistance == 'oxytoca'] =         '3'#multiclass
    resistance[resistance == 'michiganensis'] = '4'#multiclass
    resistance[resistance == 'grimontii'] =     '5'#multiclass
    resistance = resistance.astype(float)
    return resistance, names


def RSI_encoder(matrix):
    matrix = matrix.astype('|S4')
    matrix[matrix == '3'] = 'R' #resistent
    matrix[matrix == '2'] = 'R' #resistent
    matrix[matrix == '1'] = 'R' #maessig empfindlich
    matrix[matrix == '0'] = 'S' #empfindlich


    matrix[matrix == 'S*'] =     1
    matrix[matrix == 'S'] =     1
    matrix[matrix == 'S(2)'] =     1
    matrix[matrix == 'negative'] = 1
    matrix[matrix == 'NEG'] = 1

    matrix[matrix == '-1'] =     float('NaN')
    matrix[matrix == '-'] =     float('NaN')
    matrix[matrix == ''] =         float('NaN')
    matrix[matrix == ' '] =     float('NaN')
    matrix[matrix == 'U'] =     float('NaN')
    matrix[matrix == 'nan'] =     float('NaN')
    matrix[matrix == 'R(1)- I(1)'] = float('NaN')
    matrix[matrix == 'I(1)- S(1)'] = float('NaN')
    matrix[matrix == 'R(1)- S(1)'] = float('NaN')
    matrix[matrix == 'R(1)- I(1)- S(1)'] = float('NaN')
    matrix[matrix == 'N/R'] = float('NaN')
    matrix[matrix == 'BLEE'] = float('NaN')
    matrix[matrix == 'EBL?'] = float('NaN')
    
    matrix[matrix == 'R'] = 0
    matrix[matrix == 'R(1)'] = 0
    matrix[matrix == 'R(2)'] = 0
    matrix[matrix == 'I'] = 0
    matrix[matrix == 'I(1)'] = 0
    matrix[matrix == 'I*'] = 0
    matrix[matrix == 'RES'] = 0
    matrix[matrix == 'INT'] = 0
    matrix[matrix == 'positive'] = 0
    matrix[matrix == 'POS'] = 0


    # print np.unique(matrix, return_counts=True)
    matrix = matrix.astype(float)
    # set(x).issubset([float('nan'),1.,0.])  # doesn't work with nans yet
    return matrix


def min_num_class_memb(X, y, min_num):
    # multiclass setting, remove samples classes having fewer members than min_num 
    assert np.shape(X)[0] == np.shape(y)[0]
    if not isinstance(y, list):
        y = list(y)
    idx_remove = []
    dict_counts = Counter(y)
    remove_keys = [key for key,val in dict_counts.items() if val < min_num]
    [idx_remove.append(False) if s in remove_keys else idx_remove.append(True) for s in y]
    y_min_num = np.asarray(list(compress(y, idx_remove)))
    X_min_num = X[idx_remove,:]
    return X_min_num, y_min_num


#-----------------------------------------------------------------------------
# Performance measure functions
#-----------------------------------------------------------------------------


def binary_performance_measures(clf, X_test, y_test):
    roc_auc = round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]), 3)
    #fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    prec, rec, thresholds = precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])
    # pr_auc = auc(prec, rec, reorder=True)
    pr_auc = average_precision_score(y_test, clf.predict_proba(X_test)[:, 1], average='weighted')
    tn, fp, fn, tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()
    recall = round(tp / float(tp+fn), 3)
    precision = round(tp / float(tp+fp), 3)
    accuracy = round((tp+tn) / float(tn+fp+fn+tp), 3)
    f1 = round(f1_score(y_test, clf.predict(X_test), average='binary'), 3)
    return [roc_auc, pr_auc, accuracy, precision, recall, f1, tn, fp, fn, tp]


def multiclass_performance_measures(clf, X_test, y_test):
    f1 = round(f1_score(y_test, clf.predict(X_test), average='micro'), 3)
    #roc_auc = round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]), 3)
    #fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    # prec, rec, thresholds = precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])
    # pr_auc = auc(prec, rec, reorder=True)
    # pr_auc = average_precision_score(y_test, clf.predict_proba(X_test)[:, 1], average='weighted')
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
    return [f1, conf_matrix]


def results_sample(y_true,y_pred,sample_id):
    y_true = list(y_true)
    y_pred = list(y_pred)
    sample_id = list(sample_id)
    assert np.shape(y_true)==np.shape(y_pred)==np.shape(sample_id)
    y_match = [1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))]
    assert np.shape(y_true)==np.shape(y_pred)==np.shape(sample_id)==np.shape(y_match)
    results = np.column_stack(([y_true,y_pred,sample_id,y_match]))
    return results

#-----------------------------------------------------------------------------
# Miscellaneous functions
#-----------------------------------------------------------------------------

def find_common_pheno_set(list_MaldiAI_exp_types):
    list_phenotype_sets = []
    for exp_types in list_MaldiAI_exp_types:
        current_exp_type_phenotypes = [l.phenotype for l in exp_types]
        list_phenotype_sets.append(set(current_exp_type_phenotypes))
    
    for i in range(1,len(list_phenotype_sets)):
        if i == 1:
            common_set = list_phenotype_sets[0]&list_phenotype_sets[i]
        else:
            common_set = common_set&list_phenotype_sets[i]

    return common_set


def count_elements(array):
    counts = np.unique(array, return_counts=True)
    zipped = zip(counts[0],counts[1])
    return sorted(zipped, key=operator.itemgetter(1), reverse=True)


    
def count_values_per_key(dictionary):
    list_len = []
    for key in dictionary.keys():
        list_len.append(len(dictionary[key]))
    return np.unique(list_len, return_counts=True)



def get_antibiotics_name_matching(match_from='LIESTAL', match_to='USB'):
    assert match_from in ['LIESTAL','USB','Viollier','Aarau','Madrid']
    assert match_to in ['LIESTAL','USB','Viollier','Aarau','Madrid']
    #csvname = '/links/groups/borgwardt/Data/ms_diagnostics/validation/AB-matching.csv'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    csvname = dir_path+'/files/AB-matching.csv'

    with open(csvname,'r') as f:
        ff = csv.reader(f, delimiter=',', dialect=csv.excel)
        list_antibiotics = []
        
        d_naming = ddict(list)

        for j, row in enumerate(ff):
            if j==0:
                print row
                inidx = row.index(match_from)
                outidx = row.index(match_to)
            else:
                inname = row[inidx]
                outname = row[outidx]
                if inname == '':
                    continue
                else:
                    d_naming[inname] = outname
    return d_naming


def anglicize_labels(label):
    if label == 'Amoxicillin...Clavulansaeure.bei.unkompliziertem.HWI': return 'Amoxicillin-Clavulans.unkompl.HWI'
    elif label == 'Ampicillin...Amoxicillin': return 'Ampicillin-Amoxicillin'
    elif label == 'Piperacillin...Tazobactam': return 'Piperacillin-Tazobactam'
    elif label == 'Amoxicillin...Clavulansaeure': return 'Amoxicillin-Clavulanic acid'
    elif label == 'Fusidinsaeure': return 'Fusidic acid'
    elif label == 'Ceftazidim.1': return 'Ceftazidime'
    elif label == 'Ceftazidim.Avibactam': return 'Ceftazidime-Avibactam'
    elif label == 'X5.Fluorocytosin': return '5-Fluorocytosin'
    elif label == 'Fosfomycin.Trometamol': return 'Fosfomycin-Trometamol'
    elif label == 'Ceftolozan...Tazobactam': return 'Ceftolozane-Tazobactam'
    elif label == 'Cefepim': return 'Cefepime'
    elif label == 'Posaconazol': return 'Posaconazole'
    elif label == 'Tigecyclin': return 'Tigecycline'
    elif label == 'Cefpodoxim': return 'Cefpodoxime'
    elif label == 'Ceftobiprol': return 'Ceftobiprole'
    elif label == 'Fluconazol': return 'Fluconazole'
    elif label == 'Cefuroxim': return 'Cefuroxime'
    elif label == 'Tetracyclin': return 'Tetracycline'
    elif label == 'Ceftriaxon': return 'Ceftriaxone'
    elif label == 'Itraconazol': return 'Itraconazole'
    elif label == 'Cotrimoxazol': return 'Trimethoprim-Sulfamethoxazole'
    elif label == 'Minocyclin': return 'Minocycline'    
    elif label == 'Voriconazol': return 'Voriconazole'
    elif label == 'Metronidazol': return 'Metronidazole'
    elif label == 'Aminoglykoside': return 'Aminoglycosides'    
    elif label == 'Chinolone': return 'Quinolones'
    elif label == 'Doxycyclin': return 'Doxycycline'
    elif label == 'Cefixim': return 'Cefixime'
    else: return label

def shorten_labels(label):
    if label == 'Piperacillin-Tazobactam': return 'Pip.-Tazo.'
    elif label == 'Amoxicillin-Clavulanic acid': return 'Amox.-Clav.'
    else: return label


def PosRatio_pval(Dataset_ref, Dataset_comp):
    # check if Datasets in right format
    Dataset_ref.check_is_singletask()
    Dataset_ref.check_contains_nans()
    Dataset_comp.check_is_singletask()
    Dataset_comp.check_contains_nans()
    
    # calculate frequencies
    f_obs_ref = np.sum(Dataset_ref.y == 1)
    f_obs_comp = np.sum(Dataset_comp.y == 1)


    n_P_total = np.sum(Dataset_ref.y == 1)+np.sum(Dataset_comp.y == 1)
    n_total = len(Dataset_ref.y)+len(Dataset_comp.y)
    p_exp = n_P_total/float(n_total)

    f_exp_ref = p_exp*len(Dataset_ref.y)
    f_exp_comp = p_exp*len(Dataset_comp.y)
    
    # chi-squared test
    f_obs = [f_obs_ref, f_obs_comp]
    f_expect = [f_exp_ref, f_exp_comp]
    chi_results = chisquare(f_obs, f_exp=f_expect)

    return chi_results.pvalue


def list_phenotype(list_Experiments):
    return [e.phenotype for e in list_Experiments]


def remove_ab_from_list(input_list, order_phenotype, remove_phenotype_list):
    remove_idx = []
    for ab in remove_phenotype_list:
        if ab in order_phenotype:
            remove_idx.append(order_phenotype.index(ab))
    return [elem for i, elem in enumerate(input_list) if i not in remove_idx]


def remove_ab_from_list_index(order_phenotype, remove_phenotype_list):
    remove_idx = []
    for ab in remove_phenotype_list:
        if ab in order_phenotype:
            remove_idx.append(order_phenotype.index(ab))
    return remove_idx


def calc_pvalue(Y_exp1, Y_exp2):
    '''
    Takes two input matrics Y_exp1 and Y_exp2, where each column represent the results of repeated experiments for exp1 and exp2 respectively.
    Calculates the p-value of a T-test for the means of two independent samples of scores.
    '''

    assert Y_exp1.shape == Y_exp2.shape, 'Input matrices have different lengths.'

    list_pvals = []
    for i in range(Y_exp1.shape[1]):
        _, p_val = ttest_ind(Y_exp1[:, i], Y_exp2[:, i], equal_var=False)
        list_pvals.append(p_val)

    assert len(list_pvals) == Y_exp1.shape[1]
    return list_pvals



def count_clf(X_clf, clf_type='LogisticRegression'):
    '''
    Takes input matrix n_exp x n_abs with strings of classifiers. Counts how often the string of a clf appears in one antibiotic, and return values in list.
    '''
    assert clf_type=='LogisticRegression' or clf_type=='RandomForest'
    return list(np.sum(np.array(l) == clf_type, axis=0))


def count_species_class_pairs(Dataset):
    '''
    Takes Dataset object and returns a list of unique species-class combinations in the Dataset, along with its counts.
    '''
    species_class_tuples = zip(Dataset.sample_species, Dataset.y)

    counts = Counter(species_class_tuples) 
    unique_pairs = list(set(species_class_tuples)) 
      
    list_counts = []
    for i in unique_pairs: 
        list_counts.append(counts[i])
    return unique_pairs, list_counts


#-----------------------------------------------------------------------------
# s1 utils
#-----------------------------------------------------------------------------

def get_s1_spectra_avg(num_reps, pickle_dir='/links/groups/borgwardt/Projects/maldi_tof_diagnostics/pub1/pickles/'):     
    list_roc_spectra = []
    list_prauc_spectra = []

    with open(pickle_dir+'s1_AMRpred_baseline/s1_AMRpred_baseline_rep0.pkl', 'rb') as infile:
        dp = pickle.load(infile) 
        list_datasets_spectra = dp['list_datasets']
        list_test_datasets_spectra = dp['list_test_datasets']
        list_experiments_spectra = dp['list_experiments']  
    l_pheno = list_phenotype(list_experiments_spectra)
    
    # --------------------------
    # get repetition values from compressed dataset
    for r in range(num_reps):
        with open(pickle_dir+'s1_AMRpred_baseline/list_metrics/list_metric_s1_AMRpred_baseline_rep{}.pkl'.format(r), 'rb') as infile:
            dp = pickle.load(infile)   
            list_roc_spectra.append(dp['list_rocauc_values'])
            list_prauc_spectra.append(dp['list_prauc_values'])
            assert dp['l_pheno'] == l_pheno
            infile.close()
        
    d_out = {
    'l_pheno': l_pheno,
    'list_roc_spectra': list_roc_spectra,
    'list_prauc_spectra': list_prauc_spectra,
    'list_std_rocauc_spectra': list(np.std(np.array(list_roc_spectra), axis=0)),
    'list_mean_rocauc_spectra': list(np.mean(np.array(list_roc_spectra), axis=0)),
    'list_std_prauc_spectra': list(np.std(np.array(list_prauc_spectra), axis=0)),
    'list_mean_prauc_spectra': list(np.mean(np.array(list_prauc_spectra), axis=0)),
    'list_datasets_spectra': list_datasets_spectra,
    'list_test_datasets_spectra': list_test_datasets_spectra,
    'list_experiments_spectra': list_experiments_spectra,
    }
    return d_out


def get_s1_species_avg(num_reps, pickle_dir='/links/groups/borgwardt/Projects/maldi_tof_diagnostics/pub1/pickles/'):
    list_roc_species = []
    list_prauc_species = []

    with open(pickle_dir+'s1_AMRpred_baseline/s1_AMRpred_from_species_rep0.pkl', 'rb') as infile:
        dp = pickle.load(infile) 
        list_datasets_species = dp['list_datasets']
        list_test_datasets_species = dp['list_test_datasets']
        list_experiments_species = dp['list_experiments']  
    l_pheno = list_phenotype(list_experiments_species)

    # --------------------------
    # get repetition values from compressed dataset
    for r in range(num_reps):
        with open(pickle_dir+'s1_AMRpred_baseline/list_metrics/list_metric_s1_AMRpred_from_species_rep{}.pkl'.format(r), 'rb') as infile:
            dp = pickle.load(infile)   
            list_roc_species.append(dp['list_rocauc_values'])
            list_prauc_species.append(dp['list_prauc_values'])
            assert dp['l_pheno'] == l_pheno
            infile.close()

    d_out = {
    'l_pheno': l_pheno,
    'list_roc_species': list_roc_species,
    'list_prauc_species': list_prauc_species,
    'list_std_rocauc_species': list(np.std(np.array(list_roc_species), axis=0)),
    'list_mean_rocauc_species': list(np.mean(np.array(list_roc_species), axis=0)),
    'list_std_prauc_species': list(np.std(np.array(list_prauc_species), axis=0)),
    'list_mean_prauc_species': list(np.mean(np.array(list_prauc_species), axis=0)),
    'list_datasets_species': list_datasets_species,
    'list_test_datasets_species': list_test_datasets_species,
    'list_experiments_species': list_experiments_species,
    }
    return d_out


def conf_matrix_threshold(list_ytrue, list_yprob, threshold):
    list_ypredict = [1 if prob >= threshold else 0 for prob in list_yprob]
    TN, FP, FN, TP = confusion_matrix(list_ytrue, list_ypredict, labels=[0,1]).ravel()
    return TN, FP, FN, TP


def get_auccurve_values_4Bokeh(list_Experiments, list_test_datasets, site='USB', species='all'):

    dd = ddict(list)

    for i, exp in enumerate(list_Experiments):

        data = list_test_datasets[i]

        list_ytrue = data.y
        list_yprob = exp.estimator.predict_proba(data.X)[:, 1]

        # phenotype can be in 'antibiotic' or 'antibiotic-species' format
        if len(exp.phenotype.split('-'))>1:
            ab_name = anglicize_labels(exp.phenotype.split('-')[0])
        else:
            ab_name = anglicize_labels(exp.phenotype)
        assert len(list_ytrue) == len(list_yprob)


        # ------------
        # ROC curve
        # ------------
        fpr, tpr, thresholds1 = roc_curve(list_ytrue, list_yprob, pos_label=1)
        AUROC = round(roc_auc_score(list_ytrue, list_yprob), 3)

        # ------------
        # PRAUC-1 curve
        # ------------
        precision, recall, thresholds2 = precision_recall_curve(list_ytrue, list_yprob)
        AUPRC = round(average_precision_score(list_ytrue, list_yprob, average='weighted'), 3)

        # ------------
        # VME curve
        # ------------
        vme, me_inv, thresholds3 = vme_auc_curve(list_ytrue, list_yprob)
        me = 1-me_inv
        AUVME = round(vme_auc_score(list_ytrue, list_yprob),3)
        
        # ------------
        # TN, FP, FN, TP
        # ------------

        TN1, FP1, FN1, TP1 = [], [], [], []
        TN2, FP2, FN2, TP2 = [], [], [], []
        TN3, FP3, FN3, TP3 = [], [], [], []

        for thresh in thresholds1:
            TN, FP, FN, TP = conf_matrix_threshold(list_ytrue, list_yprob, thresh)
            TN1.append(TN)
            FP1.append(FP)
            FN1.append(FN)
            TP1.append(TP)
        dd['TN1'].append(TN1)
        dd['FP1'].append(FP1)
        dd['FN1'].append(FN1)
        dd['TP1'].append(TP1)

        for thresh in thresholds2:
            TN, FP, FN, TP = conf_matrix_threshold(list_ytrue, list_yprob, thresh)
            TN2.append(TN)
            FP2.append(FP)
            FN2.append(FN)
            TP2.append(TP)
        dd['TN2'].append(TN2)
        dd['FP2'].append(FP2)
        dd['FN2'].append(FN2)
        dd['TP2'].append(TP2)

        for thresh in thresholds3:
            TN, FP, FN, TP = conf_matrix_threshold(list_ytrue, list_yprob, thresh)
            TN3.append(TN)
            FP3.append(FP)
            FN3.append(FN)
            TP3.append(TP)
        dd['TN3'].append(TN3)
        dd['FP3'].append(FP3)
        dd['FN3'].append(FN3)
        dd['TP3'].append(TP3)
       

        # ------------
        # build dict
        # ------------
        dd['false_positive_rate'].append(list(fpr))
        dd['true_positive_rate'].append(list(tpr))
        dd['thresholds1'].append(list(thresholds1))

        dd['precision'].append(list(precision))
        dd['recall'].append(list(recall))
        dd['thresholds2'].append(list(thresholds2))

        dd['very_major_error'].append(list(vme))
        dd['major_error'].append(list(me))
        dd['thresholds3'].append(list(thresholds3))

        dd['antibiotic'].extend([ab_name])
        dd['AUROC'].extend([AUROC])
        dd['AUPRC'].extend([AUPRC])
        dd['AUVME'].extend([AUVME])

        dd['site'].extend([site])
        dd['species'].extend([species])   
    
    return dd




def vme_auc_curve(y_true, y_prob, pos_label=1, sample_weight=None):
    fps, tps, thresholds = _binary_clf_curve(y_true, y_prob, pos_label=pos_label, sample_weight=sample_weight)
    vme = fps / fps[-1]
    me =  (tps[-1] -  tps) / tps[-1]
    return vme, 1-me, thresholds

def vme_auc_score(y_true, y_prob, pos_label=1, sample_weight=None):
    vme, me_inv, thresholds = vme_auc_curve(y_true, y_prob, pos_label=1, sample_weight=None)
    vme_auc_score = auc(vme, me_inv)
    return vme_auc_score


# ----------------
# colors
# ----------------



col_list = [u'orangered', u'purple', u'royalblue', u'olivedrab', u'orange', u'violet',  u'red', u'turquoise', u'chartreuse', u'black', u'saddlebrown', u'salmon', u'mediumvioletred', u'seagreen', u'skyblue', u'slateblue', u'darkgrey', u'springgreen', u'teal', u'tomato', u'peru', u'yellowgreen', u'aqua', u'aquamarine', u'blue', u'blueviolet', u'brown', u'burlywood', u'cadetblue', u'chocolate', u'coral', u'cornflowerblue', u'crimson', u'darkblue', u'darkcyan', u'darkgoldenrod', u'darkgreen', u'darkgrey', u'darkmagenta', u'hotpink', u'darkolivegreen', u'yellow']

col_map = {
    'Ceftriaxon': col_list[0],
    'Oxacillin': col_list[1],
    'Amoxicillin...Clavulansaeure': col_list[2],
    'Meropenem': col_list[3],
    'Piperacillin...Tazobactam': col_list[4],
    'Ciprofloxacin': col_list[5],
    'Colistin': col_list[6],
    'Fluconazol': col_list[7],
    'Fusidinsaeure': col_list[9],
    'Cefepim': col_list[8],
    'Penicillin': col_list[10],
    'Imipenem': col_list[11],
    'Gentamicin': col_list[12],
    'Tetracyclin': col_list[13],
    'Vancomycin': col_list[14],
    'Clindamycin': col_list[15],
    'Nitrofurantoin': col_list[16],
    'Tigecyclin': col_list[17],
    'Tobramycin': col_list[18],    
    'Amikacin': col_list[19],
    'Amoxicillin': col_list[20],
    'Ampicillin...Amoxicillin': col_list[21],
    'Anidulafungin': col_list[22],
    'Aztreonam': col_list[23],
    'Caspofungin': col_list[24],
    'Cefazolin': col_list[25],
    'Cefpodoxim': col_list[26],
    'Ceftazidim': col_list[27], 
    'Cefuroxim': col_list[28], 
    'Cotrimoxazol': col_list[29],
    'Daptomycin': col_list[30],
    'Ertapenem': col_list[31],
    'Erythromycin': col_list[32],
    'Fosfomycin.Trometamol': col_list[33],
    'Itraconazol': col_list[34],
    'Levofloxacin': col_list[35],
    'Micafungin': col_list[36],
    'Norfloxacin': col_list[37],
    'Rifampicin': col_list[38],
    'Teicoplanin': col_list[39],
    'Voriconazol': col_list[40],
    'X5.Fluorocytosin': col_list[41]
    }

col_map_anglicize = {}
for key in col_map.keys():
    col_map_anglicize[anglicize_labels(key)] = col_map[key]

# col_map_anglicize = {
#     'Ceftriaxone': col_list[0],
#     'Oxacillin': col_list[1],
#     'Amoxicillin-Clavulanic acid': col_list[2],
#     'Meropenem': col_list[3],
#     'Piperacillin-Tazobactam': col_list[4],
#     'Ciprofloxacin': col_list[5],
#     'Colistin': col_list[6],
#     'Fluconazole': col_list[7],
#     'Fusidic acid': col_list[9],
#     'Cefepime': col_list[8],
#     'Penicillin': col_list[10],
#     'Imipenem': col_list[11],
#     'Gentamicin': col_list[12],
#     'Tetracycline': col_list[13],
#     'Vancomycin': col_list[14],
#     'Clindamycin': col_list[15],
#     'Nitrofurantoin': col_list[16],
#     'Tigecycline': col_list[17],
#     'Tobramycin': col_list[18],    
#     'Amikacin': col_list[19],
#     'Amoxicillin': col_list[20],
#     'Ampicillin-Amoxicillin': col_list[21],
#     'Anidulafungin': col_list[22],
#     'Aztreonam': col_list[23],
#     'Caspofungin': col_list[24],
#     'Cefazolin': col_list[25],
#     'Cefpodoxime': col_list[26],
#     'Ceftazidim': col_list[27], 
#     'Cefuroxime': col_list[28], 
#     'Trimethoprim-Sulfamethoxazole': col_list[29],
#     'Daptomycin': col_list[30],
#     'Ertapenem': col_list[31],
#     'Erythromycin': col_list[32],
#     'Fosfomycin-Trometamol': col_list[33],
#     'Itraconazole': col_list[34],
#     'Levofloxacin': col_list[35],
#     'Micafungin': col_list[36],
#     'Norfloxacin': col_list[37],
#     'Rifampicin': col_list[38],
#     'Teicoplanin': col_list[39],
#     'Voriconazole': col_list[40],
#     '5-Fluorocytosin': col_list[41]
#     }


cmap_site = {
    'USB': 'cornflowerblue',
    'KSBL': 'coral',
    'KSA': 'forestgreen',
    'Viollier': 'firebrick'
}
