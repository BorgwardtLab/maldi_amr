# -*- coding: utf-8 -*-
# @Author: weisc
# @Date:   2019-08-30 16:28:01
# @Last Modified by:   weisc
# @Last Modified time: 2019-09-18 17:28:12


import pickle
import numpy as np
import h5py
# from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
from MaldiAI_pub1.utils import idres_metadata_BrID, read_and_bin_preprocessed_spectra, RSI_encoder
from MaldiAI_pub1.datasets import Dataset


def read_MaldiAIobject(datanames, 
                       from_pickle=True, 
                       pickle_dir='/links/groups/borgwardt/Projects/maldi_tof_diagnostics/pub1/pickles/MaldiAIobjects/', 
                       bin_size=20):
    if not isinstance(datanames, list):
        datanames = [datanames]
    
    if from_pickle:
        with open(pickle_dir+'MaldiAIobject_{}_{}.pkl'.format(','.join(datanames),bin_size), 'rb') as infile:
            X = pickle.load(infile)
            y = pickle.load(infile)
            list_AB = pickle.load(infile)
            sample_ids = pickle.load(infile)   
            genus = pickle.load(infile)
            species = pickle.load(infile) 
    else:
        dat = DataPreprocessing(datanames, bin_size=bin_size)
        X = dat.match_int
        y = dat.match_resist
        list_AB = dat.list_AB
        sample_ids = dat.match_ID
        genus = dat.match_genus
        species = dat.match_species


    data = Dataset(X, y, list_AB, sample_ids, genus, species)
    data.check()
    return data


def read_MaldiAIhdf5(datanames, 
                   pickle_dir='/links/groups/borgwardt/Projects/maldi_tof_diagnostics/pub1/pickles/MaldiAIobjects/'):
    
    if isinstance(datanames, list):
        datanames = datanames[0]

    with h5py.File(pickle_dir+'MaldiAIobject_{}_allbins.hdf5'.format(datanames), 'r') as f:
        X = f['X'][:,:]
        y = f['y'][:,:]
        list_AB = list(f['tasks'][:])
        sample_ids = f['sample_ids'][:]
        sample_genera = f['sample_genera'][:]
        sample_species = f['sample_species'][:]

    y = map(lambda x: max(x,0),y)
    print('check if y is the same!!')
    data = Dataset(X, y, list_AB, sample_ids, sample_genera, sample_species)
    data.check()
    return data


def write_MaldiAIobject(datanames, 
                        pickle_dir='/links/groups/borgwardt/Projects/maldi_tof_diagnostics/pub1/pickles/MaldiAIobjects/', 
                        bin_size=20, 
                        verbose=False):
    if not isinstance(datanames, list):
        datanames = [datanames]
    dat = DataPreprocessing(datanames, bin_size=bin_size, verbose=verbose)
    X = dat.match_int
    y = dat.match_resist
    list_AB = dat.list_AB
    sample_ids = dat.match_ID
    genus = dat.match_genus
    species = dat.match_species

    with open(pickle_dir+'MaldiAIobject_{}_{}.pkl'.format(','.join(datanames), bin_size), 'wb') as outfile:
        pickle.dump(X, outfile)
        pickle.dump(y, outfile)
        pickle.dump(list_AB, outfile)
        pickle.dump(sample_ids, outfile)
        pickle.dump(genus, outfile)
        pickle.dump(species, outfile)
    return



class DataPreprocessing():

    def __init__(self, datanames, bin_size=20, verbose=False):
        if isinstance(datanames, list):
            self.datanames = datanames
        else:
            self.datanames = [datanames]
        self.bin_size = bin_size
        self.set_ab = set()
        self.spectra_dict = {}
        self.AMR_dict = {}
        self.verbose = verbose

        self.__get_MALDI_MS()
        self.__get_AMR()
        self.__match()



    def __get_MALDI_MS(self, 
                       spectra_dir='/links/groups/borgwardt/Projects/maldi_tof_diagnostics/preprocessed_spectra/'):

        for name in self.datanames:
            if self.verbose: print('\nProcessing MALDI MS spectra from {}...'.format(name))

            input_dir=spectra_dir+'{}'.format(name)
            d_binned_spectra = read_and_bin_preprocessed_spectra(input_dir, self.bin_size, id_lower=20, id_upper=100, verbose=self.verbose)

            # remove common spectra keys between datanames
            current = self.spectra_dict.keys()
            new = d_binned_spectra.keys()
            common_IDs = set(current).intersection(set(new))
            if self.verbose: print('\nNumber of common sampleIDs found between datasets: {}'.format(len(common_IDs)))
            for common_id in list(common_IDs):
                d_binned_spectra.pop(common_id, None)
                self.spectra_dict.pop(common_id, None)

            self.spectra_dict.update(d_binned_spectra)
            self.spectra_ID = self.spectra_dict.keys()



    def __get_AMR(self):

        for name in self.datanames:
            if name in ['2015', '2016', '2017', '2018']:
                if self.verbose: print('\nProcessing AMR profiles from {}...'.format(name))
                resist_dir='/links/groups/borgwardt/Data/ms_diagnostics/USB/IDRES_new_Caroline/{}/{}_pub1_IDRES.csv'.format(name, name)
                new_dict = idres_metadata_BrID(resist_dir, verbose=self.verbose)

                current_id = self.AMR_dict.keys()
                new_id = new_dict.keys()
                common_IDs = set(current_id).intersection(set(new_id))
                if self.verbose: print('\nNumber of common sampleIDs found between datasets: {}'.format(len(common_IDs)))

                for common_id in common_IDs:
                    new_dict.pop(common_id, None)
                    self.AMR_dict.pop(common_id, None)

                self.AMR_dict.update(new_dict)
                self.AMR_ID = self.AMR_dict.keys()

                if self.set_ab == set():
                    # list_AB the same for all entries in self.AMR_dict
                    self.set_ab = set(self.AMR_dict[self.AMR_ID[0]].list_AB)
                else:
                    present_ab = self.set_ab
                    new_ab = set(self.AMR_dict[self.AMR_ID[0]].list_AB)
                    # this list is not equal to the order within the AMR_dict, will be matched in self.__match()
                    self.set_ab = present_ab.intersection(new_ab)
                    if self.verbose: print('\nAntibiotic names without match: {} {}'.format(new_ab-present_ab, present_ab-new_ab))



            ##
            ## still needs to be adjusted and cleaned
            ##
            # if name in ['KSBL','KSA','Viollier']:
            #   print(name)
            #   if name=='Aarau_PQN':
            #       resist_dir='/links/groups/borgwardt/Data/ms_diagnostics/validation/Aarau/Aarau_IDRES_converted.csv'
            #   elif name=='Madrid':
            #       resist_dir='/links/groups/borgwardt/Data/ms_diagnostics/validation/Barcelona/Barcelona_IDRES_converted.csv'
            #   else:
            #       resist_dir='/links/groups/borgwardt/Data/ms_diagnostics/validation/{}/{}_IDRES_converted.csv'.format(name,name)

                
            #   self.AMR_dict.update(utils.idres_metadata_validation(resist_dir))
            #   self.AMR_ID = self.AMR_dict.keys()

            #   if self.list_AB == []:
            #       self.list_AB = self.AMR_dict[self.AMR_ID[0]].list_AB
            #   else:
            #       present_ab = self.list_AB
            #       new_ab = self.AMR_dict[self.AMR_ID[0]].list_AB
            #       self.list_AB = list(set(present_ab).union(set(new_ab)))
            #       print('\nAntibiotic names without match: {} {}'.format(set(new_ab)-set(present_ab), set(new_ab)-set(present_ab)))


    def __match(self):
        match_int = []
        match_resist = []
        match_genus = []
        match_species = []
        match_ID = []

        # get list of sampleIDs present in both metadata and spectral data
        match_ID_set = set(self.AMR_ID) & set(self.spectra_ID)
        if self.verbose: print('\nlength IDs AMR profiles: {}'.format(len(set(self.AMR_ID))))
        if self.verbose: print('\nlength IDs MALDI MS spectra: {}'.format(len(set(self.spectra_ID))))
        if self.verbose: print('\nnumber IDs that match between AMR profiles and MALDI MS spectra: {}'.format(len(match_ID_set)))
        nonmatch_tagesnr = set(self.AMR_ID) - set(self.spectra_ID)
        # if self.verbose: print('\nnonmatch IDs: {}'.format(list(nonmatch_tagesnr)[:10]))

        # make fixed reference list of antibiotics in the dataset
        antibiotics_ordered = list(self.set_ab)
        self.list_AB = antibiotics_ordered

        for sid in match_ID_set:
            md = self.AMR_dict[sid]
            match_int.append(self.spectra_dict[sid])
            match_genus.append(md.genus)
            match_species.append(md.species)
            match_ID.append(sid)

            current_ab = np.full_like(antibiotics_ordered,'nan', dtype='|S4')
            for i, ab in enumerate(md.list_AB):
                if ab in antibiotics_ordered:
                    idx = antibiotics_ordered.index(ab)
                    current_ab[idx] = md.resist[i]  
            match_resist.append(list(current_ab))

        self.match_resist = np.array(match_resist)
        self.match_resist = RSI_encoder(self.match_resist)
        self.match_int = np.array(match_int)
        if self.verbose: print('\ndim match_int: {}'.format(np.shape(self.match_int)))
        
        self.match_ID = np.array(match_ID)
        self.match_genus = np.array(match_genus)
        self.match_species = np.array(match_species)

