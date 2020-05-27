"""
Script to extract the following dataset statistics: 

    (1) The number of spectra in each DRIAMS dataset. 
        We define this number as the number of fid datafiles
    (2) The number of antimicrobial resistance labels in each DRIAMS
        dataset. 
"""

import glob
import os

import pandas as pd
from dotenv import load_dotenv
from maldi_learn.driams import load_driams_dataset
from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')
explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

list_sites = ['DRIAMS-A', 
              'DRIAMS-B', 
              'DRIAMS-C', 
              'DRIAMS-D', 
              'DRIAMS-E', 
              'DRIAMS-F']

map_raw_datapaths = {
        'DRIAMS-A': [
 '/links/groups/borgwardt/Data/ms_diagnostics/USB/2015-01-12-monthly_added_not_renamed',
 '/links/groups/borgwardt/Data/ms_diagnostics/USB/2016_01-12-monthly_added_not_renamed',
 '/links/groups/borgwardt/Data/ms_diagnostics/USB/spectra_folder_for_IDRES/2017',
 '/links/groups/borgwardt/Data/ms_diagnostics/USB/spectra_folder_for_IDRES/2018',
            ],
        'DRIAMS-B': [
 '/links/groups/borgwardt/Data/ms_diagnostics/validation/KSBL/encoded'
            ],
        'DRIAMS-C': [
 '/links/groups/borgwardt/Data/ms_diagnostics/validation/Aarau/2018'
            ],
        'DRIAMS-D': [
 '/links/groups/borgwardt/Data/ms_diagnostics/validation/Viollier/Viollier'
            ],
        'DRIAMS-E': [
 '/links/groups/borgwardt/Data/maldi_repetitions/sa_ec/usb_ec_sa_spectra'
            ],
        'DRIAMS-F': [
 '/links/groups/borgwardt/Data/maldi_repetitions/sa_ec/ksbl_ec_sa_spectra'
            ],
    }


def count_number_of_spectra(datapaths):
    fid_count = 0

    for path in datapaths:
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('fid'):
                    fid_count+=1
        print(f'{path} contains {fid_count} fid files.')
    return fid_count

def count_amr_labels(site):
    label_count = 0
    list_antibiotics = explorer.available_antibiotics(site)
    
    for year in explorer.available_years(site): 
        driams = load_driams_dataset(
        DRIAMS_ROOT,
        site,
        year,
        '*',
        list_antibiotics[year],
        encoder=DRIAMSLabelEncoder(),
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
        )

        antibiotics = [c for c in driams.y.columns if not c[0].islower()]
        size = driams.y[antibiotics].size
        nans = driams.y[antibiotics].isna().values.sum() 
        label_count += size - nans 
    return label_count



if __name__=='__main__':
    
    for site in list_sites:
        fid_count = count_number_of_spectra(map_raw_datapaths[site])
        print(f'In {site} a total of {fid_count} fid files were found.')
        label_count = count_amr_labels(site)
        print(f'In {site} a total of {label_count} AMR labels were found.')
