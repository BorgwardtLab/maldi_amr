"""
Script to extract the following dataset statistics: 

    (1) The number of spectra in each DRIAMS dataset. 
        We define this number as the number of fid datafiles
    (2) The number of antimicrobial resistance labels in each DRIAMS
        dataset. 
"""

import glob
import os

list_sites = ['DRIAMS-A', 'DRIAMS-B', 'DRIAMS-C', 'DRIAMS-D']

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
    }

def count_number_of_spectra(datapaths):
    for path in datapaths:
        for filename in glob.glob('**/fid'):
            print(filename)
    pass

def count_amr_labels():
    pass



if __name__=='__main__':
    
    for site in list_sites:
        count_number_of_spectra(map_raw_datapaths[site])
    count_amr_labels()
