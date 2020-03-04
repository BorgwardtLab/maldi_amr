"""
Maldi-Tof Spectra folder have overlapping spectra. Remove overlap from id files.
"""

import pandas as pd


def remove_overlap(id_2017, id_2018):
    df_2017 = pd.read_csv(id_2017)    
    df_2018 = pd.read_csv(id_2018)    

if __name__ == '__main__':
    
    clean_id_2017 = '/links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/id/2017/2017_clean.csv'
    clean_id_2018 = '/links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/id/2018/2018_clean.csv'
    
    remove_overlap(clean_id_2017, clean_id_2018)
