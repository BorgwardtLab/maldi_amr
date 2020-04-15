"""
Maldi-Tof Spectra folder have overlapping spectra. Remove overlap from id files.
"""

import pandas as pd


def remove_overlap(id_2017, id_2018, keep='first'):

    df_2017 = pd.read_csv(id_2017, low_memory=False)    
    df_2018 = pd.read_csv(id_2018, low_memory=False)    
    print('shape df_2017 {}'.format(df_2017.shape))
    print('shape df_2018 {}'.format(df_2018.shape))

    # TODO implement if not first
    if keep == 'first':
        overlap = df_2018['code'].isin(df_2017['code']) == True
        df_2018.drop(df_2018[overlap].index, inplace=True)

    # check that no overlaps exist anymore
    assert sum(df_2018['code'].isin(df_2017['code']) == True) == 0
    assert sum(df_2017['code'].isin(df_2018['code']) == True) == 0
    print('\nAfter removal:')
    print('shape df_2017 {}'.format(df_2017.shape))
    print('shape df_2018 {}'.format(df_2018.shape))

    # write output to file
    if keep == 'first':
        df_2018.to_csv(id_2018)

if __name__ == '__main__':
    
    clean_id_2017 = '/links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/id/2017/2017_clean.csv'
    clean_id_2018 = '/links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/id/2018/2018_clean.csv'

    remove_overlap(clean_id_2017, clean_id_2018)
