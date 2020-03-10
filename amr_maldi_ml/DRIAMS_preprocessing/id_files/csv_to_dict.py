"""
Print antibiotic naming dictionary to file
based on input from AB-matching.csv

The resulting dictionary was appended to
amr_maldi_ml/utilities.py to serve as a 
renaming map for all sites.
"""

import numpy as np
import pandas as pd

df = pd.read_csv('AB-matching_utf8.csv')
list_included = []

with open('ab_name_map.txt', 'w') as f:

    f.write('ab_name_map = {\n')
    for index, row in df.iterrows():
        match_set = set()
        match_set.add(row['USB'])
        match_set.add(row['LIESTAL'])
        match_set.add(row['Aarau'])
        match_set.add(row['Viollier'])

        if np.nan in list(match_set):
            match_set.remove(np.nan)
        match_list = list(match_set)

        match_entry = row['MATCH']

        for name in match_list:
            if f'{name}' not in list_included:
                f.write(f'\'{name}\': \'{match_entry}\',\n')
                list_included.append(f'{name}')

    f.write('}')
    f.close()
