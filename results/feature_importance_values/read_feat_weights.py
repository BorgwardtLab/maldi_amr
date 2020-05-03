import json

import numpy as np
import pandas as pd


filenames = [
    'Site_DRIAMS-A_Model_lr_Species_Escherichia_coli_Antibiotic_Ceftriaxone_Seed_164-172-188-270-344-35-409-480-545-89_average.json',
    'Site_DRIAMS-A_Model_lr_Species_Staphylococcus_aureus_Antibiotic_Oxacillin_Seed_164-172-188-270-344-35-409-480-545-89_average.json',
    'Site_DRIAMS-A_Model_lr_Species_Klebsiella_pneumoniae_Antibiotic_Meropenem_Seed_164-172-188-270-344-35-409-480-545-89_average.json',
]


for filename in filenames:
    with open(filename) as f:
        data = json.load(f)

    df = pd.DataFrame({k: data[k][0] for k in ['mean_feature_importance', 'std_feature_importance']})

    rank = [sorted(
       -np.abs(df['mean_feature_importance'])
                   ).index(x) for x in -np.abs(df['mean_feature_importance'])
            ]
    df['rank'] = [r+1 for r in rank]

    df['bin_start'] = np.linspace(2000, 19997, 6000).astype(int).tolist()
    df['bin_end'] = np.linspace(2003, 20000, 6000).astype(int).tolist()

    outname = filename.replace('.json', '.csv')
    df.to_csv(outname, index=False)
