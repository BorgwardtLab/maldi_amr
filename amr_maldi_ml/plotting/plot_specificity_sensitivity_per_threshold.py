""" Script to plot sensitivity-specificity pairs of rejection threshold. """

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO make configurable
tablename = '/links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/amr_maldi_ml/tables/Rejection_table_assymetric_finegrid_Escherichia_coli_Cefepime_calibrated_lightgbm.csv'
df = pd.read_csv(tablename)

# plot sensitivity vs. specificity
plt.figure(figsize=(10,10))
sns.set(style="whitegrid")

plt.scatter(x=df['specificity'], 
            y=df['sensitivity'], 
            s=4, 
            cmap='tab10',
            c=df['percentage rejected samples'],
            )
plt.xlabel('specificity (percentage of susceptible samples correctly identified)')
plt.ylabel('sensitivity (percentage of resistant samples correctly identified)')

plt.colorbar()

filename = os.path.basename(tablename).replace('.csv', '')
plt.savefig(f'../plots/sensitivity_vs_specificity/{filename}.png')
