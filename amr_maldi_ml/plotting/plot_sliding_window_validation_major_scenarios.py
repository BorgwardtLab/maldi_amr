"""Plot sliding window validation experiment data."""

import argparse
import json
import glob

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import seaborn as sns

#pd.set_option('display.max_rows', 1000)
if __name__ == '__main__':

    metrics = ['auroc', 'auprc']

    prjdir = '../../results/sliding_window_validation' 
    input_files = glob.glob(prjdir+'/lightgbm/Site_DRIAMS-A_Model_lightgbm_Species_Staphylococcus_aureus_Antibiotic_Oxacillin_*[0-9].json', recursive=True) + \
                  glob.glob(prjdir+'/lightgbm/Site_DRIAMS-A_Model_lightgbm_Species_Escherichia_coli_Antibiotic_Ceftriaxone_*[0-9].json', recursive=True) + \
                  glob.glob(prjdir+'/mlp/Site_DRIAMS-A_Model_mlp_Species_Klebsiella_pneumoniae_Antibiotic_Ceftriaxone_*[0-9].json', recursive=True)

    # Create new data frame by concatenating all the JSON files there
    # are. This assumes that the fields are the same in all files. It
    # will inevitably *fail* if this is not the case.
    rows = []
    for filename in input_files:
        with open(filename) as f:
            data = json.load(f)
            rows.append(data)

    df = pd.DataFrame(rows)

    # Convert all columns to dates. This simplifies the `seaborn`
    # handling below as it will recognise the special structure of
    # the data frame.
    for column in df.columns:
        if column.endswith('_to') or column.endswith('_from'):
            df[column] = pd.to_datetime(df[column])

    # Create new column that describes the whole scenario.
    df[f'scenario'] = df['species'] + ' (' + df['antibiotic'] + ')' + ' (' + df['model']  + ')'
    df = df.sort_values([f'scenario', 'train_to'])

    # Drop unnecessary columns.
    cols_to_del = [
       'years', 
       'auroc',
       'auprc',
       'accuracy',
       'feature_weights',
       'best_params',
       'prevalence_train', 
       'prevalence_test',
       'oversampling', 
       'metadata_versions', 
       'scoring', 
       'y_score', 
       'y_pred', 
       'y_pred_calibrated', 
       'y_score_calibrated',
       'train_accuracy', 
       'train_auprc', 
       'train_auroc', 
       'train_sample_size', 
    ]

    df = df.drop(columns=cols_to_del)


    # Plot lineplot.
    plt.close('all')
    sns.set(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(2, 1, figsize=(30,15))

    for i, metric in enumerate(metrics):
        # Some debug output, just so all values can be seen in all their
        # glory.
        print(
            df.groupby(['train_to', 'species', 'antibiotic']).agg(
                {
                    'test_calibrated_'+metric: [np.mean, np.std]
                }
            )
        )

        sns.lineplot(
            x='train_to',
            y='test_calibrated_'+metric,
            data=df,
            ax=ax[i],
            hue=f'scenario',
        )
        ax[i].set_xlabel('last month of 8-month training interval')
        ax[i].set_ylabel(f'{metric}'.upper())
        if metric == 'auroc': ax[i].axhline(0.5, color='darkgrey', linestyle='--')

        # Minor ticks every month.
        fmt_month = mdates.MonthLocator(interval=1)
        ax[i].xaxis.set_major_locator(fmt_month)
        
        # Format dates in xticks.
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.xticks(rotation=45)

    # Hide x labels and tick labels for all but bottom plot.
    for axis in ax:
        axis.label_outer()
    plt.subplots_adjust(hspace=0)
    plt.savefig('../plots/sliding_window_validation/sliding_window_validation.png')
    plt.show()
