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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--metric',
        type=str,
        default='auroc',
        help='Metric to use for plotting'
    )

    parser.add_argument(
        '-d', '--date-column',
        type=str,
        default='train_to',
        help='Date column to use for visualisation'
    )

    parser.add_argument(
        '-s', '--suffix',
        type=str,
        default='',
        help='Suffix to be added to plot name'
    )
    args = parser.parse_args()

    
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
    df = df.sort_values([f'scenario', args.date_column])

    if args.metric != 'train_sample_size':
        df[args.metric] *= 100

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

    # Some debug output, just so all values can be seen in all their
    # glory.
    print(
        df.groupby([args.date_column, 'species', 'antibiotic']).agg(
            {
                'test_calibrated_'+args.metric: [np.mean, np.std]
            }
        )
    )

    # Plot lineplot.
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15,6))

    g = sns.lineplot(
        x=args.date_column,
        y='test_calibrated_'+args.metric,
        data=df,
        hue=f'scenario',
    )
    plt.xlabel('last month of 8-month training interval')
    plt.ylabel(f'{args.metric}'.upper())

    # Minor ticks every month.
    fmt_month = mdates.MonthLocator(interval=1)
    ax.xaxis.set_major_locator(fmt_month)
    
    # Format dates in xticks.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    if args.metric == 'auprc':
        plt.ylim((0.0,0.4))
    else:
        plt.ylim((0.3,1.0))
        plt.axhline(0.5, color='darkgrey', linestyle='--')

    if args.suffix != '':
        suffix = '_' + args.suffix
    else:
        suffix = args.suffix

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'../plots/sliding_window_validation/sliding_window_validation_Metric_{args.metric}.png')
    plt.show()
