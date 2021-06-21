"""Plot sliding window validation experiment data."""

import json
import glob

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns

from utils import scenario_map

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
    df['scenario'] = df['species'].str.replace(' ','_')
    df['scenario'] = df['scenario'].apply(lambda x: scenario_map[x])
    df['scenario'] = df['scenario'] + ' (' + df['model']  + ')'
    df = df.sort_values(['scenario', 'train_to'])

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
    ]

    df = df.drop(columns=cols_to_del)

    # Plot lineplot.
    plt.close('all')
    sns.set(style="whitegrid", font_scale=1.6)
    fig, ax = plt.subplots(1, 2, figsize=(25,14))

    for i, metric in enumerate(metrics):
        # Some debug output, just so all values can be seen in all their
        # glory.
        df_ = df.groupby(['train_to', 'species', 'antibiotic']).agg(
                {
                    'test_calibrated_'+metric: np.mean,
                    'train_sample_size': np.mean,
                    'scenario': pd.Series.unique,
                }
            )
        
        print(df_)

        sns.scatterplot(
            x='train_sample_size',
            y='test_calibrated_'+metric,
            data=df_,
            ax=ax[i],
            hue='scenario',
        )
        ax[i].set_xlabel('sample size')
        ax[i].set_ylabel(metric.upper())
        if metric == 'auprc':
            ax[i].yaxis.tick_right()
            ax[i].yaxis.set_label_position('right')

        xpos = []
        ypos = []
        xdir = []
        ydir = []

        # extract plotting values for arrows
        for scenario in df['scenario'].unique(): 
            df__ = df_.loc[df_['scenario'] == scenario]

            # check that train_to is in order
            x = df__['train_sample_size']
            y = df__['test_calibrated_'+metric]

            # calculate position and direction vectors
            x0 = x.iloc[range(len(x)-1)].values
            x1 = x.iloc[range(1,len(x))].values
            y0 = y.iloc[range(len(y)-1)].values
            y1 = y.iloc[range(1,len(y))].values
            xpos.extend(list((x0+x1)/2))
            ypos.extend(list((y0+y1)/2))
            xdir.extend(list(x1-x0))
            ydir.extend(list(y1-y0))
            # plot each line between points
            ax[i].plot(x,y)

        arrows = zip(xpos, ypos, xdir, ydir) 

        if metric == 'auroc':
            ax[i].set_ylim((0.4,0.8))
        elif metric == 'auprc':
            ax[i].set_ylim((0.0,0.4))

        # plot arrow on each line
        for (x, y, x_dir, y_dir) in arrows:
            ax[i].annotate(
                    '', 
                    xytext=(x,y), 
                    xy=(
                        x+0.001*x_dir,
                        y+0.001*y_dir
                        ), 
                    arrowprops=dict(
                            arrowstyle='->', 
                            color='black',
                        ), 
                    size=20,
            )

    # Hide x labels and tick labels for all but bottom plot.
    #for axis in ax:
    #    axis.label_outer()
    plt.subplots_adjust(wspace=0.01)
    plt.savefig('../plots/sliding_window_validation/sliding_window_scatterplot.png')
    plt.show()
