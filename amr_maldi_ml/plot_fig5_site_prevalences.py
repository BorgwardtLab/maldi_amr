"""
"""

import os
import json
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from heatmap import heatmap
import pandas as pd


def plot_figure5(args):
    
    PATH_TABLE = '../results/DRIAMS_summary'

    LIST_SPECIES = ['Escherichia coli',
                    'Staphylococcus epidermidis',
                    'Staphylococcus aureus',
                    'Pseudomonas aeruginosa',
                    'Klebsiella pneumoniae']
    
    # get list of all .json files in directory
    file_list = []
    for (_, _, filenames) in os.walk(PATH_TABLE):
        [file_list.append(f) for f in filenames if '.json' in f]
        break

    table = pd.DataFrame(columns=['antibiotic',
                                  'site',
                                  'number of samples',
                                  'percentage positive',
                                  ])    

    # read data and append to dataframe
    for filename in file_list:
        with open(os.path.join(PATH_TABLE, filename)) as f:
            data = json.load(f)
            if data['number spectra with AMR profile'] == 0:
                continue

            # construct string of common species
            species = data['species']
            y = data['label']

            for s in LIST_SPECIES:
                index = [1 if sp==s else 0 for sp in species]
                assert len(index) == len(species)
                species_s = [species[i] for i, idx in enumerate(index) if idx==1]
                y_s = [y[i] for i, idx in enumerate(index) if idx==1]

                if len(y_s) != 0:
                    pos_class_ratio = float(sum(y_s)/len(y_s))
                    num_samples = len(y_s)
                else:
                    pos_class_ratio = 0
                    num_samples = 0

                d = {'antibiotic' : [data['antibiotic']],
                    'site': [data['site']],
                    'species': [s],
                    'number of samples': [num_samples],
                    'percentage positive': [round(pos_class_ratio, 3)],
                    }
                table = table.append(pd.DataFrame(d), ignore_index=True)    

    table = table.sort_values(by=['number of samples'], ascending=False)

    # subset by antibiotic
    table = table.loc[table['antibiotic'] == args.antibiotic]
    print(table) 
    
    # -----------------------------------
    # plot bubble heat map
    # -----------------------------------
    plt.close('all')
    fig, ax = plt.subplots()

    #plt.figure(figsize=(12,7))
    

    heatmap(x=table['species'],
            y=table['site'],
            size=table['number of samples'],
            size_scale=1500,
            #Used to scale the size of the shapes in the
            #plot to make them fit the size of the fields in the matrix.
            #Default value is 500. You will likely need to fiddle with this
            #parameter in order to find the right value for your figure size
            #and the size range applied.
            marker='o',
            color=table['percentage positive'],
            # containing values based on which to apply the heatmap color.
            color_range=(0.0, 1.0),
            # A tuple (color_min, color_max) that enables capping the values of
            # color being mapped to palette. All color values less than color_min
            # are capped to color_min, and all color values larger than color_max
            # are capped to color_max.
            palette=sns.diverging_palette(20, 220, n=10),
           )
    plt.grid(False) # Hide grid
    plt.savefig(f'{args.outfile}')



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--antibiotic',
                        type=str,
                        default='Ciprofloxacin')
    parser.add_argument('--outfile',
                        type=str,
                        default='figure5_prevalence.png')
    args = parser.parse_args()

    plot_figure5(args)
