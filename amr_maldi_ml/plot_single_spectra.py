
"""
Plot a single raw MALDI-TOF mass spectra.
"""
import glob
import os

import pandas as pd
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt

from maldi_learn.driams import load_driams_dataset
from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder

load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

pair = ('Escherichia coli', 'Ciprofloxacin')

def load_single_raw_spectrum(pair_, year, site):

    driams = load_driams_dataset(
    DRIAMS_ROOT,
    site,
    year,
    pair_[0],
    pair_[1],
    encoder=DRIAMSLabelEncoder(),
    handle_missing_resistance_measurements='remove_if_any_missing',
    spectra_type='raw',
    )

    # return first spectrum from list
    return driams.X[0]



if __name__=='__main__':
    
    raw_spectrum = load_single_raw_spectrum(pair, '2018', 'DRIAMS-A')
    
    sns.set(
        style='whitegrid',
        font_scale=2,
                )
    fig, ax = plt.subplots(figsize=(10,7))
    
    ax.plot(raw_spectrum[:,0], raw_spectrum[:,1])
    ax.set_xlim(2000,15000)
    ax.set_ylabel('intensity')
    ax.set_xlabel('m/z')    
    
    plt.tight_layout()
    plt.savefig('./plots/example_spectra.png')

