import json
import sys

import matplotlib.pyplot as plt


if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        data = json.load(f)

    plt.hist(data['theta_bootstrap'], bins=20)
    plt.axvline(data['theta_original'], c='k', ls='dotted')
    plt.xscale('log')
    plt.show()
