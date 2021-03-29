"""Analyse results of metadata prediction."""

import argparse
import json
import tabulate

import numpy as np

from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str)

    args = parser.parse_args()

    classes = None
    confusion_matrix = []

    for filename in tqdm(args.INPUT, desc='Loading'):
        with open(filename) as f:
            data = json.load(f)

            if classes is None:
                classes = data['classes']

            matrix = np.asarray(data['confusion_matrix'])
            n = len(classes)
            matrix = matrix.reshape(n, n)

            confusion_matrix.append(matrix)

    confusion_matrix = sum(confusion_matrix) / len(confusion_matrix)

    print(
        tabulate.tabulate(
            confusion_matrix,
            showindex=classes,
            headers=classes
        )
    )
