"""MMD-based statistical hypothesis testing and distribution comparison."""

import argparse
import dotenv
import functools
import os

import numpy as np

from sklearn.metrics import pairwise_kernels

from maldi_learn.driams import load_driams_dataset

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')


def gaussian_kernel(x, y, sigma=1.0, **kwargs):
    """Gaussian (RBF) kernel."""
    return np.exp(-sigma * np.dot(x - y, x - y))


class MetaKernel:
    """Wraps a kernel for simplified evaluation later on."""

    def __init__(self, kernel, **kwargs):
        # Wrap the original kernel such that it can always be evaluated
        # later on.
        self.kernel = functools.partial(
            kernel,
            **kwargs
        )

    def __call__(self, X, Y=None):
        return pairwise_kernels(X, Y, metric=self.kernel)

    def diag(self, X, Y=None):
        if Y is None:
            Y = X

        return [self.original_kernel(x, y) for x, y in zip(X, Y)]


def calculate_kernel_matrix(X, Y, kernel, **kwargs):
    """Calculate kernel matrix between two sets of elements.

    Evaluates a kernel matrix between two sets of elements and returns
    it. The kernel matrix dimension will be the product of each of the
    set cardinalities.

    Parameters
    ----------
    X : `np.array`
        First set of elements

    Y : `np.array`
        Second set of elements

    kernel : callable
        Kernel function; must be compatible with the data type of `X`
        and `Y`.

    **kwargs
        Additional keyword arguments that are presented to the kernel
        function or used in the kernel matrix calculation itself. The
        function currently supports the global parameter `n_jobs`; if
        set to a non-zero value, jobs are being spawned in parallel.

    Returns
    -------
    Kernel matrix between `X` and `Y`.
    """
    return pairwise_kernels(X, Y, metric=kernel, **kwargs)


def mmd(X, Y, kernel):
    """Calculate MMD between two sets of samples, using a kernel.

    This function calculates the maximum mean discrepancy between two
    distributions `X` and `Y` by means of a kernel function, which is
    assumed to be *compatible* with the input data.

    Parameters
    ----------
    X : `array_like`
        First distribution

    Y : `array_like`
        Second distribution

    kernel : callable
        Kernel function for evaluating the similarity between samples
        from `X` and `Y`. The kernel function must support the *type*
        of `X` and `Y`, respectively. It is supposed to return a real
        value. On the technical side, the kernel must be PSD, i.e. it
        must be a positive semi-definite function.

    Returns
    -------
    Maximum mean discrepancy value between `X` and `Y`.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Following the original notation of the paper
    m = X.shape[0]
    n = Y.shape[0]

    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)

    # The paper 'Integrating structured biological data by kernel
    # maximum mean discrepancy' claims that one should ignore all
    # the diagonal elements in the sum (i.e. remove the trace). I
    # find that this does *not* result in a metric, though. Thus,
    # I am following 'A Kernel Method for the Two-Sample Problem'
    # and implement Eq. 6.
    k_XX = np.sum(K_XX)
    k_YY = np.sum(K_YY)
    k_XY = np.sum(K_XY)

    mmd = 1 / m**2 * k_XX \
        + 1 / n**2 * k_YY \
        - 2 / (m * n) * k_XY

    return mmd


def mmd_linear_approximation(X, Y, kernel):
    """Calculate linear approximation to MMD between two sets of samples.

    This function calculates the maximum mean discrepancy between two
    distributions `X` and `Y` by means of a kernel function, which is
    assumed to be *compatible* with the input data. Notice that there
    are two caveats here:

    1. `X` and `Y` must have the same size
    2. Only an approximation to MMD is calculated, but the upshot is
       that this approximation can be calculated in linear time.

    Parameters
    ----------
    X : `array_like`
        First distribution; must match cardinality of `Y`

    Y : `array_like`
        Second distribution; must match cardinality of `X`

    kernel : callable following the sklearn.gaussian_process.Kernel API or
        a general callable which compares two samples from `X` and `Y`.
        The kernel function must support the *type* of `X` and `Y`,
        respectively. It is supposed to return a real value. On the technical
        side, the kernel must be PSD, i.e. it must be a positive semi-definite
        function.

    Returns
    -------
    Maximum mean discrepancy value between `X` and `Y`.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Following the original notation of the paper
    m = X.shape[0]
    n = Y.shape[0]

    assert m == n, RuntimeError('Cardinalities must coincide')

    n = (n // 2) * 2

    # The main functionality of this code is taken from Dougal
    # Sutherland's repository:
    #
    #  https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
    #
    # This is a more generic version.
    K_XX = np.sum(kernel.diag(X[:n:2], X[1:n:2]))
    K_YY = np.sum(kernel.diag(Y[:n:2], Y[1:n:2]))

    K_XY = np.sum(kernel.diag(X[:n:2], Y[1:n:2])) + \
        np.sum(kernel.diag(X[1:n:2], Y[:n:2]))

    mmd = (K_XX + K_YY - K_XY) / (n // 2)
    return mmd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-a', '--antibiotic',
        type=str,
        help='Antibiotic for which to run the experiment',
        default='Ceftriaxone',
    )

    parser.add_argument(
        '-s', '--species',
        type=str,
        help='Species for which to run the experiment',
        default='Klebsiella pneumoniae'
    )

    parser.add_argument(
        '-S', '--seed',
        type=int,
        help='Random seed to use for the experiment',
        required=True
    )

    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        help='Number of samples for the comparison',
    )

    args = parser.parse_args()

    # TODO: make configurable
    sites = ['DRIAMS-A', 'DRIAMS-B']

    # Will contain all data sets of the respective sites, but only the
    # spectra themselves *without* any labels.
    data = []

    for site in sites:
        dataset = load_driams_dataset(
            DRIAMS_ROOT,
            site,
            '*',
            species=args.species,
            antibiotics=args.antibiotic,
            spectra_type='binned_6000',
            nrows=1000,
        )

        X = np.asarray([spectrum.intensities for spectrum in dataset.X])
        data.append(X)

    print(len(data[0]), len(data[1]))

    # TODO: make configurable?
    kernel = functools.partial(gaussian_kernel, sigma=1.0)

    # This is the difference between the two distributions under the
    # assumption that the labels are *known*.
    theta_0 = mmd(data[0], data[1], kernel=kernel)

    print(theta_0)
