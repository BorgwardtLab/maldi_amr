"""Utility functions."""


import os


def _encode(s):
    """Encode string for filename generation."""
    return s.replace(' ', '_')


def generate_output_filename(root, data):
    """Generate output filename for dictionary.

    Given a root folder for storing output files, creates a filename
    based on a data dictionary.

    Parameters
    ----------
    root : str
        Output directory for storing the data

    data : dict
        Data dictionary; containing a few required keys, such as
        `antibiotic`, and many optional ones.

    Returns
    -------
    Filename for storing the output. This filename is not guaranteed to
    be unique and it is the client's responsibility to check whether it
    exists.
    """
    # Default to using *all* species if the key is not present in the
    # data dictionary.
    species = _encode(data.get('species', 'all'))

    antibiotic = _encode(data['antibiotic'])
    seed = data['seed']
    site = data['site']

    filename = f'Site_{site}_' \
               f'Species_{species}_Antibiotic_{antibiotic}_Seed_{seed}.json'
    filename = os.path.join(root, filename)

    return filename
