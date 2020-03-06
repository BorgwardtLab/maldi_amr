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



# define color maps
col_list = [u'orangered', u'purple', u'royalblue', u'olivedrab', u'orange', u'violet',  u'red', u'turquoise', u'chartreuse', u'black', u'saddlebrown', u'salmon', u'mediumvioletred', u'seagreen', u'skyblue', u'slateblue', u'darkgrey', u'springgreen', u'teal', u'tomato', u'peru', u'yellowgreen', u'aqua', u'aquamarine', u'blue', u'blueviolet', u'brown', u'burlywood', u'cadetblue', u'chocolate', u'coral', u'cornflowerblue', u'crimson', u'darkblue', u'darkcyan', u'darkgoldenrod', u'darkgreen', u'darkgrey', u'darkmagenta', u'hotpink', u'darkolivegreen', u'yellow']


#TODO adjust antibiotics to general naming
maldi_col_map = {
    'Ceftriaxon': col_list[0],
    'Oxacillin': col_list[1],
    'Amoxicillin-Clavulanic acid': col_list[2],
    'Meropenem': col_list[3],
    'Piperacillin...Tazobactam': col_list[4],
    'Ciprofloxacin': col_list[5],
    'Colistin': col_list[6],
    'Fluconazol': col_list[7],
    'Fusidinsaeure': col_list[9],
    'Cefepim': col_list[8],
    'Penicillin': col_list[10],
    'Imipenem': col_list[11],
    'Gentamicin': col_list[12],
    'Tetracyclin': col_list[13],
    'Vancomycin': col_list[14],
    'Clindamycin': col_list[15],
    'Nitrofurantoin': col_list[16],
    'Tigecyclin': col_list[17],
    'Tobramycin': col_list[18],
    'Amikacin': col_list[19],
    'Amoxicillin': col_list[20],
    'Ampicillin...Amoxicillin': col_list[21],
    'Anidulafungin': col_list[22],
    'Aztreonam': col_list[23],
    'Caspofungin': col_list[24],
    'Cefazolin': col_list[25],
    'Cefpodoxim': col_list[26],
    'Ceftazidim': col_list[27],
    'Cefuroxim': col_list[28],
    'Cotrimoxazol': col_list[29],
    'Daptomycin': col_list[30],
    'Ertapenem': col_list[31],
    'Erythromycin': col_list[32],
    'Fosfomycin.Trometamol': col_list[33],
    'Itraconazol': col_list[34],
    'Levofloxacin': col_list[35],
    'Micafungin': col_list[36],
    'Norfloxacin': col_list[37],
    'Rifampicin': col_list[38],
    'Teicoplanin': col_list[39],
    'Voriconazol': col_list[40],
    'X5.Fluorocytosin': col_list[41]
    }
