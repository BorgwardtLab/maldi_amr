"""Utility functions."""


import os


def _encode(s):
    """Encode string for filename generation."""
    return s.replace(' ', '_')


def generate_output_filename(root, data, suffix=None):
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

    suffix : str, optional
        Contains a suffix that is appended to the filename before the
        extension. This is useful when describing certain experiments
        that cannot be fully described by `data`.

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
               f'Species_{species}_Antibiotic_{antibiotic}_Seed_{seed}'

    # Ensures that the suffix is only added if it exists. Else, we will
    # add spurious underscores.
    if suffix:
        filename += f'_{suffix}'

    filename += '.json'
    filename = os.path.join(root, filename)

    return filename



# define color maps
col_list = [u'orangered', u'purple', u'royalblue', u'olivedrab', u'orange', u'violet',  u'red', u'turquoise', u'chartreuse', u'black', u'saddlebrown', u'salmon', u'mediumvioletred', u'seagreen', u'skyblue', u'slateblue', u'darkgrey', u'springgreen', u'teal', u'tomato', u'peru', u'yellowgreen', u'aqua', u'aquamarine', u'blue', u'blueviolet', u'brown', u'burlywood', u'cadetblue', u'chocolate', u'coral', u'cornflowerblue', u'crimson', u'darkblue', u'darkcyan', u'darkgoldenrod', u'darkgreen', u'darkgrey', u'darkmagenta', u'hotpink', u'darkolivegreen', u'yellow']


#TODO adjust antibiotics to general naming
maldi_col_map = {
    'Ceftriaxone': col_list[0],
    'Oxacillin': col_list[1],
    'Amoxicillin-Clavulanic acid': col_list[2],
    'Meropenem': col_list[3],
    'Piperacillin-Tazobactam': col_list[4],
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


# antibiotic naming
ab_name_map = {
    'AN-Amikacin': 'Amikacin',
    'Amikacin': 'Amikacin',
    'Amikacin 01 mg/l': 'Amikacin_1mg-l',
    'Amikacin 04 mg/l': 'Amikacin_4mg-l',
    'Amikacin 20 mg/l': 'Amikacin_20mg-l',
    'Aminoglykoside': 'Aminoglycosides',
    'Amoxicillin...Clavulansaeure.bei.unkompliziertem.HWI': 'Amoxicillin-Clavulanic acid_uncomplicated_HWI',
    'Amoxicillin-Clavulansaeure.unkompl.HWI': 'Amoxicillin-Clavulanic acid_uncomplicated_HWI',
    'Amoxicillin-Clavulan': 'Amoxicillin-Clavulanic acid',
    'AMC-Amoxicillin/Clavulans\xc3\xa4ure': 'Amoxicillin-Clavulanic acid',
    'Amoxicillin-Clavulansaeure': 'Amoxicillin-Clavulanic acid',
    'Amoxicillin/Clavulansäure': 'Amoxicillin-Clavulanic acid',
    'Amoxicillin...Clavulansaeure': 'Amoxicillin-Clavulanic acid',
    'Amoxicillin/Clavulansaeure': 'Amoxicillin-Clavulanic acid',
    'Amoxicillin': 'Amoxicillin',
    'AMX-Amoxicillin': 'Amoxicillin',
    'Ampicillin': 'Ampicillin',
    'AM-Ampicillin': 'Ampicillin',
    'P-Benzylpenicillin': 'Benzylpenicillin',
    'Benzylpenicillin': 'Benzylpenicillin',
    'Benzylpenicillin andere': 'Benzylpenicillin_others',
    'Benzylpenicillin bei Meningitis': 'Benzylpenicillin_with_meningitis',
    'Benzylpenicillin bei Pneumonie': 'Benzylpenicillin_with_pneumonia',
    'Amphotericin.B': 'Amphotericin B',
    'Amphothericin B': 'Amphotericin B',
    'Ampicillin...Amoxicillin': 'Ampicillin-Amoxicillin',
    'SAM-Ampicillin/Sulbactam': 'Ampicillin-Sulbactam',
    'Ampicillin...Sulbactam': 'Ampicillin-Sulbactam',
    'Anidulafungin': 'Anidulafungin',
    'Azithromycin': 'Azithromycin',
    'ATM-Aztreonam': 'Aztreonam',
    'Aztreonam': 'Aztreonam',
    'Bacitracin': 'Bacitracin',
    'Caspofungin': 'Caspofungin',
    'Cefalotin/Cefazolin': 'Cefalotin-Cefazolin',
    'Cefamandol': 'Cefamandole',
    'Cefazolin': 'Cefazolin',
    'FEP-Cefepim': 'Cefepime',
    'Cefepim': 'Cefepime',
    'Cefepime': 'Cefepime',
    'Cefepim.1': 'Cefepime',
    'Cefixim': 'Cefixime',
    'Cefoperazon-Sulbactam': 'Cefoperazon-Sulbactam',
    'Cefoperazon-Sulbacta': 'Cefoperazon-Sulbactam',
    'CTX-Cefotaxim': 'Cefotaxime',
    'Cefotaxim': 'Cefotaxime',
    'Cefoxitin Screening Staph': 'Cefoxitin_screen',
    'Cefoxitin.Screen': 'Cefoxitin_screen',
    'OXSF-Cefoxitin-Screen': 'Cefoxitin_screen',
    'FOX-Cefoxitin': 'Cefoxitin',
    'Cefoxitin': 'Cefoxitin',
    'CPD-Cefpodoxim': 'Cefpodoxime',
    'Cefpodoxim': 'Cefpodoxime',
    'Ceftarolin': 'Ceftarolin',
    'CAZ-Ceftazidim': 'Ceftazidime',
    'Ceftazidim.1': 'Ceftazidime',
    'Ceftazidim': 'Ceftazidime',
    'Ceftazidim.Avibactam': 'Ceftazidime-Avibactam',
    'Ceftazidim-Avibactam': 'Ceftazidime-Avibactam',
    'Ceftibuten': 'Ceftibuten',
    'Ceftobiprol': 'Ceftobiprole',
    'Ceftolozan...Tazobactam': 'Ceftolozane-Tazobactam',
    'Ceftolozan-Tazobacta': 'Ceftolozane-Tazobactam',
    'Ceftriaxon': 'Ceftriaxone',
    'CRO-Ceftriaxon': 'Ceftriaxone',
    'CXMA-Cefuroxim-Axetil': 'Cefuroxime',
    'Cefuroxim.Axetil': 'Cefuroxime',
    'Cefuroxim iv': 'Cefuroxime',
    'CXM-Cefuroxim': 'Cefuroxime',
    'Cefuroxim': 'Cefuroxime',
    'Cefuroxim oral': 'Cefuroxime',
    'Chinolone': 'Quinolones',
    'C-Chloramphenicol': 'Chloramphenicol',
    'Chloramphenicol': 'Chloramphenicol',
    'Ciprofloxacin': 'Ciprofloxacin',
    'CIP-Ciprofloxacin': 'Ciprofloxacin',
    'Clarithromycin': 'Clarithromycin',
    'Clarithromycin 04': 'Clarithromycin_4mg-l',
    'Clarithromycin 16': 'Clarithromycin_16mg-l',
    'Clarithromycin 32': 'Clarithromycin_32mg-l',
    'Clarithromycin 64': 'Clarithromycin_64mg-l',
    'Clindamycin': 'Clindamycin',
    'CM-Clindamycin': 'Clindamycin',
    'Clindamycin ind.': 'Clindamycin_induced',
    'ICR-Induzierbare Clindamycin Resistenz': 'Clindamycin_induced',
    'Clofazimin': 'Clofazimine',
    'Clofazimin 0.25 mg/l': 'Clofazimine_.25mg-l',
    'Clofazimin 0.5 mg/l': 'Clofazimine_.5mg-l',
    'Clofazimin 1.0 mg/l': 'Clofazimine_1mg-l',
    'Clofazimin 2.0 mg/l': 'Clofazimine_2mg-l',
    'Clofazimin 4.0 mg/l': 'Clofazimine_4mg-l',
    'Colistin': 'Colistin',
    'CS-Colistin': 'Colistin',
    'Cotrimoxazol': 'Cotrimoxazol',
    'Trimethoprim/Sulfamethoxazol': 'Cotrimoxazol',
    'SXT-Trimethoprim/Sulfamethoxazol': 'Cotrimoxazol',
    'Trimethoprim-Sulfame': 'Cotrimoxazol',
    'DAP-Daptomycin': 'Daptomycin',
    'Daptomycin': 'Daptomycin',
    'ESBL': 'ESBL',
    'Doxycyclin': 'Doxycycline',
    'Dummy': 'Dummy',
    'Ertapenem': 'Ertapenem',
    'ETP-Ertapenem': 'Ertapenem',
    'E-Erythromycin': 'Erythromycin',
    'Erythromycin': 'Erythromycin',
    'Ethambutol': 'Ethambutol',
    'Ethambutol 02.5': 'Ethambutol_2mg-l',
    'Ethambutol 05.0': 'Ethambutol_5mg-l',
    'Ethambutol.5.0.mg.l': 'Ethambutol_5mg-l',
    'Ethambutol 07.5': 'Ethambutol_7.5mg-l',
    'Ethambutol 12.5': 'Ethambutol_12.5mg-l',
    'Ethambutol 50': 'Ethambutol_50mg-l',
    'Fluconazol': 'Fluconazole',
    'Fosfomycin.Trometamol': 'Fosfomycin-Trometamol',
    'FOS-Fosfomycin': 'Fosfomycin',
    'Fosfomycin': 'Fosfomycin',
    'FA-Fusidins\xc3\xa4ure': 'Fusidic acid',
    'Fusidinsaeure': 'Fusidic acid',
    'Fusidins\xc3\xa4ure': 'Fusidic acid',
    'Fusidinsäure': 'Fusidic acid',
    'GHLR-High-Level-Resistenz gegen Gentamicin': 'Gentamicin_high_level',
    'Gentamicin High Level': 'Gentamicin_high_level',
    'Gentamicin.High.level': 'Gentamicin_high_level',
    'HLG-Gentamicin, High-Level (Synergie)': 'Gentamicin_high_level',
    'HLS-Streptomycin, High-Level (Synergie)': 'Streptomycin_high_level',
    'Gentamicin': 'Gentamicin',
    'GM-Gentamicin': 'Gentamicin',
    'Imipenem': 'Imipenem',
    'IPM-Imipenem': 'Imipenem',
    'Isavuconazol': 'Isavuconazole',
    'Isoniazid': 'Isoniazid',
    'Isoniazid.0.1.mg.l': 'Isoniazid_.1mg-l',
    'Isoniazid 0\t1 mg/l': 'Isoniazid_.1mg-l',
    'Isoniazid.0.4.mg.l': 'Isoniazid_.4mg-l',
    'Isoniazid 0\t4 mg/l': 'Isoniazid_.4mg-l',
    'Isoniazid 1.0 mg/l': 'Isoniazid_1mg-l',
    'Isoniazid 10  mg/l': 'Isoniazid_10mg-l',
    'Isoniazid 3.0 mg/l': 'Isoniazid_3mg-l',
    'Itraconazol': 'Itraconazole',
    'Ketoconazol': 'Ketoconazole',
    'LEV-Levofloxacin': 'Levofloxacin',
    'Levofloxacin': 'Levofloxacin',
    'LNZ-Linezolid': 'Linezolid',
    'Linezolid': 'Linezolid',
    'Linezolid 01 mg/l': 'Linezolid_1mg-l',
    'Linezolid 04 mg/l': 'Linezolid_4mg-l',
    'Linezolid 16 mg/l': 'Linezolid_16mg-l',
    'MRSA': 'MRSA',
    'Meropenem.bei.Meningitis': 'Meropenem_with_meningitis',
    'Meropenem.bei.Pneumonie': 'Meropenem_with_pneumonia',
    'Meropenem.ohne.Meningitis': 'Meropenem_without_meningitis',
    'Meropenem': 'Meropenem',
    'MEM-Meropenem': 'Meropenem',
    'Meropenem-Vaborbactam': 'Meropenem-Vaborbactam',
    'Meropenem-Vaborbacta': 'Meropenem-Vaborbactam',
    'Metronidazol': 'Metronidazole',
    'Miconazol': 'Miconazole',
    'Micafungin': 'Micafungin',
    'Minocyclin': 'Minocycline',
    'MXF-Moxifloxacin': 'Moxifloxacin',
    'Moxifloxacin': 'Moxifloxacin',
    'Moxifloxacin 0.5': 'Moxifloxacin_.5mg-l',
    'Moxifloxacin 02.5': 'Moxifloxacin_2.5mg-l',
    'Moxifloxacin 10': 'Moxifloxacin_10mg-l',
    'MUP-Mupirocin': 'Mupirocin',
    'Mupirocin': 'Mupirocin',
    'Nalidixinsaeure': 'Nalidixin acid',
    'Nitrofurantoin': 'Nitrofurantoin',
    'FT-Nitrofurantoin': 'Nitrofurantoin',
    'Norfloxacin': 'Norfloxacin',
    'NOR-Norfloxacin': 'Norfloxacin',
    'Novobiocin': 'Novobiocin',
    'Ofloxacin': 'Ofloxacin',
    'OFL-Ofloxacin': 'Ofloxacin',
    'Oxacillin': 'Oxacillin',
    'Oxa/Flucloxacil.': 'Oxacillin',
    'OX1-Oxacillin': 'Oxacillin',
    'Pefloxacin': 'Pefloxacin',
    'Penicillin.bei.anderen.Infekten': 'Penicillin_with_other_infections',
    'Penicillin.bei.Endokarditis': 'Penicillin_with_endokarditis',
    'Penicillin.bei.Meningitis': 'Penicillin_with_meningitis',
    'Penicillin.bei.Pneumonie': 'Penicillin_with_pneumonia',
    'Penicillin.ohne.Endokarditis': 'Penicillin_without_endokarditis',
    'Penicillin.ohne.Meningitis': 'Penicillin_without_meningitis',
    'Penicillin': 'Penicillin',
    'PIP-Piperacillin]': 'Piperacillin',
    'Piperacillin/Tazobactam': 'Piperacillin-Tazobactam',
    'TZP-Piperacillin/Tazobactam': 'Piperacillin-Tazobactam',
    'Piperacillin...Tazobactam': 'Piperacillin-Tazobactam',
    'Piperacillin-Tazobac': 'Piperacillin-Tazobactam',
    'PT-Pristinamycin': 'Pristinamycine',
    'Polymyxin.B': 'Polymyxin B',
    'Polymyxin B': 'Polymyxin B',
    'Posaconazol': 'Posaconazole',
    'Pyrazinamid.100.0.mg.l': 'Pyrazinamide',
    'Pyrazinamid 100\t0 mg': 'Pyrazinamide',
    'Pyrazinamid': 'Pyrazinamide',
    'QDA-Quinupristin/Dalfopristin': 'Quinupristin-Dalfopristin',
    'Quinupristin-Dalfopr': 'Quinupristin-Dalfopristin',
    'Rifabutin 0.1 mg/l': 'Rifabutin_.1mg-l',
    'Rifabutin 0.4 mg/l': 'Rifabutin_.4mg-l',
    'Rifabutin 2 mg/l': 'Rifabutin_2mg-l',
    'Rifampicin 01.0 mg/l': 'Rifampicin_1mg-l',
    'Rifampicin.1.0.mg.l': 'Rifampicin_1mg-l',
    'RA-Rifampicin': 'Rifampicin',
    'Rifampicin': 'Rifampicin',
    'Rifampicin 02.0 mg/l': 'Rifampicin_2mg-l',
    'Rifampicin 04 mg/l': 'Rifampicin_4mg-l',
    'Rifampicin 20 mg/l': 'Rifampicin_20mg-l',
    'SPX-Sparfloxacin': 'Sparfloxacin',
    'Roxithromycin': 'Roxithromycin',
    'Spectinomycin': 'Spectinomycin',
    'Streptomycin.1.0.mg.l': 'Streptomycin',
    'Streptomycin': 'Streptomycin',
    'Strepomycin High Level': 'Strepomycin_high_level',
    'Streptomycin.High.level': 'Strepomycin_high_level',
    'Teicoplanin.GRD': 'Teicoplanin_GRD',
    'Teicoplanin': 'Teicoplanin',
    'TEC-Teicoplanin': 'Teicoplanin',
    'Tetracyclin': 'Tetracycline',
    'TE-Tetracyclin': 'Tetracycline',
    'TIC-Ticarcillin': 'Ticarcillin',
    'TCC-Ticarcillin/Clavulans\xc3\xa4ure': 'Ticarcillin-Clavulan acid',
    'Ticarcillin...Clavulansaeure': 'Ticarcillin-Clavulan acid',
    'TEL-Telithromycin': 'Telithromycin',
    'Tigecyclin': 'Tigecycline',
    'TGC-Tigecycline': 'Tigecycline',
    'Tobramycin': 'Tobramycin',
    'TM-Tobramycin': 'Tobramycin',
    'Vancomycin.GRD': 'Vancomycin_GRD',
    'Vancomycin': 'Vancomycin',
    'VA-Vancomycin': 'Vancomycin',
    'Voriconazol': 'Voriconazole',
    'X5.Fluorocytosin': '5-Fluorocytosine',
    '5-Fluorocytosin': '5-Fluorocytosine',
}
