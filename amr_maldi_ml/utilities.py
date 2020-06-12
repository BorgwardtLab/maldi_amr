"""Utility functions."""

import os
import seaborn as sns


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
    model = data['model']

    # Single site exists in the input dictionary. Create the path
    # accordingly.
    if 'site' in data.keys():
        site = data['site']

        # We are training for different time periods. Adjust everything
        # accordingly.
        if 'train_years' in data.keys():
            train = '_'.join(data['train_years'])
            test = '_'.join(data['test_years'])

            filename = f'Site_{site}_'         \
                       f'Train_years_{train}_' \
                       f'Test_years_{test}_'   \
                       f'Model_{model}_'       \
                       f'Species_{species}_Antibiotic_{antibiotic}_Seed_{seed}'

        # Regular training
        else:
            filename = f'Site_{site}_'   \
                       f'Model_{model}_' \
                       f'Species_{species}_Antibiotic_{antibiotic}_Seed_{seed}'

    # Except `train_site` and `test_site` to highlight different
    # scenarios.
    else:
        train_site = data['train_site']
        test_site = data['test_site']

        filename = f'Train_site_{train_site}_' \
                   f'Test_site_{test_site}_'   \
                   f'Model_{model}_'           \
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
    'Fluconazole': col_list[7],
    'Fusidic acid': col_list[9],
    'Cefepime': col_list[8],
    'Penicillin': col_list[10],
    'Imipenem': col_list[11],
    'Gentamicin': col_list[12],
    'Tetracycline': col_list[13],
    'Vancomycin': col_list[14],
    'Clindamycin': col_list[15],
    'Nitrofurantoin': col_list[16],
    'Tigecycline': col_list[17],
    'Tobramycin': col_list[18],
    'Amikacin': col_list[19],
    'Amoxicillin': col_list[20],
    'Ampicillin-Amoxicillin': col_list[21],
    'Anidulafungin': col_list[22],
    'Aztreonam': col_list[23],
    'Caspofungin': col_list[24],
    'Cefazolin': col_list[25],
    'Cefpodoxime': col_list[26],
    'Ceftazidime': col_list[27],
    'Cefuroxime': col_list[28],
    'Cotrimoxazol': col_list[29],
    'Daptomycin': col_list[30],
    'Ertapenem': col_list[31],
    'Erythromycin': col_list[32],
    'Fosfomycin-Trometamol': col_list[33],
    'Itraconazole': col_list[34],
    'Levofloxacin': col_list[35],
    'Micafungin': col_list[36],
    'Norfloxacin': col_list[37],
    'Rifampicin': col_list[38],
    'Teicoplanin': col_list[39],
    'Voriconazole': col_list[40],
    '5-Fluorocytosine': col_list[41]
    }

maldi_col_map_seaborn = {
    'Oxacillin': sns.color_palette()[0],
    'Ceftriaxone': sns.color_palette()[1],
    'Amoxicillin-Clavulanic acid': sns.color_palette()[7],
    'Meropenem': sns.color_palette()[3],
    'Piperacillin-Tazobactam': sns.color_palette()[4],
    'Ciprofloxacin': sns.color_palette()[5],
    'Fusidic acid': sns.color_palette()[6],
    'Cefepime': sns.color_palette()[2],
    'Penicillin': sns.color_palette()[8],
    'Tobramycin': sns.color_palette()[9],
    }


# antibiotic categorization
ab_cat_map = {'5-Fluorocytosine': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Amikacin': 'AMINOGLYCOSIDE ANTIBACTERIALS',
              'Aminoglycosides': 'AMINOGLYCOSIDE ANTIBACTERIALS',
              'Amoxicillin': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Amoxicillin-Clavulanic acid': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Amoxicillin-Clavulanic acid_uncomplicated_HWI': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Amphotericin B': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Ampicillin-Amoxicillin': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Anidulafungin': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Azithromycin': 'MACROLIDES, LINCOSAMIDES AND STREPTOGRAMINS',
              'Aztreonam': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Caspofungin': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Cefazolin': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Cefepime': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Cefixime': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Cefoxitin_screen': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Cefpodoxime': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Ceftazidime': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Ceftriaxone': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Cefuroxime': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Chloramphenicol': 'AMPHENICOLS',
              'Ciprofloxacin': 'QUINOLONE ANTIBACTERIALS',
              'Clarithromycin': 'MACROLIDES, LINCOSAMIDES AND STREPTOGRAMINS',
              'Clindamycin': 'MACROLIDES, LINCOSAMIDES AND STREPTOGRAMINS',
              'Colistin': 'OTHER ANTIBACTERIALS',
              'Cotrimoxazole': 'SULFONAMIDES AND TRIMETHOPRIM',
              'Daptomycin': 'OTHER ANTIBACTERIALS',
              'Doxycycline': 'TETRACYCLINES',
              'Ertapenem': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Erythromycin': 'MACROLIDES, LINCOSAMIDES AND STREPTOGRAMINS',
              'Fluconazole': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Fosfomycin': 'OTHER ANTIBACTERIALS',
              'Fosfomycin-Trometamol': 'OTHER ANTIBACTERIALS',
              'Fusidic acid': 'OTHER ANTIBACTERIALS',
              'Gentamicin': 'AMINOGLYCOSIDE ANTIBACTERIALS',
              'Gentamicin_high_level': 'AMINOGLYCOSIDE ANTIBACTERIALS',
              'Imipenem': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Itraconazole': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Levofloxacin': 'QUINOLONE ANTIBACTERIALS',
              'Linezolid': 'OTHER ANTIBACTERIALS',
              'Meropenem': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Meropenem_with_meningitis': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Meropenem_with_pneumonia': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Meropenem_without_meningitis': 'OTHER BETA-LACTAM ANTIBACTERIALS',
              'Metronidazole': 'OTHER ANTIBACTERIALS',
              'Micafungin': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Minocycline': 'TETRACYCLINES',
              'Moxifloxacin': 'QUINOLONE ANTIBACTERIALS',
              'Mupirocin': 'ANTIBIOTICS FOR TOPICAL USE',
              'Nitrofurantoin': 'OTHER ANTIBACTERIALS',
              'Norfloxacin': 'QUINOLONE ANTIBACTERIALS',
              'Oxacillin': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin_with_endokarditis': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin_with_meningitis': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin_with_other_infections': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin_with_pneumonia': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Penicillin_without_endokarditis': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Piperacillin-Tazobactam': 'BETA-LACTAM ANTIBACTERIALS, PENICILLINS',
              'Posaconazole': 'ANTIMYCOTICS FOR SYSTEMIC USE',
              'Quinolones': 'QUINOLONE ANTIBACTERIALS',
              'Rifampicin': 'DRUGS FOR TREATMENT OF TUBERCULOSIS',
              'Rifampicin_1mg-l': 'DRUGS FOR TREATMENT OF TUBERCULOSIS',
              'Teicoplanin': 'OTHER ANTIBACTERIALS',
              'Teicoplanin_GRD': 'OTHER ANTIBACTERIALS',
              'Tetracycline': 'TETRACYCLINES',
              'Tigecycline': 'TETRACYCLINES',
              'Tobramycin': 'AMINOGLYCOSIDE ANTIBACTERIALS',
              'Vancomycin': 'OTHER ANTIBACTERIALS',
              'Vancomycin_GRD': 'OTHER ANTIBACTERIALS',
              'Voriconazole': 'ANTIMYCOTICS FOR SYSTEMIC USE',
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
    'Cotrimoxazol': 'Cotrimoxazole',
    'Trimethoprim/Sulfamethoxazol': 'Cotrimoxazole',
    'SXT-Trimethoprim/Sulfamethoxazol': 'Cotrimoxazole',
    'Trimethoprim-Sulfame': 'Cotrimoxazole',
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
