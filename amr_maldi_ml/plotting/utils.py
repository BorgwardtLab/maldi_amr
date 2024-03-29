# define color maps
import seaborn as sns

col_list = [u'orangered', u'purple', u'royalblue', u'olivedrab', u'orange', u'violet',  u'red', u'turquoise', u'chartreuse', u'black', u'saddlebrown', u'salmon', u'mediumvioletred', u'seagreen', u'skyblue', u'slateblue', u'darkgrey', u'springgreen', u'teal', u'tomato', u'peru', u'yellowgreen', u'aqua', u'aquamarine', u'blue', u'blueviolet', u'brown', u'burlywood', u'cadetblue', u'chocolate', u'coral', u'cornflowerblue', u'crimson', u'darkblue', u'darkcyan', u'darkgoldenrod', u'darkgreen', u'darkgrey', u'darkmagenta', u'hotpink', u'darkolivegreen', u'yellow']

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

scenario_map = {
    'Escherichia_coli': 'E-CEF',
    'Klebsiella_pneumoniae': 'K-CEF',
    'Staphylococcus_aureus': 'S-OXA',
}
