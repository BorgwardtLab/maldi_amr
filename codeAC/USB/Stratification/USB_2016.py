# coding: utf-8
# Pythonn3.6
# Aline Cuénod


########## 2016

# Load packages
import json
import os
from os.path import join
import numpy as np
import re
import pandas as pd

## Define paths
# Path to MALDI spectra
PATH_to_spectra_MALDI1 = ''

#Path to Bruker reports
PATH_to_Bruker_reports_16 = '../Bruker_reports/'

# Path to the file extracted from the USB laboratory information system containing the AMR profiles acquired in routine diagnostics
PATH_TO_IDRES_MLAB = './ID_RES_2016.csv'

#Path to the output file
PATH_TO_Output = './2016-01-12_IDRES_AB_not_summarised.csv'

# Create dictionary, containing the Brukercode and the TGNR per spectra
# Read metadata (.json files), which are by default in the same directory as rawspectra
# read two variables in dictonary: (i) TGNR (= laboratory ID assigned in clinical routine) and (ii) Brukercode: coode assigned during the spectra acquisition.

return_list_1 = []
runinfo_all_1 = []

for root, dirs, files in os.walk(PATH_to_spectra_MALDI1):
    for name in files:
        if name.startswith(("info")):
            with open(os.path.join(root, name)) as json_file:
                runinfo_all_1.append(json.load(json_file))

dicts_1 = []
dicts_all = []
for runinfo in runinfo_all_1:
        dicts_1.append({runinfo['AnalyteUid']:runinfo['AnalyteId']})
dicts_all = {}
for d in dicts_1:
    dicts_all.update(d)

# Save dictionary as dataframe and add 'year' column
dicts_16_df = pd.DataFrame(dicts_all, index=[0]).transpose()
dicts_16_df['YEAR'] = '2016'

dicts_all_df = dicts_16_df

# rename columns and drop duplicates
dicts_all_df.index.name = 'code'
dicts_all_df.reset_index(inplace=True)
dicts_all_df.columns=['code', 'strain', 'YEAR']

# Read Bruker reports. These contain the microbial species identification per spectra, which was assigned by comparing each spectra to the Biotyper Database v.8.0.
report_12 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_12.csv'), error_bad_lines=False, sep=';', header=None)
report_11 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_11.csv'), error_bad_lines=False, sep=';', header=None)
report_10 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_10.csv'), error_bad_lines=False, sep=';', header=None)
report_09 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_09.csv'), error_bad_lines=False, sep=';', header=None)
report_08 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_08.csv'), error_bad_lines=False, sep=';', header=None)
report_07 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_07.csv'), error_bad_lines=False, sep=';', header=None)
report_06 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_06.csv'), error_bad_lines=False, sep=';', header=None)
report_05 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_05.csv'), error_bad_lines=False, sep=';', header=None)
report_04 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_04.csv'), error_bad_lines=False, sep=';', header=None)
report_03 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_03.csv'), error_bad_lines=False, sep=';', header=None)
report_02 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_02.csv'), error_bad_lines=False, sep=';', header=None)
report_01 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_16, '2016_01.csv'), error_bad_lines=False, sep=';', header=None)



# Combine the reports from all months and rename columns
report=pd.DataFrame()
report=report_12.append(report_11)
report=report.append(report_10)
report=report.append(report_09)
report=report.append(report_08)
report=report.append(report_07)
report=report.append(report_06)
report=report.append(report_05)
report=report.append(report_04)
report=report.append(report_03)
report=report.append(report_02)
report=report.append(report_01)
report=report.append(report_05)
report=report.append(report_04)
report=report.append(report_03)
report=report.append(report_02)
report=report.append(report_01)

#Drop 8th column as it is empty
report = report.drop(report.columns[7], axis=1)


report.columns = ['Brukername', 'Value','A','Organism_best_match', 'Score1', 'Organism(second best match)', 'Score2']

# Drop duplicated 'Brukername'.
report = report.drop_duplicates()

# Add 'code' column in order to merge with 'dicts_all_df'
report['code'] = [row if re.match(r'\w{8}\-\w{4}\-\w{4}-\w{4}-\w{12}', row) else 'NA' for row in report.Brukername]

# Save 'code' as string
report['code'] = report['code'].astype(str)
report = report.reset_index(drop=True)

# Extract 6-digit TGNR from strainname in order to match with the AMR profile acquired at USB
dicts_all_df['TAGESNUMMER'] = [re.findall(r'\d{6}', row) if re.match(r'\d{6}', row) else 'NA' for row in dicts_all_df.strain]
dicts_all_df['TAGESNUMMER'] = [match[0] for match in dicts_all_df.TAGESNUMMER]

# Merge dicts_all_df with Bruker reports
report_MALDI = pd.merge(dicts_all_df, report, on=['code'], how='outer')
report_MALDI = report_MALDI.drop_duplicates()

# Drop rows with empty TGNR
report_MALDI.TAGESNUMMER.fillna('', inplace= True)
report_MALDI= report_MALDI[report_MALDI.TAGESNUMMER.notnull()]

# Add '2016' string to TGNR to make it unique before combining it with data from other years
report_MALDI['TAGESNUMMER'] = '2016' + report_MALDI['TAGESNUMMER'].astype(str)

# Drop rows with empty TGNR
report_MALDI = report_MALDI[-report_MALDI['TAGESNUMMER'].isin(['2016N'])]

# Import AMR File, which was extracted from USB laboratory information system
ID_RES = pd.DataFrame(pd.read_csv(PATH_TO_IDRES_MLAB, error_bad_lines=False, sep=';', low_memory=False))

# Drop rows with 'NA' as TAGESNUMMER
ID_RES['TAGESNUMMER'] = ID_RES['TAGESNUMMER'].astype(str)
ID_RES = ID_RES[ID_RES['TAGESNUMMER'].str.contains("NA")== False]

# Add 'GENUS' column for matching
ID_RES['GENUS'] = ID_RES['KEIM'].str.split('\s').str[0]
# Add 'SPEZIES_MLAB column
ID_RES['SPEZIES_MLAB'] = ID_RES['KEIM'].str.split('\s').str[1]
cols = ID_RES.columns.tolist()
cols = cols[-1:] + cols[:-1]
ID_RES = ID_RES[cols]
ID_RES['SPEZIES_MLAB']=ID_RES['SPEZIES_MLAB'].str.replace('\xa0',' ')

# Add 'GENUS' column for matching
report_MALDI['Organism_best_match']=report_MALDI['Organism_best_match'].str.strip(' ')
# Add 'SPEZIES_MLAB colum
report_MALDI['SPEZIES_MALDI'] = report_MALDI['Organism_best_match'].str.split('\s').str[1]
report_MALDI['GENUS'] = report_MALDI['Organism_best_match'].str.split('\s').str[0]
report_MALDI['GENUS_match'] = report_MALDI['GENUS'].str.replace('MIX!', '')

# Merge with TGNR and GENUS in order to account for species differences between MLAB am MALDI
result_IDRES = pd.merge(report_MALDI, ID_RES, left_on=['GENUS_match', 'TAGESNUMMER'], right_on=['GENUS', 'TAGESNUMMER'], how = 'left')
result_IDRES = result_IDRES.drop(['GENUS_y', 'GENUS_match'], axis=1)
result_IDRES.rename(columns={'GENUS_x':'GENUS'}, inplace=True)

# Remove rows containing'not reliable identification', 'no peaks found' as species identification
result_IDRES = result_IDRES[-result_IDRES['Organism_best_match'].isin(['not reliable identification', 'no peaks found'])]


# Harmonize columnnames
result_IDRES.columns = result_IDRES.columns.str.replace("ä", "ae")
result_IDRES.columns = result_IDRES.columns.str.replace("Amoxicillin...Clavulansaeure.bei.unkompliziertem.HWI", "Amoxicillin-Clavulansaeure.unkompl.HWI")
result_IDRES.columns = result_IDRES.columns.str.replace("Organism_best_match", "Organism(best match)")
result_IDRES.drop('Brukername', axis=1, inplace=True)
result_IDRES.drop('YEAR', axis=1, inplace=True)


result_IDRES = result_IDRES[['Value', 'code', 'strain', 'A', 'Organism(best match)', 'Score1', 'Organism(second best match)', 'Score2', 'TAGESNUMMER', 'SPEZIES_MALDI', 'GENUS', 'SPEZIES_MLAB', 'MATERIAL', 'KEIM', 'AUFTRAGSNUMMER', 'STATION', 'PATIENTENNUMMER', 'GEBURTSDATUM', 'GESCHLECHT', 'EINGANGSDATUM', 'LOKALISATION', 'Ciprofloxacin', 'Cefepim', 'Meropenem', 'Piperacillin...Tazobactam', 'Cotrimoxazol', 'Ceftazidim', 'Levofloxacin', 'Colistin', 'Tobramycin', 'Ceftriaxon', 'Imipenem', 'Amikacin', 'Tigecyclin', 'Clindamycin', 'Amoxicillin...Clavulansaeure', 'Amoxicillin', 'Posaconazol', 'Itraconazol', 'Voriconazol', 'Caspofungin', 'Amphotericin.B', 'Penicillin', 'Vancomycin', 'Ertapenem', 'Metronidazol', 'Moxifloxacin', 'Rifampicin', 'Erythromycin', 'Fluconazol', 'Anidulafungin', 'X5.Fluorocytosin', 'Micafungin', 'Ampicillin...Amoxicillin', 'Norfloxacin', 'Fosfomycin.Trometamol', 'Cefpodoxim', 'Chloramphenicol', 'Aminoglykoside', 'Chinolone', 'Daptomycin', 'Teicoplanin', 'Linezolid', 'Gentamicin', 'Gentamicin.High.level', 'Nitrofurantoin', 'Cefuroxim', 'Meropenem.ohne.Meningitis', 'Meropenem.bei.Meningitis', 'Ceftazidim.1', 'Fosfomycin', 'Aztreonam', 'Cefazolin', 'Tetracyclin', 'Fusidinsaeure', 'Oxacillin', 'Clarithromycin', 'Isoniazid.0.1.mg.l', 'Streptomycin.1.0.mg.l', 'Rifampicin.1.0.mg.l', 'Ethambutol.5.0.mg.l', 'Pyrazinamid.100.0.mg.l', 'Azithromycin', 'Cefixim', 'Doxycyclin', 'Mupirocin', 'Vancomycin.GRD', 'Teicoplanin.GRD', 'Cefoxitin.Screen', 'Ceftarolin', 'Ticarcillin...Clavulansaeure', 'Penicillin.bei.Endokarditis', 'Penicillin.ohne.Meningitis', 'Penicillin.ohne.Endokarditis', 'Dummy', 'Penicillin.bei.Pneumonie', 'Penicillin.bei.anderen.Infekten', 'Penicillin.bei.Meningitis', 'Meropenem.bei.Pneumonie', 'Cefepim.1', 'Minocyclin', 'Cefuroxim.Axetil', 'Amoxicillin-Clavulansaeure.unkompl.HWI', 'Ceftazidim.Avibactam', 'Ceftolozan...Tazobactam', 'Ampicillin...Sulbactam', 'Ceftobiprol', 'Bacitracin', 'Isoniazid.0.4.mg.l', 'Streptomycin.High.level', 'Isavuconazol', 'PATIENTENNUMMER_id', 'FALLNUMMER_id', 'AUFTRAGSNUMMER_id']]

# Write out resulting dataframe
result_IDRES = result_IDRES.replace(';',',')
pd.DataFrame.to_csv(result_IDRES, PATH_TO_Output ,sep=',',index=False)
