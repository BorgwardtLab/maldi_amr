
# coding: utf-8
# Pythonn3.6
# Aline Cuénod



########## 2018

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
PATH_to_spectra_MALDI2 = ''

#Path to Bruker reports
PATH_to_Bruker_reports_18 = '../Bruker_reports/'

# Path to the file extracted from the USB laboratory information system containing the AMR profiles acquired in routine diagnostics
PATH_TO_IDRES_MLAB = './ID_RES_2018-01-08.csv'

#Path to the output file
PATH_TO_Output = './2018_01-08_IDRES_AB_not_summarised.csv'


# make dicts, translating from Bruker encoding to TGNR for MALDI 1
# Read metadata (.json files), which are by default in the same directory as rawspectra
# read two variables in dictonary: (i) TGNR (= laboratory ID assigned in clinical routine) and (ii) Brukercode: coode assigned during the spectra acquisition.
## MALDI1
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

#Add string to code, to make it unambiguous, wehn combining MALDI1 and MALDI2
dicts_all_18_1 = dict(("{}{}".format(k,'_MALDI1'),v) for k,v in dicts_all.items())

## MALDI2
# make dicts, translating from Bruker encoding to TGNR for MALDI 1
return_list_1 = []
runinfo_all_1 = []

for root, dirs, files in os.walk(PATH_to_spectra_MALDI2):
    for name in files:
        if name.startswith(("info")):
            with open(os.path.join(root, name)) as json_file:
                runinfo_all_1.append(json.load(json_file))

dicts_1 = []
dicts_all_1= []
for runinfo in runinfo_all_1:
        dicts_1.append({runinfo['AnalyteUid']:runinfo['AnalyteId']})
dicts_all_1 = {}
for d in dicts_1:
    dicts_all_1.update(d)


#Add string to code, to make it unambiguous, wehn combining MALDI1 and MALDI2
dicts_all_18_2 = dict(("{}{}".format(k,'_MALDI2'),v) for k,v in dicts_all_1.items())

#Combine dicts from MALDI1 and MALDI2
dicts_all = {}
dicts_all = {**dicts_all_18_1, **dicts_all_18_2}

# Save dictionary as dataframe and add 'year' column
dicts_18_df = pd.DataFrame(dicts_all, index=[0]).transpose()
dicts_18_df['YEAR'] = '2018'

dicts_all_df = dicts_18_df

# rename columns and drop duplicates
dicts_all_df.index.name = 'code'
dicts_all_df.reset_index(inplace=True)
dicts_all_df.columns=['code', 'strain', 'YEAR']

# Read Bruker reports. These contain the microbial species identification per spectra, which was assigned by comparing each spectra to the Biotyper Database v.8.0.
## MALDI1
report_08_07 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_18, '2018_m1_07-08.csv'), error_bad_lines=False, sep=';', header=None)
report_05_06 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_18, '2018_m1_05-06.csv'), error_bad_lines=False, sep=';', header=None)
report_04_03 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_18, '2018_m1_03-04.csv'), error_bad_lines=False, sep=';', header=None)
report_01 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_18, '2018_m1_01.csv'), error_bad_lines=False, sep=';', header=None)
report_02 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_18, '2018_m1_02.csv'), error_bad_lines=False, sep=';', header=None)

# Combine reports from different months
report=pd.DataFrame()
report=report_08_07.append(report_05_06)
report=report.append(report_04_03)
report=report.append(report_01)
report=report.append(report_02)

#Drop 8th column as it is empty
report_m1 = report
report_m1 = report.drop(report.columns[7], axis=1)

#Rename columns
report_m1.columns = ['Brukername', 'Value','A','Organism_best_match', 'Score1', 'Organism(second best match)', 'Score2']

# Add '_MALDI1' to make it unique before matching with reports from MALDI2
report_m1['code'] = [row if re.match(r'\w{8}\-\w{4}\-\w{4}-\w{4}-\w{12}', row) else 'NA' for row in report_m1['Brukername']]
report_m1['code'] = [row + '_MALDI1' if not re.match('NA', row) else 'NA' for row in report_m1['code']]

# Remove the rows which contain 'not (yet) present' as identification, they have been reanalised
report_m1 = report_m1.drop(report_m1.index[report_m1.Organism_best_match == 'not (yet) present'])

#drop duplicated entries in case they have been multiple times in Brukeroutput
report_m1 = report_m1.drop_duplicates()

## MALDI2
report_08_07 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_18, '2018_m2_01-02.csv'), error_bad_lines=False, sep=';', header=None)
report_05_06 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_18, '2018_m2_03-04.csv'), error_bad_lines=False, sep=';', header=None)
report_04_03 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_18, '2018_m2_05-06.csv'), error_bad_lines=False, sep=';', header=None)
report_01_02 = pd.read_csv(os.path.join(PATH_to_Bruker_reports_18, '2018_m2_07-08.csv'), error_bad_lines=False, sep=';', header=None)
report_missing = pd.read_csv(os.path.join(PATH_to_Bruker_reports_18, '2018_m2-missing.csv'), error_bad_lines=False, sep=';', header=None)

# Combine reports from different months
report=pd.DataFrame()
report=report_08_07.append(report_05_06)
report=report.append(report_04_03)
report=report.append(report_01_02)
report=report.append(report_missing)

#Drop 8th column as it is empty
report_m2 = report
report_m2 = report.drop(report.columns[7], axis=1)

#Rename columns
report_m2.columns = ['Brukername', 'Value','A','Organism_best_match', 'Score1', 'Organism(second best match)', 'Score2']

# Add '_MALDI2' to make it unique before matching with reports from MALDI2
report_m2['code'] = [row if re.match(r'\w{8}\-\w{4}\-\w{4}-\w{4}-\w{12}', row) else 'NA' for row in report_m2['Brukername']]
report_m2['code'] = [row + '_MALDI2' if not re.match('NA', row) else 'NA' for row in report_m2['code']]

# Remove the rows which contain 'not (yet) present' as identification, they have been reanalised
report_m2 = report_m2.drop(report_m2.index[report_m2.Organism_best_match == 'not (yet) present'])

#drop duplicated entries in case they have been multiple times in Brukeroutput
report_m2 = report_m2.drop_duplicates()

report = pd.concat([report_m1, report_m2])
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

# Add '2018' string to TGNR to make it unique before combining it with data from other years
report_MALDI['TAGESNUMMER'] = '2018' + report_MALDI['TAGESNUMMER'].astype(str)

# Drop rows with empty TGNR
report_MALDI = report_MALDI[-report_MALDI['TAGESNUMMER'].isin(['2018N'])]

# Import AMR File which was extracted from USB laboratory information system
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
# Add 'SPEZIES_MLAB column
report_MALDI['SPEZIES_MALDI'] = report_MALDI['Organism_best_match'].str.split('\s').str[1]
report_MALDI['GENUS'] = report_MALDI['Organism_best_match'].str.split('\s').str[0]
report_MALDI['GENUS_match'] = report_MALDI['GENUS'].str.replace('MIX!', '')

# Merge with TGNR and GENUS in order to account for species differences between MLAB am MALDI
result_IDRES = pd.merge(report_MALDI, ID_RES, left_on=['GENUS_match', 'TAGESNUMMER'], right_on=['GENUS', 'TAGESNUMMER'], how = 'left')
result_IDRES = result_IDRES.drop(['GENUS_y', 'GENUS_match'], axis=1)
result_IDRES.rename(columns={'GENUS_x':'GENUS'}, inplace=True)

# Remove rows containing'not reliable identification', 'no peaks found' as species identification
result_IDRES = result_IDRES[-result_IDRES['Organism_best_match'].isin(['not reliable identification', 'no peaks found'])]

# Harmonise column names
result_IDRES.columns = result_IDRES.columns.str.replace("ä", "ae")
result_IDRES.columns = result_IDRES.columns.str.replace("Amoxicillin...Clavulansaeure.bei.unkompliziertem.HWI", "Amoxicillin-Clavulansaeure.unkompl.HWI")
result_IDRES.columns = result_IDRES.columns.str.replace("5.Fluorocytosin", "X5.Fluorocytosin")
result_IDRES.columns = result_IDRES.columns.str.replace("Organism_best_match", "Organism(best match)")
result_IDRES.drop('Brukername', axis=1, inplace=True)
result_IDRES.drop('YEAR', axis=1, inplace=True)

result_IDRES = result_IDRES[['Value', 'code', 'strain', 'A', 'Organism(best match)', 'Score1', 'Organism(second best match)', 'Score2', 'TAGESNUMMER', 'SPEZIES_MALDI', 'GENUS', 'SPEZIES_MLAB', 'MATERIAL', 'KEIM', 'AUFTRAGSNUMMER', 'STATION', 'PATIENTENNUMMER', 'GEBURTSDATUM', 'GESCHLECHT', 'EINGANGSDATUM', 'LOKALISATION', 'Penicillin', 'Ceftriaxon', 'Vancomycin', 'Piperacillin...Tazobactam', 'Ciprofloxacin', 'Cefepim', 'Cotrimoxazol', 'Meropenem', 'Moxifloxacin', 'Amoxicillin...Clavulansaeure', 'Colistin', 'Tobramycin', 'Ceftazidim', 'Ceftolozan...Tazobactam', 'Ceftazidim.Avibactam', 'Ceftobiprol', 'Chinolone', 'Ceftazidim.1', 'Tigecyclin', 'Levofloxacin', 'Fosfomycin', 'Amikacin', 'Imipenem', 'Minocyclin', 'Gentamicin', 'Ceftarolin', 'Ampicillin...Sulbactam', 'Gentamicin.High.level', 'Aztreonam', 'Clindamycin', 'Amoxicillin', 'Metronidazol', 'Daptomycin', 'Ampicillin...Amoxicillin', 'Caspofungin', 'Voriconazol', 'Posaconazol', 'Amphotericin.B', 'Itraconazol', 'Fluconazol', 'Erythromycin', 'Doxycyclin', 'Isavuconazol', 'Anidulafungin', 'X5.Fluorocytosin', 'Micafungin', 'Cefepim.1', 'Tetracyclin', 'Azithromycin', 'Ertapenem', 'Fosfomycin.Trometamol', 'Norfloxacin', 'Cefpodoxim', 'Nitrofurantoin', 'Dummy', 'Aminoglykoside', 'Chloramphenicol', 'Rifampicin.1.0.mg.l', 'Rifampicin', 'Linezolid', 'Amoxicillin-Clavulansaeure.unkompl.HWI', 'Streptomycin.High.level', 'Teicoplanin', 'Cefuroxim', 'Penicillin.bei.Endokarditis', 'Penicillin.ohne.Endokarditis', 'Meropenem.bei.Meningitis', 'Meropenem.ohne.Meningitis', 'Cefazolin', 'Oxacillin', 'Fusidinsaeure', 'Streptomycin.1.0.mg.l', 'Isoniazid.0.1.mg.l', 'Pyrazinamid.100.0.mg.l', 'Ethambutol.5.0.mg.l', 'Cefixim', 'Mupirocin', 'Vancomycin.GRD', 'Teicoplanin.GRD', 'Cefoxitin.Screen', 'Penicillin.bei.Meningitis', 'Clarithromycin', 'Penicillin.bei.anderen.Infekten', 'Penicillin.bei.Pneumonie', 'Meropenem.bei.Pneumonie', 'PATIENTENNUMMER_id', 'FALLNUMMER_id', 'AUFTRAGSNUMMER_id']]

# Write out resulting dataframe
result_IDRES = result_IDRES.replace(';',',')
pd.DataFrame.to_csv(result_IDRES, PATH_TO_Output  ,sep=',', index=False)
