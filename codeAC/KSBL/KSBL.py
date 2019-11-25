# coding: utf-8
# Python3.6
# Aline Cuénod


# Load packages
import json
import os
from os.path import join
import numpy as np
import re
import pandas as pd


# # KSBL
# Define input paths
PATH_to_spectra = ''

# Define input report
PATH_to_report = './Bruker_reports/'

# Path to the file extracted from the USB laboratory information system containing the AMR profiles acquired in routine diagnostics
PATH_TO_IDRES_MLAB = './DatenexportResiKSBLJanbisJun2018fuerMALDIStudie.csv'

# Define output paths
PATH_to_output = './KSBL_res_report.csv'

# Create dictionary, containing the Brukercode and the TGNR per spectra
# Read metadata (.json files), which are by default in the same directory as rawspectra
# Read two variables in dictonary: (i) TGNR (= laboratory ID assigned in clinical routine) and (ii) Brukercode: coode assigned during the spectra acquisition. 

return_list_1 = []
runinfo_all_1 = []

for root, dirs, files in os.walk(PATH_to_spectra):
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
    
# Create a second dictionary, including the date the spectra was acquired
dicts_2 = []
for runinfo in runinfo_all_1: 
        dicts_2.append({runinfo['AnalyteUid']:runinfo['ProjectName']})

dicts_all_2 = {}
for d in dicts_2:
    dicts_all_2.update(d)

# Add string to code to make it unambiguous when combining with spectra from other devices
dicts_all_15 = dict(("{}{}".format(k,'_MALDI1'),v) for k,v in dicts_all_1.items())

# Build dataframes from dictionaries which contain: Brukercode, samplename and acquisition date
df_date = pd.DataFrame([dicts_all_2]).T
df_date.index.name = 'Bruker'
df_date.reset_index(inplace=True)
df_date.columns = ['Bruker', 'Projectname']

df_samplename = pd.DataFrame([dicts_all_1]).T
df_samplename.index.name = 'Bruker'
df_samplename.reset_index(inplace=True)
df_samplename.columns = ['Bruker', 'Samplename']

df = pd.DataFrame.merge(df_samplename,df_date, on='Bruker')


# Load species identification reports, outputted from the Bruker Database
report_01 = pd.read_csv(os.path.join(PATH_to_report, 'KSBL-1.csv'),  sep=';', header = None)
report_02 = pd.read_csv(os.path.join(PATH_to_report, 'KSBL-2.csv'),  sep=';', header = None)
report_03 = pd.read_csv(os.path.join(PATH_to_report, 'KSBL-3.csv'),  sep=';', header = None)
report_04 = pd.read_csv(os.path.join(PATH_to_report, 'KSBL-4.csv'),  sep=';', header = None)


report=pd.DataFrame()
report=report.append(report_01)
report=report.append(report_02)
report=report.append(report_03)
report=report.append(report_04)


#Drop 8th column as it is empty
report = report.drop(report.columns[7], axis=1)

# Rename columns and drop duplicates
report.columns = ['Bruker', 'Value','A','Organism_best_match', 'Score1', 'Organism(second best match)', 'Score2']
report = pd.DataFrame(report.drop_duplicates())


# Merge report to dict_df_all using Bruker code
report_TGNR = pd.merge(df, report, how='right', on='Bruker')


# Load in AMR profiles
#res = pd.read_csv(PATH_to_report),  sep=';', header = None)
res = pd.read_csv(PATH_TO_IDRES_MLAB )
res['Auftrag'] = res['Auftrag'].astype(str)
res['Auftrag'] = res['Auftrag'].str.extract('(\d{7})', expand=False).str.strip()
res['Keim']=res['Keim'].str.strip(' ')
res['SPEZIES_RES'] = res['Keim'].str.split('\s').str[1]
res['GENUS'] = res['Keim'].str.split('\s').str[0]


# Extract species and genus identifies by the Bruker Database. Create 'GENUS' column for matching
report_TGNR['Organism_best_match']=report_TGNR['Organism_best_match'].str.strip(' ')
report_TGNR['SPEZIES_MALDI'] = report_TGNR['Organism_best_match'].str.split('\s').str[1]
report_TGNR['GENUS'] = report_TGNR['Organism_best_match'].str.split('\s').str[0]
report_TGNR['Auftrag'] = report_TGNR['Samplename'].str.split('\-').str[0]


# Merge report_TGNR with res file using the Auftragsnummer and Genus
res_report = pd.merge(res,report_TGNR, on=('Auftrag', 'GENUS'), how='right')


# Write output
res_report.to_csv(PATH_to_output, sep=';')

