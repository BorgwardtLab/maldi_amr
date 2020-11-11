#!/usr/bin/env python
# coding: utf-8
# Aline Cuénod
# 2020 / 05 / 12


# Load packages
import PyPDF2
import textract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from os.path import join
import re
import pandas as pd
import json
import glob
import numpy as np


# Define paths

PATH_spectra_usb = './usb_ec_sa_spectra/'
PATH_res_usb = './usb_ec_sa_res/'
PATH_output_usb = './Res_EcSa_USB.csv'

PATH_spectra_ksbl = './ksbl_ec_sa_spectra/'
PATH_res_ksbl = './ksbl_ec_sa_res/'
PATH_to_report_ksbl = './ksbl_ec_sa_bruker.csv'
PATH_output_ksbl = './Res_EcSa_ksbl.csv'


# # # # # # USB

#list all PDF Vitek2 reports
dir = PATH_res_usb
filenames = list()
for file in os.listdir(dir):
    if file.endswith(".pdf"):
        filenames = filenames + [os.path.join(dir, file)]


#read all text per file in dict entry, with ID as key
all = dict()
for filename in filenames:
    pdfFileObj = open(filename,'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj,strict = False)
    num_pages = pdfReader.numPages

    count = 0
    text = ""
    while count < num_pages:
        pageObj = pdfReader.getPage(count)
        count +=1
        text += pageObj.extractText()
        text = text
        ID = re.sub(r'.*\/','',filename)
        ID = re.sub(r'\..*','',ID)
        all[ID] = text
        all[ID] = re.sub(r'\n','_',all[ID])

# define Antibiotics to be read
AB = ['Ampicilin',
'Amoxicillin Clavulansäure',
'Amoxicillin/Clavulansäure',
'Piperacillin Tazobactam',
'Piperacillin/Tazobactam',
'Cefuroxim-Axetil',
'Cefuroxim',
'Ceftazidim',
'Ceftriaxon',
'Cefpodoxim',
'Cefoxitin',
'Cefepim',
'Ertapenem',
'Imipenem',
'Meropenem',
'Tobramycin',
'Amikacin',
'Trimethoprim/Sulfamethoxazol',
'Nitrofurantoin',
'Norfloxacin',
'Ciprofloxacin',
'Levofloxacin',
'Fosfomycin',
'Colistin',
'Benzylpenicillin',
'Cefoxitin-Screen',
'Oxacillin',
'Cefazolin',
'Erythromycin',
'Tetracyclin',
'Tigecycline',
'Induzierbare Clindamycin__Resistenz',
'InduzierbareClindamycinResistenz',
'Clindamycin',
'Gentamycin',
'Gentamicin',
'Vancomycin',
'Teicoplanin',
'Fusidinsäure',
'Rifampicin',
'Linezolid',
'Daptomycin',
'Mupirocin']


# extract info (i) sample ID, (ii) Species, and (iii) AB to dict
info = dict()
for ID in list(all):
    info[ID]= dict()
    if re.search(re.escape('Kommentar'), all[ID]):
        Kommentar = re.sub(r'\_Infos zur Identifizierung\_.*','',all[ID])
        Kommentar = re.sub(r'Kommentare\:\_*','',Kommentar)
        info[ID]['Kommentar'] = Kommentar
    Spezies = re.sub(r'.*\_Gew\ählter Keim\:\s','',all[ID])
    Spezies = re.sub(r'\_.*','',Spezies)
    Status = re.sub(r'.*Status\:','',all[ID])
    Status = re.sub(r'Analysen\-.*','',Status)
    info[ID]['Spezies'] = Spezies
    info[ID]['Status'] = Status

    info[ID]['Spezies'] = Spezies

    ab_dict = dict()
    for ab in AB:
        if re.search(re.escape(ab), all[ID]):
            ab_dict[ab] = dict()
            pattern1 = re.escape(ab) + r'(\d|\<|\>)([^_|^S|^R]{0,9})([S|R|I]{1})(.*)'
            pattern2 = re.escape(ab) + r'(\_*|NEG*|POS*)([S|R|I|\-|\+]{1})(.*)'
            pattern3 = re.escape(ab) + r'(\_*)(NEG|POS)([S|R|I|\-|\+]{1})(.*)'
            if re.search(pattern1, all[ID]):
                ab_dict[ab]['MHK'] = ''.join(re.search(pattern1, all[ID]).group(1,2))
            else:
                ab_dict[ab]['MHK'] = ''
            if re.search(pattern1, all[ID]):
                ab_dict[ab]['Interpretation'] = re.search(pattern1, all[ID]).group(3)
            elif re.search(pattern2, all[ID]):
                ab_dict[ab]['Interpretation'] = re.search(pattern2, all[ID]).group(2)
            elif re.search(pattern3, all[ID]):
                ab_dict[ab]['Interpretation'] = re.search(pattern3, all[ID]).group(3)
    info[ID]['AB'] = ab_dict


# build dataframe
df_all= pd.DataFrame()
for ID in list(all):
    df = pd.DataFrame(info[ID]['AB']).transpose().reset_index()
    df['TGNR'] = str(ID)
    df['Species'] = str(info[ID]['Spezies'])
    df['Status'] = str(info[ID]['Status'])
    df_all = df_all.append(df, sort=True)

# drop 'Cefoxitin -' for S. aureus, there were not measured, but are anartefact from the regex matching
df_all = df_all[~((df_all['Species'] == 'Staphylococcus aureus') & (df_all['index'] == 'Cefoxitin')& (df_all['Interpretation'] == '-'))]


# import usb spectra into dicts, translating from Brukercode to TGNR
return_list_1 = []
runinfo_all_1 = []

for root, dirs, files in os.walk(PATH_spectra_usb):
    for name in files:
        if name.startswith(("info")):
            with open(os.path.join(root, name)) as json_file:
                runinfo_all_1.append(json.load(json_file))

#
dicts_1 = []
dicts_all_1= []
for runinfo in runinfo_all_1:
        dicts_1.append({runinfo['AnalyteUid']:runinfo['AnalyteId']})

dicts_all_1 = {}
for d in dicts_1:
    dicts_all_1.update(d)


# Build dataframe and extract 6 - 10 digit sample ID
usb_Bruker = pd.DataFrame(dicts_all_1, index=[0]).transpose().reset_index()
usb_Bruker.columns = ['Bruker', 'TGNR']
usb_Bruker['ID'] = usb_Bruker['TGNR'].str.extract(r'(\d{6,15})')

# remove duplicate measurements of the same strain (same TGNR)
usb_Bruker = usb_Bruker.drop_duplicates('TGNR')

# translate typos
typos = dict()
typos['10062449flach-1'] =   '10062449Flach-1'
typos['10062449rund-1'] =   '10062449Rund-1'
typos['10067754flach-1'] =   '10067754Flach-1'
typos['10067754rund-1'] =   '10067754Rund-1'
typos['10070663gross-1'] =   '10070663Gross-1'
typos['10070663klein-1'] =   '10070663Klein-1'
typos['10071362rund-1'] =    '10071362Rund-1'
typos['10073097dunkle-1'] =   '10073097Dunkle-1'
typos['10073097hell-1'] =   '10073097Helle-1'
typos['3152402gross-1'] =   '3152402Gross-1'
typos['3152402klein-1'] =   '3152402Klein-1'
typos['8586683flach-1'] =  '8586683Flach-1'
typos['8586683rund-1'] =   '8586683Rund-1'
typos['8593577grau-1'] =   '8593577Grau-1'
typos['8593577weiss-1'] =   '8593577Weiss-1'
typos['8596102grau-1'] =   '8596102Grau-1'
typos['8596102weisss-1'] =   '8596102Weiss-1'

# harmonise and exctract 6-10 digit sample ID from res dataframe
res_usb = df_all.copy(deep=True)
res_usb['TGNR'] = res_usb['TGNR'].replace(typos, regex=True)
res_usb['ID'] = res_usb['TGNR'].str.extract(r'(\d{6,15})')

# only keep finished runs
res_usb = res_usb[res_usb['Status'] == '_Fertig_']
res_usb = res_usb.drop(['Status'], 1)

# merge the res and the Brukercode dataframes
usb_merge = res_usb.merge(usb_Bruker, left_on=['ID','TGNR'], right_on=['ID','TGNR'], how = 'inner')
usb_merge = usb_merge.rename({'index': 'AB'}, axis=1)

#exclude strains for which multiple morphotypes have been measured
usb_merge['count_TGNR'] = usb_merge.groupby('TGNR')['TGNR'].transform('count')
usb_merge['count_ID'] = usb_merge.groupby('ID')['ID'].transform('count')
usb_merge = usb_merge[usb_merge['count_TGNR']  >= usb_merge['count_ID']]
usb_merge = usb_merge.drop(['count_ID','count_TGNR'], 1)

# Harmonise spelling
usb_merge['AB'] = usb_merge['AB'].str.replace('InduzierbareClindamycinResistenz', 'Induzierbare Clindamycin__Resistenz')
usb_merge['AB'] = usb_merge['AB'].str.replace('Gentamicin', 'Gentamycin')
usb_merge['AB'] = usb_merge['AB'].str.replace('Amoxicillin Clavulansäure', 'Amoxicillin...Clavulansaeure')
usb_merge['AB'] = usb_merge['AB'].str.replace('Amoxicillin/Clavulansäure', 'Amoxicillin...Clavulansaeure')
usb_merge['AB'] = usb_merge['AB'].str.replace('Piperacillin Tazobactam', 'Piperacillin...Tazobactam')
usb_merge['AB'] = usb_merge['AB'].str.replace('Piperacillin/Tazobactam', 'Piperacillin...Tazobactam')
usb_merge['AB'] = usb_merge['AB'].str.replace('Trimethoprim/Sulfamethoxazol', 'Cotrimoxazol')
usb_merge['AB'] = usb_merge['AB'].str.replace('Cefoxitin-Screen', 'Cefoxitin.Screen')
usb_merge['AB'] = usb_merge['AB'].str.replace('Fusidinsäure', 'Fusidinsaeure')


# # # # # # KSBL


#list all PDF Vitek2 reports

dir = PATH_res_ksbl
filenames = list()
for file in os.listdir(dir):
    if file.endswith(r".pdf"):
        filenames = filenames + [os.path.join(dir, file)]

#read all text per file in dict entry, with sample ID as key
all = dict()
for filename in filenames:
    pdfFileObj = open(filename,'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj,strict = False)
    num_pages = pdfReader.numPages

    count = 0
    text = ""
    while count < num_pages:
        pageObj = pdfReader.getPage(count)
        count +=1
        text += pageObj.extractText()
        text = text
        ID = re.sub(r'.*\/','',filename)
        ID = re.sub(r'\..*','',ID)
        all[ID] = text
        all[ID] = re.sub(r'\n','_',all[ID])


# extract info  (i) sample ID, (ii) Species, and (iii) AB to dict

info = dict()
for ID in list(all):
    info[ID]= dict()
    if re.search(re.escape('Kommentar'), all[ID]):
        Kommentar = re.sub(r'InfoszurIdentifizierung.*','',all[ID])
        Kommentar = re.sub(r'Kommentare\*','',Kommentar)
        info[ID]['Kommentar'] = Kommentar
    Spezies = re.sub(r'.*Gew\ählterKeim\:','',all[ID])
    Spezies = re.sub(r'InstallierteVITEK2.*','',Spezies)
    Spezies = re.sub(r'Eingegeben.*','',Spezies)
    Status = re.sub(r'.*Status\:','',all[ID])
    Status = re.sub(r'Analysen\-Dauer.*','',Status)
    info[ID]['Spezies'] = Spezies
    info[ID]['Status'] = Status

    ab_dict = dict()
    for ab in AB:
        if re.search(re.escape(ab), all[ID]):
            ab_dict[ab] = dict()
            pattern1 = re.escape(ab) + r'(\d|\<|\>)([^_|^S|^R]{0,9})([S|R|I]{1})(.*)'
            pattern2 = re.escape(ab) + r'(\_*|NEG*|POS*)([S|R|I|\-|\+]{1})(.*)'
            pattern3 = re.escape(ab) + r'(\_*)(NEG|POS)([S|R|I|\-|\+]{1})(.*)'
            if re.search(pattern1, all[ID]):
                ab_dict[ab]['MHK'] = ''.join(re.search(pattern1, all[ID]).group(1,2))
            else:
                ab_dict[ab]['MHK'] = ''
            if re.search(pattern1, all[ID]):
                ab_dict[ab]['Interpretation'] = re.search(pattern1, all[ID]).group(3)
            elif re.search(pattern2, all[ID]):
                ab_dict[ab]['Interpretation'] = re.search(pattern2, all[ID]).group(2)
            elif re.search(pattern3, all[ID]):
                ab_dict[ab]['Interpretation'] = re.search(pattern3, all[ID]).group(3)
    info[ID]['AB'] = ab_dict


# build dataframe
df_all= pd.DataFrame()
for ID in list(all):
    df = pd.DataFrame(info[ID]['AB']).transpose().reset_index()
    ID2 = re.sub(r'Isolate\_','',ID)
    ID2 = re.sub(r'\_.*','',ID2)
    df['TGNR'] = str(ID2)
    df['Species'] = str(info[ID]['Spezies'])
    df['Status'] = str(info[ID]['Status'])
    df_all = df_all.append(df, sort=True)


# Add ID column
res_ksbl = df_all.copy(deep=True)

#some strains have been measured twice, remove duplicate entries
res_ksbl = res_ksbl.drop_duplicates()

# correct typos
res_ksbl['ID'] = res_ksbl['TGNR'].str.extract(r'(\d{6,15})')

# Include only antibiotics profiles of the species E. coli and S. aureus
res_ksbl = res_ksbl.loc[res_ksbl['Species'].isin(['Escherichiacoli','Staphylococcusaureus']),:]
res_ksbl.loc[:,'Species'] = res_ksbl.loc[:,'Species'].str.replace('Staphylococcusaureus', 'Staphylococcus aureus')
res_ksbl.loc[:,'Species'] = res_ksbl.loc[:,'Species'].str.replace('Escherichiacoli', 'Escherichia coli')

# only keep finished runs
res_ksbl = res_ksbl[res_ksbl['Status'] == 'Fertig']
res_ksbl = res_ksbl.drop(['Status'], 1)

# drop 'Cefoxitin -' for S. aureus, there were not measured, but are anartefact from the regex matching
res_ksbl[~((res_ksbl['Species'] == 'Staphylococcus aureus') & (res_ksbl['index'] == 'Cefoxitin')& (res_ksbl['Interpretation'] == '-'))]

res_ksbl['ID_AB'] = res_ksbl['ID'] + res_ksbl['index']
res_ksbl = res_ksbl[~res_ksbl['ID_AB'].isin(res_ksbl['ID_AB'][res_ksbl['ID_AB'].duplicated()])]


# import ksbl spectra into dicts, translating from Bruker encoding to TGNR for Brukercode
return_list_1 = []
runinfo_all_1 = []

for root, dirs, files in os.walk(PATH_spectra_ksbl):
    for name in files:
        if name.startswith(("info")):
            with open(os.path.join(root, name)) as json_file:
                runinfo_all_1.append(json.load(json_file))

#
dicts_1 = []
dicts_all_1= []
for runinfo in runinfo_all_1:
        dicts_1.append({runinfo['AnalyteUid']:runinfo['AnalyteId']})

dicts_all_1 = {}
for d in dicts_1:
    dicts_all_1.update(d)


# build dataframe and extract 6 - 10 digit sample ID
ksbl_Bruker = pd.DataFrame(dicts_all_1, index=[0]).transpose().reset_index()
ksbl_Bruker.columns = ['Bruker', 'TGNR_ksbl']
ksbl_Bruker['TGNR_ksbl'] = ksbl_Bruker['TGNR_ksbl'].replace(typos, regex=True)
ksbl_Bruker['ID'] = ksbl_Bruker['TGNR_ksbl'].str.extract(r'(\d{6,15})')


#import Bruker reports
ksbl_report = pd.read_csv(PATH_to_report_ksbl, sep=';', header= None)
ksbl_report = ksbl_report.drop(7, 1)
ksbl_report.columns = ['Bruker', '+', 'A', 'Organism_best_match', 'Score', 'Organism_second_best_match', 'Score2']

# Include only spectra of E. coli and S. aureus
ksbl_report.Organism_best_match.isin(['Staphylococcus aureus', 'Escherichia coli']).sum()
ksbl_report_sa_ec = ksbl_report.loc[ksbl_report.Organism_best_match.isin(['Staphylococcus aureus', 'Escherichia coli']),:]

# merge spectra to bruker reports
ksbl_spectra = ksbl_report_sa_ec.merge(ksbl_Bruker, on=['Bruker'], how = 'inner')

# only keep one spectra per ID. keep the one TGNR-ksbl for which the res profile has been acquired
# add 'merge' column, including TGNR_ksbl if TGNR:ksbl in res_ksbl, if not then ID
# remove duplicate with respectr to this column
ksbl_spectra['merge'] = np.where(ksbl_spectra['TGNR_ksbl'].isin(set(res_ksbl['TGNR'])), ksbl_spectra['TGNR_ksbl'],ksbl_spectra['ID'])
ksbl_spectra = ksbl_spectra.drop_duplicates(['merge'], keep= 'first')
ksbl_spectra = ksbl_spectra.drop(['+', 'A', 'Organism_best_match', 'Score', 'Organism_second_best_match', 'Score2'], 1)

# Merge res and spectra and export
# add 'merge' column, including TGNR_ksbl if TGNR:ksbl in res_ksbl, if not then ID, merge by this column
res_ksbl['merge'] = np.where(res_ksbl['TGNR'].isin(set(ksbl_spectra['TGNR_ksbl'])), res_ksbl['TGNR'],res_ksbl['ID'])

ksbl_merge = res_ksbl.merge(ksbl_spectra, on = ['ID','merge'], how='inner')
ksbl_merge = ksbl_merge.rename({'index': 'AB'}, axis=1)

# Harmonise spelling
ksbl_merge['AB'] = ksbl_merge['AB'].str.replace('InduzierbareClindamycinResistenz', 'Induzierbare Clindamycin__Resistenz')
ksbl_merge['AB'] = ksbl_merge['AB'].str.replace('Gentamicin', 'Gentamycin')
ksbl_merge['AB'] = ksbl_merge['AB'].str.replace('Amoxicillin Clavulansäure', 'Amoxicillin...Clavulansaeure')
ksbl_merge['AB'] = ksbl_merge['AB'].str.replace('Amoxicillin/Clavulansäure', 'Amoxicillin...Clavulansaeure')
ksbl_merge['AB'] = ksbl_merge['AB'].str.replace('Piperacillin Tazobactam', 'Piperacillin...Tazobactam')
ksbl_merge['AB'] = ksbl_merge['AB'].str.replace('Piperacillin/Tazobactam', 'Piperacillin...Tazobactam')
ksbl_merge['AB'] = ksbl_merge['AB'].str.replace('Trimethoprim/Sulfamethoxazol', 'Cotrimoxazol')
ksbl_merge['AB'] = ksbl_merge['AB'].str.replace('Cefoxitin-Screen', 'Cefoxitin.Screen')
ksbl_merge['AB'] = ksbl_merge['AB'].str.replace('Fusidinsäure', 'Fusidinsaeure')


#exclude strains for which multiple morphotypes have been measured
ksbl_merge['count_TGNR'] = ksbl_merge.groupby('TGNR')['TGNR'].transform('count')
ksbl_merge['count_ID'] = ksbl_merge.groupby('ID')['ID'].transform('count')
ksbl_merge = ksbl_merge[ksbl_merge['count_TGNR']  >= ksbl_merge['count_ID']]
ksbl_merge = ksbl_merge.drop(['count_ID','count_TGNR', 'merge', 'TGNR_ksbl'], 1)

#Only keep entries which are unambiguously present in both datasets

ksbl_merge['ID_AB'] = ksbl_merge['ID'] + ksbl_merge['AB']
usb_merge['ID_AB'] = usb_merge['ID'] + usb_merge['AB']

ksbl_merge = ksbl_merge[ksbl_merge['ID_AB'].isin(set(ksbl_merge['ID_AB']).intersection(set(usb_merge['ID_AB'])))]
usb_merge = usb_merge[usb_merge['ID_AB'].isin(set(ksbl_merge['ID_AB']).intersection(set(usb_merge['ID_AB'])))]

# remove ID_AB column
ksbl_merge = ksbl_merge.drop('ID_AB', 1)
usb_merge = usb_merge.drop('ID_AB', 1)

ksbl_merge.to_csv(PATH_output_ksbl)
usb_merge.to_csv(PATH_output_usb)
