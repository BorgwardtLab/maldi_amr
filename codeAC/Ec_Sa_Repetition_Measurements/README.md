The sa_ec.py script matches spectra and antimicrobial resistance (AMR) profiles of S. aureus and E. coli strains.
MALDI-TOF MS spectra and antibiotic resistance profiles have been acquired at the Kantonsspital Basellandschaft (ksbl) and the University Hospital of Basel (usb).
Both laboratories used the Microflex Biotyper (Bruker Daltonics, Bremen, Germany) to acquire MALDI-TOF MS spectra, which are saved as fid files.
Each fid file has a unique 36 - digit identifier ('Brukercode'). With every MALDI-TOF MS measurement, a .json file is created, containing the translation of Brukercode to user defined sample ID. The Sample ID is a 6-10 digit number.  

Both laboratories use the Vitek2 (Biomérieux, Marcy-d'Etoile, France) to measure minimal inhibitory concentrations (MIC) and interpret them to sensitive, intermediate and resistance.
All AMR profiles are available on PDF files (Vitek2 reports).

This script reads the the Vitek2 reports and extracts the MIC and interpretation for every strain measured at the usb.
Further, the MALDI-TOF MS .json files acquired at the usb are read and the translation of Brukercode to sample ID is extracted.
Both dataframes are merged using the sample ID.
Strains for which multiple morphotypes / colonies have been measured are excluded from further analysis as these cannot unambiguously be assigned to the measurements at KSBL.

Subsequently, the Vitek2 reports acquired at the ksbl are read and the MIC and interpretation are extracted for every strain measured at the ksbl.
Further, the MALDI-TOF MS .json files acquired at the ksbl are read and the translation of Brukercode to sample ID is extracted.
At ksbl all MALDI-TOF MS spectra have been acquired in routine diagnostic. Here, all strains isolated from the same patient material are assigned the same sample ID. Therefor, the sample ID is not a unique identifier and spectra of multiple bacterial species can have the same sample ID.
In order to exclusively include spectra of the E.coli and S.aureus strains, the MALDI-TOF MS spectra acquired at ksbl were analysed using the Bruker Datebase (v.3) and the software flexControl. The results of this species identification have been saved at .csv.
This .csv file (ksbl_report) is imported and used to filter for MALDI-TOF MS spectra if the two species of interest.
Tis spectra information is then merged to the AMR profiles using the sample ID.
Strains for which multiple morphotypes / colonies have been measured are excluded from further analysis as these cannot unambiguously be assigned to the measurements at USB.

This script can be run using:
python3.6 ec_sa.py

The following paths need to me defined:
- Directory, where the MALDI-TOF MS spectra acquired at usb are stored, including the .json file
- Directory, where the Vitek2 reports acquired at usb are stored
- Outputpath for the merged usb dataframe

- Directory, where the MALDI-TOF MS spectra acquired at ksbl are stored, including the .json file
- Directory, where the Vitek2 reports acquired at ksbl are stored
- Path to the Bruker report .csv file, including species identification of ksbl spectra
- Outputpath for the merged ksbl dataframe
