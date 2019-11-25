This scripts matches MALDI-TOF mass spectra to Antimicrobial resistance (AMR) profiles of the microbial strains which were routinely analysed at the Cantonal Hospital of Baselland (KSBL) between 06/2018 and 12/2018. 

For microbial species identification have all mass spectra been compared to the Bruker Datebase (v.3) using the software flexControl. This comparison has beed done in portions of 4'000 - 10'000 spectra and the resulting species identification tables were saved to separate .csv files. 
AMR profiles measured in routine diagnostic have been queried from the laboratory information system (LIS) of the KSBL. 

This script uses three input arguments, for which the paths need to ba adjusted: 

(i)   Paths to the MALDI-TOF mass spectra
(ii)  Paths to the species identification tables
(iii) Paths to the AMR profiles. 

It produces and output file called 'KSBL_res_report.csv'. 
After the paths have been adjusted this script can be run with: 

python3.6 KSBL.py

MALDI-TOF mass spectra are labelled with a 36 digit code ('Brukercode') as they are acquired. 
With every MALDI-TOF MS target plate measured, a .json file is created which contains the information which Brukercode translates to which Sample ID. 
In order to match the mass spectra to the AMR profiled of the same sample, these .json files are imported and a dataframe is created including a unique Brukercode and the assigned sample ID per row. 
These dataframes are then merged to the species identification using the Brukercodes. The resulting dataframe includes sample ID, Brukercode and species identification per row. 

The next step is to combine this dataframe to the AMR profiles acquired in routine disgnostics. 
All AMR profiles of microbial strains which are isolated from the same patient material are assigned the same 10 digit sample ID. 
For example a Klebsiella pneumoniae strain and Escherichia coli strain isolated from the same urine sample are assigned the same 10 digit sample ID:

1000000000; Escherichia coli; 			R;	AB1;
1000000000; Escherichia coli; 			S;	AB2;
1000000000; Escherichia coli; 			S;	AB3;
1000000000; Klebsiella pneumoniae; 		S;	AB1;
1000000000; Klebsiella pneumoniae; 		S;	AB2;
1000000000; Klebsiella pneumoniae; 		R;	AB3;

The species identification by MALDI-TOF MS does not always match the species assigned in the LIS. A possible reason for this can be, that the MALDI-TOF MS system has assigned a species for which in house experience / research studies have shown that current MALDI-TOF MS approaches have difficulties to distinguish it from other closely related species (eg Burkholderia cepacia / Burkholderia cenocepacia or Streptococcus mitis / Streprococcus pneumoniae). 
These species also most often raise a warning by the MALDI-TOF MS system. Such strains get either (a) assigned a higher level identification in the LIS or (b), if the distinction between these closely related species has a clinical impact, further diagnostic tests (e.g. biochemical profiling) are performed to identify the strain on species level.
	
	Example: 
	(a) Species identification by MALDI-TOF MS: Burkholderia cenocepacia -> Species identity in LIS: Burkholderia cepacia complex
	(b) Species identification by MALDI-TOF MS: Streprococcus mitis 	 -> Species identity in LIS: Streptococcus pneumoniae (additional diagnostic tests performed)
	
Whereas the species identity of a strain can vary between the two datasets (MALDI-TOF MS and LIS), the assigned genus is matching in most cases and polymicrobial infections with strains of the same genus and different species, isolated from the same patient material, are rare. 
Thus, the MALDI-TOF MS dataframes and the AMR profiles are merged using the sample ID and the genus of the stain of interest. 
The resulting dataframe includes the Brukercode, the species ID and the AMR profile per row. 

	
	
	
	
