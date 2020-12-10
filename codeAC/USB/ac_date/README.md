The script 'read_acquisition_tim_from_fid.R ectracts the acquisition time and date from MALDI-TOF mass spectra whichwere acquired on a Bruker microflex system and are stored as fid files. 
It requires an input argument (directory in which the fid files are stored) and an output path, to where the resulting .csv should be written to.
This script was applied onto the DRIAMS-S dataset for each year and device separately, resulting in the6 files: 

acquisition_dates_2015.csv
acquisition_dates_2016.csv
acquisition_dates_2017_m1.csv
acquisition_dates_2017_m2.csv
acquisition_dates_2018_m1.csv
acquisition_dates_2018_m2.csv

The script 'Add_acquisition_date.R' adds the acquisition time and date information to the ID_RES files. 
If a measurement has been repeated (same 'brukercode'), it adds the time and date of later measurement.