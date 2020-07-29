# Direct Antimicrobial Resistance Prediction from MALDI-TOF mass spectra profile in clinical isolates through Machine Learning

[![N|Solid](https://ethz.ch/services/en/service/communication/corporate-design/logo/_jcr_content/par/twocolumn_1/par_left/fullwidthimage/image.imageformat.lightbox.1923118499.jpg)](https://bsse.ethz.ch/mlcb)

This code accompanies the paper &ldquo;Direct Antimicrobial Resistance Prediction from MALDI-TOF mass spectra profile in clinical isolates through Machine Learning&rdquo;
by Caroline Weis et al.


#### Preprocessing 

##### KSBL
TODO Aline's KSBL script

Conversion to IDRES_convert.csv, input for AMR and spectra matching
Run following Jupyter notebook (Contains Bash magic. Run as notebook, don't convert to Python script.)
[KSBL_convert_to_IDRES_input_format.ipynb][KSBL_convert]

##### USB
##### KSA
Run following Jupyter notebooks (Contain Bash magic. Run as notebooks, don't convert to Python scripts.)
[Aarau_PP1_divide_Aarau_spectra_into_packages_for_Bruker_DB_analysis.ipynb][KSA_pp1]
[Aarau_PP2_match_with_Bruker_DB_output_files.ipynb][KSA_pp2]

##### Viollier
Run following Jupyter notebook (Contains R magic. Run as notebook, don't convert to Python script.)
[Viollier_dataset_preprocessing.ipynb][Viollier_pp]

#### Main analysis 

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [KSBL_convert]: <https://github.com/cvweis/link/to/KSBL_convert_to_IDRES_input_format.ipynb>
   [KSA_pp1]: <https://github.com/cvweis/link/to/Aarau_PP1_divide_Aarau_spectra_into_packages_for_Bruker_DB_analysis.ipynb>
   [KSA_pp2]: <https://github.com/cvweis/link/to/Aarau_PP2_match_with_Bruker_DB_output_files.ipynb>
   [Viollier_pp]: <https://github.com/cvweis/link/to/Viollier_dataset_preprocessing.ipynb>
