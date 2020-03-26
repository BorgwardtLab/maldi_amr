#!/bin/bash

# DRIAMS-A
python DRIAMS-A_clean_id_file.py /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/2018/2018_01-08_IDRES_AB_not_summarised.csv \
                        /links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/id/2018/2018_clean.csv
python DRIAMS-A_clean_id_file.py /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/2017/2017-01-12_IDRES_AB_not_summarised.csv \
                        /links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/id/2017/2017_clean.csv
python DRIAMS-A_clean_id_file.py /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/2016/2016-01-12_IDRES_AB_not_summarised.csv \
                        /links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/id/2016/2016_clean.csv
python DRIAMS-A_clean_id_file.py /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/2015/2015-01-12_IDRES_AB_not_summarised.csv \
                        /links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/id/2015/2015_clean.csv

python remove_overlap_17_18.py

# DRIAMS-B
# python DRIAMS-B_clean_id_file.py /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/KSBL/KSBL_res_report.csv \
#                         /links/groups/borgwardt/Data/DRIAMS/DRIAMS-B/id/2018/2018_clean.csv 
