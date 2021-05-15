#!/bin/bash

# DRIAMS-A clean stratification IDRES files
#python id_file_with_strat.py /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/ac_add_workstation/2018_01-08_IDRES_AB_not_summarised.csv \
#                        ./2018_strat.csv
#python id_file_with_strat.py /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/ac_add_workstation/2017-01-12_IDRES_AB_not_summarised.csv \
#                        ./2017_strat.csv
#python id_file_with_strat.py /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/ac_add_workstation/2016-01-12_IDRES_AB_not_summarised.csv \
#                        ./2016_strat.csv
#python id_file_with_strat.py /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/ac_add_workstation/2015-01-12_IDRES_AB_not_summarised.csv \
#                        ./2015_strat.csv
#python ./remove_overlap_17_18_strat.py
#python ./remove_missing_spectra_17_strat.py

# DRIAMS-A print some summary statistics for consistency checks
python check_consistency_strat_data.py --prev /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/2018/2018_01-08_IDRES_AB_not_summarised.csv \
                        --strat /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/ac_add_workstation/2018_01-08_IDRES_AB_not_summarised.csv
python check_consistency_strat_data.py --prev /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/2017/2017-01-12_IDRES_AB_not_summarised.csv \
                        --strat /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/ac_add_workstation/2017-01-12_IDRES_AB_not_summarised.csv
python check_consistency_strat_data.py --prev /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/2016/2016-01-12_IDRES_AB_not_summarised.csv \
                        --strat /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/ac_add_workstation/2016-01-12_IDRES_AB_not_summarised.csv
python check_consistency_strat_data.py --prev /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/2015/2015-01-12_IDRES_AB_not_summarised.csv \
                        --strat /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/codeAC/USB/ac_add_workstation/2015-01-12_IDRES_AB_not_summarised.csv

# DRIAMS-A create a csv that connects Bruker codes with duplication info
#python strat_summary_stats.py --path /links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/amr_maldi_ml/DRIAMS_preprocessing/id_files/stratification \
#                              --write_checkfile
