# determine spectra codes in id file with no corresponding spectra in /preprocessed
python ~/maldi-learn/utilities/list_missing_spectra.py /links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/id/2017/2017_clean.csv /links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/preprocessed/ > missing_spectra.txt

sed "s/$/',/" missing_spectra.txt >> missing_spectra_2.txt
rm missing_spectra.txt
sed "s/^/'/" missing_spectra_2.txt >> missing_spectra.txt
rm missing_spectra_2.txt

