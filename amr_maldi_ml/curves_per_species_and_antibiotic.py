"""
"""

import dotenv
import os

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSDataset
from maldi_learn.driams import DRIAMSLabelEncoder

from maldi_learn.driams import load_driams_dataset

from maldi_learn.vectorization import BinningVectorizer
from maldi_learn.utilities import stratify_by_species_and_label

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')
       
explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)


driams_dataset = load_driams_dataset(
            explorer.root,
            'DRIAMS-A',
            ['2015', '2017'],
            'Staphylococcus aureus',
            ['Ciprofloxacin', 'Penicillin.ohne.Meningitis'],
            encoder=DRIAMSLabelEncoder(),
            handle_missing_resistance_measurements='remove_if_all_missing',
)


# bin spectra
bv = BinningVectorizer(100, min_bin=2000, max_bin=20000)
X = bv.fit_transform(driams_dataset.X)

# train-test split
index_train, index_test = stratify_by_species_and_label(driams_dataset.y, antibiotic='Ciprofloxacin')
print(index_train)
print(index_test)

y = driams_dataset.to_numpy('Ciprofloxacin')
print(y[index_train].dtype)
print(X[index_train].shape)

lr = LogisticRegression()
lr.fit(X[index_train], y[index_train])
y_pred = lr.predict(X[index_test])

print(accuracy_score(y_pred, y[index_test]))
