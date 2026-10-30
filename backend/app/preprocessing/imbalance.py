from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier


def make_smote(random_state=42):
    return SMOTE(random_state=random_state)


def make_baseline(random_state=42):
    return DummyClassifier(strategy="most_frequent", random_state=random_state)
