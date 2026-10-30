from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def get_model():
    return BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6), n_estimators=40, n_jobs=1, random_state=42)
