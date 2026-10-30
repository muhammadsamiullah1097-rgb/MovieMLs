from sklearn.ensemble import RandomForestClassifier

def get_model():
    return RandomForestClassifier(n_estimators=120, max_depth=None, n_jobs=1, random_state=42, class_weight="balanced_subsample")
