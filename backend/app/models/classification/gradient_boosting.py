from sklearn.ensemble import GradientBoostingClassifier

def get_model():
    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=2, random_state=42)
