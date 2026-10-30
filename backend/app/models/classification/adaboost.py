from sklearn.ensemble import AdaBoostClassifier

def get_model():
    return AdaBoostClassifier(n_estimators=100, learning_rate=0.7, random_state=42)
