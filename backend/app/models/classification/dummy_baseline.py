from sklearn.dummy import DummyClassifier

def get_model():
    return DummyClassifier(strategy="most_frequent")
