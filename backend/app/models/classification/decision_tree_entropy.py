from sklearn.tree import DecisionTreeClassifier

def get_model():
    return DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42)
