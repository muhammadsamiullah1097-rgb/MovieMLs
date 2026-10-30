from sklearn.tree import DecisionTreeClassifier

def get_model():
    return DecisionTreeClassifier(criterion="gini", max_depth=6, random_state=42)
