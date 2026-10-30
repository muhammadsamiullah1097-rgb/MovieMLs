from sklearn.tree import DecisionTreeRegressor

def get_model():
    return DecisionTreeRegressor(max_depth=8, random_state=42)
