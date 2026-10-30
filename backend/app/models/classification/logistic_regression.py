from sklearn.linear_model import LogisticRegression

def get_model():
    return LogisticRegression(max_iter=1500, random_state=42)
