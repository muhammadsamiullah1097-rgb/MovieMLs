from sklearn.ensemble import RandomForestRegressor

def get_model():
    return RandomForestRegressor(n_estimators=120, max_depth=None, n_jobs=1, random_state=42)
