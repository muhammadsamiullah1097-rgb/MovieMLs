from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def get_model():
    return Pipeline([("polynomial", PolynomialFeatures(degree=2, include_bias=False)), ("regression", LinearRegression())])
