from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def select_features(x, y, count=8):
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in x.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical))
    prep = ColumnTransformer(transformers)
    matrix = prep.fit_transform(x)
    names = prep.get_feature_names_out()
    count = max(1, min(int(count), matrix.shape[1]))
    selector = RFE(LogisticRegression(max_iter=1000, class_weight="balanced"), n_features_to_select=count, step=0.2)
    selector.fit(matrix, y)
    selected = names[selector.support_].tolist()
    return selected, selector.ranking_.tolist()
