from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def make_one_hot_encoder():
    return OneHotEncoder(handle_unknown="ignore")


def encode_target(values):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(values.astype(str))
    return encoded, encoder


def frequency_encode(series):
    counts = series.astype(str).value_counts(normalize=True)
    return series.astype(str).map(counts).astype(float)
