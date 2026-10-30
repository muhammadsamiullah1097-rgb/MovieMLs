import pandas as pd


def clean_data(data):
    df = data.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].replace(r"^\s*$", pd.NA, regex=True)
    return df


def fill_missing(data, strategy="median"):
    df = data.copy()
    if strategy == "drop":
        return df.dropna().reset_index(drop=True)
    if strategy == "forward_fill":
        return df.ffill().bfill()
    if strategy == "interpolate":
        numeric = df.select_dtypes(include="number").columns
        df[numeric] = df[numeric].interpolate().bfill().ffill()
    elif strategy == "knn":
        from sklearn.impute import KNNImputer
        numeric = df.select_dtypes(include="number").columns
        if len(numeric):
            df[numeric] = KNNImputer(n_neighbors=5).fit_transform(df[numeric])
    else:
        for column in df.select_dtypes(include="number").columns:
            value = df[column].mean() if strategy == "mean" else df[column].median()
            df[column] = df[column].fillna(value)
    for column in df.select_dtypes(exclude="number").columns:
        mode = df[column].mode()
        df[column] = df[column].fillna(mode.iloc[0] if not mode.empty else "unknown")
    return df
