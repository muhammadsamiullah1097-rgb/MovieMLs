from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def get_scaler(name="standard"):
    choices = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    return choices.get(name, StandardScaler())
