import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(matrix, components=2):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = StandardScaler().fit_transform(matrix)
    components = max(1, min(int(components), matrix.shape[1], matrix.shape[0]))
    model = PCA(n_components=components, random_state=42)
    transformed = model.fit_transform(matrix)
    return {
        "components": components,
        "explained_variance": model.explained_variance_ratio_.round(4).tolist(),
        "total_explained_variance": round(float(model.explained_variance_ratio_.sum()), 4),
        "sample_projection": transformed[:10].round(4).tolist(),
    }
