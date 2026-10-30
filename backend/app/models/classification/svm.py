from sklearn.svm import SVC

def get_model():
    return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)
