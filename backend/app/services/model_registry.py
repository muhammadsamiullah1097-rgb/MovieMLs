from app.models.classification import logistic_regression, balanced_logistic, decision_tree_gini, decision_tree_entropy
from app.models.classification import random_forest, bagging, adaboost, gradient_boosting, svm, naive_bayes, knn, dummy_baseline
from app.models.regression import linear_regression, multiple_linear_regression, polynomial_regression
from app.models.regression import ridge_regression, lasso_regression, decision_tree_regression, random_forest_regression, knn_regression
from app.models.text import tfidf_naive_bayes, countvectorizer_naive_bayes

CLASSIFIERS = {
    "logistic_regression": ("Logistic Regression", logistic_regression.get_model),
    "balanced_logistic": ("Logistic Regression (Class Weight)", balanced_logistic.get_model),
    "decision_tree_gini": ("Decision Tree (Gini)", decision_tree_gini.get_model),
    "decision_tree_entropy": ("Decision Tree (Entropy)", decision_tree_entropy.get_model),
    "random_forest": ("Random Forest", random_forest.get_model),
    "bagging": ("Bagging Classifier", bagging.get_model),
    "adaboost": ("AdaBoost", adaboost.get_model),
    "gradient_boosting": ("Gradient Boosting", gradient_boosting.get_model),
    "svm": ("Support Vector Machine", svm.get_model),
    "naive_bayes": ("Gaussian Naive Bayes", naive_bayes.get_model),
    "knn": ("K-Nearest Neighbours", knn.get_model),
    "dummy_baseline": ("Majority Class Baseline", dummy_baseline.get_model),
}

REGRESSORS = {
    "linear_regression": ("Simple Linear Regression", linear_regression.get_model),
    "multiple_linear_regression": ("Multiple Linear Regression", multiple_linear_regression.get_model),
    "polynomial_regression": ("Polynomial Regression", polynomial_regression.get_model),
    "ridge_regression": ("Ridge Regression", ridge_regression.get_model),
    "lasso_regression": ("Lasso Regression", lasso_regression.get_model),
    "decision_tree_regression": ("Decision Tree Regressor", decision_tree_regression.get_model),
    "random_forest_regression": ("Random Forest Regressor", random_forest_regression.get_model),
    "knn_regression": ("K-Nearest Neighbours Regressor", knn_regression.get_model),
}

TEXT_MODELS = {
    "tfidf_naive_bayes": ("TF-IDF + Multinomial Naive Bayes", tfidf_naive_bayes.get_model),
    "countvectorizer_naive_bayes": ("CountVectorizer + Multinomial Naive Bayes", countvectorizer_naive_bayes.get_model),
}
