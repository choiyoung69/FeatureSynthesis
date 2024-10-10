###############################################################################
#Feature Synthesis
#Feature Selection 함수 모음
###############################################################################

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_squared_log_error,
    explained_variance_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm  # tqdm 임포트
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.base import is_classifier
from sklearn.datasets import make_classification
import FeatureSelection.OriginalDataFeatureSelection as odfs
import FeatureSelection.FundamentalFeatureSelection as ffs
from itertools import combinations  # combinations를 임포트합니다.


# 평가 방법에 따라 다른 return 값
def evaluate_model(y_true, y_pred, scoring):
    if scoring == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif scoring == 'precision':
        return precision_score(y_true, y_pred, average='weighted')
    elif scoring == 'recall':
        return recall_score(y_true, y_pred, average='weighted')
    elif scoring == 'f1':
        return f1_score(y_true, y_pred, average='weighted')
    elif scoring == 'roc_auc':
        return roc_auc_score(y_true, y_pred)
    elif scoring == 'mean_squared_error':
        return mean_squared_error(y_true, y_pred)
    elif scoring == 'mean_absolute_error':
        return mean_absolute_error(y_true, y_pred)
    elif scoring == 'r2':
        return r2_score(y_true, y_pred)
    elif scoring == 'mean_squared_log_error':
        return mean_squared_log_error(y_true, y_pred)
    elif scoring == 'explained_variance':
        return explained_variance_score(y_true, y_pred)
    else:
        raise ValueError("Unsupported scoring method")

def forward_selection(X, y, model, k_features, scoring):
    best_score = -np.inf
    best_features = None

    for k in range(1, k_features + 1):
        selector = SequentialFeatureSelector(model, n_features_to_select=k, direction='forward', scoring=scoring)
        selector.fit(X, y)
        X_selected = X.loc[:, selector.get_support()]  # 올바르게 필터링된 DataFrame
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = evaluate_model(y_test, y_pred, scoring)
        if score > best_score:
            best_score = score
            best_features = X_selected.columns.tolist()

    return best_features


def backward_elimination(X, y, model, k_features, scoring):
    if k_features >= X.shape[1]:
        raise ValueError("k_features must be less than the number of features in X")

    best_score = -np.inf
    best_features = None

    # Initialize list of features
    features = X.columns.tolist()

    for k in range(X.shape[1], k_features - 1, -1):
        # Ensure that n_features_to_select is less than current number of features
        if k <= k_features:
            continue
        selector = SequentialFeatureSelector(model, n_features_to_select=k, direction='backward', scoring=scoring)
        X_selected = X[features]
        selector.fit(X_selected, y)
        selected_features = X_selected.columns[selector.get_support()].tolist()

        if len(selected_features) <= k_features:
            break

        X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = evaluate_model(y_test, y_pred, scoring)
        
        if score > best_score:
            best_score = score
            best_features = selected_features

    return best_features

def rfe_selection(X, y, model, k_features, scoring):
    best_score = 0
    best_features = []

    # 모든 조합을 반복
    for k in range(1, k_features + 1):
        # tqdm을 사용하여 진행 상황 표시
        for feature_combination in tqdm(combinations(X.columns, k), desc=f'Combining {k} features'):
            X_subset = X[list(feature_combination)]
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            score = evaluate_model(y, y_pred, scoring)

            # 최고 점수와 피처 조합 갱신
            if score > best_score:
                best_score = score
                best_features = feature_combination

    print(f"Best score: {best_score}")
    return best_features


def featureSelection(X, y, method='forward', model=None, k_features=None, scoring='accuracy'):
    if model is None:
        raise ValueError("input model")

    is_class = is_classifier(model)
    
    if is_class:
        if scoring not in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            raise ValueError("Invalid scoring method for classification")
    else:
        if scoring not in ['mean_squared_error', 'mean_absolute_error', 'r2', 'mean_squared_log_error', 'explained_variance']:
            raise ValueError("Invalid scoring method for regression")

    if k_features is None:
        k_features = X.shape[1]

    if method == 'forward':
        selected_features = forward_selection(X, y, model, k_features, scoring)
    elif method == 'backward':
        selected_features = backward_elimination(X, y, model, k_features, scoring)
    elif method == 'rfe':
        selected_features = rfe_selection(X, y, model, k_features, scoring)
    else:
        raise ValueError("please selection [forward Selection, backward Selection, RFE]")

    return selected_features  