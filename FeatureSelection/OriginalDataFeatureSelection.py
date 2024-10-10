###############################################################################
#Feature Synthesis
#original feature에서 feature selection 진행
#N개의 Feature를 선택
###############################################################################
import pandas as pd
import FeatureSelection
from sklearn.model_selection import cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE
import numpy as np


def originalDataFeatureSelection(X, y,  model, scoring, selection_method = 'rfe', k_features = 10) :
    selected_feature_names = list()
    if selection_method == 'forward':
        selected_feature_names = featureSelection_forward(X, y, model, scoring, k_features)

    elif selection_method == 'backward':
        selected_feature_names = featureSelection_backward(X, y, model, scoring, k_features)

    elif selection_method == 'rfe' :
        selected_feature_names = featureSelection_RFE(X, y, model, scoring, k_features)

    return selected_feature_names
    
def featureSelection_forward(X, y, model, scoring, k_features):
    sfs = SFS(
        model,
        k_features = k_features,
        floating = False,
        scoring = scoring,
        cv = 5
    )
    sfs.fit(X, y)
    selected_features = list(sfs.k_feature_idx_)
    feature_names = list(np.array(X.columns)[selected_features])

    return feature_names

def featureSelection_backward(X, y, model, scoring, k_features):
    sfs = SFS(
        model,
        k_features = k_features,
        forward = False,
        scoring = scoring,
        cv = 5
    )

    sfs.fit(X, y)
    selected_features = list(sfs.k_feature_idx_)
    feature_names = list(np.array(X.columns)[selected_features])

    return feature_names


def featureSelection_RFE(X, y, model, scoring, k_features):
    rfe = RFE(model, n_features_to_select=k_features)
    rfe.fit(X, y)

    selected_features = np.where(rfe.support_ == True)[0]
    feature_names = list(np.array(X.columns)[selected_features])

    return feature_names