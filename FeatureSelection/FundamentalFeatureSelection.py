import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# 낮은 정보를 주는 feature 삭제
# 모든 값이 null 이거나 unique한 값이 1개 이하
def remove_lowInformation_features(df):
    low_info_columns = []

    for col in df.columns:
        if(df[col].isnull().all()):
            low_info_columns.append(col)
        elif len(df[col].unique()) < 2:
            low_info_columns.append(col)

    return df.drop(columns = low_info_columns)


# VIF를 계산해주는 함수
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# 다중공선성이 높은 feature 제거
def remove_high_vif_feature(X, threshold = 10):
    while True:
        vif_data = calculate_vif(X)
        max_vif = vif_data["VIF"].max()
        if max_vif < threshold :
            break

        max_vif_feature = vif_data.sort_values("VIF", ascending=False)["feature"].iloc[0]
        X = X.drop(columns = [max_vif_feature])

    return X

# null 값이 일정 수준 이상인 column 제거 
def remove_highlyNull_features(df, null_threshold=0.5):
    null_ratio = df.isnull().mean()
    # null_ratio가 null_threshold를 초과하는 컬럼만 제거
    high_null_columns = null_ratio[null_ratio > null_threshold].index
    return df.drop(columns=high_null_columns)

