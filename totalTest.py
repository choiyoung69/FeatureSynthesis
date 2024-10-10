import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import FeatureSelection.OriginalDataFeatureSelection as odfs
import featureSynthesis as fs
import AutoEncoding as ae
import FeatureSelection.FundamentalFeatureSelection as ffs
import FeatureSelection.FeatureSelection as fsfs
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("C:\\Users\\young\\Downloads\\heart.csv")

X = df.drop(columns=['output'])
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
scoring = 'accuracy'

selected_features = odfs.originalDataFeatureSelection(X_train, y_train, model, scoring, selection_method='backward', k_features=10)


# 초기 선택된 feature들
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# feature들을 통한 변환
X_train_transformed_features = fs.integrated_transformations(X_train_selected)
X_test_transformed_features = fs.integrated_transformations(X_test_selected)

# AutoEncoding.py를 사용하여 boolean type encoding
boolean_columns = ae.transform_boolean(X_train_transformed_features)
X_train_transfomred_encoding = boolean_columns

# AutoEncoding.py를 사용하여 boolean type encoding
boolean_columns_test = ae.transform_boolean(X_test_transformed_features)
X_test_transfomred_encoding = boolean_columns_test


#############################################################################
## feature selection step 1
## 기본적인 featureSelection 진행
X_train_transformed_fs = ffs.remove_lowInformation_features(X_train_transfomred_encoding )
X_train_transformed_fs  = ffs.remove_highlyNull_features(X_train_transformed_fs )


# 합성된 피처에서 nan 값 제거
X_train_transformed_fs  = X_train_transformed_fs .apply(lambda col: col.fillna(col.sum()))


#############################################################################
## feature selection step 2
# 낮은 중요성을 가진 피처 제거
X_train_selectKBest = fs.remove_low_importance_features(X_train_selected, X_train_transformed_fs , y_train, 'classification')
print(X_train_selectKBest)


##############################################################################
## feature selection step 3
# Permutation feature importance 계산
top_features, importance = fs.compute_permutation_importance(X_train_selectKBest, y_train, model, top_n=10)
top_features

# 합성된 피처와 원본 피처를 합치기
X_permutation_importance = X_train_selectKBest.iloc[:, top_features]
X_permutation_importance

# 행을 drop한 후 인덱스를 재설정
X_train_selected_reset = X_train_selected.reset_index(drop=True)
X_permutation_importance_reset = X_permutation_importance.reset_index(drop=True)

X_train_concat = pd.concat([X_train_selected_reset, X_permutation_importance_reset], axis=1)
X_train_concat.columns
##############################################################################
## feature selection step 4
# 최종적인 feature Selection
selected_feature_final = fsfs.featureSelection(X_train_concat, y_train, method = 'rfe', model=model, k_features = 10, scoring = 'accuracy')
X_train_final = X_train_concat[selected_feature_final]

##############################################################################
from sklearn.metrics import accuracy_score
# 성능 평가: feature 합성 전
model.fit(X_train, y_train)
y_pred_initial = model.predict(X_test)
accuracy_initial = accuracy_score(y_test, y_pred_initial)
print(f"Initial model accuracy (without feature synthesis): {accuracy_initial:.4f}")


# MinMaxScaler 초기화
scaler = MinMaxScaler()
X_test_transfomred_encoding.fillna(X_test_transfomred_encoding .sum(), inplace=True)
X_test_scaled = scaler.fit_transform(X_test_transfomred_encoding)

# 변환된 배열을 DataFrame으로 변환
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_transfomred_encoding.columns)

# 원본 피처랑 합성 피처를 합치기
X_test_concat = pd.concat([X_test_selected, X_test_scaled_df], axis=1)

# X_train_final의 열만 선택
X_test_final = X_test_concat[X_train_final.columns]