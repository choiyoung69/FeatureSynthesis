import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler


# 분모가 0인 경우 0으로 처리하는 함수
def safe_divide(numerator, denominator):
    return np.where(denominator != 0, numerator / denominator, np.nan)  # 나누기 안전 처리

# 역수 변환을 적용할 때 안전하게 나눗셈을 수행하는 함수
def apply_inverse(series):
    return pd.Series(safe_divide(1, series), index=series.index)

# 연속형 변수들을 변환하는 함수
def transform_Dataframe(df):
    transformations = {
        'log10({})': lambda x: pd.Series(np.log10(x.replace(0, np.nan)), index=x.index),  # 상용로그 (0을 np.nan으로 대체)
        '√{}': lambda x: pd.Series(np.sqrt(x.clip(lower=0)), index=x.index),              # 제곱근 (음수를 0으로 대체)
        '1/{}': lambda x: apply_inverse(x),                     # 역수 (분모가 0인 경우 0으로 처리)
        '{}^2': lambda x: pd.Series(np.square(x), index=x.index),           # 제곱
        '{}^3': lambda x: pd.Series(np.power(x, 3), index=x.index),  # 세제곱
        '|{}|': lambda x: pd.Series(np.abs(x), index=x.index),              # 절대값
        'exp({})': lambda x: pd.Series(np.exp(x), index=x.index),           # 지수 함수
        '2{}': lambda x: pd.Series(2 * x, index=x.index),      # 두 배
        'sin({})': lambda x: pd.Series(np.sin(x), index=x.index),           # 사인 함수
        'cos({})': lambda x: pd.Series(np.cos(x), index=x.index),           # 코사인 함수
        'ln({})': lambda x: pd.Series(np.log(x.replace(0, np.nan)), index=x.index),  # 자연 로그 (0을 np.nan으로 대체)
        '-{}': lambda x: pd.Series(-x, index=x.index)          # 부호 변환 (Negate)
    }

    result_list = []  # 변환된 데이터를 저장할 리스트

    # 각 변환 적용
    for col in tqdm(df.columns, desc="Applying transformations to each column"):
        for name, func in transformations.items():
            transformed_col = func(df[col])
            transformed_col.name = f'{name.format(col)}'  # 컬럼 이름 변경
            result_list.append(transformed_col)

    # 변환된 모든 컬럼을 한 번에 DataFrame으로 결합
    transformed_df = pd.concat(result_list, axis=1)
    return transformed_df

# 연속형 데이터를 기본 연산 처리하는 함수
def arithmetic_operations(df):
    operations = []

    for i, col1 in tqdm(enumerate(df.columns), total=len(df.columns), desc="Applying arithmetic operations"):
        for j, col2 in enumerate(df.columns):
            if i < j:  # 덧셈과 곱셈은 대칭적인 쌍을 제외
                operations.append((f'{col1} + {col2}', df[col1] + df[col2].fillna(np.nan)))  # NaN 유지
                operations.append((f'{col1} * {col2}', df[col1] * df[col2].fillna(np.nan)))  # NaN 유지
                operations.append((f'{col1} - {col2}', df[col1] - df[col2].fillna(np.nan)))  # NaN 유지
                operations.append((f'{col2} - {col1}', df[col2] - df[col1].fillna(np.nan)))  # NaN 유지
                operations.append((f'{col1} / {col2}', safe_divide(df[col1], df[col2].fillna(0))) )  # NaN 유지
                operations.append((f'{col2} / {col1}', safe_divide(df[col2], df[col1].fillna(0))) )  # NaN 유지
                operations.append((f'{col1} % {col2}', df[col1] % df[col2].fillna(np.nan)))  # NaN 유지
                operations.append((f'{col2} % {col1}', df[col2] % df[col1].fillna(np.nan)))  # NaN 유지

    # 연산 결과 DataFrame 생성
    result_df = pd.DataFrame({name: result for name, result in operations})

    return result_df

# 연속형 변수 비교하는 함수
def comparison_operations(df):
    comparisons = []

    for i, col1 in tqdm(enumerate(df.columns), total=len(df.columns), desc="Applying comparison operations"):
        for j, col2 in enumerate(df.columns):
            if i < j:  # 대칭적인 비교 연산을 제외
                comparisons.append((f'{col1} == {col2}', df[col1] == df[col2]))  # 같은지 비교
                comparisons.append((f'{col1} != {col2}', df[col1] != df[col2]))  # 다른지 비교
                comparisons.append((f'{col1} > {col2}', df[col1] > df[col2]))  # 큰지 비교
                comparisons.append((f'{col1} < {col2}', df[col1] < df[col2]))  # 작은지 비교
                comparisons.append((f'{col1} >= {col2}', df[col1] >= df[col2]))  # 크거나 같은지 비교
                comparisons.append((f'{col1} <= {col2}', df[col1] <= df[col2]))  # 작거나 같은지 비교

    # 비교 결과 DataFrame 생성
    result_df = pd.DataFrame({name: result for name, result in comparisons})

    return result_df

def apply_cumulative_transformations(df):
    cumulative_ops = []

    for col in tqdm(df.columns, desc="Applying cumulative transformations"):
        cumulative_ops.append((f'{col}_diff_1', df[col].diff(periods=1)))
        cumulative_ops.append((f'{col}_diff_2', df[col].diff(periods=2)))
        cumulative_ops.append((f'{col}_CumCount', df.groupby(col).cumcount() + 1))
        cumulative_ops.append((f'{col}_CumSum', df[col].cumsum()))
        cumulative_ops.append((f'{col}_CumMean', df[col].expanding().mean()))
        cumulative_ops.append((f'{col}_CumMin', df[col].expanding().min()))
        cumulative_ops.append((f'{col}_CumMax', df[col].expanding().max()))

    # 누적 결과 DataFrame 생성
    result_df = pd.DataFrame({name: result for name, result in cumulative_ops})

    return result_df

def integrated_transformations(df):
    # 변환된 데이터프레임 생성
    transformed_df = transform_Dataframe(df)
    
    # 산술 연산
    arithmetic_df = arithmetic_operations(transformed_df)
    
    # 비교 연산
    comparison_df = comparison_operations(transformed_df)
    
    # 누적 변환
    cumulative_df = apply_cumulative_transformations(transformed_df)
    
    # 모든 결과 데이터프레임을 병합
    result_df = pd.concat([transformed_df, arithmetic_df, comparison_df, cumulative_df], axis=1)
    
    return result_df

def remove_low_importance_features(X_original, X_synthesized, y, task_type):
    # 무한대 값을 가진 피처를 찾고 제거
    infinite_features = X_synthesized.columns[np.isinf(X_synthesized).any(axis=0)]
    X_synthesized_cleaned = X_synthesized.drop(columns=infinite_features)

    # MinMaxScaler로 X_original 스케일링
    scaler = MinMaxScaler()
    X_original_scaled = scaler.fit_transform(X_original)
    X_original_scaled_df = pd.DataFrame(X_original_scaled, columns=X_original.columns)

    # MinMaxScaler로 X_synthesized 스케일링
    X_synthesized_scaled = scaler.fit_transform(X_synthesized_cleaned)
    X_synthesized_scaled_df = pd.DataFrame(X_synthesized_scaled, columns=X_synthesized_cleaned.columns)

    # 상수 피처 제거
    constant_features = X_synthesized_scaled_df.columns[X_synthesized_scaled_df.nunique() <= 1]
    X_synthesized_scaled_df_cleaned = X_synthesized_scaled_df.drop(columns=constant_features)

    # 첫 번째 단계: 원본 데이터로 카이 제곱 검정 수행
    if task_type == 'regression':
        func1 = f_regression
        func2 = mutual_info_regression
    elif task_type == 'classification':
        func1 = f_classif
        func2 = mutual_info_classif
    else:
        raise ValueError("잘못된 type 입력")

    # 원본 데이터에서 상수 피처 제거
    constant_features_original = X_original_scaled_df.columns[X_original_scaled_df.nunique() <= 1]
    X_original_scaled_df = X_original_scaled_df.drop(columns=constant_features_original)

    # 원본 데이터 피처 선택 (첫 번째 단계)
    selector_original = SelectKBest(score_func=func1, k='all')
    selector_original.fit(X_original_scaled_df, y)
    
    # 원본 데이터의 최대 통계량 확인
    max_score_original = selector_original.scores_.max()
    print(f"Maximum score from original data (first step): {max_score_original}")

    # 합성된 데이터 피처 선택 (첫 번째 단계)
    selector_synthesized = SelectKBest(score_func=func1, k='all')
    selector_synthesized.fit(X_synthesized_scaled_df_cleaned, y)
    scores_synthesized = selector_synthesized.scores_

    # 원본 데이터의 최대 값보다 작은 합성 피처 필터링
    low_importance_features = [
        feature for feature, score in zip(X_synthesized_scaled_df_cleaned.columns, scores_synthesized) 
        if score < max_score_original
    ]

    # 중요도가 낮은 피처 제거
    X_synthesized_reduced = X_synthesized_scaled_df_cleaned.drop(columns=low_importance_features)

    # 원본 데이터 피처 선택 (두 번째 단계)
    selector_original_final = SelectKBest(score_func=func2, k='all')
    selector_original_final.fit(X_original_scaled_df, y)
    
    # 원본 데이터의 최대 통계량 확인
    max_score_original2 = selector_original_final.scores_.max()
    print(f"Maximum score from original data (second step): {max_score_original2}")

    # 줄인 합성 데이터 피처 선택 (두 번째 단계)
    selector_synthesized_final = SelectKBest(score_func=func2, k='all')
    selector_synthesized_final.fit(X_synthesized_reduced, y)
    scores_synthesized2 = selector_synthesized_final.scores_

    # 원본 데이터의 최대 값보다 작은 합성 피처 필터링
    low_importance_features2 = [
        feature for feature, score in zip(X_synthesized_reduced.columns, scores_synthesized2) 
        if score < max_score_original2
    ]
    
    # 중요도가 낮은 피처 제거
    X_final = X_synthesized_reduced.drop(columns=low_importance_features2)
    
    return X_final


from sklearn.inspection import permutation_importance
from sklearn.inspection import permutation_importance as skl_permutation_importance

def compute_permutation_importance(X, y, model, top_n=10):
    model.fit(X, y)
    
    # sklearn의 permutation_importance 사용
    results = skl_permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
    importance = results.importances_mean
    
    # 중요도 순위 정렬
    sorted_indices = np.argsort(importance)[::-1]
    
    # 상위 top_n 중요 피처 선택
    top_features = sorted_indices[:top_n]
    
    return top_features, importance
