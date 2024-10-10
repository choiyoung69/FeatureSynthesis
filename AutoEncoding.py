# 필요한 패키지 import
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import pandas as pd


### Data Encoding
def transform_categories(df, columnList, type = "OneHot") :
    df_selected = df[columnList]

    if(type == "Label"):
        encoder = LabelEncoder()
        for column in columnList:
            df[column] = encoder.fit_transform(df[column])
        return df

    elif(type == "OneHot") :
        # 행렬 반환이 아닌 벡터 반환
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df[columnList])
        encoded_df = pd.DataFrame(encoded, columns = encoder.get_feature_names_out(columnList))
        return pd.concat([df_selected, encoded_df], axis=1)
    
    elif(type == "Ordinal"):
        encoder = OrdinalEncoder()
        df[columnList] = encoder.fit_transform(df[columnList])
        return df
    
    else :
        raise ValueError("지원되지 않는 인코딩 방식")

#boolean 값을 가지는 columnList
def get_boolean_columns(df):
    boolean_columns = df.select_dtypes(include='bool').columns.tolist()  # Boolean 열 선택
    return boolean_columns

### Boolean Data Data Encoding
def transform_boolean(df):
    columnList = get_boolean_columns(df)
    for col in columnList:
        df[col] = df[col].astype(int)
    return df