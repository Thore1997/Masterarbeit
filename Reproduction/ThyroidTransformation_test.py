import pandas as pd

file_path = 'data/thyroid0387.data'
df = pd.read_csv(file_path, sep=',', header=None)
df_numeric = df.apply(pd.to_numeric, errors='coerce')
cols = [col for col in df_numeric.columns if df_numeric[col].nunique() > 2] #we want columns with more than 2 unique values


column_mapping = {
    0:  'Age',
    17: 'TSH',
    19: 'T3',
    21: 'TT4',
    23: 'T4U',
    25: 'FTI',
    27: 'TBG'
}


df_1 = df_numeric[cols]
df_2 = df_1[list(column_mapping.keys())].rename(columns=column_mapping)
df_raw_features = df_2.dropna()
#df_3 = df_2.fillna(df_2.median())
number_nan = df_2.isnull().sum()


#print(df_raw_features.head())
print(f"Most important stats {df_2.describe()}")
print(f"Missing values per column {number_nan}")
#print(f"The dataset has {df_raw_features.shape[0]} rows and {df_raw_features.shape[1]} columns.")


#df[0] = df[0] * 100
#target_columns = [0,16,17,18,19,20,21]
#target_df = df.iloc[:,target_columns].copy()
#print(target_df.head())