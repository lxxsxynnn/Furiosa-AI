import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
import joblib, pickle

# 데이터프레임을 joblib으로 저장하기 - iris
# 1. 데이터
datasets = load_iris()
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets['data']    # 딕셔너리 형태
y = datasets.target     # 클래스 형태

df = pd.DataFrame(x, columns=datasets.feature_names)
df['Target'] = y
print(df)
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target
0                  5.1               3.5                1.4               0.2       0
1                  4.9               3.0                1.4               0.2       0
2                  4.7               3.2                1.3               0.2       0
3                  4.6               3.1                1.5               0.2       0
...
149                5.9               3.0                5.1               1.8       2

[150 rows x 5 columns]
'''

# joblib, pickle로 데이터 세이브하기
path1 = './_save/joblib/'
# path2 = './_save/pickle/'

joblib.dump(df, path1 + 'm19_correlation_coefficient1.joblib')
# pickle.dump(df, open(path2 + 'm19_correlation_coefficient1.pickle', 'wb'))
print('저장 완료')