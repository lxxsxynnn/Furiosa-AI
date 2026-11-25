import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
import joblib, pickle

# 저장한 데이터프레임을 불러와 상관계수 보기 - iris
# 1. 데이터
# joblib, pickle로 데이터 로드하기
path1 = './_save/joblib/'
# path2 = './_save/pickle/'

df = joblib.load(path1 + 'm19_correlation_coefficient1.joblib')
# df = pickle.load(open(path2 + 'm19_correlation_coefficient1.pickle', 'rb'))
print('로드 완료')

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

print('================ 상관관계 히트맵 ================')
print(df.corr())
'''
================ 상관관계 히트맵 ================
                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
Target                      0.782561         -0.426658           0.949035          0.956547  1.000000
'''

import matplotlib.pyplot as plt
import seaborn as sns
# pip install seaborn
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()