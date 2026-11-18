import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# IterativeImputer로 채우기 - 다른 열로 예측
data = pd.DataFrame([[2, np.nan, 6, 8, 10, 8],
                    [2, 4, np.nan, 8, np.nan, 4],
                    [2, 4, 6, 8, 10, 12],
                    [np.nan, 4, np.nan, 8, np.nan, 8],
                    ]).T
data.columns= ['x1', 'x2', 'x3', 'x4']
print(data)

from xgboost import XGBRegressor
xgb = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    random_state=333,
)
# xgb계열 n_estimators default: 100

imputer = IterativeImputer(
    estimator=xgb,
    max_iter=10,
    tol=0.001,   # default
    random_state=333,
)

data2 = imputer.fit_transform(data)
print(data2)
'''
[[ 2.          2.          2.          7.99865484]
 [ 9.98037434  4.          4.          4.        ]
 [ 6.          2.01480126  6.          7.99865484]
 [ 8.          8.          8.          8.        ]
 [10.          7.97927809 10.          4.01578808]
 [ 8.          4.         12.          8.        ]]
'''
print("실제 반복 횟수: ", imputer.n_iter_)  # 실제 반복 횟수:  3
# 알아서 early stopping까지 해주고 있음