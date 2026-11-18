import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# KNNImputer로 채우기 - 비슷한 행의 평균
data = pd.DataFrame([[2, np.nan, 6, 8, 10, 8],
                    [2, 4, np.nan, 8, np.nan, 4],
                    [2, 4, 6, 8, 10, 12],
                    [np.nan, 4, np.nan, 8, np.nan, 8],
                    ]).T
data.columns= ['x1', 'x2', 'x3', 'x4']
print(data)

# imputer = KNNImputer()
# imputer = KNNImputer(n_neighbors=5) # default
imputer = KNNImputer(n_neighbors=2)
data2 = imputer.fit_transform(data)
print(data2)
'''
[[ 2.  2.  2.  6.]
 [ 4.  4.  4.  4.]
 [ 6.  6.  6.  6.]
 [ 8.  8.  8.  8.]
 [10.  6. 10.  8.]
 [ 8.  4. 12.  8.]]
'''