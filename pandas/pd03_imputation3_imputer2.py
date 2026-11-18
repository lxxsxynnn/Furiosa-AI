import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# SimpleImputer로 채우기 - 최빈값·지정한 값·함수
data = pd.DataFrame([[2, np.nan, 6, 8, 10, 8],
                    [2, 4, np.nan, 8, np.nan, 4],
                    [2, 4, 6, 8, 10, 12],
                    [np.nan, 4, np.nan, 8, np.nan, 8],
                    ]).T
data.columns= ['x1', 'x2', 'x3', 'x4']
print(data)

imputer = SimpleImputer(strategy='most_frequent')
data2 = imputer.fit_transform(data)
print(data2)
'''
[[ 2.  2.  2.  8.]
 [ 8.  4.  4.  4.]
 [ 6.  4.  6.  8.]
 [ 8.  8.  8.  8.]
 [10.  4. 10.  8.]
 [ 8.  4. 12.  8.]]
'''

imputer2 = SimpleImputer(strategy='constant', fill_value=777)
data3 = imputer2.fit_transform(data)
print(data3)
'''
[[  2.   2.   2. 777.]
 [777.   4.   4.   4.]
 [  6. 777.   6. 777.]
 [  8.   8.   8.   8.]
 [ 10. 777.  10. 777.]
 [  8.   4.  12.   8.]]
'''

imputer3 = SimpleImputer(
#     strategy=lambda x: np.max(x)  # 최댓값
    strategy=lambda x: np.min(x)    # 최솟값
)
data4 = imputer3.fit_transform(data)
print(data4)
'''
[[ 2.  2.  2.  4.]
 [ 2.  4.  4.  4.]
 [ 6.  2.  6.  4.]
 [ 8.  8.  8.  8.]
 [10.  2. 10.  4.]
 [ 8.  4. 12.  8.]]
'''