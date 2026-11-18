import numpy as np
import pandas as pd
# IQR로 이상치 찾고 처리하기 - 2차원
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]])

def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])
    print('1사분위: ', quartile_1)
    print('q2: ', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3 - quartile_1
    print('IQR: ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data > upper_bound) | (data < lower_bound)), \
            iqr, lower_bound, upper_bound

outlier_loc1, iqr1, low1, up1 = outlier(aaa[0])
outlier_loc2, iqr2, low2, up2 = outlier(aaa[1])
print('이상치의 위치: ', outlier_loc1, outlier_loc2)

import matplotlib.pyplot as plt

# 두 행의 범위가 달라서(-10~50, -70000~1000) 한 축에 그리면 1행이 눌림
# subplot으로 나누면 y축이 각자 잡히고 경계선도 자기 박스에만 그려짐
plt.subplot(1, 2, 1)
plt.boxplot(aaa[0])
plt.axhline(up1, color='pink', label='upper bound')
plt.axhline(low1, color='skyblue', label='lower bound')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(aaa[1])
plt.axhline(up2, color='red', label='upper bound')
plt.axhline(low2, color='blue', label='lower bound')
plt.legend()

plt.show()
'''
1사분위:  4.0
q2:  7.0
3사분위:  10.0
IQR:  6.0
1사분위:  200.0
q2:  400.0
3사분위:  600.0
IQR:  400.0
이상치의 위치:  (array([ 0, 12]),) (array([6]),)
>> 1행은 -10과 50, 2행은 -70000이 이상치
'''

# 이상치 처리
df = pd.DataFrame(aaa.T, columns=['x1', 'x2'])

df.loc[outlier_loc1[0], 'x1'] = np.nan
df.loc[outlier_loc2[0], 'x2'] = np.nan
df.loc[2, 'x2'] = np.nan     # -30은 IQR 경계 안이지만 음수라 직접 제외

df = df.interpolate()
df = df.bfill()
print(df)
'''
      x1      x2
0    2.0   100.0
1    2.0   200.0
2    3.0   300.0
3    4.0   400.0
4    5.0   500.0
5    6.0   600.0
6    7.0   700.0
7    8.0   800.0
8    9.0   900.0
9   10.0  1000.0
10  11.0   210.0
11  12.0   420.0
12  12.0   350.0
>> x1의 0번은 앞 값이 없어 interpolate로 못 채우고 bfill이 채움
>> 12번은 뒤 값이 없어 앞의 값 12.0이 그대로 들어감
'''