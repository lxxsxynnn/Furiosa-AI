import pandas as pd

# 상관계수가 직선 관계만 보는지 확인해 보기
df = pd.DataFrame({'X1': [1, 2, 3, 4, 5],
                   'X2': [5, 4, 3, 2, 1],
                   'X3': [-1000, 50, 7, 100, -1100],
                   'Y': [2, 3, 4, 5, 6],
                   })

print(df)
'''
   X1  X2     X3  Y
0   1   5  -1000  2
1   2   4     50  3
2   3   3      7  4
3   4   2    100  5
4   5   1  -1100  6
'''

import matplotlib.pyplot as plt
import seaborn as sns
# pip install seaborn
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()