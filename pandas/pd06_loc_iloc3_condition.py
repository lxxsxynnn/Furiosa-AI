import pandas as pd
print(pd.__version__)   # 3.0.6

# 조건으로 행 고르기
data = [
    ['삼성', 1000, 2000],
    ['현대', 1100, 3000],
    ['LG', 2000, 500],
    ['아모레', 3500, 6000],
    ['네이버', 100, 1500],
]

index = ['031', '059', '033', '045', '023']
columns = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)
'''
     종목명    시가    종가
031  삼성      1000   2000
059  현대      1100   3000
033  LG        2000    500
045  아모레    3500   6000
023  네이버     100   1500
'''

print("================================")
print("시가가 1100원 이상인 행을 모두 출력")
cond1 = df['시가'] >= 1100
print(cond1)
'''
031    False
059     True
033     True
045     True
023    False
Name: 시가, dtype: bool
'''
print(df[cond1])
'''
     종목명    시가    종가
059   현대  1100  3000
033   LG  2000   500
045  아모레  3500  6000
'''
print(df['시가'] >= 1100)

################ 위에서 종가만 뽑기 ################
print(df[df['시가'] >= 1100]['종가'])
'''
059    3000
033     500
045    6000
Name: 종가, dtype: int64
'''
# print(df[df['시가'] >= 1100].loc['종가'])   # raise KeyError(key) from err
