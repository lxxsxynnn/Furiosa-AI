import pandas as pd
print(pd.__version__)   # 3.0.6

# loc와 iloc으로 행 고르기
data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '3000'],
    ['LG', '2000', '500'],
    ['아모레', '3500', '6000'],
    ['네이버', '100', '1500'],
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

# print(df[0])        # raise KeyError(key) from err
# print(df['031'])    # raise KeyError(key) from err
print(df['종목명'])     # 판다스 열행       컬럼이 기준
'''
031     삼성
059     현대
033     LG
045    아모레
023    네이버
Name: 종목명, dtype: str
'''

######## 아모레 시가 출력 ########
# print(df[3, 1]) # raise KeyError(key) from err
# print(df[3][1]) # raise KeyError(key) from err
# print(df['045', '시가'])    # raise KeyError(key) from err
# print(df['045']['시가'])    # raise KeyError(key) from err
print(df['시가']['045'])  # 3500

################################################
# loc: 인덱스 기준으로 행 데이터 추출
# iloc: 행 번호를 기준으로 행 데이터 추출
        # int loc로 외울 것
################################################
print("============ 아모레 뽑기 ============")
# print(df.iloc[3])
'''
종목명     아모레
시가     3500
종가     6000
Name: 045, dtype: str
'''
# print(df.iloc['045'])   # TypeError: Cannot index by location index with a non-integer key
# print(df.loc[3])    # raise KeyError(key) from err
print(df.loc['045'])

print("============ 네이버 뽑기 ============")
print(df.iloc[4])
# print(df.iloc['023'])     # error
# print(df.loc[4])          # error
print(df.loc['023'])

print("============ 네이버 종가 뽑기 ============")
# print(df.iloc[4][2])       # error(pandas 2버전대에서는 가능)
print(df.iloc[4, 2])         # 1500
print(df.iloc[4]['종가'])    # 1500
# print(df.iloc[4, '종가'])  # error

# print(df.loc['023'][2])     # error
# print(df.loc['023', 2])     # error
print(df.loc['023']['종가'])    # 1500
print(df.loc['023', '종가'])    # 1500

print("====================================")
print(df.iloc[4].iloc[2])       # 1500
print(df.iloc[4].loc['종가'])   # 1500

print(df.loc['023'].loc['종가'])    # 1500
print(df.loc['023'].iloc[2])        # 1500