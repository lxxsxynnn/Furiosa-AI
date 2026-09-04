# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
# path = "./_data/ddarung/"            # 상대경로
# path = "C:\study\_data/ddarung/"     # 절대경로
# path = "C:\study\_data\ddarung/"     # '/', '\' 상관없음
# path = "C:\\study//_data//ddarung/"
# path = "C:\\study\\_data\\ddarung\\" # '//', '\\'도 가능
path = "C:\\study\_data\\ddarung/"    # 섞어써도 되는데 가급적 하나만 사용 권장

 
train_csv = pd.read_csv(path + "train.csv", index_col=0) # index_col - 인덱스 컬럼(제외)
test = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path + "submission.csv")

print(train_csv)
print(train_csv.shape)      # [1459 rows x 10 columns]
print(train_csv.columns)    # (1459, 10)
print(train_csv.info())     # Non-Null Count로 컬럼별 결측치 개수 확인

############ 결측치 처리 1.삭제 ############
train_csv = train_csv.dropna()
print(train_csv) # [1328 * 10]

########### train_csv를 x와 y로 분리 ###########
x = train_csv.drop(['count'], axis=1) # count 컬럼을 제거한 나머지 / axis=1 열(컬럼) 삭제
print("x : ", x) # x : [1328 rows x 9 columns] 

y = train_csv['count'] # count 컬럼만 선택
print(y)
print(y.shape) # (1328,) -> y가 벡터 형태로 빠짐

# model.predict()에 넣을 값
test_csv = pd.read_csv(path + "test.csv", index_col= 0) # id 컬럼은 순번(인덱스)인데 이걸 불러오면 id가 실제 데이터처럼 포함됨 -> index_col=0으로 인덱스 지정하면 데이터에서 제외됨
print(test_csv) #[715 * 9] 
print(test_csv.shape)
print(test_csv.columns)
print(test_csv.info())

#여기에 model.predict(submission) 값을 넣음
submission = pd.read_csv(path + "submission.csv" , index_col= 0)
print(submission) # [715 * 9] / NaN: 결측치 
print(submission.shape)
print(submission.columns)

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, test_size=0.2)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

#4. 평가, 예측
print("========================================")
y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)