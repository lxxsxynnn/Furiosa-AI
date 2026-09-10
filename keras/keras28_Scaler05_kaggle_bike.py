# https://www.kaggle.com/competitions/bike-sharing-demand/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 스케일링 효과 확인해보기 - 자전거(캐글)
# 1. 데이터
path = "C:/study/_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 날짜 시간 데이터를 인덱스로
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(train_csv.describe())

############## 결측치 확인 ##############
print(train_csv.isna().sum())   # 결측치가 있는지 확인 info랑 둘 중에 편한 거 골라서 사용하면 됨
print(test_csv.isnull().sum())  # 마찬가지로 결측치 확인하는 방법

############## x, y 분리 ##############
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   # test에 없는 컬럼 제거
print(x)    # [10886 rows x 8 columns]
y = train_csv['count']  # train_csv에서 count만 사용
print(y, y.shape)       # (10886,) > 벡터 반환

# test, train 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=23)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))
model.add(Dense(64, activation='relu',))
model.add(Dense(120, activation='relu',))
model.add(Dense(96, activation='relu',))
model.add(Dense(48, activation='relu',))
model.add(Dense(6, activation='relu',))
model.add(Dense(1)) # 마지막은 굳이 안넣어도 됨

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
hist = model.fit(x_train, y_train, epochs=200, batch_size=8, validation_split=0.2)

# 4. 평가, 예측
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)      # r2 :  0.3217502236366272

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)    # mse :  23356.876953125

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)  # rmse :  152.82956832080956

# y_submit = model.predict(test_csv)
# submission['count'] = y_submit

# from datetime import datetime
# now = datetime.now().strftime('%m%d_%H%M')
# submission.to_csv(path + "submit/" + "submit_" + now + ".csv")