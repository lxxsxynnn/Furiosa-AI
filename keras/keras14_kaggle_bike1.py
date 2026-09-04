# https://www.kaggle.com/competitions/bike-sharing-demand/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = "C:/study/_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    # 날짜 시간 데이터를 인덱스로
print(train_csv)    # 인덱스 처리된 거 확인용   [10886 rows x 11 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)     # [6493 rows x 8 columns]

submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
print(submission)   # [6493 rows x 1 columns]

print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(submission.shape)     # (6493, 1)

print(train_csv.info())     # pandas에서는 info()를 통해 결측치 확인 가능 / 지금 데이터에서는 없음
print(test_csv.info())

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
model.fit(x_train, y_train, epochs=500, batch_size=8, validation_split=0.2)

# 4. 평가, 훈련
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)

y_submit = model.predict(test_csv)
submission['count'] = y_submit

from datetime import datetime
now = datetime.now().strftime('%m%d_%H%M')
submission.to_csv(path + "submit/" + "submit_" + now + ".csv")

#  kaggle 점수 0.33 나올 것

"""
[1st try]
0.8 51 100 16 0.2 16
layer: 16 32 128 32 12 6 1

r2 :  0.243391752243042
mse :  26438.78515625
rmse :  162.60007735622392

[2nd try]
0.8 51 100 16 0.2 16
layer: 16 64 128 256 120 96 48 24 12 6 1

r2 :  0.30594444274902344
mse :  24252.955078125
rmse :  155.73360291897507

[3rd try]
0.8 51 1000 16 0.2 16
layer: 16 64 128 256 120 96 48 24 12 6 1

r2 :  0.09419381618499756
mse :  31652.330078125
rmse :  177.91101730394607

[4th try]
0.8 51 500 16 0.2 16
layer: 16 64 128 256 120 96 48 24 12 6 1

r2 :  -0.0009036064147949219
mse :  34975.3984375
rmse :  187.01710733914157

[5th try]
0.8 51 500 16 0.2 16
16 64 256 120 96 48 6 1

r2 :  0.23778235912322998
mse :  26634.798828125
rmse :  163.20171208699068

[6th try]
0.8 23 500 16 0.2 16
16 64 120 96 48 6 1

r2 :  0.2100335955619812
mse :  27204.060546875
rmse :  164.93653490623294

가상환경 하나 더 만들어서 keras14까지 직접 쳐볼것
"""