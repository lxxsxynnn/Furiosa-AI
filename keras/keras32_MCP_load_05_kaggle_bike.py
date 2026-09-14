# https://www.kaggle.com/competitions/bike-sharing-demand/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 최고 가중치 불러오기 - 자전거(캐글)
# 1. 데이터
data_path = "C:/study/_data/kaggle_bike/"

train_csv = pd.read_csv(data_path + "train.csv", index_col=0)    # 날짜 시간 데이터를 인덱스로
test_csv = pd.read_csv(data_path + "test.csv", index_col=0)
submission = pd.read_csv(data_path + "sampleSubmission.csv", index_col=0)
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

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
save_path = 'C:/study/_save/keras31/'
model = load_model(save_path + 'k31_0914_1417_05_bike_0035-22708.7090.keras')
# 3. 컴파일, 훈련

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)

'''
loss :  22943.3203125
r2 :  0.3337591886520386
mse :  22943.32421875
rmse :  151.47053911157113
'''

# y_submit = model.predict(test_csv)
# submission['count'] = y_submit

# from datetime import datetime
# now = datetime.now().strftime('%m%d_%H%M')
# submission.to_csv(data_path + "submit/" + "submit_" + now + ".csv")