# 19-1 참조
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

# 스케일링 - 캘리포니아 주택 가격
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (20640, 8) (20640,)

'''
x의 비율을 동일하게
x 데이터를 0에서 1사이의 수치로 바꿔야 함 > 가장 큰 수로 나눠주기
=> MinMaxScaler

이렇게 되면 가장 큰 값은 1이 되는데 가장 작은 값은 0이 나오지 않을 수도 있음
(val - MIN) / (MAX - MIN)
로 보정해서 0에서 1사이에 숫자만 나올 수 있게 수정
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x)
print(np.min(x), np.max(x))     # 0.0 1.0000000000000002 (부동소숫점 연산 방식 때문에 발생하는 오차)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=121)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,  
         )      # epoch 단위로 출력을 해주고 있음

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  0.9900057911872864 > loss :  0.5106000304222107

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)           # r2:  0.25423430595539853 > r2:  0.6153678981162408

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)         # mse:  0.9900055964814629 > mse:  0.5105999598696697

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)      # RMSE :  0.9949902494403967 > RMSE :  0.7145627753176551