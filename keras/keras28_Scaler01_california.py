# 27-1 참조
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

# 스케일링 효과 확인해보기 - 캘리포니아 주택 가격
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (20640, 8) (20640,)

'''
전체를 한 번에 스케일링하면 test 정보가 새어 나가 평가가 부풀려짐 → train/test를 먼저 분리하고 train 기준으로만 fit → test는 그 기준으로 transform만 (0~1을 벗어나도 정상)
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=121  )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train), np.max(x_train)) # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))   # -0.0012367054167697258 1.0
# exit()

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
print("loss : ", loss)      # loss :  0.5106000304222107 > loss :  0.5158395171165466

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)           # r2:  0.6153678981162408 > r2:  0.6114210351869949

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)         # mse:  0.5105999598696697 > mse:  0.5158394290752149

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)      # RMSE :  0.7145627753176551 > RMSE :  0.7182196245405822