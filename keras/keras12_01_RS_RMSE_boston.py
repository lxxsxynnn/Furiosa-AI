from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np

# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=500, batch_size=2)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)       # 매 epoch마다 train 값과 train 예측값의 차이를 갖고 loss를 계산하고 있음, 여기 있는 값은 test 값과 test 예측값의 차이
print("loss : ", loss)      # loss : 24.472475051879883 (500, 2)

y_predict = model.predict(x_test)          # 예측값 생성

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)           # 원값과 예측값 비교
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)

def RMSE(y_test, y_predict):  #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)