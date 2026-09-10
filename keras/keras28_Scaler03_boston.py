from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# 스케일링 효과 확인해보기 - 보스턴 주택 가격
# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()   #  tensorflow는 가져오면서 분할됨(훈련/학습 묶음으로 나오니까 주의할 것)
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_split=0.25
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss : 22.92780113220215 (1000, 1) > loss :  21.666061401367188