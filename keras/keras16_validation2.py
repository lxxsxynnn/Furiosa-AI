from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

# 검증 데이터 직접 명시해서 자르기
# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] 8개, 4개, 4개 잘라보기
x_train = x[:8]
y_train = y[:8]

x_valid = x[8:12]
y_valid = y[8:12]

x_test = x[12:]
y_test = y[12:]

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=100, batch_size=2,
          validation_data = (x_valid, y_valid))

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  0.0003835715469904244

# loss가 내려가는데 val_loss가 더 내려가면? val_loss를 믿어야 함 항상 val_loss가 우선