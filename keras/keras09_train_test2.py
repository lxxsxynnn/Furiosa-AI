import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 슬라이싱
# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# x_train = np.array([1, 2, 3, 4, 5, 6, 7])
# y_train = np.array([1, 2, 3, 4, 5, 6, 7])

# x_test = np.array([8, 9, 10])
# y_test = np.array([8, 9, 10])

# [찾아보기] 넘파이 리스트의 슬라이싱 => 7 : 3으로 나누기
x_train = x[0:7]
y_train = y[:7]     # 0은 항상 시작이기 때문에 생략도 가능

x_test = x[7:10]
y_test = y[7:]

# 2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)                  # loss :  0.0008748897234909236