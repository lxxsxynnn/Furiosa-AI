import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# validation - 훈련 중에 검증
# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x_train = np.array([1, 2, 3, 4, 5, 6])
y_train = np.array([1, 2, 3, 4, 5, 6])

x_val = np.array([7, 8])
y_val = np.array([7, 8])

x_test = np.array([9, 10])
y_test = np.array([9, 10])

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
model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=1,
          validation_data=(x_val, y_val),
         )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)                  # loss :  0.0071001313626766205

# 모델을 제대로 판단하려면 validation과 test를 분리하는 게 좋음
# epoch에 대한 검증으로 판단할 것