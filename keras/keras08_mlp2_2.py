import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터를 생성하는 방법 - range 메서드 활용
# 1. 데이터
x = np.array(range(10))
print(x)  # [0 1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 10))
print(x)  # [1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 11))
print(x)  # [1 2 3 4 5 6 7 8 9 10]

x = np.array([range(10), range(21,31,), range(201,211)]).T
print(x.shape)  # (3, 10) -> (10, 3)

y = np.array(range(1, 11))
print(y.shape)  # (10,)

# 실습
# [10, 31, 211]

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 3))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, epochs=500, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)          # loss : 2.2525128429151664e-0
results = model.predict(np.array([[10, 31, 211]]))
print("results : ", results)    # results : [[11.000064]]