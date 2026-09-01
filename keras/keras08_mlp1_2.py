import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 데이터 리셰이핑
# 1. 데이터
x = np.array([[1, 2, 3, 4, 5],
             [6, 7, 8, 9, 10]])
x = x.T                         # 행 <-> 렬 전환 방법 1
# x = x.transpose()               # 행 <-> 렬 전환 방법 2
y = np.array([1, 2, 3, 4, 5])

print(x.shape)  # (5, 2)
print(y.shape)  # (5,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 2)) # input_dim은 열의 개수
                                   # 행무시열우선
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, epochs=100, batch_size=3)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)                          # loss :  0.009501990862190723
results = model.predict(np.array([[6, 11]]))
print("results : ", results)                    # results :  [[6.174222]]