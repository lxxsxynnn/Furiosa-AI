import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 리셰이핑 + 노이즈가 있는 데이터
# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
             ])
x = x.T
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 3))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 학습
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)                              # loss : 0.00030513718957081437
results = model.predict(np.array([[10, 1.3, 0]]))
print("results : ", results)                        # results :  [[10.032502]]