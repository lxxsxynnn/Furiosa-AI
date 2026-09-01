import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 반환해야 하는 y값이 2개일 때
# 1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)]).T
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]).transpose()
print(x.shape, y.shape)     #(10, 3) (10, 2)

# [실습]
# [10, 31, 211]의 예측값
# 요구사항 [11.00, 0.00]

# 2. 모델 구성
model = Sequential()
model.add(Dense(10,input_dim = 3))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(2))

# 3. 컴파일, 학습
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, epochs=500, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)          # loss : 4.845403939190973e-11
np.set_printoptions(suppress=True)
results = model.predict(np.array([[10, 31, 211]]))
print("results : ", results)    # results : [[11.000007    0.00022031]]