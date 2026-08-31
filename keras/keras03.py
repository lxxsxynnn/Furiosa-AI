from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])
#이전 데이터(x=y)와 달리 완벽한 비례 관계가 아님 (3,4번째 값이 순서가 어긋남)
#-> 모든 점을 정확히 지나는 "직선"이 존재하지 않음
#-> Dense(1) 하나(=직선 하나, y=wx+b)로는 5개 점을 다 못 맞춤
#-> loss가 0에 수렴하지 못하고, 오차가 가장 작아지는 "최적의 직선"에서 수렴함

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 8000)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict(np.array([1, 2, 3, 4, 5]))
print("예측값 : ", result)