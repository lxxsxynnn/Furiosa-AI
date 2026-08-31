from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 4, 5, 6])

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2500)

#4. 평가, 예측
loss = model.evaluate(x, y)
#컴파일에서 넣은 loss값이 나옴
#compile에서 지정한 방식(mse)으로, 지금 모델 상태의 오차를 다시 계산해줌

print("loss : ", loss)
result = model.predict(np.array([1, 2, 3, 4, 5, 6, 7]))
#2개 이상은 list
print("7의 예측값 : ", result)