from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 레이어 추가 간결하게 하기
# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 5, 4, 6])

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim = 1))  # 맨 처음 층만 입력 개수를 알려줘야 함
model.add(Dense(5))                 # 이후로는 안 써도 자동으로 이어짐
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 2000)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
# result = model.predict(np.array([1, 2, 3, 4, 5, 6]))
# print("예측값 : ", result)