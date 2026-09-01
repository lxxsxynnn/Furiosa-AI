from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 딥러닝(다층 레이어)
# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim = 1))
# 첫 번째 파라미터: 하위 Layer의 아웃풋 노드수
# 두 번째 파라미터: 상위 Layer의 인풋 노드수
model.add(Dense(6, input_dim=3))
model.add(Dense(7, input_dim=6))
model.add(Dense(6, input_dim=7))
model.add(Dense(5, input_dim=6))
model.add(Dense(1, input_dim=5))

# 층 개수, 각 층의 노드 개수 -> 사람이 직접 정하는 "하이퍼파라미터"
# (w, b처럼 학습으로 찾아지는 값이 아니라, 학습 전에 설계자가 미리 정해야 하는 값)
# 이 조합을 바꿔가며 loss가 더 잘 줄어드는 구조를 찾는 과정(하이퍼파라미터 튜닝)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 5000)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict(np.array([1, 2, 3, 4, 5]))
print("예측값 : ", result)