from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 데이터 추가해서 정확성 높이기
# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 4, 5, 6])

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2500)

# 4. 평가, 예측
loss = model.evaluate(x, y)
# x와 정답 y를 둘 다 넣음
# 반환값: 컴파일에서 넣은 loss값
# 지금 이 모델이 얼마나 잘 맞추고 있는지를 점수로 확인
# compile에서 지정한 방식(mse)으로, 지금 모델 상태의 오차를 다시 계산해줌

print("loss : ", loss)
result = model.predict(np.array([1, 2, 3, 4, 5, 6, 7]))
# 2개 이상은 list
# x만 넣음(정답 y는 필요 없음)
# 새로운 입력에 대해 모델이 어떻게 예측하는지를 값으로 확인
# 학습된 w,b로 실제 예측값을 계산해서 뽑아줌
print("7의 예측값 : ", result)