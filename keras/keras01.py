import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 인공지능 코딩의 4단계
# 1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
# units=1  -> 이 층의 출력 노드가 1개 (예측하려는 값이 숫자 하나이므로)
#             지금은 은닉층 없이 입력->출력 직결 구조 = 단순 선형 회귀
#             (x,y가 단순 비례관계라 이 구조만으로 충분히 표현 가능)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# mse : 평균제곱 오차 / mae : 절댓값
# optimizer : 로스율 최적화
model.fit(x, y, epochs= 1000)
# epochs -> 선을 처음에 제대로 긋지 못해서 여러번 찍어가면서 오차를 줄여나감
#          (무조건 많이 돌린다고 좋은 건 아님 -> 어느 지점부터 정체되거나 과적합 위험)

# 4. 평가 예측
result = model.predict(np.array([4]))
# model.predict -> 최적의(마지막) 훈련값 대입해서 계산
print("4의 예측값 : ", result)