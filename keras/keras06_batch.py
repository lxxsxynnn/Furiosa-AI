from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 5, 4, 6])

# ===== epoch vs batch_size 정리 =====

# epoch: 전체 데이터를 처음부터 끝까지 한 번 다 도는 것
# batch_size: 가중치를 1번 업데이트할 때 사용하는 데이터 묶음 크기
# iteration: 1epoch 안에서 가중치가 업데이트되는 횟수 = 전체 데이터 수 / batch_size

# 예) 데이터 9만개
# 9만개 통으로 넣고 1epoch  -> 가중치 업데이트 1번  (batch_size = 9만, 기본값 안 씀)
# 3만개씩 잘라서 3epoch     -> 가중치 업데이트 3번  (batch_size = 3만)
# -> 같은 데이터를 같은 만큼 봤어도, 후자가 가중치를 더 자주 고쳐나감

# 왜 잘라서(mini-batch) 하는 게 더 효율적인가?
# 1. 업데이트 횟수 증가
#    - 기울기 계산 1번당 가중치 업데이트는 딱 1번
#    - 조금씩 자주 걸으면서 방향을 재확인하는 게 더 빠르고 정확하게 수렴함

# 2. 노이즈 낀 기울기가 오히려 도움
#    - 전체 데이터로 계산한 기울기는 정확하지만 지역 최솟값/안장점에 갇히면 못 빠져나옴
#    - mini-batch는 매번 다른 부분집합이라 기울기가 조금씩 달라짐(노이즈)
#    - 이 노이즈가 얕은 지역 최솟값을 흔들어서 탈출하게 도와줌

# 3. 메모리 문제
#    - 순전파/역전파 시 각 층의 출력값을 다 들고 있어야 함
#    - 배치가 너무 크면 메모리 초과(OOM) 위험
#    - (백만 x 백만 연산 같은 경우가 여기 해당)

# 트레이드오프
# batch 크다  -> 기울기 안정적, 메모리 많이 씀, 업데이트 적음, 지역 최솟값에 갇히기 쉬움
# batch 작다  -> 업데이트 자주, 메모리 적게 씀, 기울기 노이즈 심함, 너무 작으면 불안정

# 실무 관행: 2의 거듭제곱 (32, 64, 128 ...) - GPU 메모리 구조상 효율적

# keras 코드에서 제어 방법
# model.fit(x, y, epochs=10, batch_size=32)
# batch_size 생략 시 기본값 32
# batch_size = len(x) 로 주면 지금까지처럼 데이터를 통으로 학습하는 것과 동일

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 300, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
# result = model.predict(np.array([1, 2, 3, 4, 5, 6]))
# print("예측값 : ", result)