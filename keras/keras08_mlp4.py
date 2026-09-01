import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 하나의 x값에 반환하는 y값이 여러 개일 때
# 1. 데이터
x = np.array(range(10))
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]).transpose()
print(x.shape, y.shape)     # (10,) (10, 3)

# [실습]
# x = [10]의 예측값
# 요구사항 y: [11, 0, -1]

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(3))

# 3. 컴파일, 학습
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, epochs=500, batch_size=5)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)          # loss : 2.1113331527544688e-08
np.set_printoptions(suppress=True)
results = model.predict(np.array([[10]]))
print("results : ", results)    # results : [[11.000417 0.00182354 -1.0019611 ]]


####################################################################
# cf) train / test 분리
####################################################################
# 지금은 fit(x, y) 하고 evaluate(x, y) -> 훈련 데이터를 그대로 시험에 냄
# = 시험 문제 미리 보고 푼 점수. loss 2.1e-08은 성능이 아니라 암기력 점수
#
# [과적합은 왜 생기나]
# 1. 모델 표현력 > 데이터
#    - 파라미터 204개(20+66+49+48+21) vs 학습에 쓴 y값 30개(10행 x 3열)
#    - 미지수 204에 식 30 -> 훈련 데이터를 통과하는 해가 무수히 많음
#    - 남는 파라미터로 규칙 대신 데이터를 외움
# 2. 목표가 "훈련 데이터 loss 최소화"뿐 -> 암기가 더 빠르면 암기를 택함
# 3. 노이즈까지 학습 -> 진짜 규칙은 초반에 배우고, 남은 epoch는 노이즈 맞추는 데 씀
#
# [그래서 나눈다]
# - 1~3은 evaluate(x, y)로 탐지 불가 -> 안 본 데이터로 평가해야 구분됨
# - train 70% : test 30%
#     x_train, y_train -> fit
#     x_test,  y_test  -> evaluate
# - test가 fit에 한 번이라도 들어가면 평가 의미 사라짐
# - train loss만 작다 -> 과적합 / 둘 다 크다 -> 과소적합
#
# 주의) x가 0~9로 정렬 -> 앞 7 / 뒤 3으로 자르면 test가 훈련 범위 밖(외삽)
#      -> 자르기 전에 shuffle
#
# 이 파일이 정답을 맞힌 건 y가 노이즈 없는 완벽한 선형이라 운이 좋았을 뿐
