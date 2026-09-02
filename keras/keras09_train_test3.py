import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 라이브러리(sklearn) 활용해서 데이터 분리하기
# 1, 2에서 한 작업의 문제점 > 순차적으로 나열된 데이터를 활용해서 데이터가 편중되어 있음
# 데이터를 랜덤하게 뽑아서 처리하는 게 신뢰도가 높아짐
# 데이터 분리를 할 때 안에서 랜덤으로 뽑아서 사용하기 위해 sklearn을 사용

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# [검색] train과 test를 섞어서 7:3 나누기
# 힌트: 사이킷런
x_train, x_test, y_train, y_test = train_test_split(x, y,              # x와 y를 자름
                                                    train_size=0.7,    # 훈련 데이터의 비중을 0.7로 (기본값 : 0.75)
                                                  # test_size=0.3,     # 테스트 데이터의 비중을 0.3으로 (비중은 둘 중 하나만 적어도 됨 / 단, 두 개의 합이 전체를 초과할 수 없음)
                                                  # shuffle=True,      # 기본값
                                                    random_state=123,  # 호출할 때마다 동일한 학습/테스트용 데이터 세트를 생성하기 위해 주어지는 난수 값(값을 바꾸면 새로운 데이터를 가지고 훈련을 진행함)
                                                    )
# 값을 반환하는 함수이기 때문에 반환 값을 저장할 변수가 필요함

print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=200, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)           # loss:  1.4086682043057408e-08
results = model.predict(np.array([[11]]))
print("results : ", results)    # results :  [[10.999918]]