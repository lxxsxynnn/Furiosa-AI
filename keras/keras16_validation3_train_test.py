from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

# 검증 데이터 함수 사용해서 자르기
# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] train_test_split으로 자르기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=12, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=8, random_state=2)
# random_state는 두 호출이 서로 독립적이라 같은 값일 필요 없음 (1, 2처럼 달라도 됨)
# 중요한 건 '값을 주는 것' 자체 - 그래야 매번 같은 분할이 재현됨

print(x_train)
print(x_val)
print(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=100, batch_size=2,
          validation_data = (x_val, y_val)
         )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  0.06897628307342529