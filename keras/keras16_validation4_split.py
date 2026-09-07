from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

# 검증 데이터 함수 사용해서 자르기
# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

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
        #   validation_data = (x_val, y_val)
            validation_split=0.33,      # 훈련 데이터 중에 33%를 검증에 사용해라 라는 의미 / train_test_split()으로 나눠줘도 상관없음 편한 방법 사용하면 됨
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  0.013258373364806175