from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 실제 데이터 셋으로 검증해보기 - 캘리포니아 주택 가격
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=121)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=1000, batch_size=10,
          validation_split=0.2,  
         )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  0.5714614391326904 (1000번 시행, 배치 10, 랜덤시드 121)