# 19-1 참조
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 얼리스탑핑 실습 - 캘리포니아 주택 가격
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

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',         # 기준
    mode='min',                 # auto로 하면 accuracy는 max, loss는 min으로 자동 설정됨
    patience=100,                # 갱신이 안되면 10번 움직임
    restore_best_weights=True,  # 기본값 False / 이론상으로는 True로 했을 때 성능이 가장 좋아야하는데 아닐 때도 있음
)

hist = model.fit(x_train, y_train,
                 epochs=500,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[es],    # 리스트로 받음 > es 말고 다른 요소들도 들어올 수 있다는 얘기
                 )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  0.9900057911872864

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)           # r2:  0.25423430595539853

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)         # mse:  0.9900055964814629

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)      # RMSE :  0.9949902494403967

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False      # 음수 부호 깨짐 방지

plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss')           # y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('California Loss 그래프')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()          # 격자표시 추가
plt.show()