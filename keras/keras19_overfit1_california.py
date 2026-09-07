# 17-1 참조
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

# 과적합 눈으로 확인해보기 - 캘리포니아 주택 가격
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
hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,  
         )      # epoch 단위로 출력을 해주고 있음

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

print("============= history =============")
print(hist)             # <keras.src.callbacks.history.History object at 0x000001B647F84D50>
print("=========== hist.history ===========")
print(hist.history)     # {'loss': [5.202541828155518, 1.1687170267105103, 1.6816571950912476, 1.504464864730835, 1.2467339038848877, 1.3960566520690918, 0.803559422492981, 0.9427419900894165, 1.2987130880355835, 0.8533874154090881],
                        #  'val_loss': [1.0669183731079102, 1.2030078172683716, 4.273839473724365, 0.8472128510475159, 0.8789268136024475, 2.308255434036255, 0.7464414238929749, 0.739704430103302, 1.4063829183578491, 0.9026002883911133]}
# {'key': value} 형태의 자료구조 > 딕셔너리
print("============== loss ==============")
print(hist.history['loss']) # 딕셔너리에 있는 loss 값만 리스트 형태로 빼기  [380.23468017578125, 13.053133010864258, 11.87856674194336, 7.972664833068848, 10.172430038452148, 39.329925537109375, 13.205758094787598, 2.8293049335479736, 3.3304946422576904, 3.573814868927002]
print("=========== val_loss ===========")
print(hist.history['val_loss'])
print("=================================")

import matplotlib.pyplot as plt

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'][2:], c='red', label='loss')           # y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('California Loss 그래프')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()          # 격자표시 추가
plt.show()