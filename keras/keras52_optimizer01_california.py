# 28-1 참조
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os

# 옵티마이저 - 캘리포니아 주택 가격
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (20640, 8) (20640,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=121)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1,
                   )

path = 'C:/study/_save/keras52/'
os.makedirs(path, exist_ok=True)    # 폴더가 없으면 ModelCheckpoint가 저장할 때 에러
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=path + 'keras52_01_california.keras'
)

from tensorflow.keras.optimizers import Adam
learning_rate1 = 0.01
learning_rate2 = 0.001  # Adam default
learning_rate3 = 0.0001
learning_rate4 = 0.005
learning_rate5 = 0.05
learning_rate6 = 0.009

model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate2))
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.2,
          callbacks=[es, mcp]
         )
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  0.5106000304222107 > 0.5050846338272095

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)           # r2:  0.6153678981162408 > 0.6195226915201146

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)         # mse:  0.5105999598696697 > 0.5050844625024586

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)      # RMSE :  0.7145627753176551 > 0.7106929453023004

print("time : ", round(end - start, 2), "sec")  # 324.66sec