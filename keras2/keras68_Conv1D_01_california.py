from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import time

# DNN 모델에 Conv1D 적용해보기 - 캘리포니아 주택 가격
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=121)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 1) # (16512, 8, 1)
x_test = x_test.reshape(-1, 8, 1)   # (4128, 8, 1)

print(x_train.shape)
print(x_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(100, kernel_size=4, input_shape=(8, 1)))
model.add(GlobalAveragePooling1D())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=50,
                   restore_best_weights=True,
                   verbose=1,
                   )

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,
)

from tensorflow.keras.optimizers import Adam
learning_rate1 = 0.01
learning_rate2 = 0.001  # Adam default
learning_rate3 = 0.0001
learning_rate4 = 0.005
learning_rate5 = 0.05
learning_rate6 = 0.009

model.compile(loss="mse", optimizer=Adam(learning_rate=0.0009))
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.2,
          callbacks=[es, rlr],
         )
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  0.38592788577079773 > 0.3836520314216614

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)           # r2:  0.7092826865107242 > 0.7109970863774868

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)         # mse:  0.3859278720472035 > 0.3836520024594018

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)      # RMSE :  0.6212309329445883 > 0.6193964824402878

print("time : ", round(end - start, 2), "sec")  # time :  363.56 sec > 714.15 sec