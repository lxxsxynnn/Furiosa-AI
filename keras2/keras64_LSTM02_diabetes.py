from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np
import time

# DNN 모델을 RNN으로 바꿔보기 - 당뇨
# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=45)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(5, input_shape=(10, 1)))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=45,
                   restore_best_weights=True,
                   verbose=1,
                   )
rlr = ReduceLROnPlateau(
    monitor='val_loss,',
    mode='min',
    patience=30,
    verbose=1,
    factor=0.5,
)

from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001

model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
start = time.time()
hist = model.fit(x_train, y_train, epochs=640, batch_size=16,
          validation_split=0.33,
          callbacks=[es, rlr]
         )
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  4027.325927734375 > 3946.05908203125

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)           # r2:  0.21134505258133196 > 0.22725928402961326

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)         # mse:  4027.3261133813453 > 3946.0588873329507

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)      # RMSE :  63.4612173959919 > 62.817663816262304

print("time : ", round(end - start, 2), "sec")  # time :  54.98 sec > 60.8 sec