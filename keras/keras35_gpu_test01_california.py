# 30-1 참조
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import time

path = 'C:\study\_save\keras35/'

# CPU / GPU 실행 시간 비교 - 캘리포니아 주택
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=121  )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

from tensorflow.keras.callbacks import ModelCheckpoint
# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
# es = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=30,
#                    restore_best_weights=True,
#                    verbose=1,
#                    )

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=path + 'keras35_01_california.keras',
    verbose=1,
)

start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs=1000, batch_size=32,
          validation_split=0.2,
          callbacks=[mcp],
         )  # 성능이 붙어있는 상태
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print("loss : ", loss)
print("r2: ", r2)
print("mse: ", mse)
print("RMSE : ", rmse)

print('실행 시간: ', round(end_time - start_time, 2), '초')

'''
실행 시간:  631.2초 >> 실행 시간:  995.79 초
'''