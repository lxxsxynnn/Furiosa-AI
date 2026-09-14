# 30-1 참조
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

path = 'C:\study\_save\keras30/'

# 최고의 가중치 저장하기(ModelCheckPoint)
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=121  )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# model = Sequential()
# model.add(Dense(5, input_dim=8))
# model.add(Dense(6))
# model.add(Dense(7))
# model.add(Dense(7))
# model.add(Dense(1))
model = load_model(path + 'keras30_mcp1.keras')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 3. 컴파일, 훈련
# model.compile(loss="mse", optimizer="adam")
# es = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=30,
#                    restore_best_weights=True,
#                    verbose=1,
#                    )

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     filepath=path + 'keras30_mcp1.keras',
#     verbose=1,
# )

# start_timne = time.time()
# hist = model.fit(x_train, y_train,
#                  epochs=1000, batch_size=32,
#           validation_split=0.2,
#           callbacks=[es, mcp],  
#          )  # 성능이 붙어있는 상태
# end_timne = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)              # loss :  3.2042396068573

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)                   # r2:  -1.4137360856877286

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)                 # mse:  3.2042399541070834

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)              # RMSE :  1.7900390928991141