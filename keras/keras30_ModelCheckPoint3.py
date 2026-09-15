# 30-1 참조
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

path = 'C:\study\_save\keras30/'

# 체크포인트 파일명 만들기 - 캘리포니아 주택
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
print(np.min(x_train), np.max(x_train)) # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))   # -0.0012367054167697258 1.0
# exit()

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))

# exit()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1,
                   )

################ mcp 세이브 파일명 만들기 ################
import datetime
date = datetime.datetime.now()      # 현재 시간 반환
print(date)                         # 2026-09-14 11:41:00.132990
print(type(date))                   # <class 'datetime.datetime'> 클래스 형태
date = date.strftime('%m%d_%H%M_')
print(date)                         # 0914_1147
print(type(date))                   # <class 'str'>

filename = '{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
filepath = ''.join([path, 'k30_', date, filename])
#######################################################

# exit()

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath,
    verbose=1,
)

start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs=1000, batch_size=16,
          validation_split=0.2,
          callbacks=[es, mcp],  
         )  # 성능이 붙어있는 상태
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)              # loss :  0.5186867713928223

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)                   # r2:  -3.377297237332068

mse = mean_squared_error(y_test, y_predict)
print("mse: ", mse)                 # mse:  5.810871694726173

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)              # RMSE :  2.410574971811948