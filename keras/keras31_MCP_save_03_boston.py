from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np

save_path = 'C:/study/_save/keras31/'

# 최고 가중치 저장하기 - 보스턴 주택 가격
# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()   #  tensorflow는 가져오면서 분할됨(훈련/학습 묶음으로 나오니까 주의할 것)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1
                   )

import datetime
date = datetime.datetime.now()      # 현재 시간 반환
print(date)                         # 2026-09-14 11:41:00.132990
print(type(date))                   # <class 'datetime.datetime'> 클래스 형태
date = date.strftime('%m%d_%H%M_')
print(date)                         # 0914_1147
print(type(date))                   # <class 'str'>

filename = '03_boston_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
filepath = ''.join([save_path, 'k31_', date, filename])

mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath=filepath,
                      verbose=1,
                      )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_split=0.25,
          callbacks=[es, mcp]
          )

# 이번 실행에서 생긴 체크포인트 중 최고(마지막 저장) 파일만 남김
import glob, os
files = sorted(glob.glob(save_path + 'k31_' + date + '03_boston_*.keras'))
for f in files[:-1]:
    os.remove(f)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  22.366579055786133

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)          # r2 :  0.7313125397984366

mse = mean_squared_error(y_test, y_pred)
print("mse : ", mse)        # mse :  22.366578236188758

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)      # RMSE :  4.729331690227358