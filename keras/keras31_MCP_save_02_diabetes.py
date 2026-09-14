from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np

# 최고 가중치 저장하기 - 당뇨
# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=45)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train)) # -0.13776722569000302 0.19878798965729408

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(6))
model.add(Dense(6))
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

save_path = 'C:/study/_save/keras31/'

filename = '02_diabetes_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
filepath = ''.join([save_path, 'k31_', date, filename])

mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath=filepath,
                      verbose=1,
                      )

hist = model.fit(x_train, y_train, epochs=640, batch_size=16,
          validation_split=0.33,
          callbacks=[es, mcp]
          )

# 이번 실행에서 생긴 체크포인트 중 최고(마지막 저장) 파일만 남김
import glob, os
files = sorted(glob.glob(save_path + 'k31_' + date + '02_diabetes_*.keras'))
for f in files[:-1]:
    os.remove(f)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : " , loss)     # loss :  2598.738525390625

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)          # r2 :  0.4910995855751583

mse = mean_squared_error(y_test, y_pred)
print("mse : ", mse)        # mse :  2598.738440470018

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)      # RMSE :  50.977823025998454