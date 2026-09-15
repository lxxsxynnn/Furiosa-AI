# 30-1 참조
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import time

save_path = 'C:/study/_save/keras33/'

# Drop Out 실습해보기 - 캘리포니아 주택
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
# exit()

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

import datetime
date = datetime.datetime.now()      # 현재 시간 반환
date = date.strftime('%m%d_%H%M_')

filename = '01_california_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
filepath = ''.join([save_path, 'k33_', date, filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath,
    verbose=1,
)

start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs=1000, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],  
         )  # 성능이 붙어있는 상태
end_time = time.time()

# 이번 실행에서 생긴 체크포인트 중 최고(마지막 저장) 파일만 남김
import glob, os
files = sorted(glob.glob(save_path + 'k33_' + date + '01_california_*.keras'))
for f in files[:-1]:
    os.remove(f)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print("loss : ", loss)              # loss :  0.5186867713928223
print("r2: ", r2)                   # r2:  -3.377297237332068
print("mse: ", mse)                 # mse:  5.810871694726173
print("RMSE : ", rmse)              # RMSE :  2.410574971811948

'''
loss :  0.34304144978523254
r2:  0.741588686569804
mse:  0.34304158602758045
RMSE :  0.5856975209334426
'''