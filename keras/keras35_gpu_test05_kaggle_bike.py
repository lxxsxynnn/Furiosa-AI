# https://www.kaggle.com/competitions/bike-sharing-demand/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import time

# CPU / GPU 실행 시간 비교 - 자전거(캐글)
# 1. 데이터
data_path = "C:/study/_data/kaggle_bike/"

train_csv = pd.read_csv(data_path + "train.csv", index_col=0)    # 날짜 시간 데이터를 인덱스로
test_csv = pd.read_csv(data_path + "test.csv", index_col=0)
submission = pd.read_csv(data_path + "sampleSubmission.csv", index_col=0)
# print(train_csv.describe())

############## 결측치 확인 ##############
print(train_csv.isna().sum())   # 결측치가 있는지 확인 info랑 둘 중에 편한 거 골라서 사용하면 됨
print(test_csv.isnull().sum())  # 마찬가지로 결측치 확인하는 방법

############## x, y 분리 ##############
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   # test에 없는 컬럼 제거
print(x)    # [10886 rows x 8 columns]
y = train_csv['count']  # train_csv에서 count만 사용
print(y, y.shape)       # (10886,) > 벡터 반환

# test, train 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=23)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',))
model.add(Dropout(0.5))
model.add(Dense(120, activation='relu',))
model.add(Dropout(0.5))
model.add(Dense(96, activation='relu',))
model.add(Dropout(0.5))
model.add(Dense(48, activation='relu',))
model.add(Dense(6, activation='relu',))
model.add(Dense(1))

# input1 = Input(shape=(8,))
# dense1 = Dense(16, activation='relu')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(64, activation='relu')(drop1)
# drop2 = Dropout(0.5)(dense2)
# dense3 = Dense(120, activation='relu')(drop2)
# drop3 = Dropout(0.5)(dense3)
# dense4 = Dense(96, activation='relu')(drop3)
# drop4 = Dropout(0.5)(dense4)
# dense5 = Dense(48, activation='relu')(drop4)
# dense6 = Dense(6, activation='relu')(dense5)
# output1 = Dense(1)(dense6)

# model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

# es = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=20,
#                    restore_best_weights=True,
#                    verbose=1
#                    )

# import datetime

# save_path = 'C:/study/_save/keras34/'
# date = datetime.datetime.now()      # 현재 시간 반환
# print(date)                         # 2026-09-14 11:41:00.132990
# print(type(date))                   # <class 'datetime.datetime'> 클래스 형태
# date = date.strftime('%m%d_%H%M_')
# print(date)                         # 0914_1147
# print(type(date))                   # <class 'str'>

# filename = '05_bike_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
# filepath = ''.join([save_path, 'k34_', date, filename])

# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath=filepath,
#                       verbose=1,
#                       )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=8, validation_split=0.2,
                # callbacks=[es, mcp],
                 )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("loss : ", loss)
print("r2 : ", r2)
print("mse : ", mse)
print("rmse : ", rmse)

print('실행 시간: ', round(end_time - start_time, 2), '초')

'''
실행 시간:  513.01 초 >> 실행 시간:  405.62 초
'''

# y_submit = model.predict(test_csv)
# submission['count'] = y_submit

# from datetime import datetime
# now = datetime.now().strftime('%m%d_%H%M')
# submission.to_csv(data_path + "submit/" + "submit_" + now + ".csv")