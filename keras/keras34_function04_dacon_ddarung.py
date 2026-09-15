from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

save_path = 'C:/study/_save/keras34/'

# Sequential 모델을 함수형 모델로 바꿔보기 - 따릉이(데이콘)
# 1. 데이터
data_path = "C:/study/_data/ddarung/"

train_csv = pd.read_csv(data_path + "train.csv", index_col=0) # index_col - 인덱스 컬럼(제외)

############ 결측치 처리 1.삭제 ############
train_csv = train_csv.dropna()
print(train_csv) # [1328 * 10]

########### train_csv를 x와 y로 분리 ###########
x = train_csv.drop(['count'], axis=1) # count 컬럼을 제거한 나머지 / axis=1 열(컬럼) 삭제
print("x : ", x) # x : [1328 rows x 9 columns] 

y = train_csv['count'] # count 컬럼만 선택
print(y)
print(y.shape) # (1328,) -> y가 벡터 형태로 빠짐

# model.predict()에 넣을 값
test_csv = pd.read_csv(data_path + "test.csv", index_col= 0) # id 컬럼은 순번(인덱스)인데 이걸 불러오면 id가 실제 데이터처럼 포함됨 -> index_col=0으로 인덱스 지정하면 데이터에서 제외됨

#여기에 model.predict(submission) 값을 넣음
submission = pd.read_csv(data_path + "submission.csv")

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, test_size=0.2, random_state=333)

# 2. 모델 구성
# model = Sequential()
# model.add(Dense(64, input_dim=9))
# model.add(Dropout(0.2))
# model.add(Dense(256))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dropout(0.2))
# model.add(Dense(32))
# model.add(Dense(1))

input1 = Input(shape=(9,))
dense1 = Dense(64)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(256)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(128)(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(64)(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(32)(drop4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)

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

filename = '04_ddarung_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
filepath = ''.join([save_path, 'k34_', date, filename])

mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath=filepath,
                      verbose=1,
                      )

hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp]
          )

# 이번 실행에서 생긴 체크포인트 중 최고(마지막 저장) 파일만 남김
import glob, os
files = sorted(glob.glob(save_path + 'k34_' + date + '04_ddarung_*.keras'))
for f in files[:-1]:
    os.remove(f)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)
print("mse : ", mse)
print("RMSE : ", rmse)

'''
loss :  2884.355224609375
r2 :  0.5717144546277286
mse :  2884.3549223991245
RMSE :  53.706190726946225

>>

loss :  2888.400146484375
r2 :  0.5711137742636446
mse :  2888.400296761288
RMSE :  53.74383961684621
'''