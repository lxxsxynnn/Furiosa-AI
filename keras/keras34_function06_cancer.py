import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer     # 유방암 관련 데이터셋 불러오기(이진분류, y가 0/1)

# Sequential 모델을 함수형 모델로 바꿔보기 - 유방암
# 1. 데이터
datasets = load_breast_cancer()     # sklearn에서 제공하는 교육용 데이터셋, 실무에서 쓸 일이 없음
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=234,
    stratify=y,         # 범주형 데이터에서는 불균형하게 나눠지는 걸 막기 위한 기능이 있음 / y를 기준으로 동일한 비율로 데이터를 잘라줌
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델링
# model = Sequential()
# model.add(Dense(32, input_dim = 30, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(72, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(36, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(1, activation='sigmoid'))

input1 = Input(shape=(30,))
dense1 = Dense(32, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(64, activation='relu')(drop1)
drop2 = Dropout(0.4)(dense2)
dense3 = Dense(128, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(72, activation='relu')(drop3)
drop4 = Dropout(0.5)(dense4)
dense5 = Dense(36, activation='relu')(drop4)
drop5 = Dropout(0.25)(dense5)
output1 = Dense(1, activation='sigmoid')(drop5)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',     # loss는 w 갱신할 때 사용하는데 이진분류는 무조건 binary_crossentropy
            #   metrics=['accuracy'],
              metrics=['acc'],      # train data 지표
             )  

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=20,
                   restore_best_weights=True,
                  )

import datetime

save_path = 'C:/study/_save/keras34/'
date = datetime.datetime.now()      # 현재 시간 반환
date = date.strftime('%m%d_%H%M_')

filename = '06_cancer_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
filepath = ''.join([save_path, 'k34_', date, filename])

mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath=filepath,
                      verbose=1,
                      )

hist = model.fit(x_train, y_train, 
          epochs=1000,
          batch_size=32,
          verbose=1,
          validation_split=0.3,
          callbacks=[es, mcp],
         )

# 이번 실행에서 생긴 체크포인트 중 최고(마지막 저장) 파일만 남김
import glob, os
files = sorted(glob.glob(save_path + 'k34_' + date + '06_cancer_*.keras'))
for f in files[:-1]:
    os.remove(f)

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print('loss : ', loss[0])
print('acc : ', round(loss[1], 4))
print('acc_score : ', acc_score)

'''
loss :  0.10264232754707336
acc :  0.9532
acc_score :  0.9532163742690059

>>

loss :  0.1285681128501892
acc :  0.9649
acc_score :  0.9649122807017544
'''