from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import numpy as np
import pandas as pd

# Sequential 모델을 함수형 모델로 바꿔보기 - 산림 피복 유형
# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)                 # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

y = pd.get_dummies(y, dtype=float).values
print(y.shape)  # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    stratify=y,
    random_state=12,
)

print(x_train.shape, x_test.shape)  # (464809, 54) (116203, 54)
print(y_train.shape, y_test.shape)  # (464809, 7) (116203, 7)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# model = Sequential()
# model.add(Dense(60, input_dim=54, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(120, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(70, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(7, activation='softmax'))

input1 = Input(shape=(54,))
dense1 = Dense(60, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(120, activation='relu')(drop1)
drop2 = Dropout(0.5)(dense2)
dense3 = Dense(70, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(50, activation='relu')(drop3)
drop4 = Dropout(0.1)(dense4)
dense5 = Dense(20, activation='relu')(drop4)
output1 = Dense(7, activation='softmax')(dense5)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'],
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

save_path = 'C:/study/_save/keras34/'
date = datetime.datetime.now()      # 현재 시간 반환
date = date.strftime('%m%d_%H%M_')

filename = '09_fetch_covtype_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
filepath = ''.join([save_path, 'k34_', date, filename])

mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath=filepath,
                      verbose=1,
                      )

model.fit(x_train, y_train,
          epochs=200,
          batch_size=2048,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp],
          )

# 이번 실행에서 생긴 체크포인트 중 최고(마지막 저장) 파일만 남김
import glob, os
files = sorted(glob.glob(save_path + 'k34_' + date + '09_fetch_covtype_*.keras'))
for f in files[:-1]:
    os.remove(f)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', result[0])
print('acc : ', round(result[1], 2))
print('acc_score : ', acc_score)

'''
loss :  0.38224494457244873
acc :  0.85
acc_score :  0.8505374215811984

>>

loss :  0.38543593883514404
acc :  0.85
acc_score :  0.8474049723329001
'''