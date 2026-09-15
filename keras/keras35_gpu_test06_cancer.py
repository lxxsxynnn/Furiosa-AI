import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer     # 유방암 관련 데이터셋 불러오기(이진분류, y가 0/1)
import time

# CPU / GPU 실행 시간 비교 - 유방암
# 1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=234,
    stratify=y,
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델링
model = Sequential()
model.add(Dense(32, input_dim = 30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(72, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(36, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',     # loss는 w 갱신할 때 사용하는데 이진분류는 무조건 binary_crossentropy
            #   metrics=['accuracy'],
              metrics=['acc'],      # train data 지표
             )  

# es = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=20,
#                    restore_best_weights=True,
#                   )

# import datetime

# save_path = 'C:/study/_save/keras34/'
# date = datetime.datetime.now()      # 현재 시간 반환
# date = date.strftime('%m%d_%H%M_')

# filename = '06_cancer_{epoch:04d}-{val_loss:.4f}.keras'    # history에서 가져옴
# filepath = ''.join([save_path, 'k34_', date, filename])

# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath=filepath,
#                       verbose=1,
#                       )

start_time = time.time()
hist = model.fit(x_train, y_train, 
          epochs=1000,
          batch_size=32,
          verbose=1,
          validation_split=0.3,
        #   callbacks=[es, mcp],
         )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print('loss : ', loss[0])
print('acc : ', round(loss[1], 4))
print('acc_score : ', acc_score)

print('실행 시간: ', round(end_time - start_time, 2), '초')

'''
실행 시간:  61.14 초 >> 실행 시간:  58.94 초
'''