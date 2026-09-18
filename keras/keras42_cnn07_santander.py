# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime

# 2차원 데이터를 CNN 모델로 처리해보기 - 산탄데르
# 1. 데이터
data_path = "C:/study/_data/kaggle_santander/"

train_csv = pd.read_csv(data_path + "train.csv", index_col=0)
test_csv = pd.read_csv(data_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(data_path + "sample_submission.csv", index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)

print(np.unique(y, return_counts=True)) # (array([0, 1]), array([179902,  20098]))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=100, train_size=0.7,
                                                    stratify=y)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 20, 10, 1)
x_test = x_test.reshape(-1, 20, 10, 1)

print(x_train.shape, x_test.shape)  # 

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(200, (2, 2), input_shape=(20, 10, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(400, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(200, (2,2), activation='relu', padding='same'))
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))     # 이진분류 출력층

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

model.fit(x_train, y_train, 
          epochs=100,
          batch_size=32,
          verbose=1,
          validation_split=0.2,
          callbacks=[es,],
         )

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print('loss : ', loss[0])
print('acc : ', round(loss[1], 4))
print('acc_score : ', acc_score)

'''
loss :  0.26940223574638367
acc :  0.8995
acc_score :  0.8995166666666666

>>

loss :  0.2632559537887573
acc :  0.9062
acc_score :  0.9062333333333333
'''

# y_submit = model.predict(test_csv)
# submission_csv['target'] = np.round(y_submit).astype(int)

# from datetime import datetime
# now = datetime.now().strftime('%m%d_%H%M')
# submission_csv.to_csv(data_path + "submit/" + "submit_" + now + ".csv")