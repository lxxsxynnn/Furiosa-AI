# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터
path = "C:/study/_data/kaggle_santander/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)      # (200000, 201)
print(test_csv.shape)       # (200000, 200)
print(submission_csv.shape) # (200000, 1)

print(train_csv.isna().sum())   # Length: 201, dtype: int64
print(test_csv.isna().sum())    # Length: 200, dtype: int64

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)

print(np.unique(y, return_counts=True)) # (array([0, 1]), array([179902,  20098]))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=100, train_size=0.7,
                                                    stratify=y)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(400, input_dim = 200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

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
          callbacks=[es],
         )

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss[0])           # loss :  0.21817214787006378 > loss :  0.23302412033081055
print('acc : ', round(loss[1], 4))  # acc :  0.9298 > acc :  0.9138

y_predict = model.predict(x_test)

y_predict = np.round(y_predict)
print(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print('acc_score : ', acc_score)        # acc_score :  0.935672514619883 > acc_score :  0.9138

# y_submit = model.predict(test_csv)
# submission_csv['target'] = np.round(y_submit).astype(int)

# from datetime import datetime
# now = datetime.now().strftime('%m%d_%H%M')
# submission_csv.to_csv(path + "submit/" + "submit_" + now + ".csv")