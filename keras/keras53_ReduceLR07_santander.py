# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# ReduceLR - 산탄데르
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

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(400, input_dim = 200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.012),
              metrics=['acc'],
             )  

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=20,
                   restore_best_weights=True,
                  )

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=30,
    verbose=1,
    factor=0.6,
)

start = time.time()
model.fit(x_train, y_train, 
          epochs=100,
          batch_size=32,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, rlr],
         )
end = time.time()

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print('loss : ', loss[0])
print('acc : ', round(loss[1], 4))
print('acc_score : ', acc_score)
print("time : ", round(end - start, 2), "sec")

'''
loss :  0.3261420428752899
acc :  0.8995
acc_score :  0.8995166666666666
time :  243.71 sec

>>

loss :  0.32614579796791077
acc :  0.8995
acc_score :  0.8995166666666666
time :  539.66 sec
'''

# y_submit = model.predict(test_csv)
# submission_csv['target'] = np.round(y_submit).astype(int)

# from datetime import datetime
# now = datetime.now().strftime('%m%d_%H%M')
# submission_csv.to_csv(data_path + "submit/" + "submit_" + now + ".csv")