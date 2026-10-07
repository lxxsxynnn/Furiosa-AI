from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import numpy as np
import pandas as pd

# DNN 모델에 Conv1D 적용해보기 - 산림 피복 유형
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

x_train = x_train.reshape(-1, 54, 1)
x_test = x_test.reshape(-1, 54, 1)

# 2. 모델 구성
model = Sequential()
# model.add(LSTM(60, input_shape=(54, 1), activation='relu'))
model.add(Conv1D(60, kernel_size=3, input_shape=(54, 1)))
model.add(GlobalAveragePooling1D())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(70, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.015),
              metrics=['acc'],
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=20,
    factor=0.45,
    verbose=1,
)

start = time.time()
model.fit(x_train, y_train,
          epochs=200,
          batch_size=2048,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, rlr],
          )
end = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', result[0])                     # loss :  0.7251755595207214 > 0.689144492149353
print('acc : ', round(result[1], 2))            # acc :  0.69 > 0.7
print('acc_score : ', acc_score)                # acc_score :  0.692572480917016 > 0.696126606025662
print("time : ", round(end - start, 2), "sec")  # time :  822.57 sec > 126.22 sec