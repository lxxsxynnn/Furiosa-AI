from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
import numpy as np
import pandas as pd

# 1. 데이터
datasets = load_digits()
x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)                 # (1797, 64) (1797,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y.reshape(-1 , 1))
print(y.shape)  # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8,
    random_state=100,
    shuffle=True,
    stratify=y,
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=64, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
)

start = time.time()
model.fit(x_train, y_train,
          epochs=1000,
          validation_split=0.2,
          callbacks=[es],
          )
end = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0]) # loss :  0.12074317038059235 > loss :  0.16659173369407654
print('acc : ', result[1])  # acc :  0.9722222089767456 > acc :  0.9527778029441833

y_pred = model.predict(x_test)      # evaluate(채점)가 아니라 predict(예측값) - y 없이 부르면 ValueError

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)    # acc_score :  0.9722222222222222 > acc_score :  0.9527777777777777
print('time : ', round(end - start, 2), 'sec')  # time :  19.44 sec > time :  13.09 sec

# acc : 1.0