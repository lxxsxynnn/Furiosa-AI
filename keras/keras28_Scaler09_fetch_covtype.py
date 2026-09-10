from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
import numpy as np
import pandas as pd

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

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(60, input_dim=54, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='softmax'))

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

start = time.time()
model.fit(x_train, y_train,
          epochs=200,
          batch_size=2048,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end = time.time()

# acc : 0.93
# 4. 예측, 훈련
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])             # loss :  0.25401097536087036 > loss :  0.258330762386322
print('acc : ', round(result[1], 2))    # acc :  0.9 > acc :  0.9

y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score : ', accuracy_score)           # acc_score :  0.8998562859822896 > acc_score :  0.8967410479935974
print('time : ', round(end - start, 2), 'sec')  # time :  339.34 sec > time :  165.86 sec