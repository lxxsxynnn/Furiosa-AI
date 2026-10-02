from sklearn.datasets import load_wine
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

# DNN 모델을 RNN으로 바꿔보기 - 와인
# 1 . 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48]))

y = pd.get_dummies(y, dtype=float).values
print(y.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8,
    random_state=100,
    shuffle=True,
    stratify=y
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(20, input_shape=(13, 1), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.02), 
              metrics=['acc']
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=50,
    factor=0.5,
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, rlr],
          )
end = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc_score = accuracy_score(y_test, y_predict)

print('loss : ', result[0])
print('acc : ', round(result[1], 2))
print('acc_score : ', acc_score)
print("time : ", round(end - start, 2), "sec")

'''
loss :  1.3628830909729004
acc :  0.97
acc_score :  0.9722222222222222
time :  36.09 sec

>>

loss :  0.046640750020742416
acc :  1.0
acc_score :  1.0
time :  159.03 sec
'''