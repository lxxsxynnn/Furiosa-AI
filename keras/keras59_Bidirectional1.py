import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# RNN 모델 양방향 학습시키기 - Bidirectional
# 1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x = np.array([[1, 2, 3], 
              [2, 3, 4], 
              [3, 4, 5], 
              [4, 5, 6], 
              [5, 6, 7], 
              [6, 7, 8], 
              [7, 8, 9], 
              ])
# 데이터가 적으므로 직접 구현
# [8, 9, 10] 부터는 뒤에 예측할 값이 없으므로 x에 추가하지 않음

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  # (7, 3, 1)

# 2. 모델 구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(units=10), input_shape=(3, 1)))   # Bidirectional 자체는 혼자 동작하지 않아서 다른 레이어를 감싸서 사용(Wrapper Layer)
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# model.summary()
'''
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ bidirectional (Bidirectional)        │ (None, 20)                  │             240 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 50)                  │           1,050 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 100)                 │           5,100 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 200)                 │          20,200 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 200)                 │          40,200 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (Dense)                      │ (None, 100)                 │          20,100 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_5 (Dense)                      │ (None, 50)                  │           5,050 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_6 (Dense)                      │ (None, 10)                  │             510 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_7 (Dense)                      │ (None, 1)                   │              11 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 92,461 (361.18 KB)
 Trainable params: 92,461 (361.18 KB)
 Non-trainable params: 0 (0.00 B)
 
 > 파라미터가 2배
 '''

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    restore_best_weights=True,
    patience=100,
)

model.fit(x, y, epochs=1500,
          callbacks=[es],
          )

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)   # loss :  6.4963906649950776e-12 > 6.301498819277773e-12

x_predict = np.array([8, 9, 10]).reshape(1, 3, 1)
y_predict = model.predict(x_predict)

print('Predict [8, 9, 10] : ', y_predict)   # Predict [8, 9, 10] :  [[10.776364]] > [[10.603219]]