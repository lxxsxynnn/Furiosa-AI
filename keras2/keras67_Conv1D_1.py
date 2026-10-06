import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Flatten

# 54-1 카피
# RNN > CNN 바꿔보기(Conv1D 적용)
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

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  # (7, 3, 1)

# 2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3, 1)))   
# kernel_size : 모델 안에서 몇 칸씩 묶어 볼지. timesteps(3)가 데이터를 몇 개씩 잘랐는지라면, kernel_size(2)는 그 3칸을 다시 2칸씩 묶음
model.add(Conv1D(10, 2))
# Conv1D 출력이 3차원이라 Dense를 바로 붙이면 마지막 축에만 적용돼 3차원이 남음. Flatten 또는 GlobalAveragePooling1D로 2차원으로 펴야 함
model.add(Flatten())
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
│ conv1d (Conv1D)                      │ (None, 2, 10)               │              30 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_1 (Conv1D)                    │ (None, 1, 10)               │             210 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 10)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 50)                  │             550 │
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
 Total params: 91,961 (359.22 KB)
 Trainable params: 91,961 (359.22 KB)
 Non-trainable params: 0 (0.00 B)
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
print('loss : ', results)   # loss :  6.4963906649950776e-12 > 2.7274779768049484e-08

x_predict = np.array([8, 9, 10]).reshape(1, 3, 1)  # x가 (7, 3, 1)이므로 모양을 맞춰줘야 함
y_predict = model.predict(x_predict)

print('Predict [8, 9, 10] : ', y_predict)   # Predict [8, 9, 10] :  [[10.776364]] > [[11.019807]]