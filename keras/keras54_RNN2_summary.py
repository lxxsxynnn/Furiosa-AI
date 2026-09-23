import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

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
model.add(SimpleRNN(units=5, input_shape=(3, 1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()
# 파라미터의 개수 = units * feature + units * bias + unit * units

'''
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ simple_rnn (SimpleRNN)               │ (None, 5)                   │              35 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 7)                   │              42 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1)                   │               8 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 (5 * 5) + (5 * 1) + 5 = 35
 (5 + 1 + 1) * 5
 
현재 아웃풋 * 현재 아웃풋 + 현재 아웃풋 * 이전 층 아웃풋 + 현재 아웃풋(b)
=> Dh * Dh + Dh * d + Dh

 Total params: 85 (340.00 B)
 Trainable params: 85 (340.00 B)
 Non-trainable params: 0 (0.00 B)
'''