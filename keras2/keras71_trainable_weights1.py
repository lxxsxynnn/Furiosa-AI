import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 가중치 꺼내서 동결하기
print(tf.__version__)   # 2.21.0

# 실행할 때 같은 랜덤값이 나오게 고정하는 방법 3가지
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
print("=============== model.weights ===============")
print(model.weights)
'''
[<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, value=[[ 0.8516406  -0.52920127 -0.9112464 ]]>, 
 <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, value=[0. 0. 0.]>, ** bias 초기값은 0
 <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, value=[[ 0.6380346  -1.0862008 ] [-0.38601977 -0.21482044] [ 0.20393836 -0.87937653]]>, 
 <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, value=[0. 0.]>, 
 <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, value=[[-0.5044266 ] [ 0.90831745]]>, 
 <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, value=[0.]>]
'''

print("=============== model.trainable.weights ===============")
print(model.trainable_weights)
'''
[<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, value=[[ 0.8516406  -0.52920127 -0.9112464 ]]>, 
 <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, value=[0. 0. 0.]>, 
 <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, value=[[ 0.6380346  -1.0862008 ] [-0.38601977 -0.21482044] [ 0.20393836 -0.87937653]]>, 
 <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, value=[0. 0.]>, 
 <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, value=[[-0.5044266 ] [ 0.90831745]]>, 
 <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, value=[0.]>]
'''

print("========================================================")
print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 6

################ 동결 ################
model.trainable = False
print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 0

print("=============== model.weights ===============")
print(model.weights)
print("=============== model.trainable.weights ===============")
print(model.trainable_weights)  # []


model.summary()

'''
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 3)                   │               6 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 2)                   │               8 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 1)                   │               3 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 17 (68.00 B)
 Trainable params: 17 (68.00 B)
 Non-trainable params: 0 (0.00 B)

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 3)                   │               6 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 2)                   │               8 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 1)                   │               3 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 17 (68.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 17 (68.00 B)
'''