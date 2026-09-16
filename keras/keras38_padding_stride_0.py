import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# padding과 strides의 개념
# 2. 모델 구성
model = Sequential()
# model.add(Conv2D(10, (2, 2), input_shape=(10, 10, 1),   # (9, 9, 10)
#                  padding='valid',
#                  ))
# model.add(Conv2D(filters=9, kernel_size=(3, 3),         # (7, 7, 9)
#                  padding='valid',                       # default
#                  ))

model.add(Conv2D(10, (2, 2), input_shape=(10, 10, 1),   # (10, 10, 10)
                 strides=2,                             # 를 적용하면 (5, 5, 10) >>>> 가급적 권하지 않는 설정
                 padding='same',
                 ))
model.add(Conv2D(filters=9, kernel_size=(3, 3),         # (10, 10, 9)
                 strides=2,                             # 를 적용하면 (3, 3, 9)
                 padding='same',
                 ))

model.summary()