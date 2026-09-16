import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# MaxPooling의 개념
# 2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2, 2), input_shape=(10, 10, 1),
                 strides=2,
                 padding='same',
                 ))

model.add(MaxPooling2D())

model.add(Conv2D(filters=9, kernel_size=(3, 3),
                 strides=2,
                 padding='same',
                 ))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                  Output Shape              Param #   
=================================================================
 conv2d (Conv2D)               (None, 5, 5, 10)          50                                                      
 max_pooling2d (MaxPooling2D)  (None, 2, 2, 10)          0         
 conv2d_1 (Conv2D)             (None, 1, 1, 9)           819       
=================================================================
Total params: 869
Trainable params: 869

적용 전과 비교해서 연산량이 확 줄어들었음
'''