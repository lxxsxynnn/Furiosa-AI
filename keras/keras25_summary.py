from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 파라미터 개수 세는 방법
# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))    # Param: 6
model.add(Dense(4))                 # Param: 16
model.add(Dense(3))                 # Param: 15
model.add(Dense(1))                 # Param: 4
# Total params : 41

'''
y = X w + b

h1 = x1 * w + b1
h2 = x2 * w + b2
h3 = x3 * w + b3

-> x값 개수 + bias 개수 합쳐서 6개
'''

model.summary()
