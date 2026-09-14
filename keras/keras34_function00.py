from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input

# 함수형 모델 만들어보기
# 2-1. 순차적 모델
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dropout(0.2))
model.add(Dense(9))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

# 2-2. 함수형 모델
input1 = Input(shape=(3,))
dense1 = Dense(10, name='1st')(input1)     # name은 안써도 됨, 끝에 (input1)을 달아줘야 앞에 변수 input1과 연결됨
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(9)(drop1)
drop2 = Dropout(0.2)(dense2)
output1 = Dense(1)(drop2)

# model = Sequential()처럼 모델에 대한 정의 추가 필요
model2 = Model(inputs=input1, outputs=output1)

model2.summary()
'''
│ input_layer_1 (InputLayer)           │ (None, 3)                   │               0 │

함수형 모델은 인풋 파라미터에 대해 명시가 되어있음
'''