import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 다중 입력 모델 구현(입력 2개 → 출력 1개)
# 1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T     # (100, 2)
x2_datasets = np.array([range(101, 201), range(411, 511),
                        range(150, 250)]).transpose()       # (100, 3)
y = np.array(range(3001, 3101))

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y, train_size=0.8, random_state=123)

# 2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)

# 2-2. 모델
input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(3, name='han25')(dense24)
# model1 = Model(inputs=input21, outputs=output21)

# 2-3. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate     # keras 2.9에는 layers.merge 모듈이 없음
# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21])
merge2 = Dense(5, name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)

last_output = Dense(1, name='last')(merge3)
model = Model(inputs=[input1, input21], outputs=last_output)
model.summary()
'''
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)    │ (None, 3)                 │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ input_layer (InputLayer)      │ (None, 2)                 │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han21 (Dense)                 │ (None, 50)                │             200 │ input_layer_1[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han1 (Dense)                  │ (None, 10)                │              30 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han22 (Dense)                 │ (None, 40)                │           2,040 │ han21[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han2 (Dense)                  │ (None, 20)                │             220 │ han1[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han23 (Dense)                 │ (None, 30)                │           1,230 │ han22[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han3 (Dense)                  │ (None, 30)                │             630 │ han2[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han24 (Dense)                 │ (None, 20)                │             620 │ han23[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han4 (Dense)                  │ (None, 5)                 │             155 │ han3[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han25 (Dense)                 │ (None, 3)                 │              63 │ han24[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mg1 (Concatenate)             │ (None, 8)                 │               0 │ han4[0][0], han25[0][0]    │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mg2 (Dense)                   │ (None, 5)                 │              45 │ mg1[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mg3 (Dense)                   │ (None, 10)                │              60 │ mg2[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ last (Dense)                  │ (None, 1)                 │              11 │ mg3[0][0]                  │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 5,304 (20.72 KB)
 Trainable params: 5,304 (20.72 KB)
 Non-trainable params: 0 (0.00 B)
 '''

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8)

# 4. 평가, 예측
result = model.evaluate([x1_test, x2_test], y_test)
print('loss: ', result)     # loss:  0.02441691793501377

x1_pred = np.array([range(100, 106), range(400, 406)]).T
x2_pred = np.array([range(200, 206), range(510, 516),
                   range(249, 255)]).T

y_predict = model.predict([x1_pred, x2_pred])
print('y_predict: ', y_predict)
'''
y_predict:  [[3101.8247]
 [3103.547 ]
 [3105.27  ]
 [3106.9924]
 [3108.7148]
 [3110.44  ]]
'''