import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 다중 입력 모델 구현(입력 3개 → 출력 2개)
# 1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T     # (100, 2)
                        # 삼성 종가   하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511),
                            # 원유가          환율 
                        range(150, 250)]).transpose()       # (100, 3)
                            # 금 시세
x3_datasets = np.array([range(100), range(301, 401),
                        range(77, 177), range(33, 133)]).T  # (100, 4)
y1 = np.array(range(3001, 3101))
             # 화성의 화씨 온도
y2 = np.array(range(13001, 13101))  # 비트코인 가격

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2, train_size=0.8, random_state=123
)

# 2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, name='han1')(input1)
dense2 = Dense(20, name='han2')(dense1)
dense3 = Dense(30, name='han3')(dense2)
output1 = Dense(5, name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)

# 2-2. 모델
input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(3, name='han25')(dense24)
# model1 = Model(inputs=input21, outputs=output21)

# 2-3. 모델
input31 = Input(shape=(4,))
dense31 = Dense(50, name='han31')(input31)
dense32 = Dense(30, name='han32')(dense31)
output31 = Dense(3, name='han35')(dense32)

# 2-4. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate     # keras 2.9에는 layers.merge 모듈이 없음
# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21, output31])
merge2 = Dense(5, name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)

# 2-5. 분기 1
last_dense1 = Dense(10, name='ld1')(merge3)
last_dense2 = Dense(10, name='ld2')(last_dense1)
last_output1 = Dense(1, name='last')(last_dense2)

# 2-6. 분기 2
last_output2 = Dense(1, name='last2')(merge3)


model = Model(inputs=[input1, input21, input31],
              outputs=[last_output1, last_output2])

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=300, batch_size=8)

# 4. 평가, 예측
result = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print('loss: ', result) # loss:  [1.0627508345351089e-05, 9.435415449843276e-06, 1.1920928955078125e-06]

x1_pred = np.array([range(100, 106), range(400, 406)]).T
x2_pred = np.array([range(200, 206), range(510, 516),
                   range(249, 255)]).T
x3_pred = np.array([range(100, 106), range(400, 406),
                   range(177, 183), range(133, 139)]).T

y_predict = model.predict([x1_pred, x2_pred, x3_pred])
print('y_predict: ', y_predict)
'''
y_predict: [array([[3095.7378], [3096.739 ], [3097.7373], [3098.737 ], [3099.7378], [3100.7383]], dtype=float32),
       array([[13075.845], [13076.846], [13077.844], [13078.846], [13079.846], [13080.846]], dtype=float32)]
'''