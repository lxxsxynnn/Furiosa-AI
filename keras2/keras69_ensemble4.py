import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 다중 출력 모델 구현(입력 1개 → 출력 2개)
# 1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T     # (100, 2)
                        # 삼성 종가   하이닉스 종가
y1 = np.array(range(3001, 3101))
             # 화성의 화씨 온도
y2 = np.array(range(13001, 13101))  # 비트코인 가격

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, y1, y2, train_size=0.8, random_state=123
)

# 2-1. 모델 (공통)
input1 = Input(shape=(2,))
dense1 = Dense(10, name='han1')(input1)
dense2 = Dense(20, name='han2')(dense1)
dense3 = Dense(30, name='han3')(dense2)
output1 = Dense(10, name='han4')(dense3)
# 입력이 하나라 합칠(Concatenate) 것이 없음. 공통 층 출력에서 바로 분기

# 2-2. 분기 1
last_dense1 = Dense(10, name='ld1')(output1)
last_dense2 = Dense(10, name='ld2')(last_dense1)
last_output1 = Dense(1, name='last')(last_dense2)

# 2-3. 분기 2
last_output2 = Dense(1, name='last2')(output1)

model = Model(inputs=input1, outputs=[last_output1, last_output2])   # 입력 하나는 리스트 없이

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1_train, [y1_train, y2_train], epochs=300, batch_size=8)

# 4. 평가, 예측
result = model.evaluate(x1_test, [y1_test, y2_test])
print('loss: ', result) # loss:  [535.2418823242188, 503.18377685546875, 32.058082580566406]

x1_pred = np.array([range(100, 106), range(401, 407)]).T     # (6, 2)

y_predict = model.predict(x1_pred)
print('y_predict: ', y_predict)
# y_predict:  [array([[3062.8616], [3063.0652], [3063.2683], [3063.4712], [3063.675 ], [3063.8774]], dtype=float32), 
# array([[13113.179], [13114.377], [13115.573], [13116.77 ], [13117.967], [13119.164]], dtype=float32)]