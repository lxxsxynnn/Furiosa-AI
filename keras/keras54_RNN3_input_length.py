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
model.add(SimpleRNN(units=10, input_length=3, input_dim=1))
# input_shape=(3, 1)을 timesteps(input_length)와 feature(input_dim)로 나눠 쓴 것. 파라미터 120개로 동일
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

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
          validation_split=0.2,   # val_loss를 모니터하려면 검증 데이터가 있어야 es가 동작함
          callbacks=[es],
          )

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)   # loss :  6.4963906649950776e-12

x_predict = np.array([8, 9, 10]).reshape(1, 3, 1)  # x가 (7, 3, 1)이므로 모양을 맞춰줘야 함
y_predict = model.predict(x_predict)

print('Predict [8, 9, 10] : ', y_predict)   # Predict [8, 9, 10] :  [[10.776364]]