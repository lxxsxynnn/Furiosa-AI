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
model.add(SimpleRNN(units=10, input_shape=(3, 1)))
# model.add(SimpleRNN(10, input_shape=(3, 1)))  # 같은 표현
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결 가능
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
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
          callbacks=[es],
          )

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)   # loss :  6.4963906649950776e-12

x_predict = np.array([8, 9, 10]).reshape(1, 3, 1)  # x가 (7, 3, 1)이므로 모양을 맞춰줘야 함
y_predict = model.predict(x_predict)

print('Predict [8, 9, 10] : ', y_predict)   # Predict [8, 9, 10] :  [[10.776364]]