import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12],
              [20, 30, 40], [30, 40, 50], [40, 50, 60]
              ])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x_predict = np.array([50, 60, 70])

x = x.reshape(-1, 3, 1)
y = y.reshape(-1, 1)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(3, 1), return_sequences=True))
model.add(LSTM(5))
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')

es = EarlyStopping(
    monitor='loss',
    mode='auto',
    restore_best_weights=True,
    patience=200,
)

model.fit(x, y, epochs=7500,
          callbacks=[es],
          batch_size=16,
          )

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)   # loss :  0.0027872261125594378

x_predict = x_predict.reshape(-1, 3, 1)
y_predict = model.predict(x_predict)

print('Predict [50, 60, 70] : ', y_predict) # Predict [50, 60, 70] :  [[79.68305]]