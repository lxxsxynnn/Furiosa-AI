import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터
a = np.array(range(1, 11))
size = 5    # timestep size

print(a.shape)  # (10,)
print(len(a) - size + 1)

def split_x(dataset, size):
    arr = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i: (i + size)]
        arr.append(subset)
    return np.array(arr)

arr2 = split_x(a, size)
print(arr2)
'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
'''
print(arr2.shape)   # (6, 5)

x = arr2[:, :-1]
y = arr2[:, -1]

print(x.shape)  # (6, 4)
print(y.shape)  # (6,)

x = x.reshape(-1, 4, 1)
y = y.reshape(-1, 1)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape=(4, 1)))
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
          batch_size=3,
          )

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)   # loss :  0.0004233139625284821

x_predict = np.array([[7, 8, 9, 10]])
x_predict = x_predict.reshape(-1, 4, 1)
y_predict = model.predict(x_predict)

print('Predict [7, 8, 9, 10] : ', y_predict) # Predict [7, 8, 9, 10] :  [[10.900864]]