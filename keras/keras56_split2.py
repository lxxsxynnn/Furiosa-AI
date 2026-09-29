import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# 시계열 데이터 가공해보기 - 2개 이상의 데이터에서 하나의 값만 알고 싶을 때
# 1. 데이터
a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]).T

size = 5

print(a.shape)  # (10, 2)

def split(dataset, size):
    arr = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : i + size]
        arr.append(subset)
    return np.array(arr)

arr2 = split(a, size)
print(arr2)
'''
[
 [[ 1  9]  [ 2  8]  [ 3  7]  [ 4  6]  [ 5  5]]
 [[ 2  8]  [ 3  7]  [ 4  6]  [ 5  5]  [ 6  4]]
 [[ 3  7]  [ 4  6]  [ 5  5]  [ 6  4]  [ 7  3]]
 [[ 4  6]  [ 5  5]  [ 6  4]  [ 7  3]  [ 8  2]]
 [[ 5  5]  [ 6  4]  [ 7  3]  [ 8  2]  [ 9  1]]
 [[ 6  4]  [ 7  3]  [ 8  2]  [ 9  1]  [10  0]]
]
'''
print(arr2.shape)   # (6, 5, 2)

x = arr2[:, :-1]    # arr2[:, :-1, :]도 가능
y = arr2[:, -1, 1]  # arr2[:, -1, -1]도 가능

print(x.shape)  # (6, 4, 2)
print(y.shape)  # (6,)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape=((4, 2))))
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
print('loss : ', results)   # loss :  4.349492428445956e-07

x_predict = np.array([[[7, 3], [8, 2], [9, 1], [10, 0]]])
y_predict = model.predict(x_predict)

print('Predict [[7, 3], [8, 2], [9, 1], [10, 0]] : ', y_predict)   # Predict [[7, 3], [8, 2], [9, 1], [10, 0]] :  [[-0.9234167]]