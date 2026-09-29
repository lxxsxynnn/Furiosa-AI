import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 시계열 데이터 가공해보기 - 같은 수열을 특성 2개로 묶어서 자르기
# 1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))

size = 6

# data를 reshape한 후, split함수로 시계열 데이터로 반환
# (N, 10, 1) -> (N, 5, 2)

a = a.reshape(-1, 2)

def split(dataset, size):
    arr = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : i + size]
        arr.append(subset)
    return np.array(arr)

arr1 = split(a, size)

x = arr1[:, :-1]
y = arr1[:, -1, 1]

print(x.shape)  # (45, 5, 2)
print(y.shape)  # (45,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=342,
                                                    )

# 2. 모델 구성
model = Sequential()
model.add(LSTM(100, input_shape=(5, 2)))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
)

model.fit(x_train, y_train, epochs=1000,
          callbacks=[es],
          batch_size=5,
          validation_split=0.1,
          )

# 4. 평가, 예측
# loss: 0.1 이하, result = [107]의 근사치
result = model.evaluate(x_test, y_test)
print('loss : ', result)    # loss :  0.06397267431020737

arr2 = x_predict.reshape(1, 5, 2)
print(arr2)
y_predict = model.predict(arr2)

print('Predict[[96, 97], [98, 99], [100, 101], [102, 103], [104, 105]] :\n', y_predict)
'''
Predict[[96, 97], [98, 99], [100, 101], [102, 103], [104, 105]] :
 [[102.62149]]
 '''