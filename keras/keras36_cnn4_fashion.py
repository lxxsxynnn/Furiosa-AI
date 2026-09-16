import time
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# cnn 실습해보기 - fashion_mnist 데이터셋
# (28, 28, 1) > 흑백 데이터
# acc : 0.92

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(112, (4, 4), input_shape=(28, 28, 1)))
model.add(Conv2D(112, (4, 4), activation='relu'))
model.add(Conv2D(224, (4, 4), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, (2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(112, (2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(56, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(units=28, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=14, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'],
              )

es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=40,
                   restore_best_weights=True,
                   )

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()

# 4. 평가, 예측
print('================ model.evaluate ================')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss :  0.042573001235723495
print('acc : ', loss[1])                                    # acc :  0.9898999929428101

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuarcy_score : ', acc_score)                       # accuarcy_score :  0.9899
print('time : ', round(end_time - start_time, 2), 'sec')    # time :  610.07 sec (CPU) / 141.17 sec (GPU)
