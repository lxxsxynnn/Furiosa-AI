import time
import numpy as np
from keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# acc : 0.67
# cnn 실습해보기 - cifar100 데이터셋
# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(np.unique(y_train, return_counts=True))
print(y_train.shape, y_test.shape)

model = Sequential()
model.add(Conv2D(128, (8, 8), input_shape=(32, 32, 3)))
model.add(Conv2D(256, kernel_size=(4, 4), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(512, kernel_size=(2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']
              )

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_split=0.2,
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss :  18.023265838623047
print('acc : ', loss[1])                                    # acc :  0.17389999330043793

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuray_score : ', acc_score)                        # accuray_score :  0.1739
print('time : ', round(end_time - start_time, 2), 'sec')    # time :  2871.83 sec