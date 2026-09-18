import time
import numpy as np
from keras.datasets import cifar100
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, GlobalAveragePooling2D, MaxPool2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 함수형 모델로 바꿔보기 - cifar100 데이터셋
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

# model = Sequential()
# model.add(Conv2D(128, (4, 4), input_shape=(32, 32, 3)))
input1 = Input(shape=(32, 32, 3))
conv1 = Conv2D(128, (4, 4))(input1)
# model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
conv2 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
# model.add(Dropout(0.2))
dp1 = Dropout(0.2)(conv2)
# model.add(MaxPool2D())
mp1 = MaxPool2D()(dp1)
# model.add(Conv2D(512, kernel_size=(2, 2), activation='relu', padding='same'))
conv3 = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='same')(mp1)
# model.add(MaxPool2D())
mp2 = MaxPool2D()(conv3)
# model.add(Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'))
conv4 = Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same')(mp2)
# model.add(MaxPool2D())
mp3 = MaxPool2D()(conv4)
# model.add(GlobalAveragePooling2D())
gp1 = GlobalAveragePooling2D()(mp3)
# model.add(Dense(units=256, activation='relu'))
dense1 = Dense(units=256, activation='relu')(gp1)
# model.add(Dense(units=128, activation='relu'))
dense2 = Dense(units=128, activation='relu')(dense1)
# model.add(Dense(100, activation='softmax'))
output1 = Dense(100, activation='softmax')(dense2)

model = Model(input1, output1)

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
print('loss : ', loss[0])                                   # loss : 6.500567436218262 > 6.375938892364502
print('acc : ', loss[1])                                    # acc : 0.4115999937057495 > 0.41200000047683716

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuray_score : ', acc_score)                        # accuray_score : 0.4116 > 0.412
print('time : ', round(end_time - start_time, 2), 'sec')    # time : 1612.18 sec > 5331.01 sec