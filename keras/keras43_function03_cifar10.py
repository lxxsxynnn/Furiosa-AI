import time
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, GlobalAveragePooling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 함수형 모델로 바꿔보기 - cifar10 데이터셋
# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(y_train.shape, y_test.shape)      # (50000, 100) (10000, 100)

# model = Sequential()
# model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
input1 = Input(shape=(32, 32, 3))
conv1 = Conv2D(64, (3, 3), activation='relu')(input1)
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
# model.add(Dropout(0.2))
dp1 = Dropout(0.2)(conv2)
# model.add(BatchNormalization())
bn1 = BatchNormalization()(dp1)
# model.add(MaxPool2D())
mp1 = MaxPool2D()(bn1)
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(mp1)
# model.add(MaxPool2D())
mp2 = MaxPool2D()(conv3)
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
conv4 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(mp2)
# model.add(GlobalAveragePooling2D())
gp1 = GlobalAveragePooling2D()(conv4)
# model.add(Dense(units=128, activation='relu'))
dense1 = Dense(units=128, activation='relu')(gp1)
# model.add(Dropout(0.2))
dp2 = Dropout(0.2)(dense1)
# model.add(Dense(10, activation='softmax'))
output1 = Dense(10, activation='softmax')(dp2)

model = Model(input1, output1)

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=20,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=150, batch_size=128,
          verbose=1,
          callbacks=[es],
          validation_split=0.2,
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss : 0.5716548562049866 > 0.5879287719726562
print('acc : ', loss[1])                                    # acc : 0.8070999979972839 > 0.8098999857902527

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuray_score : ', acc_score)                        # accuray_score : 0.8071 > 0.8099
print('time : ', round(end_time - start_time, 2), 'sec')    # time : 136.29 sec > 540.18 sec