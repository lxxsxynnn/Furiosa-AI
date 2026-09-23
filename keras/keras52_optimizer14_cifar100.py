import os
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score


# 데이터 증폭하기 - cifar100 데이터셋
# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    shear_range=20,
    fill_mode='nearest',
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)
print(x_train.shape, x_val.shape)   # (40000, 32, 32, 3) (10000, 32, 32, 3)

augment_size=25000

randidx = np.random.randint(x_train.shape[0], size=augment_size)     # 40000개 중에 25000개

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

xy_augmented = data_gen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()

x_train = x_train / 255.
x_val = x_val / 255.
x_test = x_test / 255.

x_train = np.concatenate((x_train, xy_augmented[0]))
y_train = np.concatenate((y_train, xy_augmented[1]))

print(np.unique(y_train, return_counts=True))   # 원핫 전에 찍어야 클래스 분포가 보임

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_val = ohe.transform(y_val)
y_test = ohe.transform(y_test)

print(y_train.shape, y_val.shape, y_test.shape)     # (65000, 100) (10000, 100) (10000, 100)

model = Sequential()
model.add(Conv2D(128, (2, 2), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPool2D())
model.add(Conv2D(512, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.009),
              metrics=['acc']
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=20,
)

path = 'C:/study/_save/keras52/'
os.makedirs(path, exist_ok=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=path + 'keras52_14_cifar100.keras'
)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[es, mcp]
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss : 2.025477886199951 > 3.5501203536987305
print('acc : ', loss[1])                                    # acc : 0.4609000086784363 > 0.1632000058889389

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuray_score : ', acc_score)                        # accuray_score : 0.4609 > 0.1632
print('time : ', round(end_time - start_time, 2), 'sec')    # time : 3126.68 sec > 5220.75 sec