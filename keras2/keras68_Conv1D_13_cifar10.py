import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# CNN 모델에 Conv1D 적용해보기 - cifar10 데이터셋
# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest',
)

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train,
)
print(x_train.shape, x_val.shape)   # (45000, 32, 32, 3) (5000, 32, 32, 3)

augment_size = 25000
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 45000개 중에 25000개 랜덤뽑기

x_augmented = x_train[randidx].copy()    # (25000, 32, 32, 3)
y_augmented = y_train[randidx].copy()    # (25000, 1)

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

x_train = x_train.reshape(-1, 32, 32 * 3)
x_test = x_test.reshape(-1, 32, 32 * 3)
x_val = x_val.reshape(-1, 32, 32 * 3)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_val = ohe.transform(y_val)
y_test = ohe.transform(y_test)

print(y_train.shape, y_val.shape, y_test.shape)     # (70000, 10) (5000, 10) (10000, 10)

# 2. 모델 구성
model = Sequential()
# model.add(LSTM(64, input_shape=(32, 32 * 3)))
model.add(Conv1D(32, kernel_size=4, input_shape=(32, 32 * 3)))
model.add(GlobalAveragePooling1D())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.019),
              metrics=['acc']
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=20,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=40,
    factor=0.5,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=150, batch_size=128,
          verbose=1,
          callbacks=[es, rlr],
          validation_data=(x_val, y_val),
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss :  1.6344542503356934 > 1.9260480403900146
print('acc : ', loss[1])                                    # acc :  0.3849000036716461 > 0.3075000047683716

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)                       # accuracy_score :  0.3849 > 0.3075
print('time : ', round(end_time - start_time, 2), 'sec')    # time :  258.38 sec > 361.06 sec

'''
cf) Flatten 적용 시
loss :  1.9438347816467285
acc :  0.2994000017642975
accuracy_score :  0.2994
time :  390.52 sec
'''