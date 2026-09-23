from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ReduceLR - fashion 데이터셋
# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

data_gen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,   # 평행이동
    height_shift_range=0.1,  # 평행이동
    rotation_range=5,       # 각도 조절(입력한 값만큼 이미지 회전)
    # zoom_range=0.2,
    shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동
    fill_mode='nearest',    # 데이터를 옮기면 빈 공간이 생기게 되는데 그 근처 값으로 채우겠다는 의미
)

# 흑백이라 채널 축이 없으므로 먼저 4차원으로 맞춤
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)
print(x_train.shape, x_val.shape)   # (48000, 28, 28, 1) (12000, 28, 28, 1)

augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 48000개 중에 40000개 랜덤뽑기
print(randidx.shape)    # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28, 1) (40000,)

xy_augmented = data_gen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()

print(xy_augmented[0].shape)

x_train = x_train / 255.
x_val = x_val / 255.
x_test = x_test / 255.

x_train = np.concatenate((x_train, xy_augmented[0]))
y_train = np.concatenate((y_train, xy_augmented[1]))
print(x_train.shape, y_train.shape)     # (88000, 28, 28, 1) (88000,)

print(np.unique(y_train, return_counts=True))   # 원핫 전에 찍어야 클래스 분포가 보임

# 원핫 인코딩 - 합친 뒤에 한 번만 실행
ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train.reshape(-1, 1))     # OneHotEncoder는 2차원을 받기 때문에
y_val = ohe.transform(y_val.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))           # test는 fit 없이 transform만

print(y_train.shape, y_val.shape, y_test.shape)     # (88000, 10) (12000, 10) (10000, 10)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2, 2), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0099),
              metrics=['acc'],
              )

es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=40,
                   restore_best_weights=True,
                   )

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    factor=0.55,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[es, rlr],
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss : 0.29148000478744507 > 0.3015903830528259
print('acc : ', loss[1])                                    # acc : 0.892799973487854 > 0.902899980545044

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuarcy_score : ', acc_score)                       # accuracy_score : 0.8928 > 0.9029
print('time : ', round(end_time - start_time, 2), 'sec')    # time : 474.38 sec > 1297.43 sec