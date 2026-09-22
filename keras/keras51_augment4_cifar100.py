import os
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, GlobalAveragePooling2D, MaxPool2D
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
    # vertical_flip=True,   # 실사진이라 뒤집힌 사물은 없음. 없는 상황을 학습시키게 됨
    shear_range=20,     # 이 값은 비율이 아니라 각도. 0.7이면 0.7도라 변화가 거의 없다
    fill_mode='nearest',
)

# 검증셋을 원본에서 먼저 떼어낸다
# validation_split은 섞기 전 마지막 20%를 가져가는데, 증폭본을 뒤에 붙이면 검증셋이 전부 증폭본이 된다
# 테스트가 원본이므로 검증도 원본이어야 하고, 증폭은 훈련셋에만 적용한다
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

xy_augmented = data_gen.flow(   # rescale=1./255가 여기서 걸려 0~1이 된다
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()

# 증폭본이 0~1이므로 나머지도 맞춘다
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
model.add(Dropout(0.3))                         # loss 6.5 / acc 0.41은 과적합 신호라 규제 보강
model.add(Dense(units=256, activation='relu'))  # 출력층(100)보다 넉넉하게. 128 > 100은 너무 좁았음
model.add(Dense(100, activation='softmax'))

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

path = 'C:/study/_save/keras51/'
os.makedirs(path, exist_ok=True)    # 폴더가 없으면 ModelCheckpoint가 저장할 때 에러
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=path + 'keras51_04_cifar100.keras'
)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_data=(x_val, y_val),    # 원본에서 떼어낸 검증셋
          callbacks=[es, mcp]
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss : 6.500567436218262 > 2.025477886199951
print('acc : ', loss[1])                                    # acc : 0.4115999937057495 > 0.4609000086784363

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuray_score : ', acc_score)                        # accuray_score : 0.4116 > 0.4609
print('time : ', round(end_time - start_time, 2), 'sec')    # time : 1612.18 sec > 3126.68 sec