import os
import time
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# 1. 데이터
np_path = 'C:/study/_save/numpy/man-woman_npy/'

x_train = np.load(np_path + 'keras_04_x_train.npy')
y_train = np.load(np_path + 'keras_04_y_train.npy')
x_test = np.load(np_path + 'keras_04_x_test.npy')
y_test = np.load(np_path + 'keras_04_y_test.npy')

data_gen = ImageDataGenerator(
    # rescale 없음 : npy가 이미 0~1이라 또 나누면 255배 작아짐
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.2,
    fill_mode='nearest',
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

# woman(라벨 1)만 골라서 man과 같은 수가 되도록 증폭
woman_idx = np.where(y_train == 1)[0]
x_woman = x_train[woman_idx]
y_woman = y_train[woman_idx]

augment_size = int((y_train == 0).sum() - len(woman_idx))   # man 수 - woman 수
print('man :', int((y_train == 0).sum()), '/ woman :', len(woman_idx), '/ 증폭 :', augment_size)

randidx = np.random.choice(x_woman.shape[0], size=augment_size)

x_augmented = x_woman[randidx].copy()
y_augmented = y_woman[randidx].copy()

xy_augmented = data_gen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()

x_train = np.concatenate((x_train, xy_augmented[0]))
y_train = np.concatenate((y_train, xy_augmented[1]))

print(x_train.shape, y_train.shape)
print(np.unique(y_train, return_counts=True))

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape=(100, 100, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D())
model.add(Conv2D(200, (2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D())
model.add(Conv2D(100, (2, 2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(50, (2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(25, (2, 2), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.009), metrics=['acc'],)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   restore_best_weights=True,
                   patience=50,
                   )

save_path = 'C:/study/_save/keras52/'
os.makedirs(save_path, exist_ok=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=save_path + 'keras52_15_man_woman.keras'
)

start = time.time()
model.fit(x_train, y_train, epochs=500,
          batch_size=16,
          callbacks=[es, mcp],
          validation_split=0.1,
          )
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', loss[0])                           # loss :  0.3615022599697113 > 0.6868494153022766
print('acc : ', round(loss[1], 4))                  # acc :  0.876 > 0.6508
print('acc_score : ', acc_score)                    # acc_score :  0.8759661391240339 > 0.6507656065959952
print('time : ', round(end - start, 2), 'sec')      # time :  275.29 sec > 4725.68 sec