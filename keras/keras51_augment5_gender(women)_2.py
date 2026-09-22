import os
import time
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

np_path = 'C:/study/_save/numpy/man-woman_npy/'

start1 = time.time()
x_train = np.load(np_path + 'keras_04_x_train.npy')
y_train = np.load(np_path + 'keras_04_y_train.npy')

# npy는 keras46_03에서 이미 0.75로 나눠 저장한 것이라 train 쪽만 쓰면 20375장뿐이다
# 강사님은 전체 27167장에서 시작하시므로 test 쪽을 도로 합쳐 출발점을 맞춘다
x_train = np.concatenate((x_train, np.load(np_path + 'keras_04_x_test.npy')))
y_train = np.concatenate((y_train, np.load(np_path + 'keras_04_y_test.npy')))
print(x_train.shape, y_train.shape)     # (27167, 100, 100, 3) (27167,)

x_train_women = x_train[np.where(y_train > 0.0)]
y_train_women = y_train[np.where(y_train > 0.0)]

print(x_train_women.shape, y_train_women.shape)     # (7117, 100, 100, 3) (7117,)
print(np.unique(y_train_women, return_counts=True)) # (array([1.], dtype=float32), array([7117], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.1, random_state=231,
)

end1 = time.time()

print('data handling time: ', round(end1 - start1, 2))  # 1.33

train_datagen = ImageDataGenerator(
    # rescale=1./255,   # 이 npy는 저장할 때 이미 나눠서 0~1이라 또 나누면 255배 작아짐
    horizontal_flip=True,
    rotation_range=10,      # 각도 단위. 0.1이면 0.1도라 변화가 없다
    zoom_range=0.2,
    fill_mode='nearest',
)

augment_size = 8000

print(len(x_train_women))                       # 7117                
print(np.unique(y_train, return_counts=True))   # (array([0., 1.], dtype=float32), array([11938,  6399], dtype=int64))

randidx = np.random.choice(x_train_women.shape[0], size = augment_size)
print(randidx)                          # [5910 6745 2115 ... 3950 7089 1918]
print(len(randidx))                     # 8000
print(np.min(randidx), np.max(randidx)) # 1 7115

x_augmented = x_train_women[randidx].copy()
y_augmented = y_train_women[randidx].copy()

print(x_augmented.shape)    # (8000, 100, 100, 3)
print(y_augmented.shape)    # (8000,)

x_train = x_train.reshape(24450, 100, 100, 3)
x_test = x_test.reshape(2717, 100, 100, 3)

print(x_train.shape, x_test.shape)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

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
model.add(Dense(64, activation='relu'))     # 출력층(1)보다 넉넉하게. 10 > 5 > 1은 너무 좁았음
model.add(Dense(1, activation='sigmoid'))
# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   restore_best_weights=True,
                   patience=50,
                   )

save_path = 'C:/study/_save/keras51/'
os.makedirs(save_path, exist_ok=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=save_path + 'keras51_05_gender_2.keras'
)

model.fit(x_train, y_train, epochs=500,
          batch_size=16,
          callbacks=[es, mcp],
          validation_split=0.1,
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', loss[0])                           # 0.3615022599697113
print('acc : ', round(loss[1], 4))                  # 0.876
print('acc_score : ', acc_score)                    # 0.8759661391240339