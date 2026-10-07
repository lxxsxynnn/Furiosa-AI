import time
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv1D, Dense, GlobalAveragePooling1D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# CNN 모델에 Conv1D 적용해보기 - 성별 구분
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
x_train = x_train.reshape(-1, 100, 100 * 3)
x_val = x_val.reshape(-1, 100, 100 * 3)
x_test = x_test.reshape(-1, 100, 100 * 3)

print(x_train.shape, y_train.shape)
print(np.unique(y_train, return_counts=True))

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(100, kernel_size=5, input_shape=(100, 100 * 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.009), metrics=['acc'],)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   restore_best_weights=True,
                   patience=50,
                   )

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5
)

start = time.time()
model.fit(x_train, y_train, epochs=500,
          batch_size=16,
          callbacks=[es, rlr],
          validation_data=(x_val, y_val),    # validation_split은 뒤쪽 10%라 증폭한 woman만 들어감
          )
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', loss[0])                           # loss :  0.6724660396575928 > 0.6577674150466919
print('acc : ', round(loss[1], 4))                  # acc :  0.6508 > 0.6508
print('acc_score : ', acc_score)                    # acc_score :  0.6507656065959952 > 0.6507656065959952
print('time : ', round(end - start, 2), 'sec')      # time :  5018.35 sec > 1263.14 sec

'''
cf) GAP 적용 시
loss :  0.6577674150466919
acc :  0.4928
acc_score :  0.4927856301531213
time :  298.16 sec
'''