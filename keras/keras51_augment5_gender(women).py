import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. 데이터
np_path = 'C:/study/_save/numpy/man-woman_npy/'

x_train = np.load(np_path + 'keras_04_x_train.npy')
y_train = np.load(np_path + 'keras_04_y_train.npy')
x_test = np.load(np_path + 'keras_04_x_test.npy')
y_test = np.load(np_path + 'keras_04_y_test.npy')

data_gen = ImageDataGenerator(
    # rescale 없음 : npy가 이미 0~1이라 또 나누면 255배 작아진다
    horizontal_flip=True,
    rotation_range=10,      # 각도 단위. 0.1이면 0.1도라 변화가 없다
    zoom_range=0.2,
    fill_mode='nearest',
)

# 검증셋을 증폭 전에 떼어낸다. 뒤에 붙인 증폭본이 검증셋을 다 차지하는 것을 막는다
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
y_train = np.concatenate((y_train, xy_augmented[1]))   # x가 아니라 y를 붙인다

print(x_train.shape, y_train.shape)
print(np.unique(y_train, return_counts=True))   # man과 woman이 같은 수여야 함

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
    filepath=save_path + 'keras51_05_gender.keras'    # keras47_03을 덮어쓰지 않도록 이름을 나눔
)

model.fit(x_train, y_train, epochs=500,
          batch_size=16,
          callbacks=[es, mcp],
          validation_data=(x_val, y_val),   # 원본에서 떼어낸 검증셋
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', loss[0])                           # 0.20624126493930817 > 0.2128089964389801
print('acc : ', round(loss[1], 4))                  # 0.9134 > 0.9168
print('acc_score : ', acc_score)                    # 0.9134275618374559 > 0.916813898704358