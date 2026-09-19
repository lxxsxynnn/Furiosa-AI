import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score

# 1. 데이터
# train data 수치화 준비
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,   # 제너레이터에는 fit의 validation_split이 안 먹으므로 여기서 나눔
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)   # 테스트 데이터는 스케일링만

path_train = 'C:/study/_data/image/catdog/training_set/'
path_test = 'C:/study/_data/image/catdog/test_set/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200, 200),     # 알아서 크기 조정 해줌
    batch_size=32,              # 제너레이터를 fit에 직접 넘기므로 이 값이 곧 훈련 배치 크기가 됨
    class_mode='binary',        # 이진분류
    color_mode='rgb',           # 컬러
    shuffle=True,
    subset='training',
)
# Found 6404 images belonging to 2 classes.

xy_val = train_datagen.flow_from_directory(
    path_train,                 # train과 같은 경로, validation_split으로 갈라진 나머지 20%
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    subset='validation',
)
# Found 1601 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,              # predict 결과와 정답 순서를 맞추려면 섞으면 안 됨
)
# Found 2023 images belonging to 2 classes.

print(len(xy_train), len(xy_val), len(xy_test))     # 201 51 64   <- 배치 개수
print(xy_train[0][0].shape, xy_train[0][1].shape)   # (32, 200, 200, 3) (32,)
print(xy_train.class_indices)                       # {'cats': 0, 'dogs': 1}

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(50, (2, 2), input_shape=(200, 200, 3), padding='same'))    # color_mode='rgb'라서 채널 3
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Conv2D(50, (2, 2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(50, (2, 2), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=5,
                   restore_best_weights=True,
                  )

start = time.time()
model.fit(xy_train, epochs=30,        # 제너레이터를 통째로 넘김 > 8005장 전부 사용
          callbacks=[es],               # batch_size는 제너레이터에서 정했으므로 여기 쓰지 않음
          validation_data=xy_val,       # 제너레이터에는 validation_split을 쓸 수 없음
          )
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(xy_test)

y_pred = model.predict(xy_test)
y_pred = np.round(y_pred)
y_test = xy_test.classes                # shuffle=False라서 predict 결과와 순서가 일치

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', loss[0])                           # loss :  0.4825635552406311
print('acc : ', round(loss[1], 4))                  # acc :  0.7761
print('acc_score : ', acc_score)                    # acc_score :  0.7760751359367276
print('time : ', round(end - start, 2), 'sec')      # time :   722.26 sec
