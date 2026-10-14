import numpy as np

from tensorflow.keras.models import Sequential
from keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score

# 1. 데이터
np_path = 'C:/study/_save/numpy/horse_npy/'

x_train = np.load(np_path + 'keras_01_x_train.npy')
y_train = np.load(np_path + 'keras_01_y_train.npy')
x_test = np.load(np_path + 'keras_01_x_test.npy')
y_test = np.load(np_path + 'keras_01_y_test.npy')

print(x_train.shape, y_train.shape) # (770, 100, 100, 3) (770,)
print(x_test.shape, y_test.shape)   # (257, 100, 100, 3) (257,)

# 2. 모델 구성
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(100, 100, 3),
)

vgg16.trainable=False

model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   restore_best_weights=True,
                   patience=50,
                   )

model.fit(x_train, y_train, epochs=500,
          batch_size=32,
          callbacks=[es],
          validation_split=0.2,
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = np.round(model.predict(x_test))     # 확률 > 0/1 라벨

acc_score = accuracy_score(y_test, y_predict)

print('loss : ', loss[0])                           # loss :  0.002236940898001194 > 0.043470293283462524
print('acc : ', round(loss[1], 4))                  # acc :  1.0 > 0.9922
print('acc_score : ', acc_score)                    # acc_score :  1.0 > 0.9922178988326849