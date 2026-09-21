import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score

# 1. 데이터
np_path = 'C:/study/_save/numpy/man-woman_npy/'

x_train = np.load(np_path + 'keras_04_x_train.npy')
y_train = np.load(np_path + 'keras_04_y_train.npy')
x_test = np.load(np_path + 'keras_04_x_test.npy')
y_test = np.load(np_path + 'keras_04_y_test.npy')

print(x_train.shape, y_train.shape) # (20375, 100, 100, 3) (20375,)
print(x_test.shape, y_test.shape)   # (6792, 100, 100, 3) (6792,)

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
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 클래스 2개 > 노드 1개. 값은 woman일 확률

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   restore_best_weights=True,
                   patience=50,
                   )

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=np_path + 'keras47_03_gender.keras'
)

model.fit(x_train, y_train, epochs=500,
          batch_size=16,
          callbacks=[es, mcp],
          validation_split=0.2,
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)

print('loss : ', loss[0])                           # 0.20624126493930817
print('acc : ', round(loss[1], 4))                  # 0.9134
print('acc_score : ', acc_score)                    # 0.9134275618374559