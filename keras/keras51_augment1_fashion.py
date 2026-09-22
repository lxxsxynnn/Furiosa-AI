from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 데이터 증폭하기 - fashion 데이터셋
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

# 흑백이라 채널 축이 없으므로 먼저 4차원으로 맞춘다
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 검증셋을 원본에서 먼저 떼어낸다
# validation_split은 섞기 전 마지막 20%를 가져가는데, 증폭본을 뒤에 붙이면 검증셋이 전부 증폭본이 된다
# 테스트가 원본이므로 검증도 원본이어야 하고, 증폭은 훈련셋에만 적용한다
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

x_augmented = x_train[randidx].copy()   # 메모리가 별도로 할당된 변수 생성
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28, 1) (40000,)

xy_augmented = data_gen.flow(   # rescale=1./255가 여기서 걸려 0~1이 된다
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()

print(xy_augmented[0].shape)    # (40000, 28, 28, 1)  증폭을 거친 쪽

# 증폭본이 0~1이므로 나머지도 맞춘다
x_train = x_train / 255.
x_val = x_val / 255.
x_test = x_test / 255.

# 증폭한 데이터랑 합치기
# x_augmented는 flow에 넣기 전의 원본이므로 증폭을 거친 xy_augmented를 붙인다
x_train = np.concatenate((x_train, xy_augmented[0]))
y_train = np.concatenate((y_train, xy_augmented[1]))
print(x_train.shape, y_train.shape)     # (88000, 28, 28, 1) (88000,)

print(np.unique(y_train, return_counts=True))   # 원핫 전에 찍어야 클래스 분포가 보인다

# 원핫 인코딩 - 합친 뒤에 한 번만 한다
ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train.reshape(-1, 1))     # OneHotEncoder는 2차원을 받는다
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
model.add(Dense(units=64, activation='relu'))   # 출력층(10)보다 넓게
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))      # 클래스 10개 > 노드 10개

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'],
              )

es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=40,
                   restore_best_weights=True,
                   )

path = 'C:/study/_save/keras51/'
os.makedirs(path, exist_ok=True)    # 폴더가 없으면 ModelCheckpoint가 저장할 때 에러
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=path + 'keras51_01_fashion.keras'
)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_data=(x_val, y_val),    # 원본에서 떼어낸 검증셋
          callbacks=[es, mcp],      # 만들어만 두고 안 넘기면 체크포인트가 저장되지 않음
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])                                   # loss : 0.26139315962791443 > 0.22093379497528076
print('acc : ', loss[1])                                    # acc : 0.9081000089645386 > 0.9241999983787537

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)    # (10000, 10) 확률 > 가장 큰 자리의 번호
y_test = np.argmax(y_test, axis=1)          # y_test도 원핫이므로 같이 번호로 되돌린다

acc_score = accuracy_score(y_test, y_predict)
print('accuarcy_score : ', acc_score)                       # accuracy_score : 0.9081 > 0.9242
print('time : ', round(end_time - start_time, 2), 'sec')    # time : 261.3 sec > 275.29 sec


##################### 수정 내역 #####################
# 1) 원핫 인코딩 추가 (keras40에서 안 가져온 블록)
#    y는 (N,) 번호인데 뒤쪽 코드가 원핫 전제라 ValueError와 AxisError가 났다
#
# 2) 출력층  Dense(1) > Dense(10)
#    노드 1개에 softmax는 항상 1.0이라 기울기가 0. 학습 자체가 안 된다
#
# 3) concatenate에 x_augmented > xy_augmented[0]
#    증폭 전 원본을 붙이고 있었다. 에러가 안 나서 증폭이 통째로 무효였다

# 4) 원본에도 /255.
#    증폭본만 rescale이 걸려서 한 배열에 0~255와 0~1이 섞였다