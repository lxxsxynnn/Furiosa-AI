import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer     # 유방암 관련 데이터셋 불러오기(이진분류, y가 0/1)

# 2차원 데이터를 CNN 모델로 처리해보기 - 유방암
# 1. 데이터
datasets = load_breast_cancer()     # sklearn에서 제공하는 교육용 데이터셋, 실무에서 쓸 일이 없음
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=234,
    stratify=y,         # 범주형 데이터에서는 불균형하게 나눠지는 걸 막기 위한 기능이 있음 / y를 기준으로 동일한 비율로 데이터를 잘라줌
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)     # (398, 30)

x_train = x_train.reshape(-1, 10, 3, 1)     # 특성 30개 = 10 × 3
x_test = x_test.reshape(-1, 10, 3, 1)

print(x_train.shape, x_test.shape)  # (398, 10, 3, 1) (171, 10, 3, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape=(10, 3, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(200, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))     # 이진분류 출력층

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',     # loss는 w 갱신할 때 사용하는데 이진분류는 무조건 binary_crossentropy
            #   metrics=['accuracy'],
              metrics=['acc'],      # train data 지표
             )  

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=20,
                   restore_best_weights=True,
                  )

hist = model.fit(x_train, y_train, 
          epochs=1000,
          batch_size=32,
          verbose=1,
          validation_split=0.3,
          callbacks=[es,],
         )

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print('loss : ', loss[0])
print('acc : ', round(loss[1], 4))
print('acc_score : ', acc_score)

'''
loss :  0.10264232754707336
acc :  0.9532
acc_score :  0.9532163742690059

>>

loss :  0.11330772936344147
acc :  0.9708
acc_score :  0.9707602339181286
'''