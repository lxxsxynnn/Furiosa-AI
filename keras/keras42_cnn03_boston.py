from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np

# 2차원 데이터를 CNN 모델로 처리해보기 - 보스턴 주택 가격
# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()   #  tensorflow는 가져오면서 분할됨(훈련/학습 묶음으로 나오니까 주의할 것)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)

print(x_train.shape, x_test.shape)  # (404, 13, 1, 1) (102, 13, 1, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(13, 1, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1
                   )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_split=0.25,
          callbacks=[es, ]
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)

print("loss : ", loss)
print("r2 : ", r2)
print("mse : ", mse)
print("RMSE : ", rmse)

'''
loss :  21.785381317138672
r2 :  0.7382944060533468
mse :  21.785380819279506
RMSE :  4.6674812071693985

>>

loss :  99.16671752929688
r2 :  -0.19127983433965978
mse :  99.16671807446318
RMSE :  9.958248745359958
'''