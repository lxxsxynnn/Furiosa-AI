from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np

# 2차원 데이터를 CNN 모델로 처리해보기 - 당뇨
# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=45)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train)) # -0.13776722569000302 0.19878798965729408

x_train = x_train.reshape(-1, 5, 2, 1)
x_test = x_test.reshape(-1, 5, 2, 1)

print(x_train.shape, x_test.shape)  # (331, 5, 2, 1) (111, 5, 2, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape=(5, 2, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(200, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
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

hist = model.fit(x_train, y_train, epochs=640, batch_size=16,
          validation_split=0.2,
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

print("loss : " , loss)
print("r2 : ", r2)
print("mse : ", mse)
print("RMSE : ", rmse)

'''
loss :  3365.30712890625
r2 :  0.34098554526269853
mse :  3365.3071363417903
RMSE :  58.01126732232102

>>

loss :  2748.328369140625
r2 :  0.4618060145619539
mse :  2748.328275519947
RMSE :  52.424500717889025
'''