# 30-1 참조
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import time

# 2차원 데이터를 CNN 모델로 처리해보기 - 캘리포니아 주택
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=121  )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# exit()

x_train = x_train.reshape(-1, 4, 2, 1)
x_test = x_test.reshape(-1, 4, 2, 1)

print(x_train.shape, x_test.shape)  # (16512, 4, 2, 1) (4128, 4, 2, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(4, 2, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation='relu', padding='same'))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(32, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# exit()

from tensorflow.keras.callbacks import EarlyStopping
# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1,
                   )

start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs=1000, batch_size=32,
          validation_split=0.2,
          callbacks=[es],  
         )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test, batch_size = 32)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print("loss : ", loss)
print("r2: ", r2)
print("mse: ", mse)
print("RMSE : ", rmse)

'''
loss :  0.34304144978523254
r2:  0.741588686569804
mse:  0.34304158602758045
RMSE :  0.5856975209334426

>>

loss :  0.2565971612930298
r2:  0.8067068591800932
mse:  0.2565970688935211
RMSE :  0.5065541125028215
'''