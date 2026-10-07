from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np

# DNN 모델에 Conv1D 적용해보기 - 보스턴 주택 가격
# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()   #  tensorflow는 가져오면서 분할됨(훈련/테스트 묶음으로 나오니까 주의할 것)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(50, kernel_size=3, input_shape=(13, 1)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
learning_rate = 0.002
model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1
                   )

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1,
    factor=0.5,
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_split=0.25,
          callbacks=[es, rlr]
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

print("loss : ", loss)  # loss :  23.88810920715332 > 22.783327102661133
print("r2 : ", r2)      # r2 :  0.7130345270476302 > 0.7263061953447736
print("mse : ", mse)    # mse :  23.888110360859866 > 22.783325615527424
print("RMSE : ", rmse)  # RMSE :  4.887546456133165 > 4.773188202399673