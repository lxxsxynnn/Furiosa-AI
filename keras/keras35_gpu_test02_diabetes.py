from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import time

# CPU / GPU 실행 시간 비교 - 당뇨
# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=45)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dropout(0.3))
model.add(Dense(6))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Dropout(0.2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

# es = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=20,
#                    restore_best_weights=True,
#                    verbose=1
#                    )

# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath=path + 'keras35_02_diabetes.keras',
#                       verbose=1,
                    #   )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=640, batch_size=16,
          validation_split=0.33,
        #   callbacks=[mcp],
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)

print("loss : " , loss)     # loss :  2598.738525390625
print("r2 : ", r2)          # r2 :  0.4910995855751583
print("mse : ", mse)        # mse :  2598.738440470018
print("RMSE : ", rmse)      # RMSE :  50.977823025998454

print('실행 시간: ', round(end_time - start_time, 2), '초')

'''
실행 시간:  47.52 초 >> 실행 시간:  33.84 초
'''