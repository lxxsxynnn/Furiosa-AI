from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np

save_path = 'C:/study/_save/keras31/'

# 최고 가중치 불러오기 - 보스턴 주택 가격
# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()   #  tensorflow는 가져오면서 분할됨(훈련/학습 묶음으로 나오니까 주의할 것)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = load_model(save_path + 'k31_0914_1350_03_boston_0022-27.9234.keras')
# 3. 컴파일, 훈련

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)      # loss :  22.366579055786133

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)          # r2 :  0.7313125397984366

mse = mean_squared_error(y_test, y_pred)
print("mse : ", mse)        # mse :  22.366578236188758

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)      # RMSE :  4.729331690227358