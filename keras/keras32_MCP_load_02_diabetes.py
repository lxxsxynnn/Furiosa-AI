from sklearn.datasets import load_diabetes
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np

save_path = 'C:/study/_save/keras31/'

# 최고 가중치 불러오기 - 당뇨
# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=45)

print(np.min(x_train), np.max(x_train)) # -0.13776722569000302 0.19878798965729408

# 2. 모델 구성
model = load_model(save_path + 'k31_0914_1342_02_diabetes_0198-3227.6018.keras')

# 3. 컴파일, 훈련


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : " , loss)     # loss :  2598.738525390625

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)          # r2 :  0.4910995855751583

mse = mean_squared_error(y_test, y_pred)
print("mse : ", mse)        # mse :  2598.738440470018

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)      # RMSE :  50.977823025998454