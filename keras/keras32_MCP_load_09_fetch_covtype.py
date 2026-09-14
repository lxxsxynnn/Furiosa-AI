from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)                 # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

y = pd.get_dummies(y, dtype=float).values
print(y.shape)  # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    stratify=y,
    random_state=12,
)

print(x_train.shape, x_test.shape)  # (464809, 54) (116203, 54)
print(y_train.shape, y_test.shape)  # (464809, 7) (116203, 7)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
save_path = 'C:/study/_save/keras31/'
model = load_model(save_path + 'k31_0914_1447_09_fetch_covtype_0189-0.1841.keras')

# 3. 컴파일, 훈련

# acc : 0.93
# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', round(result[1], 2))

y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)

'''
loss :  0.17887167632579803
acc :  0.93
acc_score :  0.9311377503162569
'''