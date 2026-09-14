from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# 1 . 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48]))

y = pd.get_dummies(y, dtype=float).values
print(y.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8,
    random_state=100,
    shuffle=True,
    stratify=y
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
save_path = 'C:/study/_save/keras31/'
model = load_model(save_path + 'k31_0914_1444_08_wine_0004-0.1689.keras')

# 3. 컴파일, 훈련

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', round(result[1], 2))

y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('acc_score : ', acc_score)

'''
loss :  0.1525217890739441
acc :  0.97
acc_score :  0.9722222222222222
'''