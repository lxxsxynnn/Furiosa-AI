from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
import numpy as np

# 1. 데이터
datasets = load_digits()
x = datasets['data']
y = datasets['target']
print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y.reshape(-1 , 1))
print(y.shape)  # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8,
    random_state=100,
    shuffle=True,
    stratify=y,
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2. 모델 구성
save_path = 'C:/study/_save/keras31/'
model = load_model(save_path + 'k31_0914_1455_10_digits_0093-0.0492.keras')

# 3. 컴파일, 훈련

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)

'''
loss :  0.2014123499393463
acc :  0.9722222089767456
acc_score :  0.9722222222222222
'''