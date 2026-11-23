import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)

# XOR을 Dense 한 층으로 풀어 보기
# 1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

# 2. 모델 구성
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(1, input_dim = 2, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_data, y_data, epochs=100000)   # 10만 번을 돌려도 경계가 직선 하나라 그대로

# 4. 평가, 예측
y_predict = model.predict(x_data)

result = model.evaluate(x_data, y_data)
print("acc: ", result[1])  # acc:  0.5

acc = accuracy_score(y_data, np.round(y_predict))
print("accuracy score: ", acc)  # accuracy score:  0.5