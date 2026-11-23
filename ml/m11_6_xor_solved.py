import numpy as np
# from sklearn.linear_model import Perceptron
# from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)

# 1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

# 2. 모델 구성
# model = Perceptron()
# model = LinearSVC()
model = DecisionTreeClassifier()

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
y_predict = model.predict(x_data)

result = model.score(x_data, y_data)
print("model.score: ", result)  # model.score:  1.0

acc = accuracy_score(y_data, np.round(y_predict))
print("accuracy score: ", acc)  # accuracy score:  1.0